import time
import itertools
import contextlib
import configparser
import curses
import signal
import numpy
import pyaudio
import realtime_analysis as ra


class KnockConsole:
    def __init__(self, config_filename=None):
        config = configparser.ConfigParser()
        if config_filename is not None:
            config.read(config_filename)

        default = configparser.ConfigParser()
        default.read("default.kconfig")
        for section in default:
            if section not in config:
                config.add_section(section)
            for key in default[section]:
                if key not in config[section]:
                    config.set(section, key, default[section][key])

        self.config = config
        self.closed = False

    def close(self):
        self.closed = True

    def SIGINT_handler(self, sig, frame):
        self.close()

    @ra.DataNode.from_generator
    def get_output_node(self, knock_game):
        sound_handler = knock_game.get_sound_handler()
        sound_samplerate = knock_game.samplerate
        sound_channels = knock_game.channels

        samplerate = int(self.config["output"]["samplerate"])
        buffer_length = int(self.config["output"]["buffer_length"])
        channels = int(self.config["output"]["channels"])

        if channels != sound_channels:
            sound_handler = ra.pipe(sound_handler, lambda a: numpy.tile(a.mean(axis=1, keepdims=True), (1, channels)))
        if samplerate != sound_samplerate:
            sound_handler = ra.resample(sound_handler, ratio=(samplerate, sound_samplerate))
        sound_handler = ra.chunk(sound_handler, chunk_shape=(buffer_length, channels))

        with contextlib.closing(self), sound_handler:
            yield
            while True:
                yield sound_handler.send()

    @ra.DataNode.from_generator
    def get_input_node(self, knock_game):
        samplerate = int(self.config["input"]["samplerate"])
        buffer_length = int(self.config["input"]["buffer_length"])
        channels = int(self.config["input"]["channels"])

        time_res = float(self.config["detector"]["time_res"])
        hop_length = round(samplerate*time_res)
        freq_res = float(self.config["detector"]["freq_res"])
        win_length = round(samplerate/freq_res)
        pre_max = float(self.config["detector"]["pre_max"])
        post_max = float(self.config["detector"]["post_max"])
        pre_avg = float(self.config["detector"]["pre_avg"])
        post_avg = float(self.config["detector"]["post_avg"])
        wait = float(self.config["detector"]["wait"])
        delta = float(self.config["detector"]["delta"])

        pre_max = round(pre_max / time_res)
        post_max = round(post_max / time_res)
        pre_avg = round(pre_avg / time_res)
        post_avg = round(post_avg / time_res)
        wait = round(wait / time_res)
        delay = max(post_max, post_avg)

        knock_delay = float(self.config["controls"]["knock_delay"])
        knock_energy = float(self.config["controls"]["knock_energy"])

        knock_handler = knock_game.get_knock_handler()

        window = ra.get_half_Hann_window(win_length)
        detector = ra.pipe(ra.frame(win_length=win_length, hop_length=hop_length),
                           ra.power_spectrum(win_length=win_length,
                                             samplerate=samplerate,
                                             windowing=window,
                                             weighting=True),
                           ra.onset_strength(1),
                           (lambda a: (None, a, a)),
                           ra.pair(itertools.count(-delay), # generate index
                                   ra.delay([0.0]*delay), # delay signal
                                   ra.pick_peak(pre_max, post_max, pre_avg, post_avg, wait, delta) # pick peak
                                   ),
                           (lambda a: (a[0]*time_res-knock_delay,
                                       a[1]/knock_energy,
                                       a[2])),
                           knock_handler)

        if buffer_length != hop_length:
            detector = ra.unchunk(detector, chunk_shape=(hop_length, channels))

        with contextlib.closing(self), detector:
            while True:
                detector.send((yield))

    @ra.DataNode.from_generator
    def get_screen_node(self, knock_game):
        display_delay = float(self.config["controls"]["display_delay"])

        stdscr = curses.initscr()
        knock_handler = knock_game.get_screen_handler(stdscr)

        try:
            curses.noecho()
            curses.cbreak()
            stdscr.nodelay(True)
            stdscr.idcok(False)
            stdscr.keypad(1)
            curses.curs_set(0)

            with contextlib.closing(self), knock_handler:
                reference_time = time.time()
                curses.ungetch(curses.KEY_RESIZE)

                while True:
                    yield
                    signal.signal(signal.SIGINT, self.SIGINT_handler)
                    t = time.time() - reference_time - display_delay
                    knock_handler.send(t)
                    stdscr.refresh()

        finally:
            curses.endwin()

    def play(self, knock_game):
        input_params = dict(samplerate=int(self.config["input"]["samplerate"]),
                            buffer_shape=(int(self.config["input"]["buffer_length"]),
                                          int(self.config["input"]["channels"])),
                            format=self.config["input"]["format"],
                            device=(int(self.config["input"]["device"]) if "device" in self.config["input"] else None)
                            )
        output_params = dict(samplerate=int(self.config["output"]["samplerate"]),
                             buffer_shape=(int(self.config["output"]["buffer_length"]),
                                           int(self.config["output"]["channels"])),
                             format=self.config["output"]["format"],
                             device=(int(self.config["output"]["device"]) if "device" in self.config["output"] else None)
                             )

        display_fps = int(self.config["controls"]["display_fps"])

        try:
            manager = pyaudio.PyAudio()

            with contextlib.closing(self), knock_game:
                output_node = self.get_output_node(knock_game)
                input_node = self.get_input_node(knock_game)
                screen_node = self.get_screen_node(knock_game)

                with ra.record(manager, input_node, **input_params) as input_stream,\
                     ra.play(manager, output_node, **output_params) as output_stream:

                    input_stream.start_stream()
                    output_stream.start_stream()
                    ra.loop(screen_node, 1/display_fps, lambda: self.closed)

        finally:
            manager.terminate()

