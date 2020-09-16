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
        music_volume = float(self.config["controls"]["music_volume"])

        sound_handler = knock_game.get_sound_handler()

        with contextlib.closing(self), sound_handler:
            yield
            while True:
                data = sound_handler.send()
                data *= music_volume
                yield data

    @ra.DataNode.from_generator
    def get_input_node(self, knock_game):
        samplerate = int(self.config["input"]["samplerate"])
        hop_length = int(self.config["input"]["buffer"])
        Dt = hop_length / samplerate

        win_length = int(self.config["detector"]["win_length"])
        pre_max = float(self.config["detector"]["pre_max"])
        post_max = float(self.config["detector"]["post_max"])
        pre_avg = float(self.config["detector"]["pre_avg"])
        post_avg = float(self.config["detector"]["post_avg"])
        wait = float(self.config["detector"]["wait"])
        delta = float(self.config["detector"]["delta"])

        pre_max = round(pre_max / Dt)
        post_max = round(post_max / Dt)
        pre_avg = round(pre_avg / Dt)
        post_avg = round(post_avg / Dt)
        wait = round(wait / Dt)
        delay = max(post_max, post_avg)

        knock_delay = float(self.config["controls"]["knock_delay"])
        knock_volume = float(self.config["controls"]["knock_volume"])

        knock_handler = knock_game.get_knock_handler()

        # use halfhann window
        window = numpy.sin(numpy.linspace(0, numpy.pi/2, win_length))**2
        detector = ra.pipe(ra.frame(win_length, hop_length),
                           ra.power_spectrum(win_length, samplerate=samplerate, windowing=window, weighting=True),
                           ra.onset_strength(samplerate/win_length),
                           (lambda a: (None, a, a)),
                           ra.pair(itertools.count(-delay), # generate index
                                   ra.delay([0.0]*delay), # delay signal
                                   ra.pick_peak(pre_max, post_max, pre_avg, post_avg, wait, delta) # pick peak
                                   ),
                           (lambda a: (a[0]*hop_length/samplerate-knock_delay,
                                       a[1]*knock_volume,
                                       a[2])),
                           knock_handler)

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
            stdscr.keypad(1)
            curses.curs_set(0)

            with contextlib.closing(self), knock_handler:
                reference_time = time.time()

                while True:
                    yield
                    signal.signal(signal.SIGINT, self.SIGINT_handler)
                    t = time.time() - reference_time - display_delay
                    knock_handler.send(t)

        finally:
            curses.endwin()

    def play(self, knock_game):
        input_samplerate = int(self.config["input"]["samplerate"])
        input_buffer_length = int(self.config["input"]["buffer"])
        input_format = self.config["input"]["format"]
        output_samplerate = int(self.config["output"]["samplerate"])
        output_buffer_length = int(self.config["output"]["buffer"])
        output_format = self.config["output"]["format"]
        display_fps = int(self.config["controls"]["display_fps"])

        try:
            manager = pyaudio.PyAudio()

            with contextlib.closing(self), knock_game:
                knock_game.set_audio_params(input_samplerate, input_buffer_length)

                output_node = self.get_output_node(knock_game)
                input_node = self.get_input_node(knock_game)
                screen_node = self.get_screen_node(knock_game)

                with ra.record(manager, input_node, input_buffer_length, input_samplerate, input_format) as input_stream,\
                     ra.play(manager, output_node, output_buffer_length, output_samplerate, output_format) as output_stream:

                    input_stream.start_stream()
                    output_stream.start_stream()
                    ra.loop(screen_node, 1/display_fps, lambda: self.closed)

        finally:
            manager.terminate()

