import time
import itertools
import contextlib
import curses
import signal
import numpy
import pyaudio
import realtime_analysis as ra

RATE = 44100
HOP_LENGTH = 512
WIN_LENGTH = 512*4

PRE_MAX = int(0.03 * RATE / HOP_LENGTH)
POST_MAX = int(0.03 * RATE / HOP_LENGTH)
PRE_AVG = int(0.03 * RATE / HOP_LENGTH)
POST_AVG = int(0.03 * RATE / HOP_LENGTH)
WAIT = int(0.03 * RATE / HOP_LENGTH)
DELTA = 8.2e-06 * 20 # noise_power * 20
# frame resolution: 11.6 ms
# response delay: 30 ms
# fastest tempo: 2000 bpm

DISPLAY_FPS = 120
DISPLAY_DELAY = 0.03
KNOCK_VOLUME = HOP_LENGTH / RATE / 0.00017 # Dt / knock_max_energy
KNOCK_DELAY = 0.0
MUSIC_VOLUME = 0.5


class KnockConsole:
    def __init__(self, samplerate=RATE,
                       hop_length=HOP_LENGTH,
                       win_length=WIN_LENGTH,
                       pre_max=PRE_MAX,
                       post_max=POST_MAX,
                       pre_avg=PRE_AVG,
                       post_avg=POST_AVG,
                       wait=WAIT,
                       delta=DELTA,
                       display_fps=DISPLAY_FPS,
                       display_delay=DISPLAY_DELAY,
                       knock_volume=KNOCK_VOLUME,
                       knock_delay=KNOCK_DELAY
                       ):
        self.samplerate = samplerate
        self.hop_length = hop_length
        self.win_length = win_length

        self.pre_max = pre_max
        self.post_max = post_max
        self.pre_avg = pre_avg
        self.post_avg = post_avg
        self.wait = wait
        self.delta = delta

        self.display_fps = display_fps
        self.display_delay = display_delay
        self.knock_volume = knock_volume
        self.knock_delay = knock_delay

        self.closed = False

    def close(self):
        self.closed = True

    @ra.DataNode.from_generator
    def get_output_node(self, knock_game):
        sound_handler = knock_game.get_sound_handler(self.samplerate, self.hop_length)
        with contextlib.closing(self), sound_handler:
            yield
            while not self.closed:
                data = next(sound_handler)
                data *= MUSIC_VOLUME
                yield data

    @ra.DataNode.from_generator
    def get_input_node(self, knock_game):
        # use halfhann window
        window = numpy.sin(numpy.linspace(0, numpy.pi/2, self.win_length))**2

        knock_handler = knock_game.get_knock_handler()
        picker = ra.pick_peak(self.pre_max, self.post_max,
                              self.pre_avg, self.post_avg,
                              self.wait, self.delta)

        with picker:
            detector = ra.pipe(ra.frame(self.win_length, self.hop_length),
                               ra.power_spectrum(self.win_length, samplerate=self.samplerate,
                                                                  windowing=window,
                                                                  weighting=True),
                               ra.onset_strength(self.samplerate/self.win_length),
                               (lambda a: (None, a, a)),
                               ra.pair(itertools.count(-picker.delay), # generate index
                                       ra.delay([0.0]*picker.delay), # delay signal
                                       picker # pick peak
                                       ),
                               (lambda a: (a[0]*self.hop_length/self.samplerate-self.knock_delay,
                                           a[1]*self.knock_volume,
                                           a[2])),
                               knock_handler)

            with contextlib.closing(self), detector:
                while not self.closed:
                    detector.send((yield))

    @ra.DataNode.from_generator
    def get_screen_node(self, knock_game):
        stdscr = curses.initscr()
        knock_handler = knock_game.get_screen_handler(stdscr)

        try:
            curses.noecho()
            curses.cbreak()
            stdscr.nodelay(True)
            stdscr.keypad(1)
            curses.curs_set(0)

            with knock_handler:
                while not self.closed:
                    knock_handler.send((yield) - self.display_delay)

        finally:
            curses.endwin()

    def play(self, knock_game):
        def SIGINT_handler(sig, frame):
            self.close()

        with contextlib.closing(self), knock_game:
            signal.signal(signal.SIGINT, SIGINT_handler)

            output_node = self.get_output_node(knock_game)
            input_node = self.get_input_node(knock_game)
            screen_node = self.get_screen_node(knock_game)

            with screen_node, ra.record(input_node, self.hop_length, self.samplerate) as input_stream,\
                              ra.play(output_node, self.hop_length, self.samplerate) as output_stream:

                reference_time = time.time()
                input_stream.start_stream()
                output_stream.start_stream()

                while not self.closed:
                    signal.signal(signal.SIGINT, SIGINT_handler)
                    screen_node.send(time.time() - reference_time)
                    time.sleep(1/self.display_fps)
