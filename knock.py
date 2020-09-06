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

    def SIGINT_handler(self, sig, frame):
        self.close()

    @ra.DataNode.from_generator
    def get_output_node(self, knock_game):
        sound_handler = knock_game.get_sound_handler(self.samplerate, self.hop_length)
        with contextlib.closing(self), sound_handler:
            yield
            while True:
                data = sound_handler.send()
                data *= MUSIC_VOLUME
                yield data

    @ra.DataNode.from_generator
    def get_input_node(self, knock_game):
        # use halfhann window
        window = numpy.sin(numpy.linspace(0, numpy.pi/2, self.win_length))**2

        knock_handler = knock_game.get_knock_handler()
        delay = max(self.post_max, self.post_avg)
        detector = ra.pipe(ra.frame(self.win_length, self.hop_length),
                           ra.power_spectrum(self.win_length, samplerate=self.samplerate,
                                                              windowing=window,
                                                              weighting=True),
                           ra.onset_strength(self.samplerate/self.win_length),
                           (lambda a: (None, a, a)),
                           ra.pair(itertools.count(-delay), # generate index
                                   ra.delay([0.0]*delay), # delay signal
                                   ra.pick_peak(self.pre_max, self.post_max,
                                                self.pre_avg, self.post_avg,
                                                self.wait, self.delta) # pick peak
                                   ),
                           (lambda a: (a[0]*self.hop_length/self.samplerate-self.knock_delay,
                                       a[1]*self.knock_volume,
                                       a[2])),
                           knock_handler)

        with contextlib.closing(self), detector:
            while True:
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

            with contextlib.closing(self), knock_handler:
                reference_time = time.time()

                while True:
                    yield
                    signal.signal(signal.SIGINT, self.SIGINT_handler)
                    knock_handler.send(time.time() - reference_time - self.display_delay)

        finally:
            curses.endwin()

    def play(self, knock_game):
        try:
            session = pyaudio.PyAudio()

            with contextlib.closing(self), knock_game:
                output_node = self.get_output_node(knock_game)
                input_node = self.get_input_node(knock_game)
                screen_node = self.get_screen_node(knock_game)

                with ra.record(session, input_node, self.hop_length, self.samplerate) as input_stream,\
                     ra.play(session, output_node, self.hop_length, self.samplerate) as output_stream:

                    input_stream.start_stream()
                    output_stream.start_stream()
                    ra.loop(screen_node, 1/self.display_fps, lambda: self.closed)

        finally:
            session.terminate()
