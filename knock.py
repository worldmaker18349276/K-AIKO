import time
import itertools
from contextlib import contextmanager
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

        self.pyaudio = pyaudio.PyAudio()
        self.stop = False

    def __del__(self):
        self.pyaudio.terminate()

    def get_output_stream(self, sound_handler):
        def output_callback(in_data, frame_count, time_info, status):
            data = next(sound_handler, None)
            if self.stop or data is None:
                return b'', pyaudio.paComplete
            data *= MUSIC_VOLUME
            return data.tobytes(), pyaudio.paContinue

        output_stream = self.pyaudio.open(format=pyaudio.paFloat32,
                                          channels=1,
                                          rate=self.samplerate,
                                          input=False,
                                          output=True,
                                          frames_per_buffer=self.hop_length,
                                          stream_callback=output_callback,
                                          start=False)

        return output_stream

    def get_input_stream(self, knock_handler):
        picker = ra.pick_peak(self.pre_max, self.post_max,
                              self.pre_avg, self.post_avg,
                              self.wait, self.delta)

        # use halfhann window
        window = numpy.sin(numpy.linspace(0, numpy.pi/2, self.win_length))**2
        detector = ra.pipe(ra.frame(self.win_length, self.hop_length),
                           ra.power_spectrum(self.samplerate, self.win_length, windowing=window, weighting=True),
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

        def input_callback(in_data, frame_count, time_info, status):
            data = numpy.frombuffer(in_data, dtype=numpy.float32)
            detector.send(data)
            return in_data, pyaudio.paContinue

        input_stream = self.pyaudio.open(format=pyaudio.paFloat32,
                                         channels=1,
                                         rate=self.samplerate,
                                         input=True,
                                         output=False,
                                         frames_per_buffer=self.hop_length,
                                         stream_callback=input_callback,
                                         start=False)

        return input_stream

    @contextmanager
    def get_screen(self):
        try:
            stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            stdscr.nodelay(True)
            stdscr.keypad(1)
            curses.curs_set(0)

            yield stdscr

        finally:
            curses.endwin()

    def play(self, knock_game):
        def SIGINT_handler(sig, frame):
            self.stop = True

        with knock_game, self.get_screen() as screen:
            signal.signal(signal.SIGINT, SIGINT_handler)

            sound_handler = knock_game.get_sound_handler(self.samplerate, self.hop_length)
            knock_handler = knock_game.get_knock_handler()
            screen_handler = knock_game.get_screen_handler(screen)

            output_stream = self.get_output_stream(sound_handler)
            input_stream = self.get_input_stream(knock_handler)

            try:
                reference_time = time.time()
                input_stream.start_stream()
                output_stream.start_stream()

                while output_stream.is_active() and input_stream.is_active():
                    signal.signal(signal.SIGINT, SIGINT_handler)
                    screen_handler.send(time.time() - reference_time - self.display_delay)
                    time.sleep(1/self.display_fps)

            finally:
                output_stream.stop_stream()
                input_stream.stop_stream()
                output_stream.close()
                input_stream.close()
