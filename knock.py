import os
import sys
import time
import itertools
import curses
import signal
import numpy as np
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
# delay: 30 ms
# fastest tempo: 2000 bpm

DISPLAY_FPS = 120
DISPLAY_DELAY = 0.03
KNOCK_VOLUME = HOP_LENGTH / RATE / 0.00017 # Dt / knock_max_energy
KNOCK_DELAY = 0.0


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

    def get_output_stream(self, sound_handler, sr, hop_length):
        def output_callback(in_data, frame_count, time_info, status):
            out_data = next(sound_handler, None)
            if self.stop or out_data is None:
                return b'', pyaudio.paComplete
            else:
                return out_data.tobytes(), pyaudio.paContinue

        output_stream = self.pyaudio.open(format=pyaudio.paFloat32,
                                          channels=1,
                                          rate=sr,
                                          input=False,
                                          output=True,
                                          frames_per_buffer=hop_length,
                                          stream_callback=output_callback,
                                          start=False)

        return output_stream

    def get_input_stream(self, knock_handler, sr, hop_length, win_length):
        pick_peak = ra.pick_peak(self.pre_max, self.post_max,
                                 self.pre_avg, self.post_avg,
                                 self.wait, self.delta)
        halfhann_window = np.sin(np.linspace(0, np.pi/2, win_length))**2
        dect = ra.pipe(ra.frame(win_length, hop_length),
                       ra.power_spectrum(sr, win_length, windowing=halfhann_window, weighting=True),
                       ra.onset_strength(sr/win_length),
                       (lambda a: (None, a, a)),
                       ra.pair(itertools.count(-pick_peak.delay), ra.delay(pick_peak.delay), pick_peak),
                       (lambda a: (a[0]*hop_length/sr-self.knock_delay, (a[1] or 0.0)*self.knock_volume, a[2])),
                       knock_handler)

        def input_callback(in_data, frame_count, time_info, status):
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            dect.send(audio_data)
            return in_data, pyaudio.paContinue

        input_stream = self.pyaudio.open(format=pyaudio.paFloat32,
                                         channels=1,
                                         rate=sr,
                                         input=True,
                                         output=False,
                                         frames_per_buffer=self.hop_length,
                                         stream_callback=input_callback,
                                         start=False)

        return input_stream

    def play(self, beatmap):
        def SIGINT_handler(sig, frame):
            self.stop = True

        with beatmap:
            signal.signal(signal.SIGINT, SIGINT_handler)

            try:
                stdscr = curses.initscr()
                curses.noecho()
                curses.cbreak()
                stdscr.nodelay(True)
                stdscr.keypad(1)
                curses.curs_set(0)

                sound_handler = beatmap.get_sound_handler(self.samplerate, self.hop_length)
                knock_handler = beatmap.get_knock_handler()
                screen_handler = beatmap.get_screen_handler(stdscr)

                output_stream = self.get_output_stream(sound_handler, self.samplerate, self.hop_length)
                input_stream = self.get_input_stream(knock_handler, self.samplerate, self.hop_length, self.win_length)

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

            finally:
                curses.endwin()
