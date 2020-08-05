import sys
import time
import collections
import enum
import numpy as np
import scipy
import pyaudio
import realtime_analysis as ra

CHANNELS = 1
RATE = 44100
WIN_LENGTH = 512*4
HOP_LENGTH = 512
SAMPLES_PER_BUFFER = 1024
# frame resolution: 11.6 ms

PRE_MAX = 0.03
POST_MAX = 0.03
PRE_AVG = 0.03
POST_AVG = 0.03
WAIT = 0.03
DELTA = 0.1

DISPLAY_FPS = 200
DISPLAY_DELAY = -0.03


# beats
Beat = collections.namedtuple("Beat", ["time", "key"])
Hit = collections.namedtuple("Hit", ["index", "accuracy", "key", "time"])

class Beatmap:
    def __init__(self, duration, *beats):
        self.duration = duration
        self.beats = beats

    def gen_music(self):
        times = [beat.time for beat in self.beats]
        return ra.clicks(times, sr=RATE, duration=self.duration)

class Judger:
    def __init__(self, thresholds, judgements):
        self.thresholds = thresholds
        self.judgements = judgements

    def judge(self, beatmap, scores):
        beat_times = enumerate(beat.time for beat in beatmap.beats)
        beat_index, beat_time = next(beat_times, (None, None))

        time, strength, detected = yield
        while True:
            while beat_index is not None and beat_time + self.judgements[0] <= time:
                scores.append(Hit(beat_index, 0, 0, time))
                beat_index, beat_time = next(beat_times, (None, None))

            if detected:
                key = next((i+1 for i, thr in reversed(list(enumerate(self.thresholds))) if strength >= thr), 0)

                if beat_index is not None and beat_time - self.judgements[0] <= time:
                    err = time - beat_time
                    accuracy = next((i for i, acc in reversed(list(enumerate(self.judgements))) if abs(err) < acc), 0)

                    scores.append(Hit(beat_index, accuracy, key, time))

                    beat_index, beat_time = next(beat_times, (None, None))

                else:
                    scores.append(Hit(None, 0, key, time))

            time, strength, detected = yield

class Track:
    def __init__(self, track_width, bar_offset, drop_speed):
        self.track_width = track_width
        self.bar_offset = bar_offset
        self.drop_speed = drop_speed # pixels per sec

    def draw(self, beatmap, beatmap_scores):
        dt = 1 / self.drop_speed
        beats_iter = ((int(b.time/dt),)*2 + (i,) for i, b in enumerate(beatmap.beats))
        windowed_beatmap = ra.window(beats_iter, self.track_width, self.bar_offset)
        next(windowed_beatmap)

        current_time = yield None
        while True:
            view = [" "]*self.track_width

            current_pixel = int(current_time/dt)

            hitted = [i for i, acc, key, t in beatmap_scores if i is not None and key is not 0]
            for beat_pixel, _, beat_index in windowed_beatmap.send(current_pixel):
                if beat_index not in hitted:
                    view[beat_pixel - current_pixel + self.bar_offset] = "+" # "░▄▀█", "□ ▣ ■"

            view[0] = "("
            view[self.bar_offset] = "|"
            view[-1] = ")"

            if len(beatmap_scores) > 0:
                _, _, key, hit_time = beatmap_scores[-1]
                if abs(hit_time - current_time) < 0.2:
                    if key == 1:
                        view[self.bar_offset] = "I"
                    elif key == 2:
                        view[self.bar_offset] = "H"

            current_time = yield "".join(view)


class Game:
    def __init__(self, beatmap, judger, track):
        self.beatmap = beatmap
        self.judger = judger
        self.track = track

    def get_output_callback(self):
        # output stream callback
        signal_gen = self.beatmap.gen_music()
        next(signal_gen, None)

        def output_callback(in_data, frame_count, time_info, status):
            out_data = next(signal_gen, None)
            return out_data, pyaudio.paContinue

        return output_callback

    def get_input_callback(self, beatmap_scores):
        # input stream callback
        dect = ra.pipe(ra.frame(WIN_LENGTH, HOP_LENGTH),
                       ra.power_spectrum(RATE, WIN_LENGTH),
                       ra.onset_strength(RATE/WIN_LENGTH, HOP_LENGTH/RATE),
                       ra.onset_detect(RATE, HOP_LENGTH,
                                       pre_max=PRE_MAX, post_max=POST_MAX,
                                       pre_avg=PRE_AVG, post_avg=POST_AVG,
                                       wait=WAIT, delta=DELTA),
                       self.judger.judge(self.beatmap, beatmap_scores))
        next(dect)

        def input_callback(in_data, frame_count, time_info, status):
            audio_data = np.frombuffer(in_data, dtype=np.float32)

            for i in range(0, frame_count, HOP_LENGTH):
                dect.send(audio_data[i:i+HOP_LENGTH])

            return in_data, pyaudio.paContinue

        return input_callback

    def get_view_callback(self, beatmap_scores):
        # screen output callback
        drawer = self.track.draw(self.beatmap, beatmap_scores)
        next(drawer)

        def view_callback(current_time):
            view = drawer.send(current_time)
            sys.stdout.write(view + "\r")
            sys.stdout.flush()

        return view_callback

    def play(self):
        beatmap_scores = []

        view_callback = self.get_view_callback(beatmap_scores)

        reference_time = time.time()

        p = pyaudio.PyAudio()

        input_stream = p.open(format=pyaudio.paFloat32,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              output=False,
                              frames_per_buffer=SAMPLES_PER_BUFFER,
                              stream_callback=self.get_input_callback(beatmap_scores))
        input_stream.start_stream()

        output_stream = p.open(format=pyaudio.paFloat32,
                               channels=CHANNELS,
                               rate=RATE,
                               input=False,
                               output=True,
                               frames_per_buffer=SAMPLES_PER_BUFFER,
                               stream_callback=self.get_output_callback())
        output_stream.start_stream()

        while output_stream.is_active() and input_stream.is_active():
            view_callback(time.time() - reference_time + DISPLAY_DELAY)
            time.sleep(1/DISPLAY_FPS)

        output_stream.stop_stream()
        input_stream.stop_stream()
        output_stream.close()
        input_stream.close()

        p.terminate()

        # print scores
        print()
        for score in beatmap_scores:
            print(score)


# test
def light(time):
    return Beat(time, 1)
def heavy(time):
    return Beat(time, 2)

beatmap = Beatmap(16.5,
                  light(1.0),  light(2.0),  light(3.0),  light(3.5),  heavy(4.0),
                  light(5.0),  light(6.0),  light(7.0),  light(7.5),  heavy(8.0),
                  light(9.0),  light(10.0), light(11.0), light(11.5), heavy(12.0),
                  light(13.0), light(14.0), light(15.0), light(15.5), heavy(16.0))
judger = Judger(thresholds=(0.1, 2.0), judgements=(0.14, 0.10, 0.06, 0.02))
track = Track(track_width=100, bar_offset=10, drop_speed=100.0)
game = Game(beatmap, judger, track)
game.play()
