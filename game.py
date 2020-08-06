import sys
import time
import collections
import enum
import numpy as np
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

def sign(value):
    if value > 0:
        return 1
    elif value < 0:
        return -1
    else:
        return 0


# beats and hits
class BeatType(enum.Enum):
    SOFT = 1
    LOUD = 2
    # ROLL = 3
    # FLAM = 4

def Soft(time):
    return Beat(time, BeatType.SOFT)

def Loud(time):
    return Beat(time, BeatType.LOUD)

Beat = collections.namedtuple("Beat", ["time", "type"])


class HitScore(enum.Enum):
    GREATE     = (3, 0)
    GOOD_LATE  = (2, 1)
    GOOD_EARLY = (2, -1)
    BAD_LATE   = (1, 1)
    BAD_EARLY  = (1, -1)
    MISS_LATE  = (0, 1)
    MISS_EARLY = (0, -1)
    MISS       = (0, 0)

Hit = collections.namedtuple("Hit", ["time", "strength", "score", "index"])


class Beatmap:
    def __init__(self, duration, *beats):
        self.duration = duration
        self.beats = beats

    def gen_music(self):
        times = [beat.time for beat in self.beats]
        return ra.clicks(times, sr=RATE, duration=self.duration)


def judge(beatmap, scores, thresholds, judgements):
    beat_iter = enumerate(beatmap.beats)
    index, beat = next(beat_iter, (None, None))

    while True:
        time, strength, detected = yield

        while index is not None and beat.time + judgements[0] <= time:
            scores.append(Hit(time, 0, HitScore.MISS, index))
            index, beat = next(beat_iter, (None, None))

        if detected:
            if index is not None and beat.time - judgements[0] <= time:
                err = time - beat.time
                loudness = next((i+1 for i, thr in reversed(list(enumerate(thresholds))) if strength >= thr), 0)
                accuracy = next((i for i, acc in reversed(list(enumerate(judgements))) if abs(err) < acc), 0)
                sgn = sign(err) if accuracy != len(judgements)-1 else 0

                scores.append(Hit(time, strength, HitScore((accuracy, sgn)), index))

                index, beat = next(beat_iter, (None, None))

            else:
                scores.append(Hit(time, strength, HitScore.MISS, None))

def draw_track(beatmap, scores, thresholds, sustain, track_width, bar_offset, drop_speed):
    dt = 1 / drop_speed

    beats_iter = ((i, b, int(b.time/dt)) for i, b in enumerate(beatmap.beats))
    windowed_beats = ra.window(beats_iter, track_width, bar_offset, key=lambda a: a[2:]*2)
    next(windowed_beats)

    hitted_indices = set()
    scoring_index = 0

    current_time = yield None
    while True:
        view = [" "]*track_width
        current_pixel = int(current_time/dt)

        # updating hitted_indices, it also catches the last item in `scores`
        for hit_time, hit_strength, hit_score, hit_index in scores[scoring_index:]:
            if hit_index is not None and hit_score != HitScore.MISS:
                hitted_indices.add(hit_index)
        scoring_index = len(scores)

        # draw un-hitted beats
        for index, beat, beat_pixel in windowed_beats.send(current_pixel):
            if index not in hitted_indices:
                # "░▄▀█", "□ ▣ ■"
                if beat.type == BeatType.SOFT:
                    symbol = "+"
                elif beat.type == BeatType.LOUD:
                    symbol = "o"
                else:
                    symbol = "?"
                view[beat_pixel - current_pixel + bar_offset] = symbol

        view[0] = "("
        view[bar_offset] = "|"
        view[-1] = ")"

        if scoring_index > 0 and abs(hit_time - current_time) < sustain:
            # visual feedback for hit
            if hit_strength >= thresholds[1]:
                view[bar_offset] = "H"
            elif hit_strength >= thresholds[0]:
                view[bar_offset] = "I"

            # visual feedback for score
            pass

        current_time = yield "".join(view)


class Game:
    def __init__(self, thresholds=(0.1, 2.0), judgements=(0.14, 0.10, 0.06, 0.02),
                       track_width=100, bar_offset=10, drop_speed=100.0, sustain=0.2):
        self.thresholds = thresholds
        self.judgements = judgements
        self.track_width = track_width
        self.bar_offset = bar_offset
        self.drop_speed = drop_speed # pixels per sec
        self.sustain = sustain

        self.pyaudio = pyaudio.PyAudio()

    def __del__(self):
        self.pyaudio.terminate()

    def get_output_stream(self, beatmap):
        signal_gen = beatmap.gen_music()
        next(signal_gen, None)

        def output_callback(in_data, frame_count, time_info, status):
            out_data = next(signal_gen, None)
            return out_data, pyaudio.paContinue

        output_stream = self.pyaudio.open(format=pyaudio.paFloat32,
                                          channels=CHANNELS,
                                          rate=RATE,
                                          input=False,
                                          output=True,
                                          frames_per_buffer=SAMPLES_PER_BUFFER,
                                          stream_callback=output_callback,
                                          start=False)

        return output_stream

    def get_input_stream(self, beatmap, scores):
        dect = ra.pipe(ra.frame(WIN_LENGTH, HOP_LENGTH),
                       ra.power_spectrum(RATE, WIN_LENGTH),
                       ra.onset_strength(RATE/WIN_LENGTH, HOP_LENGTH/RATE),
                       ra.onset_detect(RATE, HOP_LENGTH,
                                       pre_max=PRE_MAX, post_max=POST_MAX,
                                       pre_avg=PRE_AVG, post_avg=POST_AVG,
                                       wait=WAIT, delta=DELTA),
                       judge(beatmap, scores, self.thresholds, self.judgements))
        next(dect)

        def input_callback(in_data, frame_count, time_info, status):
            audio_data = np.frombuffer(in_data, dtype=np.float32)

            for i in range(0, frame_count, HOP_LENGTH):
                dect.send(audio_data[i:i+HOP_LENGTH])

            return in_data, pyaudio.paContinue

        input_stream = self.pyaudio.open(format=pyaudio.paFloat32,
                                         channels=CHANNELS,
                                         rate=RATE,
                                         input=True,
                                         output=False,
                                         frames_per_buffer=SAMPLES_PER_BUFFER,
                                         stream_callback=input_callback,
                                         start=False)

        return input_stream

    def get_view_callback(self, beatmap, scores):
        drawer = draw_track(beatmap, scores, self.thresholds, self.sustain,
                            self.track_width, self.bar_offset, self.drop_speed)
        next(drawer)

        def view_callback(current_time):
            view = drawer.send(current_time)
            sys.stdout.write(view + "\r")
            sys.stdout.flush()

        return view_callback

    def play(self, beatmap):
        scores = []

        output_stream = self.get_output_stream(beatmap)
        input_stream = self.get_input_stream(beatmap, scores)
        view_callback = self.get_view_callback(beatmap, scores)

        reference_time = time.time()
        input_stream.start_stream()
        output_stream.start_stream()

        while output_stream.is_active() and input_stream.is_active():
            view_callback(time.time() - reference_time + DISPLAY_DELAY)
            time.sleep(1/DISPLAY_FPS)

        output_stream.stop_stream()
        input_stream.stop_stream()
        output_stream.close()
        input_stream.close()

        return scores


# test
beatmap = Beatmap(16.5,
                  Soft(1.0),  Soft(2.0),  Soft(3.0),  Soft(3.5),  Loud(4.0),
                  Soft(5.0),  Soft(6.0),  Soft(7.0),  Soft(7.5),  Loud(8.0),
                  Soft(9.0),  Soft(10.0), Soft(11.0), Soft(11.5), Loud(12.0),
                  Soft(13.0), Soft(14.0), Soft(15.0), Soft(15.5), Loud(16.0))

game = Game(thresholds=(0.1, 2.0), judgements=(0.14, 0.10, 0.06, 0.02),
            track_width=100, bar_offset=10, drop_speed=100.0, sustain=0.2)

scores = game.play(beatmap)
print()
for score in scores:
    print(score)
