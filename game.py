import sys
import time
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


# beats and hits
class Beat:
    def __init__(self, time):
        self.time = time

    def __repr__(self):
        return "Beat.{}(time={})".format(type(self).__name__, repr(self.time))

Beat.Soft = type("Soft", (Beat,), dict())
Beat.Loud = type("Loud", (Beat,), dict())


class Hit:
    def __init__(self, time, strength):
        self.time = time
        self.strength = strength

    def __repr__(self):
        return "Hit.{}(time={}, strength={})".format(type(self).__name__, repr(self.time), repr(self.strength))

Hit.Miss       = type("Miss",       (Hit,), dict(score=0, accuracy=0))
Hit.Unexpected = type("Unexpected", (Hit,), dict(score=0, accuracy=0))

Hit.Great       = type("Great",       (Hit,), dict(score=10, accuracy=3))
Hit.LateGood    = type("LateGood",    (Hit,), dict(score=5, accuracy=2))
Hit.EarlyGood   = type("EarlyGood",   (Hit,), dict(score=5, accuracy=2))
Hit.LateBad     = type("LateBad",     (Hit,), dict(score=3, accuracy=1))
Hit.EarlyBad    = type("EarlyBad",    (Hit,), dict(score=3, accuracy=1))
Hit.LateFailed  = type("LateFailed",  (Hit,), dict(score=0, accuracy=0))
Hit.EarlyFailed = type("EarlyFailed", (Hit,), dict(score=0, accuracy=0))

Hit.WrongButGreat       = type("WrongButGreat",       (Hit,), dict(score=5, accuracy=3))
Hit.WrongButLateGood    = type("WrongButLateGood",    (Hit,), dict(score=3, accuracy=2))
Hit.WrongButEarlyGood   = type("WrongButEarlyGood",   (Hit,), dict(score=3, accuracy=2))
Hit.WrongButLateBad     = type("WrongButLateBad",     (Hit,), dict(score=1, accuracy=1))
Hit.WrongButEarlyBad    = type("WrongButEarlyBad",    (Hit,), dict(score=1, accuracy=1))
Hit.WrongButLateFailed  = type("WrongButLateFailed",  (Hit,), dict(score=0, accuracy=0))
Hit.WrongButEarlyFailed = type("WrongButEarlyFailed", (Hit,), dict(score=0, accuracy=0))


class Beatmap:
    def __init__(self, duration, *beats):
        self.duration = duration
        self.beats = beats

    def gen_music(self):
        times = [beat.time for beat in self.beats]
        return ra.clicks(times, sr=RATE, duration=self.duration)


def judge(beats, hits, thresholds, judgements):
    beats = iter(beats)
    beat = next(beats, None)

    while True:
        time, strength, detected = yield

        # if next beat has passed through
        while beat is not None and beat.time + judgements[0] <= time:
            hits.append(Hit.Miss(time, 0))
            beat = next(beats, None)

        if not detected:
            continue

        # if next beat isn't in the range yet
        if beat is None or beat.time - judgements[0] > time:
            hits.append(Hit.Unexpected(time, strength))
            continue

        # judge pressed key (determined by loudness)
        loudness = next((i+1 for i, thr in reversed(list(enumerate(thresholds))) if strength >= thr), 0)

        if isinstance(beat, Beat.Soft):
            is_correct_key = loudness == 1
        elif isinstance(beat, Beat.Loud):
            is_correct_key = loudness == 2

        # judge accuracy
        err = time - beat.time
        accuracy = next((i for i, acc in reversed(list(enumerate(judgements))) if abs(err) < acc), 0)

        if accuracy == 0:
            if is_correct_key:
                hit_type = Hit.LateFailed         if err > 0 else Hit.EarlyFailed
            else:
                hit_type = Hit.WrongButLateFailed if err > 0 else Hit.WrongButEarlyFailed

        elif accuracy == 1:
            if is_correct_key:
                hit_type = Hit.LateBad            if err > 0 else Hit.EarlyBad
            else:
                hit_type = Hit.WrongButLateBad    if err > 0 else Hit.WrongButEarlyBad

        elif accuracy == 2:
            if is_correct_key:
                hit_type = Hit.LateGood           if err > 0 else Hit.EarlyGood
            else:
                hit_type = Hit.WrongButLateGood   if err > 0 else Hit.WrongButEarlyGood

        elif accuracy == 3:
            if is_correct_key:
                hit_type = Hit.Great
            else:
                hit_type = Hit.WrongButGreat

        hits.append(hit_type(time, strength))
        beat = next(beats, None)


def draw_track(beats, hits, track_width, bar_offset, drop_speed, levels, sustain):
    dt = 1 / drop_speed

    hitted_beats = set()
    hit = None
    hit_index = 0
    beat_index = -1
    index = -1

    score = 0
    progress = 0

    beats_syms = "□▣" # □ ▣ ◧ ◨
    target_sym = "⛶"
    wrong_sym = "˽"
    loudness_syms = "🞓🞒🞑🞐🞏🞎"
    accuracy_syms = "⟪⟬⟨⟩⟭⟫"

    beats_iter = ((i, b, int(b.time/dt)) for i, b in enumerate(beats))
    windowed_beats = ra.window(beats_iter, track_width, bar_offset, key=lambda a: a[2:]*2)
    next(windowed_beats)

    while True:
        current_time = yield

        view = [" "]*track_width
        current_pixel = int(current_time/dt)

        # updating `hitted_beats`, it also catches the last item in `hits`
        for hit in hits[hit_index:]:
            score += hit.score
            if not isinstance(hit, Hit.Unexpected):
                beat_index += 1
                if not isinstance(hit, Hit.Miss):
                    hitted_beats.add(beat_index)
            hit_index += 1

        # draw un-hitted beats, it also catches the last visible beat
        for index, beat, beat_pixel in windowed_beats.send(current_pixel):
            if index not in hitted_beats:
                if isinstance(beat, Beat.Soft):
                    symbol = beats_syms[0]
                elif isinstance(beat, Beat.Loud):
                    symbol = beats_syms[1]
                view[beat_pixel - current_pixel + bar_offset] = symbol

        progress = 1000*(index+1) // len(beats)

        # draw target
        view[bar_offset] = target_sym

        # visual feedback for hit loudness
        if hit is not None and current_time - hit.time < sustain:
            symbol = next((sym for thr, sym in zip(levels[5::-1], loudness_syms) if hit.strength >= thr), target_sym)
            view[bar_offset] = symbol

        # visual feedback for wrong key
        if hit is not None and current_time - hit.time < sustain:
            if not isinstance(hit, (Hit.Great, Hit.LateGood, Hit.EarlyGood,
                                               Hit.LateBad, Hit.EarlyBad,
                                               Hit.LateFailed, Hit.EarlyFailed)):
                view[bar_offset+1] = wrong_sym

        # visual feedback for hit accuracy
        if hit is not None and current_time - hit.time < sustain/3:
            if isinstance(hit, (Hit.LateGood, Hit.WrongButLateGood)):
                view[bar_offset-1] = accuracy_syms[2]
            elif isinstance(hit, (Hit.EarlyGood, Hit.WrongButEarlyGood)):
                view[bar_offset+2] = accuracy_syms[3]
            elif isinstance(hit, (Hit.LateBad, Hit.WrongButLateBad)):
                view[bar_offset-1] = accuracy_syms[1]
            elif isinstance(hit, (Hit.EarlyBad, Hit.WrongButEarlyBad)):
                view[bar_offset+2] = accuracy_syms[4]
            elif isinstance(hit, (Hit.LateFailed, Hit.WrongButLateFailed)):
                view[bar_offset-1] = accuracy_syms[0]
            elif isinstance(hit, (Hit.EarlyFailed, Hit.WrongButEarlyFailed)):
                view[bar_offset+2] = accuracy_syms[5]

        # print
        sys.stdout.write("[{:>5d}]".format(score))
        sys.stdout.write("".join(view))
        sys.stdout.write(" [{:>5.1f}%]".format(progress/10))
        sys.stdout.write("\r")
        sys.stdout.flush()


class Game:
    def __init__(self, thresholds=(0.1, 2.0), judgements=(0.14, 0.10, 0.06, 0.02),
                       track_width=100, bar_offset=10, drop_speed=100.0,
                       levels=(0.0, 0.7, 1.3, 2.0, 2.7, 3.3), sustain=0.2):
        self.thresholds = thresholds
        self.judgements = judgements
        self.track_width = track_width
        self.bar_offset = bar_offset
        self.drop_speed = drop_speed # pixels per sec
        self.levels = levels
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

    def get_input_stream(self, beatmap, hits):
        dect = ra.pipe(ra.frame(WIN_LENGTH, HOP_LENGTH),
                       ra.power_spectrum(RATE, WIN_LENGTH),
                       ra.onset_strength(RATE/WIN_LENGTH, HOP_LENGTH/RATE),
                       ra.onset_detect(RATE, HOP_LENGTH,
                                       pre_max=PRE_MAX, post_max=POST_MAX,
                                       pre_avg=PRE_AVG, post_avg=POST_AVG,
                                       wait=WAIT, delta=DELTA),
                       judge(beatmap.beats, hits, self.thresholds, self.judgements))
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

    def get_view_callback(self, beatmap, hits):
        drawer = draw_track(beatmap.beats, hits,
                            self.track_width, self.bar_offset, self.drop_speed,
                            self.levels, self.sustain)
        next(drawer)

        return drawer.send

    def play(self, beatmap):
        hits = []

        output_stream = self.get_output_stream(beatmap)
        input_stream = self.get_input_stream(beatmap, hits)
        view_callback = self.get_view_callback(beatmap, hits)

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

        return hits


# test
beatmap = Beatmap(16.5,
                  Beat.Soft(1.0),  Beat.Soft(2.0),  Beat.Soft(3.0),  Beat.Soft(3.5),  Beat.Loud(4.0),
                  Beat.Soft(5.0),  Beat.Soft(6.0),  Beat.Soft(7.0),  Beat.Soft(7.5),  Beat.Loud(8.0),
                  Beat.Soft(9.0),  Beat.Soft(10.0), Beat.Soft(11.0), Beat.Soft(11.5), Beat.Loud(12.0),
                  Beat.Soft(13.0), Beat.Soft(14.0), Beat.Soft(15.0), Beat.Soft(15.5), Beat.Loud(16.0))
# beatmap = Beatmap(10)

game = Game(thresholds=(0.1, 2.0), judgements=(0.14, 0.10, 0.06, 0.02),
            track_width=100, bar_offset=10, drop_speed=100.0,
            levels=(0.0, 0.7, 1.3, 2.0, 2.7, 3.3), sustain=0.2)

hits = game.play(beatmap)
print()
for hit in hits:
    print(hit)
