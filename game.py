import sys
import time
import numpy as np
import pyaudio
import realtime_analysis as ra

CHANNELS = 1
RATE = 44100
SAMPLES_PER_BUFFER = 1024
WIN_LENGTH = 512*4
HOP_LENGTH = 512
PRE_MAX = 0.03
POST_MAX = 0.03
PRE_AVG = 0.03
POST_AVG = 0.03
WAIT = 0.03
DELTA = 0.1
# frame resolution: 11.6 ms
# delay: 30 ms
# fastest tempo: 2000 bpm

DISPLAY_FPS = 200
DISPLAY_DELAY = 0.03
AUDIO_VOLUME = 1.2
AUDIO_DELAY = 0.03

TRACK_WIDTH = 100
BAR_OFFSET = 10
DROP_SPEED = 100.0 # pixels per sec
THRESHOLDS = (0.0, 0.7, 1.3, 2.0, 2.7, 3.3)
TOLERANCES = (0.02, 0.06, 0.10, 0.14)


# beats and hits
class Beat:
    def __init__(self, time):
        self.time = time

    def __repr__(self):
        return "Beat.{}(time={})".format(type(self).__name__, repr(self.time))

Beat.Soft = type("Soft", (Beat,), dict(score=10))
Beat.Loud = type("Loud", (Beat,), dict(score=10))
Beat.Incr = type("Incr", (Beat,), dict(score=10))

class Roll(Beat):
    def __init__(self, time, end, number):
        super().__init__(time)
        self.end = end
        self.number = number

    @property
    def score(self):
        return 10+2*(self.number-1)

    def __repr__(self):
        return "Beat.Roll(time={}, end={}, number={})".format(repr(self.time), repr(self.end), repr(self.number))
Beat.Roll = Roll


class Hit:
    def __init__(self, time, strength):
        self.time = time
        self.strength = strength

    def __repr__(self):
        return "Hit.{}(time={}, strength={})".format(type(self).__name__, repr(self.time), repr(self.strength))

Hit.Miss       = type("Miss",       (Hit,), dict(score=0))
Hit.Unexpected = type("Unexpected", (Hit,), dict(score=0))

Hit.Great       = type("Great",       (Hit,), dict(score=10))
Hit.LateGood    = type("LateGood",    (Hit,), dict(score=5))
Hit.EarlyGood   = type("EarlyGood",   (Hit,), dict(score=5))
Hit.LateBad     = type("LateBad",     (Hit,), dict(score=3))
Hit.EarlyBad    = type("EarlyBad",    (Hit,), dict(score=3))
Hit.LateFailed  = type("LateFailed",  (Hit,), dict(score=0))
Hit.EarlyFailed = type("EarlyFailed", (Hit,), dict(score=0))

Hit.WrongButGreat       = type("WrongButGreat",       (Hit,), dict(score=5))
Hit.WrongButLateGood    = type("WrongButLateGood",    (Hit,), dict(score=3))
Hit.WrongButEarlyGood   = type("WrongButEarlyGood",   (Hit,), dict(score=3))
Hit.WrongButLateBad     = type("WrongButLateBad",     (Hit,), dict(score=1))
Hit.WrongButEarlyBad    = type("WrongButEarlyBad",    (Hit,), dict(score=1))
Hit.WrongButLateFailed  = type("WrongButLateFailed",  (Hit,), dict(score=0))
Hit.WrongButEarlyFailed = type("WrongButEarlyFailed", (Hit,), dict(score=0))

class Rock(Hit):
    def __init__(self, time, strength, roll, total):
        super().__init__(time, strength)
        self.roll = roll
        self.total = total

    @property
    def score(self):
        if self.roll <= self.total:
            return 2
        elif self.roll <= self.total*2:
            return -2
        else:
            return 0

    def __repr__(self):
        return "Hit.Rock(time={}, strength={}, roll={}, total={})".format(
                    repr(self.time), repr(self.strength), repr(self.roll), repr(self.total))
Hit.Rock = Rock


class Beatmap:
    def __init__(self, duration, *beats):
        self.duration = duration
        self.beats = beats

    def gen_music(self):
        times = []
        for beat in self.beats:
            if isinstance(beat, Beat.Roll):
                step = (beat.end - beat.time)/beat.number
                for i in range(beat.number):
                    times.append(beat.time + step * i)
            else:
                times.append(beat.time)
        return ra.clicks(times, sr=RATE, duration=self.duration)


def judge(beats, hits):
    beats = iter(beats)

    prev_strength = 0.0
    incr_tol = (THRESHOLDS[3] - THRESHOLDS[0])/6
    roll_number = 0

    beat = next(beats, None)
    while True:
        time, strength, detected = yield
        time -= AUDIO_DELAY
        strength *= AUDIO_VOLUME

        # if next beat has passed through
        if roll_number != 0 and beat.end - TOLERANCES[2] < time:
            beat = next(beats, None)
            roll_number = 0

        if roll_number == 0:
            while beat is not None and beat.time + TOLERANCES[3] < time:
                hits.append(Hit.Miss(time, 0.0))
                beat = next(beats, None)

        if not detected or strength < THRESHOLDS[0]:
            continue

        # if next beat isn't in the range yet
        if beat is None or beat.time - TOLERANCES[3] >= time:
            hits.append(Hit.Unexpected(time, strength))
            continue

        # drumrolls (only for the second or later hits of rolls)
        if isinstance(beat, Beat.Roll):
            roll_number += 1
            if roll_number > 1:
                hits.append(Hit.Rock(time, strength, roll_number, beat.number))
                continue

        # judge pressed key (determined by loudness)
        if isinstance(beat, Beat.Soft):
            is_correct_key = strength < THRESHOLDS[3]
        elif isinstance(beat, Beat.Loud):
            is_correct_key = strength >= THRESHOLDS[3]
        elif isinstance(beat, Beat.Incr):
            is_correct_key = strength >= prev_strength - incr_tol
        elif isinstance(beat, Beat.Roll):
            is_correct_key = True

        # judge accuracy
        err = abs(time - beat.time)
        is_late = time > beat.time

        if err < TOLERANCES[0]:
            if is_correct_key:
                hit_type = Hit.Great
            else:
                hit_type = Hit.WrongButGreat

        elif err < TOLERANCES[1]:
            if is_correct_key:
                hit_type = Hit.LateGood           if is_late else Hit.EarlyGood
            else:
                hit_type = Hit.WrongButLateGood   if is_late else Hit.WrongButEarlyGood

        elif err < TOLERANCES[2]:
            if is_correct_key:
                hit_type = Hit.LateBad            if is_late else Hit.EarlyBad
            else:
                hit_type = Hit.WrongButLateBad    if is_late else Hit.WrongButEarlyBad

        else:
            if is_correct_key:
                hit_type = Hit.LateFailed         if is_late else Hit.EarlyFailed
            else:
                hit_type = Hit.WrongButLateFailed if is_late else Hit.WrongButEarlyFailed

        # add hit and wait for next beat
        hits.append(hit_type(time, strength))
        prev_strength = max(strength, prev_strength) if isinstance(beat, Beat.Incr) else 0.0
        if not isinstance(beat, Beat.Roll):
            beat = next(beats, None)


def draw_track(beats, hits):
    dt = 1 / DROP_SPEED
    sustain = 20 / DROP_SPEED

    hitted_beats = set()
    hit = None
    hit_index = 0
    beat_index = -1
    index = -1

    score = 0
    total_score = sum(beat.score for beat in beats)
    progress = 0

    beats_syms = "□▣◀◎"
    target_sym = "⛶"
    wrong_sym = "˽"
    loudness_syms = "🞓🞒🞑🞐🞏🞎"
    accuracy_syms = "⟪⟬⟨⟩⟭⟫"

    def range_of(b):
        if isinstance(b[1], Beat.Roll):
            return (int(b[1].time/dt), int(b[1].end/dt))
        else:
            return (int(b[1].time/dt), int(b[1].time/dt))
    windowed_beats = ra.window(enumerate(beats), TRACK_WIDTH, BAR_OFFSET, key=range_of)
    next(windowed_beats)

    while True:
        current_time = yield

        view = [" "]*TRACK_WIDTH
        current_pixel = int(current_time/dt)

        # updating `hitted_beats`, it also catches the last item in `hits`
        for hit in hits[hit_index:]:
            score += hit.score
            if not isinstance(hit, (Hit.Unexpected, Hit.Rock)):
                beat_index += 1
                if not isinstance(hit, Hit.Miss):
                    hitted_beats.add(beat_index)
            hit_index += 1

        # draw un-hitted beats, it also catches the last visible beat
        for index, beat in windowed_beats.send(current_pixel):
            start_pixel, end_pixel = range_of((index, beat))

            if isinstance(beat, Beat.Roll):
                unrolled = 0 if index not in hitted_beats else 1 if not isinstance(hit, Hit.Rock) else hit.roll+1
                step_pixel = (end_pixel - start_pixel) // beat.number

                for beat_pixel in range(start_pixel, end_pixel, step_pixel)[unrolled:beat.number]:
                    pixel = BAR_OFFSET + beat_pixel - current_pixel
                    if pixel in range(TRACK_WIDTH):
                        view[pixel] = beats_syms[3]

            elif index not in hitted_beats:
                if isinstance(beat, Beat.Soft):
                    symbol = beats_syms[0]
                elif isinstance(beat, Beat.Loud):
                    symbol = beats_syms[1]
                elif isinstance(beat, Beat.Incr):
                    symbol = beats_syms[2]

                view[BAR_OFFSET + start_pixel - current_pixel] = symbol

            progress = 1000*(index+1) // len(beats)

        # draw target
        view[BAR_OFFSET] = target_sym

        # visual feedback for hit loudness
        if hit is not None and current_time - hit.time < sustain:
            symbol = next((sym for thr, sym in zip(THRESHOLDS[::-1], loudness_syms) if hit.strength >= thr), target_sym)
            view[BAR_OFFSET] = symbol

        # visual feedback for wrong key
        if hit is not None and current_time - hit.time < sustain:
            hitted_types = (Hit.Great, Hit.Rock,
                            Hit.LateGood, Hit.EarlyGood,
                            Hit.LateBad, Hit.EarlyBad,
                            Hit.LateFailed, Hit.EarlyFailed)
            if not isinstance(hit, hitted_types) or isinstance(hit, Hit.Rock) and hit.roll > hit.total:
                view[BAR_OFFSET+1] = wrong_sym

        # visual feedback for hit accuracy
        if hit is not None and current_time - hit.time < sustain/3:
            if isinstance(hit, (Hit.LateGood, Hit.WrongButLateGood)):
                view[BAR_OFFSET-1] = accuracy_syms[2]
            elif isinstance(hit, (Hit.EarlyGood, Hit.WrongButEarlyGood)):
                view[BAR_OFFSET+2] = accuracy_syms[3]
            elif isinstance(hit, (Hit.LateBad, Hit.WrongButLateBad)):
                view[BAR_OFFSET-1] = accuracy_syms[1]
            elif isinstance(hit, (Hit.EarlyBad, Hit.WrongButEarlyBad)):
                view[BAR_OFFSET+2] = accuracy_syms[4]
            elif isinstance(hit, (Hit.LateFailed, Hit.WrongButLateFailed)):
                view[BAR_OFFSET-1] = accuracy_syms[0]
            elif isinstance(hit, (Hit.EarlyFailed, Hit.WrongButEarlyFailed)):
                view[BAR_OFFSET+2] = accuracy_syms[5]

        # print
        sys.stdout.write("[{:>5d}/{:>5d}]".format(score, total_score))
        sys.stdout.write("".join(view))
        sys.stdout.write(" [{:>5.1f}%]".format(progress/10))
        sys.stdout.write("\r")
        sys.stdout.flush()


class Game:
    def __init__(self):
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
                       judge(beatmap.beats, hits))
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
        drawer = draw_track(beatmap.beats, hits)
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
            view_callback(time.time() - reference_time - DISPLAY_DELAY)
            time.sleep(1/DISPLAY_FPS)

        output_stream.stop_stream()
        input_stream.stop_stream()
        output_stream.close()
        input_stream.close()

        return hits


# test
# beatmap = Beatmap(9.0,
#                   Beat.Soft(1.0), Beat.Soft(1.5),  Beat.Soft(2.0), Beat.Soft(2.25), Beat.Loud(2.5),
#                   Beat.Soft(3.0), Beat.Soft(3.5),  Beat.Soft(4.0), Beat.Soft(4.25), Beat.Roll(4.5, 5.0, 4),
#                   Beat.Soft(5.0), Beat.Soft(5.5),  Beat.Soft(6.0), Beat.Soft(6.25), Beat.Loud(6.5),
#                   Beat.Incr(7.0), Beat.Incr(7.25), Beat.Incr(7.5), Beat.Incr(7.75),
#                   Beat.Incr(8.0), Beat.Incr(8.25), Beat.Loud(8.5))
beatmap = Beatmap(10)

game = Game()

hits = game.play(beatmap)
print()
for hit in hits:
    print(hit)
