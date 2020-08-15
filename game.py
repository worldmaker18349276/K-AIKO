import sys
import time
import enum
import wave
import numpy as np
import pyaudio
import realtime_analysis as ra

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
BAR_OFFSET = 40
DROP_SPEED = 100.0 # pixels per sec
THRESHOLDS = (0.0, 0.7, 1.3, 2.0, 2.7, 3.3)
TOLERANCES = (0.02, 0.06, 0.10, 0.14)


# beats and performances
class Beat:
    def click(self):
        if isinstance(self, Beat.Soft):
            return ra.click(sr=RATE, freq=1000.0, decay_time=0.1, amplitude=0.5)

        elif isinstance(self, Beat.Loud):
            return ra.click(sr=RATE, freq=1000.0, decay_time=0.1, amplitude=1.0)

        elif isinstance(self, Beat.Incr):
            amplitude = 0.5 + 0.5 * (self.count-1)/self.total
            return ra.click(sr=RATE, freq=1000.0, decay_time=0.1, amplitude=amplitude)

        elif isinstance(self, Beat.Roll):
            sound = ra.click(sr=RATE, freq=1000.0, decay_time=0.1, amplitude=1.0)
            step = (self.end - self.time)/self.number
            signals = [(step*i, sound) for i in range(self.number)]
            duration = self.end-self.time
            gen = ra.merge(signals, duration, RATE, int(duration*RATE))
            next(gen)
            return next(gen)

    @property
    def total_score(self):
        if isinstance(self, (Beat.Soft, Beat.Loud, Beat.Incr)):
            return 10
        elif isinstance(self, Beat.Roll):
            return 10 + (self.number - 1) * 2

    @property
    def score(self):
        if isinstance(self, (Beat.Soft, Beat.Loud, Beat.Incr)):
            return self.perf.score if self.perf is not None else 0
        elif isinstance(self, Beat.Roll):
            if self.perf is None:
                return 0
            elif self.roll < self.number:
                return self.perf.score + (self.roll - 1) * 2
            elif self.roll < 2*self.number:
                return self.perf.score + (2*self.number - self.roll - 1) * 2
            else:
                return self.perf.score

class Soft(Beat):
    def __init__(self, time, perf=None):
        super().__init__()
        self.time = time
        self.perf = perf

    def __repr__(self):
        return "Beat.Soft(time={!r}, perf={!r})".format(self.time, self.perf)
Beat.Soft = Soft

class Loud(Beat):
    def __init__(self, time, perf=None):
        super().__init__()
        self.time = time
        self.perf = perf

    def __repr__(self):
        return "Beat.Loud(time={!r}, perf={!r})".format(self.time, self.perf)
Beat.Loud = Loud

class Incr(Beat):
    def __init__(self, time, count, total, perf=None):
        super().__init__()
        self.time = time
        self.count = count
        self.total = total
        self.perf = perf

    def __repr__(self):
        return "Beat.Incr(time={!r}, count={!r}, total={!r}, perf={!r})".format(
                        self.time, self.count, self.total, self.perf)
Beat.Incr = Incr

class Roll(Beat):
    def __init__(self, time, end, number, perf=None, roll=0):
        super().__init__()
        self.time = time
        self.end = end
        self.number = number
        self.roll = roll
        self.perf = perf

    def __repr__(self):
        return "Beat.Roll(time={!r}, end={!r}, number={!r}, perf={!r}, roll={!r})".format(
                        self.time, self.end, self.number, self.perf, self.roll)
Beat.Roll = Roll

class Performance(enum.Enum):
    MISS               = ("Miss"                      , 0)
    GREAT              = ("Great"                     , 10)
    LATE_GOOD          = ("Late Good"                 , 5)
    EARLY_GOOD         = ("Early Good"                , 5)
    LATE_BAD           = ("Late Bad"                  , 3)
    EARLY_BAD          = ("Early Bad"                 , 3)
    LATE_FAILED        = ("Late Failed"               , 0)
    EARLY_FAILED       = ("Early Failed"              , 0)
    GREAT_WRONG        = ("Great but Wrong Key"       , 5)
    LATE_GOOD_WRONG    = ("Late Good but Wrong Key"   , 3)
    EARLY_GOOD_WRONG   = ("Early Good but Wrong Key"  , 3)
    LATE_BAD_WRONG     = ("Late Bad but Wrong Key"    , 1)
    EARLY_BAD_WRONG    = ("Early Bad but Wrong Key"   , 1)
    LATE_FAILED_WRONG  = ("Late Failed but Wrong Key" , 0)
    EARLY_FAILED_WRONG = ("Early Failed but Wrong Key", 0)

    def __repr__(self):
        return "Performance." + self.name

    def __str__(self):
        return self.value[0]

    @property
    def score(self):
        return self.value[1]


# beatmap
class Beatmap:
    def __init__(self, filename, *beats):
        self.beats = beats
        self.hit = dict(number=-1, strength=0.0, beat=None)
        if isinstance(filename, float):
            self.file = None
            self.duration = filename
        else:
            self.file = wave.open(filename, "rb")
            self.duration = self.file.getnframes() / self.file.getframerate()

    def __del__(self):
        if self.file is not None:
            self.file.close()

    @property
    def total_score(self):
        return sum(beat.total_score for beat in self.beats)

    @property
    def score(self):
        return sum(beat.score for beat in self.beats)

    @property
    def progress(self):
        if len(self.beats) == 0:
            return 1000
        return int(sum(1 for beat in self.beats if beat.perf is not None) / len(self.beats) * 1000)

    def clicks(self):
        signals = [(beat.time, beat.click()) for beat in self.beats]
        return ra.merge(signals, self.duration, RATE, SAMPLES_PER_BUFFER)

    def music(self):
        if self.file is not None:
            return ra.load(self.file, RATE, SAMPLES_PER_BUFFER)
        else:
            return self.clicks()

    def judge(self):
        beats = iter(self.beats)
        INCR_TOL = (THRESHOLDS[3] - THRESHOLDS[0])/6

        incr_threshold = 0.0
        hit_number = 0

        beat = next(beats, None)
        while True:
            time, strength, detected = yield

            # if next beat has passed through
            if isinstance(beat, Beat.Roll) and beat.roll != 0 and beat.end - TOLERANCES[2] < time:
                beat = next(beats, None)

            if not isinstance(beat, Beat.Roll) or beat.roll == 0:
                while beat is not None and beat.time + TOLERANCES[3] < time:
                    beat.perf = Performance.MISS
                    beat = next(beats, None)

            # update state
            if not detected or strength < THRESHOLDS[0]:
                continue
            self.hit = dict(number=hit_number, strength=strength, beat=beat)
            hit_number += 1

            # if next beat isn't in the range yet
            if beat is None or beat.time - TOLERANCES[3] >= time:
                continue

            # drumrolls
            if isinstance(beat, Beat.Roll):
                beat.roll += 1
                if beat.roll > 1:
                    continue

            # judge pressed key (determined by loudness)
            if isinstance(beat, Beat.Soft):
                is_correct_key = strength < THRESHOLDS[3]
            elif isinstance(beat, Beat.Loud):
                is_correct_key = strength >= THRESHOLDS[3]
            elif isinstance(beat, Beat.Incr):
                is_correct_key = strength >= incr_threshold - INCR_TOL
            elif isinstance(beat, Beat.Roll):
                is_correct_key = True

            # judge accuracy
            err = abs(time - beat.time)
            too_late = time > beat.time

            if err < TOLERANCES[0]:
                if is_correct_key:
                    perf = Performance.GREAT
                else:
                    perf = Performance.GREAT_WRONG

            elif err < TOLERANCES[1]:
                if is_correct_key:
                    perf = Performance.LATE_GOOD         if too_late else Performance.EARLY_GOOD
                else:
                    perf = Performance.LATE_GOOD_WRONG   if too_late else Performance.EARLY_GOOD_WRONG

            elif err < TOLERANCES[2]:
                if is_correct_key:
                    perf = Performance.LATE_BAD          if too_late else Performance.EARLY_BAD
                else:
                    perf = Performance.LATE_BAD_WRONG    if too_late else Performance.EARLY_BAD_WRONG

            else:
                if is_correct_key:
                    perf = Performance.LATE_FAILED       if too_late else Performance.EARLY_FAILED
                else:
                    perf = Performance.LATE_FAILED_WRONG if too_late else Performance.EARLY_FAILED_WRONG

            beat.perf = perf

            # add hit and wait for next beat
            incr_threshold = max(strength, incr_threshold) if isinstance(beat, Beat.Incr) else 0.0
            if not isinstance(beat, Beat.Roll):
                beat = next(beats, None)

    def draw(self):
        dt = 1 / DROP_SPEED
        sustain = 10 / DROP_SPEED
        decay = 5 / DROP_SPEED

        beats_syms = ["□", "▣", "◀", "◎"]
        target_sym = "⛶"

        hit_number = -1
        hit_time = -1.0
        loudness_syms = ["🞎", "🞏", "🞐", "🞑", "🞒", "🞓"]
        accuracy_syms = ["⟪", "⟪", "⟨", "⟩", "⟫", "⟫"]
        correct_sym = "˽"

        def range_of(b):
            if isinstance(b, Beat.Roll):
                return (int(b.time/dt), int(b.end/dt))
            else:
                return (int(b.time/dt), int(b.time/dt))
        windowed_beats = ra.window(self.beats, TRACK_WIDTH, BAR_OFFSET, key=range_of)
        next(windowed_beats)

        while True:
            current_time = yield

            view = [" "]*TRACK_WIDTH
            current_pixel = int(current_time/dt)

            # draw un-hitted beats, it also catches the last visible beat
            for beat in windowed_beats.send(current_pixel):
                start_pixel, end_pixel = range_of(beat)

                if isinstance(beat, Beat.Roll):
                    step_pixel = (end_pixel - start_pixel) // beat.number
                    for r in range(beat.number)[beat.roll:]:
                        pixel = BAR_OFFSET + start_pixel + step_pixel * r - current_pixel
                        if pixel in range(TRACK_WIDTH):
                            view[pixel] = beats_syms[3]
                    continue

                if beat.perf in (None, Performance.MISS):
                    if isinstance(beat, Beat.Soft):
                        symbol = beats_syms[0]
                    elif isinstance(beat, Beat.Loud):
                        symbol = beats_syms[1]
                    elif isinstance(beat, Beat.Incr):
                        symbol = beats_syms[2]

                    view[BAR_OFFSET + start_pixel - current_pixel] = symbol

            # draw target
            view[BAR_OFFSET] = target_sym

            if hit_number != self.hit["number"]:
                hit_number = self.hit["number"]
                hit_time = current_time

            # visual feedback for hit loudness
            if current_time - hit_time < 6*decay:
                loudness = next(i for i, thr in reversed(list(enumerate(THRESHOLDS))) if self.hit["strength"] >= thr)
                loudness -= int((current_time - hit_time) / decay)
                if loudness >= 0:
                    view[BAR_OFFSET] = loudness_syms[loudness]

            # visual feedback for hit accuracy
            if current_time - hit_time < sustain and self.hit["beat"] is not None:
                correct_types = (Performance.GREAT,
                                 Performance.LATE_GOOD, Performance.EARLY_GOOD,
                                 Performance.LATE_BAD, Performance.EARLY_BAD,
                                 Performance.LATE_FAILED, Performance.EARLY_FAILED)
                perf = self.hit["beat"].perf
                if perf in correct_types:
                    view[BAR_OFFSET+1] = correct_sym

                if perf in (Performance.LATE_GOOD, Performance.LATE_GOOD_WRONG):
                    view[BAR_OFFSET-1] = accuracy_syms[2]
                elif perf in (Performance.EARLY_GOOD, Performance.EARLY_GOOD_WRONG):
                    view[BAR_OFFSET+2] = accuracy_syms[3]
                elif perf in (Performance.LATE_BAD, Performance.LATE_BAD_WRONG):
                    view[BAR_OFFSET-1] = accuracy_syms[1]
                elif perf in (Performance.EARLY_BAD, Performance.EARLY_BAD_WRONG):
                    view[BAR_OFFSET+2] = accuracy_syms[4]
                elif perf in (Performance.LATE_FAILED, Performance.LATE_FAILED_WRONG):
                    view[BAR_OFFSET-1] = accuracy_syms[0]
                elif perf in (Performance.EARLY_FAILED, Performance.EARLY_FAILED_WRONG):
                    view[BAR_OFFSET+2] = accuracy_syms[5]

            # print
            sys.stdout.write("[{:>5d}/{:>5d}]".format(self.score, self.total_score))
            sys.stdout.write("".join(view))
            sys.stdout.write(" [{:>5.1f}%]".format(self.progress/10))
            sys.stdout.write("\r")
            sys.stdout.flush()


# console
class KnockConsole:
    def __init__(self):
        self.pyaudio = pyaudio.PyAudio()

    def __del__(self):
        self.pyaudio.terminate()

    def get_output_stream(self, beatmap):
        signal_gen = beatmap.music()
        next(signal_gen, None)

        def output_callback(in_data, frame_count, time_info, status):
            out_data = next(signal_gen, None)
            if out_data is None:
                return b'', pyaudio.paComplete
            else:
                return out_data.tobytes(), pyaudio.paContinue

        output_stream = self.pyaudio.open(format=pyaudio.paFloat32,
                                          channels=1,
                                          rate=RATE,
                                          input=False,
                                          output=True,
                                          frames_per_buffer=SAMPLES_PER_BUFFER,
                                          stream_callback=output_callback,
                                          start=False)

        return output_stream

    def get_input_stream(self, beatmap):
        dect = ra.pipe(ra.frame(WIN_LENGTH, HOP_LENGTH),
                       ra.power_spectrum(RATE, WIN_LENGTH),
                       ra.onset_strength(RATE/WIN_LENGTH, HOP_LENGTH/RATE),
                       ra.onset_detect(RATE, HOP_LENGTH,
                                       pre_max=PRE_MAX, post_max=POST_MAX,
                                       pre_avg=PRE_AVG, post_avg=POST_AVG,
                                       wait=WAIT, delta=DELTA),
                       ra.transform(lambda _, a: (a[0]-AUDIO_DELAY, a[1]*AUDIO_VOLUME, a[2])),
                       beatmap.judge())
        next(dect)

        def input_callback(in_data, frame_count, time_info, status):
            audio_data = np.frombuffer(in_data, dtype=np.float32)

            for i in range(0, frame_count, HOP_LENGTH):
                dect.send(audio_data[i:i+HOP_LENGTH])

            return in_data, pyaudio.paContinue

        input_stream = self.pyaudio.open(format=pyaudio.paFloat32,
                                         channels=1,
                                         rate=RATE,
                                         input=True,
                                         output=False,
                                         frames_per_buffer=SAMPLES_PER_BUFFER,
                                         stream_callback=input_callback,
                                         start=False)

        return input_stream

    def get_view_callback(self, beatmap):
        drawer = beatmap.draw()
        next(drawer)

        return drawer.send

    def play(self, beatmap):
        output_stream = self.get_output_stream(beatmap)
        input_stream = self.get_input_stream(beatmap)
        view_callback = self.get_view_callback(beatmap)

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


# test
beatmap = Beatmap(9.0,
                  Beat.Soft(1.0), Beat.Loud(1.5), Beat.Soft(2.0), Beat.Soft(2.25), Beat.Loud(2.5),
                  Beat.Soft(3.0), Beat.Loud(3.5), Beat.Soft(4.0), Beat.Soft(4.25), Beat.Roll(4.5, 5.0, 4),
                  Beat.Soft(5.0), Beat.Loud(5.5), Beat.Soft(6.0), Beat.Soft(6.25), Beat.Loud(6.5),
                  Beat.Incr(7.0, 1, 6), Beat.Incr(7.25, 2, 6), Beat.Incr(7.5, 3, 6), Beat.Incr(7.75, 4, 6),
                  Beat.Incr(8.0, 5, 6), Beat.Incr(8.25, 6, 6), Beat.Loud(8.5))
# beatmap = Beatmap("test.wav")
# beatmap = Beatmap(10.0)

console = KnockConsole()

console.play(beatmap)
print()
for beat in beatmap.beats:
    print(beat)
