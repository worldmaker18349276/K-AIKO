import os
import enum
import wave
import re
import curses
import numpy
import realtime_analysis as ra
import audioread


TOLERANCES = (0.02, 0.06, 0.10, 0.14)
#             GREAT GOOD  BAD   FAILED
BEATS_SYMS = ("□", "■", "⬒", "◎", "◴◵◶◷")
#             Soft Loud Incr Roll Spin
WRONG_SYM = "⬚"
PERF_SYMS = ("\b⟪", "\b⟪", "\b⟨", "  ⟩", "  ⟫", "  ⟫")
SPIN_FINISHED_SYM = "☺"
TARGET_SYMS = ("⛶", "🞎", "🞏", "🞐", "🞑", "🞒", "🞓")

BUF_LENGTH = 512
WIN_LENGTH = 512*4
INCR_TOL = 0.1
SPEC_WIDTH = 5
SPEC_DECAY = 0.01
HIT_DECAY = 0.4
HIT_SUSTAIN = 0.1
PREPARE_TIME = 1.0


# scripts
class Event:
    # lifespan, zindex
    # def play(self, mixer, time): pass
    # def draw(self, track, time): pass
    pass

class Sym(Event):
    zindex = -2

    def __init__(self, time, symbol=None, speed=1.0, sound=None, samplerate=44100, played=False):
        self.time = time
        self.symbol = symbol
        self.speed = speed
        self.sound = sound
        self.samplerate = samplerate
        self.played = played

    @property
    def lifespan(self):
        cross_time = 1.0 / abs(0.5 * self.speed)
        return (self.time-cross_time, self.time+cross_time)

    def play(self, mixer, time):
        if self.sound is not None and not self.played:
            self.played = True
            mixer.play(self.sound, samplerate=self.samplerate, delay=self.time-time)

    def draw(self, track, time):
        if self.symbol is not None:
            pos = (self.time - time) * 0.5 * self.speed
            track.draw_sym(pos, self.symbol)

    def __repr__(self):
        return "Sym(time={!r}, symbol={!r}, speed={!r}, sound={!r}, samplerate={!r}, played={!r})".format(
                    self.time, self.symbol, self.speed, self.sound, self.samplerate, self.played)


# beats
class Beat(Event):
    # lifespan, range, score, total_score, finished
    # def hit(self, time, strength): pass
    # def finish(self): pass
    # def play(self, mixer, time): pass
    # def draw(self, track, time): pass
    # def draw_judging(self, track, time): pass
    # def draw_hitting(self, track, time): pass

    tolerances = TOLERANCES

    @property
    def zindex(self):
        return -1 if self.finished else 1

    def draw_judging(self, track, time): pass
    def draw_hitting(self, track, time): pass

class SingleBeat(Beat):
    # time, speed, perf, played, symbol, sound, samplerate
    # def hit(self, time, strength): pass

    total_score = 10
    perf_syms = PERF_SYMS
    wrong_symbol = WRONG_SYM

    def __init__(self, time, speed=1.0, perf=None, played=False):
        self.time = time
        self.speed = speed
        self.perf = perf
        self.played = played

    @property
    def range(self):
        return (self.time - self.tolerances[3], self.time + self.tolerances[3])

    @property
    def score(self):
        return self.perf.score if self.perf is not None else 0

    @property
    def finished(self):
        return self.perf is not None

    def finish(self):
        self.perf = Performance.MISS

    def hit(self, time, strength, is_correct_key):
        self.perf = Performance.judge(time - self.time, is_correct_key, self.tolerances)

    @property
    def lifespan(self):
        cross_time = 1.0 / abs(0.5 * self.speed)
        return (self.time-cross_time, self.time+cross_time)

    def play(self, mixer, time):
        if not self.played:
            self.played = True
            mixer.play(self.sound, samplerate=self.samplerate, delay=self.time-time)

    def draw(self, track, time):
        CORRECT_TYPES = (Performance.GREAT,
                         Performance.LATE_GOOD, Performance.EARLY_GOOD,
                         Performance.LATE_BAD, Performance.EARLY_BAD,
                         Performance.LATE_FAILED, Performance.EARLY_FAILED)

        if self.perf in (None, Performance.MISS):
            pos = (self.time - time) * 0.5 * self.speed
            track.draw_sym(pos, self.symbol)

        elif self.perf not in CORRECT_TYPES:
            pos = (self.time - time) * 0.5 * self.speed
            track.draw_sym(pos, self.wrong_symbol)

    def draw_hitting(self, track, time):
        self.perf.draw(track, self.speed < 0, self.perf_syms)

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

    @staticmethod
    def judge(time_diff, is_correct_key, tolerances):
        err = abs(time_diff)
        too_late = time_diff > 0

        if err < tolerances[0]:
            if is_correct_key:
                perf = Performance.GREAT
            else:
                perf = Performance.GREAT_WRONG

        elif err < tolerances[1]:
            if is_correct_key:
                perf = Performance.LATE_GOOD         if too_late else Performance.EARLY_GOOD
            else:
                perf = Performance.LATE_GOOD_WRONG   if too_late else Performance.EARLY_GOOD_WRONG

        elif err < tolerances[2]:
            if is_correct_key:
                perf = Performance.LATE_BAD          if too_late else Performance.EARLY_BAD
            else:
                perf = Performance.LATE_BAD_WRONG    if too_late else Performance.EARLY_BAD_WRONG

        else:
            if is_correct_key:
                perf = Performance.LATE_FAILED       if too_late else Performance.EARLY_FAILED
            else:
                perf = Performance.LATE_FAILED_WRONG if too_late else Performance.EARLY_FAILED_WRONG

        return perf

    def draw(self, track, flipped, perf_syms):
        LEFT_GOOD    = (Performance.LATE_GOOD,    Performance.LATE_GOOD_WRONG)
        RIGHT_GOOD   = (Performance.EARLY_GOOD,   Performance.EARLY_GOOD_WRONG)
        LEFT_BAD     = (Performance.LATE_BAD,     Performance.LATE_BAD_WRONG)
        RIGHT_BAD    = (Performance.EARLY_BAD,    Performance.EARLY_BAD_WRONG)
        LEFT_FAILED  = (Performance.LATE_FAILED,  Performance.LATE_FAILED_WRONG)
        RIGHT_FAILED = (Performance.EARLY_FAILED, Performance.EARLY_FAILED_WRONG)
        if flipped:
            LEFT_GOOD, RIGHT_GOOD = RIGHT_GOOD, LEFT_GOOD
            LEFT_BAD, RIGHT_BAD = RIGHT_BAD, LEFT_BAD
            LEFT_FAILED, RIGHT_FAILED = RIGHT_FAILED, LEFT_FAILED

        if self in LEFT_GOOD:
            track.draw_sym(0.0, perf_syms[2])
        elif self in RIGHT_GOOD:
            track.draw_sym(0.0, perf_syms[3])
        elif self in LEFT_BAD:
            track.draw_sym(0.0, perf_syms[1])
        elif self in RIGHT_BAD:
            track.draw_sym(0.0, perf_syms[4])
        elif self in LEFT_FAILED:
            track.draw_sym(0.0, perf_syms[0])
        elif self in RIGHT_FAILED:
            track.draw_sym(0.0, perf_syms[5])

class Soft(SingleBeat):
    symbol = BEATS_SYMS[0]
    sound = [ra.pulse(samplerate=44100, freq=830.61, decay_time=0.03, amplitude=0.5)]
    samplerate = 44100

    def hit(self, time, strength):
        super().hit(time, strength, strength < 0.5)

    def __repr__(self):
        return "Soft(time={!r}, speed={!r}, perf={!r}, played={!r})".format(
                     self.time, self.speed, self.perf, self.played)

class Loud(SingleBeat):
    symbol = BEATS_SYMS[1]
    sound = [ra.pulse(samplerate=44100, freq=1661.2, decay_time=0.03, amplitude=1.0)]
    samplerate = 44100

    def hit(self, time, strength):
        super().hit(time, strength, strength >= 0.5)

    def __repr__(self):
        return "Loud(time={!r}, speed={!r}, perf={!r}, played={!r})".format(
                     self.time, self.speed, self.perf, self.played)

class IncrGroup:
    def __init__(self, threshold=0.0, total=0):
        self.threshold = threshold
        self.total = total

    def add(self, time, speed=1.0, perf=None, played=False):
        self.total += 1
        return Incr(time, speed, count=self.total, group=self, perf=perf, played=played)

    def hit(self, strength):
        self.threshold = max(self.threshold, strength)

    def __repr__(self):
        return "IncrGroup(threshold={!r}, total={!r})".format(self.threshold, self.total)

class Incr(SingleBeat):
    symbol = BEATS_SYMS[2]
    samplerate = 44100
    incr_tol = INCR_TOL

    def __init__(self, time, speed=1.0, count=None, group=None, perf=None, played=False):
        super().__init__(time, speed, perf, played)
        if count is None or group is None:
            raise ValueError
        self.count = count
        self.group = group

    def hit(self, time, strength):
        super().hit(time, strength, strength >= self.group.threshold - self.incr_tol)
        self.group.hit(strength)

    @property
    def sound(self):
        amplitude = 0.2 + 0.8 * (self.count-1)/self.group.total
        return [ra.pulse(samplerate=44100, freq=1661.2, decay_time=0.03, amplitude=amplitude)]

    def __repr__(self):
        return "Incr(time={!r}, speed={!r}, count={!r}, group={!r}, perf={!r}, played={!r})".format(
                     self.time, self.speed, self.count, self.group, self.perf, self.played)

class Roll(Beat):
    symbol = BEATS_SYMS[3]
    sound = [ra.pulse(samplerate=44100, freq=1661.2, decay_time=0.01, amplitude=0.5)]
    samplerate = 44100

    def __init__(self, time, end, number, speed=1.0, roll=0, finished=False, played=False):
        self.time = time
        self.end = end
        self.number = number
        self.speed = speed
        self.roll = roll
        self.finished = finished
        self.played = played

    @property
    def step(self):
        return (self.end - self.time)/(self.number-1) if self.number > 1 else 0.0

    @property
    def range(self):
        return (self.time - self.tolerances[2], self.end + max(0.0, self.step - self.tolerances[2]))
        # return (self.time - self.tolerances[2], self.end)

    @property
    def total_score(self):
        return self.number * 2

    @property
    def score(self):
        if self.roll < self.number:
            return self.roll * 2
        elif self.roll < 2*self.number:
            return (2*self.number - self.roll) * 2
        else:
            return 0

    def hit(self, time, strength):
        self.roll += 1

    def finish(self):
        self.finished = True

    @property
    def lifespan(self):
        cross_time = 1.0 / abs(0.5 * self.speed)
        return (self.time-cross_time, self.end+cross_time)

    def play(self, mixer, time):
        if not self.played:
            self.played = True

            for r in range(self.number):
                delay = self.time + self.step * r - time
                mixer.play(self.sound, samplerate=self.samplerate, delay=delay)

    def draw(self, track, time):
        for r in range(self.number):
            if r > self.roll-1:
                pos = (self.time + self.step * r - time) * 0.5 * self.speed
                track.draw_sym(pos, self.symbol)

    def __repr__(self):
        return "Roll(time={!r}, end={!r}, number={!r}, speed={!r}, roll={!r}, finished={!r}, played={!r})".format(
                     self.time, self.end, self.number, self.speed, self.roll, self.finished, self.played)

class Spin(Beat):
    total_score = 10
    symbols = BEATS_SYMS[4]
    sound = [ra.pulse(samplerate=44100, freq=1661.2, decay_time=0.01, amplitude=1.0)]
    samplerate = 44100
    finished_sym = SPIN_FINISHED_SYM

    def __init__(self, time, end, capacity, speed=1.0, charge=0.0, finished=False, played=False):
        self.time = time
        self.end = end
        self.capacity = capacity
        self.speed = speed
        self.charge = charge
        self.finished = finished
        self.played = played

    @property
    def range(self):
        return (self.time - self.tolerances[2], self.end + self.tolerances[2])

    @property
    def score(self):
        return self.total_score if self.charge == self.capacity else 0

    def hit(self, time, strength):
        self.charge = min(self.charge + min(1.0, strength), self.capacity)
        if self.charge == self.capacity:
            self.finished = True

    def finish(self):
        self.finished = True

    @property
    def lifespan(self):
        cross_time = 1.0 / abs(0.5 * self.speed)
        return (self.time-cross_time, self.end+cross_time)

    def play(self, mixer, time):
        if not self.played:
            self.played = True

            step = (self.end - self.time)/self.capacity if self.capacity > 0.0 else 0.0
            for i in range(int(self.capacity)):
                delay = self.time + step * i - time
                mixer.play(self.sound, samplerate=44100, delay=delay)

    def draw(self, track, time):
        if self.charge < self.capacity:
            pos = 0.0
            pos += max(0.0, (self.time - time) * 0.5 * self.speed)
            pos += min(0.0, (self.end - time) * 0.5 * self.speed)
            track.draw_sym(pos, self.symbols[int(self.charge) % 4])

    def draw_judging(self, track, time):
        return True

    def draw_hitting(self, track, time):
        if self.charge == self.capacity:
            track.draw_sym(0.0, self.finished_sym)
            return True

    def __repr__(self):
        return "Spin(time={!r}, end={!r}, capacity={!r}, speed={!r}, charge={!r}, finished={!r}, played={!r})".format(
                     self.time, self.end, self.capacity, self.speed, self.charge, self.finished, self.played)


# beatmap
class BeatTrack:
    def __init__(self, width, shift, spec_width):
        self.width = width
        self.shift = shift
        self.spec_width = spec_width

        self.chars = [' ']*width
        self.spec_offset = 1
        self.score_offset = self.spec_width + 2
        self.progress_offset = self.width - 9
        self.track_offset = self.spec_width + 15
        self.track_width = self.width - 24 - self.spec_width

    def __str__(self):
        return "".join(self.chars)

    def clear(self):
        for i in range(self.width):
            self.chars[i] = ' '

    def addstr(self, index, str):
        for ch in str:
            if ch == ' ':
                index += 1
            elif ch == '\b':
                index -= 1
            else:
                if index in range(self.width):
                    self.chars[index] = ch
                index += 1

    def draw_spectrum(self, spectrum):
        self.addstr(self.spec_offset, spectrum)

    def draw_score(self, score, total_score):
        self.addstr(self.score_offset, "[{:>5d}/{:>5d}]".format(score, total_score))

    def draw_progress(self, progress):
        self.addstr(self.progress_offset, "[{:>5.1f}%]".format(progress*100))

    def draw_sym(self, pos, sym):
        index = round((pos + self.shift) * (self.track_width - 1))
        for ch in sym:
            if ch == ' ':
                index += 1
            elif ch == '\b':
                index -= 1
            else:
                if index in range(self.track_width):
                    self.chars[self.track_offset+index] = ch
                index += 1

class Beatmap:
    prepare_time = PREPARE_TIME
    buffer_length = BUF_LENGTH
    win_length = WIN_LENGTH
    spec_width = SPEC_WIDTH
    spec_decay = SPEC_DECAY

    hit_decay = HIT_DECAY
    hit_sustain = HIT_SUSTAIN
    target_syms = TARGET_SYMS

    def __init__(self, audio, events):
        # audio metadata
        self.audio = audio
        if self.audio is not None:
            with audioread.audio_open(self.audio) as file:
                self.duration = file.duration
                self.samplerate = file.samplerate
                self.channels = file.channels
        else:
            self.duration = 0.0
            self.samplerate = 44100
            self.channels = 1

        # events, beats
        self.events = list(events)
        self.beats = list(event for event in self.events if isinstance(event, Beat))
        self.start = min(0.0, min(event.lifespan[0] - self.prepare_time for event in self.events))
        self.end = max(self.duration, max(event.lifespan[1] + self.prepare_time for event in self.events))

        # hit state
        self.current_beat = None
        self.hit_index = 0
        self.hit_time = self.start - max(self.hit_decay, self.hit_sustain)*2
        self.hit_strength = 0.0
        self.hit_beat = None
        self.draw_index = self.hit_index
        self.draw_time = self.hit_time

        # spectrum show
        self.spectrum = " "*self.spec_width

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return False

    @property
    def total_score(self):
        return sum(beat.total_score for beat in self.beats)

    @property
    def score(self):
        return sum(beat.score for beat in self.beats)

    @property
    def progress(self):
        if len(self.beats) == 0:
            return 1.0
        return sum(1 for beat in self.beats if beat.finished) / len(self.beats)

    @ra.DataNode.from_generator
    def get_beats_handler(self):
        beats = iter(sorted(self.beats, key=lambda e: e.range))
        beat = next(beats, None)

        time = yield
        while True:
            while beat is not None and (beat.finished or time > beat.range[1]):
                if not beat.finished:
                    beat.finish()
                beat = next(beats, None)

            time = yield (beat if beat is not None and time > beat.range[0] else None)

    @ra.DataNode.from_generator
    def get_knock_handler(self):
        beats_handler = self.get_beats_handler()

        with beats_handler:
            while True:
                time, strength, detected = yield
                time += self.start

                self.current_beat = beats_handler.send(time)

                if not detected:
                    continue

                # hit beat
                self.hit_index += 1
                self.hit_time = time
                self.hit_strength = min(1.0, strength)
                self.hit_beat = self.current_beat

                if self.current_beat is None:
                    continue

                self.current_beat.hit(time, strength)
                self.current_beat = beats_handler.send(time)

    def get_spectrum_handler(self):
        Dt = self.buffer_length / self.samplerate
        spec = ra.pipe(ra.frame(self.win_length, self.buffer_length),
                       ra.power_spectrum(self.win_length, samplerate=self.samplerate),
                       ra.draw_spectrum(self.spec_width, win_length=self.win_length,
                                                         samplerate=self.samplerate,
                                                         decay=Dt/self.spec_decay/4),
                       lambda s: setattr(self, "spectrum", s))
        return spec

    @ra.DataNode.from_generator
    def get_sound_handler(self, mixer):
        # generate sound
        if isinstance(self.audio, str):
            music = ra.load(self.audio)

            # add spec
            music = ra.chunk(music, chunk_shape=(self.buffer_length, self.channels))
            music = ra.pipe(music, ra.branch(self.get_spectrum_handler()))

            mixer.play(music, samplerate=self.samplerate, delay=-self.start)

        elif self.audio is not None:
            raise ValueError

        events_dripper = ra.drip(self.events, lambda e: e.lifespan)

        with events_dripper:
            time = (yield) + self.start
            while time < self.end:
                for event in events_dripper.send(time):
                    event.play(mixer, time)
                time = (yield) + self.start

    def draw_target(self, track, time):
        strength = self.hit_strength - (time - self.draw_time) / self.hit_decay
        strength = max(0.0, min(1.0, strength))
        loudness = int(strength * (len(self.target_syms) - 1))
        if abs(time - self.hit_time) < self.hit_sustain:
            loudness = max(1, loudness)
        track.draw_sym(0.0, self.target_syms[loudness])

    @ra.DataNode.from_generator
    def get_view_handler(self):
        bar_shift = 0.1
        width = int(os.popen("stty size", "r").read().split()[1])
        track = BeatTrack(width, bar_shift, self.spec_width)

        events_dripper = ra.drip(self.events, lambda e: e.lifespan)

        with events_dripper:
            try:
                while True:
                    time = yield
                    time += self.start

                    if self.draw_index != self.hit_index:
                        self.draw_time = time
                        self.draw_index = self.hit_index

                    # draw events
                    track.clear()
                    events = events_dripper.send(time)
                    for event in sorted(events[::-1], key=lambda e: e.zindex):
                        event.draw(track, time)

                    # draw target
                    stop_drawing_target = False
                    if not stop_drawing_target and self.current_beat is not None:
                        stop_drawing_target = self.current_beat.draw_judging(track, time)
                    if not stop_drawing_target and self.hit_beat is not None:
                        if abs(time - self.draw_time) < self.hit_sustain:
                            stop_drawing_target = self.hit_beat.draw_hitting(track, time)
                    if not stop_drawing_target:
                        self.draw_target(track, time)

                    # draw others
                    track.draw_spectrum(self.spectrum)
                    track.draw_score(self.score, self.total_score)
                    track.draw_progress(self.progress)

                    # render
                    print('\r' + str(track) + '\r', end='', flush=True)

            finally:
                print()


def make_std_regex():
    res = dict(
        number=r"[\+\-]?\d+(\.\d+|/\d+)?",
        str=r"'((?<!\\)\\'|.)*?'",
        mstr=r"'''((?<!\\)\\'''|.|\n)*?'''",
        nl=r"((#.*?)?(\n|\r))",
        sp=r" *",
        )

    notes = {
        "rest": r" ",
        "soft": r"( | time = {number} | speed = {number} | time = {number} , speed = {number} )",
        "loud": r"( | time = {number} | speed = {number} | time = {number} , speed = {number} )",
        "incr": r" {str} (, time = {number} )?(, speed = {number} )?",
        "roll": r" {number} , {number} (, time = {number} )?(, speed = {number} )?",
        "spin": r" {number} , {number} (, time = {number} )?(, speed = {number} )?",
        "sym": r" {str} (, time = {number} )?(, speed = {number} )?",
        "pattern": r" {number} , {number} , {mstr} "
        }

    res["note"] = "(" + "|".join((name + r" \(" + args + r"\)").replace(" ", "{sp}")
                                 for name, args in notes.items()).format(**res) + ")"

    header = r"#K-AIKO-std-(?P<version>\d+\.\d+\.\d+)(\n|\r)"

    main = r"""
    {nl}*
    (sheet \. metadata = {mstr} {nl}+)?
    (sheet \. audio = {str} {nl}+)?
    (sheet \. offset = {number} {nl}+)?
    (sheet \. tempo = {number} {nl}+)?
    (sheet \[ {str} \] = {note} {nl}+)*
    (sheet \+= {note} {nl}+)*
    """
    main = main.replace("\n    ", "").replace(" ", "{sp}").format(**res)

    return re.compile(header + main)

class BeatSheetStd:
    version = "0.0.1"
    regex = make_std_regex()

    def __init__(self):
        self.metadata = ""
        self.audio = None
        self.offset = 0.0
        self.tempo = 60.0

        self.incr_groups = dict()
        self.patterns = dict()
        self.events = []

    def time(self, t):
        return self.offset+t*60.0/self.tempo

    def rest(self):
        return lambda t: []

    def soft(self, time=0, speed=1.0):
        return lambda t: [Soft(self.time(time+t), speed=speed)]

    def loud(self, time=0, speed=1.0):
        return lambda t: [Loud(self.time(time+t), speed=speed)]

    def incr(self, group, time=0, speed=1.0):
        if group not in self.incr_groups:
            self.incr_groups[group] = IncrGroup()
        return lambda t: [self.incr_groups[group].add(self.time(time+t), speed=speed)]

    def roll(self, duration, step, time=0, speed=1.0):
        number = round(duration/step)+1
        return lambda t: [Roll(self.time(time+t), self.time(time+t+duration), number=number, speed=speed)]

    def spin(self, duration, step, time=0, speed=1.0):
        capacity = duration/step
        return lambda t: [Spin(self.time(time+t), self.time(time+t+duration), capacity=capacity, speed=speed)]

    def sym(self, symbol, time=0, speed=1.0):
        return lambda t: [Sym(self.time(time+t), symbol=symbol, speed=speed)]

    def pattern(self, time, step, term):
        return lambda t: [beat for i, p in enumerate(term.split()) for beat in self.patterns[p](time+t+i*step)]

    def __setitem__(self, key, value):
        if not isinstance(key, str) or re.search(r"\s", key):
            raise KeyError("invalid key: {!r}".format(key))
        self.patterns[key] = value

    def __iadd__(self, value):
        self.events += value(0)
        return self

    def load(self, str):
        match = self.regex.fullmatch(str)
        if not match:
            raise ValueError("invalid syntax")
        if match.group("version") != self.version:
            raise ValueError("wrong version: {}".format(match.group("version")))

        terms = {
            "sheet": self,
            "rest": self.rest,
            "soft": self.soft,
            "loud": self.loud,
            "incr": self.incr,
            "roll": self.roll,
            "spin": self.spin,
            "sym": self.sym,
            "pattern": self.pattern
            }
        exec(str, dict(), terms)

