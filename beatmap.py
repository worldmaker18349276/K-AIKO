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

INCR_TOL = 0.1
SPEC_WIDTH = 5
HIT_DECAY = 0.4
HIT_SUSTAIN = 0.1
PREPARE_TIME = 1.0
SKIP_TIME = 8.0


# scripts
class Event:
    # lifespan, zindex, sounds
    # def draw(self, track, time): pass
    pass

class Sym(Event):
    zindex = -2

    def __init__(self, time, symbol=" ", speed=1.0):
        self.time = time
        self.symbol = symbol
        self.speed = speed
        self.sounds = []

    @property
    def lifespan(self):
        cross_time = 1.0 / abs(0.5 * self.speed)
        return (self.time-cross_time, self.time+cross_time)

    def draw(self, track, time):
        pos = (self.time - time) * 0.5 * self.speed
        track.addstr(pos, self.symbol)

    def __repr__(self):
        return "Sym(time={!r}, symbol={!r}, speed={!r})".format(self.time, self.symbol, self.speed)


# beats
class Beat(Event):
    # lifespan, zindex, sounds, range, time, speed, score, total_score, finished
    # def hit(self, time, strength): pass
    # def finish(self): pass
    # def draw(self, track, time): pass
    # def draw_judging(self, track, time): pass
    # def draw_hitting(self, track, time): pass

    tolerances = TOLERANCES

    @property
    def zindex(self):
        return -1 if self.finished else 1

    @property
    def lifespan(self):
        cross_time = 1.0 / abs(0.5 * self.speed)
        start, end = self.range
        return (start-cross_time, end+cross_time)

    def draw_judging(self, track, time): pass
    def draw_hitting(self, track, time): pass

class SingleBeat(Beat):
    total_score = 10
    perf_syms = PERF_SYMS
    wrong_symbol = WRONG_SYM

    def __init__(self, time, speed=1.0, perf=None):
        self.time = time
        self.speed = speed
        self.perf = perf

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

    def draw(self, track, time):
        CORRECT_TYPES = (Performance.GREAT,
                         Performance.LATE_GOOD, Performance.EARLY_GOOD,
                         Performance.LATE_BAD, Performance.EARLY_BAD,
                         Performance.LATE_FAILED, Performance.EARLY_FAILED)

        if self.perf in (None, Performance.MISS):
            pos = (self.time - time) * 0.5 * self.speed
            track.addstr(pos, self.symbol)

        elif self.perf not in CORRECT_TYPES:
            pos = (self.time - time) * 0.5 * self.speed
            track.addstr(pos, self.wrong_symbol)

    def draw_hitting(self, track, time):
        self.perf.draw(track, self.speed < 0, self.perf_syms)

class Soft(SingleBeat):
    symbol = BEATS_SYMS[0]

    @property
    def sounds(self):
        yield ([ra.pulse(samplerate=44100, freq=830.61, decay_time=0.03, amplitude=0.5)], 44100, self.time)

    def hit(self, time, strength):
        super().hit(time, strength, strength < 0.5)

    def __repr__(self):
        return "Soft(time={!r}, speed={!r}, perf={!r})".format(self.time, self.speed, self.perf)

class Loud(SingleBeat):
    symbol = BEATS_SYMS[1]

    @property
    def sounds(self):
        yield ([ra.pulse(samplerate=44100, freq=1661.2, decay_time=0.03, amplitude=1.0)], 44100, self.time)

    def hit(self, time, strength):
        super().hit(time, strength, strength >= 0.5)

    def __repr__(self):
        return "Loud(time={!r}, speed={!r}, perf={!r})".format(self.time, self.speed, self.perf)

class IncrGroup:
    def __init__(self, threshold=0.0, total=0):
        self.threshold = threshold
        self.total = total

    def add(self, time, speed=1.0, perf=None):
        self.total += 1
        return Incr(time, speed, perf, count=self.total, group=self)

    def hit(self, strength):
        self.threshold = max(self.threshold, strength)

    def __repr__(self):
        return "IncrGroup(threshold={!r}, total={!r})".format(self.threshold, self.total)

class Incr(SingleBeat):
    symbol = BEATS_SYMS[2]
    incr_tol = INCR_TOL

    def __init__(self, time, speed=1.0, perf=None, count=None, group=None):
        super().__init__(time, speed, perf)
        if count is None or group is None:
            raise ValueError
        self.count = count
        self.group = group

    def hit(self, time, strength):
        super().hit(time, strength, strength >= self.group.threshold - self.incr_tol)
        self.group.hit(strength)

    @property
    def sounds(self):
        amplitude = 0.2 + 0.8 * (self.count-1)/self.group.total
        sound = ra.pulse(samplerate=44100, freq=1661.2, decay_time=0.03, amplitude=amplitude)
        yield ([sound], 44100, self.time)

    def __repr__(self):
        return "Incr(time={!r}, speed={!r}, perf={!r}, count={!r}, group={!r})".format(
                     self.time, self.speed, self.perf, self.count, self.group)

class Roll(Beat):
    symbol = BEATS_SYMS[3]

    def __init__(self, time, end, number, speed=1.0, roll=0, finished=False):
        self.time = time
        self.end = end
        self.speed = speed
        self.number = number
        self.roll = roll
        self.finished = finished

    @property
    def range(self):
        return (self.time - self.tolerances[2], self.end)

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
    def sounds(self):
        sound = ra.pulse(samplerate=44100, freq=1661.2, decay_time=0.01, amplitude=0.5)
        step = (self.end - self.time)/(self.number-1) if self.number > 1 else 0.0
        for i in range(self.number):
            yield ([sound], 44100, self.time+step*i)

    def draw(self, track, time):
        step = (self.end - self.time)/(self.number-1) if self.number > 1 else 0.0

        for r in range(self.number):
            if r > self.roll-1:
                pos = (self.time + step * r - time) * 0.5 * self.speed
                track.addstr(pos, self.symbol)

    def __repr__(self):
        return "Roll(time={!r}, end={!r}, number={!r}, speed={!r}, roll={!r}, finished={!r})".format(
                     self.time, self.end, self.number, self.speed, self.roll, self.finished)

class Spin(Beat):
    total_score = 10
    symbols = BEATS_SYMS[4]
    finished_sym = SPIN_FINISHED_SYM

    def __init__(self, time, end, capacity, speed=1.0, charge=0.0, finished=False):
        self.time = time
        self.end = end
        self.speed = speed
        self.capacity = capacity
        self.charge = charge
        self.finished = finished

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
    def sounds(self):
        sound = ra.pulse(samplerate=44100, freq=1661.2, decay_time=0.01, amplitude=1.0)
        step = (self.end - self.time)/self.capacity if self.capacity > 0.0 else 0.0
        for i in range(int(self.capacity)):
            yield ([sound], 44100, self.time+step*i)

    def draw(self, track, time):
        if self.charge < self.capacity:
            pos = 0.0
            pos += max(0.0, (self.time - time) * 0.5 * self.speed)
            pos += min(0.0, (self.end - time) * 0.5 * self.speed)
            track.addstr(pos, self.symbols[int(self.charge) % 4])

    def draw_judging(self, track, time):
        return True

    def draw_hitting(self, track, time):
        if self.charge == self.capacity:
            track.addstr(0.0, self.finished_sym)
            return True

    def __repr__(self):
        return "Spin(time={!r}, end={!r}, capacity={!r}, speed={!r}, charge={!r}, finished={!r})".format(
                     self.time, self.end, self.capacity, self.speed, self.charge, self.finished)

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
            track.addstr(0.0, perf_syms[2])
        elif self in RIGHT_GOOD:
            track.addstr(0.0, perf_syms[3])
        elif self in LEFT_BAD:
            track.addstr(0.0, perf_syms[1])
        elif self in RIGHT_BAD:
            track.addstr(0.0, perf_syms[4])
        elif self in LEFT_FAILED:
            track.addstr(0.0, perf_syms[0])
        elif self in RIGHT_FAILED:
            track.addstr(0.0, perf_syms[5])


# beatmap
class BeatTrack:
    def __init__(self, win, width, offset, shift):
        self.win = win
        self.width = width
        self.offset = offset
        self.shift = shift
        self.chars = [' ']*width

    def clear(self):
        for i in range(self.width):
            self.chars[i] = ' '

    def refresh(self):
        for i, ch in enumerate(self.chars):
            if ch != ' ':
                self.win.addstr(0, self.offset+i, ch)

    def addstr(self, pos, msg):
        i = round((pos + self.shift) * (self.width - 1))
        for ch in msg:
            if ch == ' ':
                i += 1
            elif ch == '\b':
                i -= 1
            else:
                if i in range(self.width):
                    self.chars[i] = ch
                i += 1

class Beatmap:
    prepare_time = PREPARE_TIME
    spec_width = SPEC_WIDTH
    buffer_length = 512
    win_length = 512*4
    spec_decay = 0.01

    hit_decay = HIT_DECAY
    hit_sustain = HIT_SUSTAIN
    target_syms = TARGET_SYMS

    def __init__(self, audio, events):
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

        self.events = list(events)
        self.beats = list(event for event in self.events if isinstance(event, Beat))
        self.start = min(0.0, min(event.lifespan[0] - self.prepare_time for event in self.events))
        self.end = max(self.duration, max(event.lifespan[1] + self.prepare_time for event in self.events))

        self.spectrum = " "*self.spec_width

        self.current_beat = None
        self.hit_index = 0
        self.hit_time = self.start - max(self.hit_decay, self.hit_sustain)*2
        self.hit_strength = 0.0
        self.hit_beat = None
        self.draw_index = 0

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
            return 1000
        return sum(1 for beat in self.beats if beat.finished) * 1000 // len(self.beats)

    @ra.DataNode.from_generator
    def get_knock_handler(self):
        beats = iter(sorted(self.beats, key=lambda e: e.range[0]))
        beat = next(beats, None)

        while True:
            time, strength, detected = yield
            time += self.start

            # drip beats
            while beat is not None and (beat.finished or beat.range[1] < time):
                if not beat.finished:
                    beat.finish()
                beat = next(beats, None)

            self.current_beat = beat if beat is not None and beat.range[0] < time else None

            if not detected:
                continue

            # hit beat
            self.hit_index += 1
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

            mixer.play(music, samplerate=self.samplerate, time=-self.start)

        elif self.audio is not None:
            raise ValueError

        # events = sorted(self.events, key=lambda e: e.lifespan[0])
        # 
        # time = (yield) + self.start
        # for event in events:
        #     while time < event.lifespan[0]:
        #         time = (yield) + self.start
        # 
        #     for beat_sound, samplerate, play_time in event.sounds:
        #         mixer.play(beat_sound, samplerate=samplerate, time=play_time-self.start)
        # 
        # else:
        #     while time < self.end:
        #         time = (yield) + self.start

        events = sorted(self.events, key=lambda e: e.lifespan[0])
        for event in events:
            for beat_sound, samplerate, play_time in event.sounds:
                mixer.play(beat_sound, samplerate=samplerate, time=play_time-self.start)

        time = (yield) + self.start
        while time < self.end:
            time = (yield) + self.start

    def draw_target(self, track, time):
        strength = self.hit_strength - (time - self.hit_time) / self.hit_decay
        strength = max(0.0, min(1.0, strength))
        loudness = int(strength * (len(self.target_syms) - 1))
        if abs(time - self.hit_time) < self.hit_sustain:
            loudness = max(1, loudness)
        track.addstr(0.0, self.target_syms[loudness])

    @ra.DataNode.from_generator
    def get_screen_handler(self, scr):
        bar_offset = 0.1
        dripper = ra.drip(self.events, lambda e: e.lifespan)

        width = scr.getmaxyx()[1]

        spec_offset = 1
        score_offset = self.spec_width + 2
        track_offset = self.spec_width + 15
        progress_offset = width - 9
        track_width = width - 24 - self.spec_width

        track = BeatTrack(scr, track_width, track_offset, bar_offset)

        with dripper:
            while True:
                time = yield
                time += self.start

                if self.draw_index != self.hit_index:
                    self.hit_time = time
                    self.draw_index = self.hit_index

                # draw events
                track.clear()

                ## find visible events
                events = dripper.send(time)
                events = sorted(events, key=lambda e: -e.zindex)
                for event in events[::-1]:
                    event.draw(track, time)

                # draw target
                stop_drawing_target = False
                if not stop_drawing_target and self.current_beat is not None:
                    stop_drawing_target = self.current_beat.draw_judging(track, time)
                if not stop_drawing_target and self.hit_beat is not None:
                    if abs(time - self.hit_time) < self.hit_sustain:
                        stop_drawing_target = self.hit_beat.draw_hitting(track, time)
                if not stop_drawing_target:
                    self.draw_target(track, time)

                # draw others
                scr.clear()
                track.refresh()
                scr.addstr(0, spec_offset, self.spectrum)
                scr.addstr(0, score_offset, "[{:>5d}/{:>5d}]".format(self.score, self.total_score))
                scr.addstr(0, progress_offset, "[{:>5.1f}%]".format(self.progress/10))


class BeatmapStdSheet:
    def __init__(self):
        self.metadata = ""
        self.audio = None
        self.offset = 0.0
        self.tempo = 120.0

        self.incr_groups = dict()
        self.patterns = dict()
        self.events = []

    def time(self, t):
        return self.offset+t*60.0/self.tempo

    def skip(self):
        return lambda t: []

    def soft(self, speed=1.0):
        return lambda t: [Soft(self.time(t), speed=speed)]

    def loud(self, speed=1.0):
        return lambda t: [Loud(self.time(t), speed=speed)]

    def incr(self, group, speed=1.0):
        if group not in self.incr_groups:
            self.incr_groups[group] = IncrGroup()
        return lambda t: [self.incr_groups[group].add(self.time(t), speed=speed)]

    def roll(self, duration, step, speed=1.0):
        number = round(duration/step)+1
        return lambda t: [Roll(self.time(t), self.time(t+duration), number=number, speed=speed)]

    def spin(self, duration, step, speed=1.0):
        capacity = duration/step
        return lambda t: [Spin(self.time(t), self.time(t+duration), capacity=capacity, speed=speed)]

    def sym(self, symbol, speed=1.0):
        return lambda t: [Sym(self.time(t), symbol=symbol, speed=speed)]

    def pattern(self, offset, step, term):
        if hasattr(term, "__call__"):
            return lambda t: term(offset+t)
        elif isinstance(term, str):
            return lambda t: [beat for i, p in enumerate(term.split()) for beat in self.patterns[p](offset+t+i*step)]
        else:
            raise ValueError("invalid term: {!r}".format(term))

    def __setitem__(self, key, value):
        if not isinstance(key, str) or re.search(r"\s", key):
            raise KeyError("invalid key: {!r}".format(key))
        self.patterns[key] = self.pattern(*value)

    def __iadd__(self, value):
        self.events += self.pattern(*value)(0)
        return self

