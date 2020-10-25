import os
from enum import Enum
import inspect
import numpy
import audioread
import realtime_analysis as ra
from beatsheet import K_AIKO_STD


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
    # __init__(beatmap, context, **kwargs)
    # play(mixer, time)
    # draw(bar, time)
    pass

class Sym(Event):
    zindex = -2

    def __init__(self, beatmap, context, symbol=None, sound=None, *, beat, speed=1.0, samplerate=44100):
        self.time = beatmap.time(beat)
        self.symbol = symbol
        self.speed = speed
        self.sound = sound
        self.samplerate = samplerate
        self.played = False

    @property
    def lifespan(self):
        cross_time = 1.0 / abs(0.5 * self.speed)
        return (self.time-cross_time, self.time+cross_time)

    def play(self, mixer, time):
        if self.sound is not None and not self.played:
            self.played = True
            mixer.play(self.sound, samplerate=self.samplerate, delay=self.time-time)

    def draw(self, bar, time):
        if self.symbol is not None:
            pos = (self.time - time) * 0.5 * self.speed
            bar.draw_sym(pos, self.symbol)


# hit objects
class HitObject(Event):
    # lifespan, range, score, total_score, finished
    # __init__(beatmap, context, **kwargs)
    # hit(time, strength)
    # finish()
    # play(mixer, time)
    # draw(bar, time)
    # draw_judging(bar, time)
    # draw_hitting(bar, time)

    tolerances = TOLERANCES

    @property
    def zindex(self):
        return -1 if self.finished else 1

    def draw_judging(self, bar, time): pass
    def draw_hitting(self, bar, time): pass

class SingleHitObject(HitObject):
    # time, speed, volume, perf, played, symbol, sound, samplerate
    # hit(time, strength)

    total_score = 10
    perf_syms = PERF_SYMS
    wrong_symbol = WRONG_SYM

    def __init__(self, beatmap, context, *, beat, speed=1.0, volume=0.0):
        self.time = beatmap.time(beat)
        self.speed = speed
        self.volume = volume
        self.perf = None
        self.played = False

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
            sound = [s * 10**(self.volume/20) for s in self.sound]
            mixer.play(sound, samplerate=self.samplerate, delay=self.time-time)

    def draw(self, bar, time):
        CORRECT_TYPES = (Performance.GREAT,
                         Performance.LATE_GOOD, Performance.EARLY_GOOD,
                         Performance.LATE_BAD, Performance.EARLY_BAD,
                         Performance.LATE_FAILED, Performance.EARLY_FAILED)

        if self.perf in (None, Performance.MISS):
            pos = (self.time - time) * 0.5 * self.speed
            bar.draw_sym(pos, self.symbol)

        elif self.perf not in CORRECT_TYPES:
            pos = (self.time - time) * 0.5 * self.speed
            bar.draw_sym(pos, self.wrong_symbol)

    def draw_hitting(self, bar, time):
        self.perf.draw(bar, self.speed < 0, self.perf_syms)

class Performance(Enum):
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

    def draw(self, bar, flipped, perf_syms):
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
            bar.draw_sym(0.0, perf_syms[2])
        elif self in RIGHT_GOOD:
            bar.draw_sym(0.0, perf_syms[3])
        elif self in LEFT_BAD:
            bar.draw_sym(0.0, perf_syms[1])
        elif self in RIGHT_BAD:
            bar.draw_sym(0.0, perf_syms[4])
        elif self in LEFT_FAILED:
            bar.draw_sym(0.0, perf_syms[0])
        elif self in RIGHT_FAILED:
            bar.draw_sym(0.0, perf_syms[5])

class Soft(SingleHitObject):
    symbol = BEATS_SYMS[0]
    sound = [ra.pulse(samplerate=44100, freq=830.61, decay_time=0.03, amplitude=0.5)]
    samplerate = 44100

    def hit(self, time, strength):
        super().hit(time, strength, strength < 0.5)

class Loud(SingleHitObject):
    symbol = BEATS_SYMS[1]
    sound = [ra.pulse(samplerate=44100, freq=1661.2, decay_time=0.03, amplitude=1.0)]
    samplerate = 44100

    def hit(self, time, strength):
        super().hit(time, strength, strength >= 0.5)

class IncrGroup:
    def __init__(self, threshold=0.0, total=0):
        self.threshold = threshold
        self.total = total

    def hit(self, strength):
        self.threshold = max(self.threshold, strength)

class Incr(SingleHitObject):
    symbol = BEATS_SYMS[2]
    samplerate = 44100
    incr_tol = INCR_TOL

    def __init__(self, beatmap, context, group=None, *, beat, speed=1.0, volume=0.0):
        super().__init__(beatmap, context, beat=beat, speed=speed, volume=volume)

        if "incrs" not in context:
            context["incrs"] = OrderedDict()

        group_key = group
        if group_key is None:
            # determine group of incr note according to the context
            for key, (_, last_beat) in list(context["incrs"].items()).reverse():
                if beat - 1 <= last_beat <= beat:
                    group_key = key
                    break
            else:
                group_key = 0
                while group_key in context["incrs"]:
                    group_key += 1

        group, _ = context["incrs"].get(group_key, (IncrGroup(), beat))
        context["incrs"][group_key] = group, beat
        context["incrs"].move_to_end(group_key)

        group.total += 1
        self.count = group.total
        self.group = group

    def hit(self, time, strength):
        super().hit(time, strength, strength >= self.group.threshold - self.incr_tol)
        self.group.hit(strength)

    @property
    def sound(self):
        amplitude = (0.2 + 0.8 * (self.count-1)/self.group.total) * 10**(self.volume/20)
        return [ra.pulse(samplerate=44100, freq=1661.2, decay_time=0.03, amplitude=amplitude)]

class Roll(HitObject):
    symbol = BEATS_SYMS[3]
    sound = [ra.pulse(samplerate=44100, freq=1661.2, decay_time=0.01, amplitude=0.5)]
    samplerate = 44100

    def __init__(self, beatmap, context, density=2, *, beat, length, speed=1.0, volume=0.0):
        self.time = beatmap.time(beat)

        self.number = int(length * density)
        self.end = beatmap.time(beat+length)
        self.times = [beatmap.time(beat+i/density) for i in range(self.number)]

        self.speed = speed
        self.volume = volume
        self.roll = 0
        self.finished = False
        self.played = False

    @property
    def range(self):
        return (self.time - self.tolerances[2], self.end - self.tolerances[2])

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

            sound = [s * 10**(self.volume/20) for s in self.sound]
            for t in self.times:
                mixer.play(sound, samplerate=self.samplerate, delay=t-time)

    def draw(self, bar, time):
        for r, t in enumerate(self.times):
            if r > self.roll-1:
                pos = (t - time) * 0.5 * self.speed
                bar.draw_sym(pos, self.symbol)

class Spin(HitObject):
    total_score = 10
    symbols = BEATS_SYMS[4]
    sound = [ra.pulse(samplerate=44100, freq=1661.2, decay_time=0.01, amplitude=1.0)]
    samplerate = 44100
    finished_sym = SPIN_FINISHED_SYM

    def __init__(self, beatmap, context, density=2, *, beat, length, speed=1.0, volume=0.0):
        self.time = beatmap.time(beat)

        self.capacity = length * density
        self.end = beatmap.time(beat+length)
        self.times = [beatmap.time(beat+i/density) for i in range(int(self.capacity))]

        self.speed = speed
        self.volume = volume
        self.charge = 0.0
        self.finished = False
        self.played = False

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

            sound = [s * 10**(self.volume/20) for s in self.sound]
            for t in self.times:
                mixer.play(sound, samplerate=44100, delay=t-time)

    def draw(self, bar, time):
        if self.charge < self.capacity:
            pos = 0.0
            pos += max(0.0, (self.time - time) * 0.5 * self.speed)
            pos += min(0.0, (self.end - time) * 0.5 * self.speed)
            bar.draw_sym(pos, self.symbols[int(self.charge) % 4])

    def draw_judging(self, bar, time):
        return True

    def draw_hitting(self, bar, time):
        if self.charge == self.capacity:
            bar.draw_sym(0.0, self.finished_sym)
            return True


# beatmap
class ScrollingBar:
    def __init__(self, width, shift, spec_width):
        self.width = width
        self.shift = shift
        self.spec_width = spec_width

        self.chars = [' ']*width
        self.spec_offset = 1
        self.score_offset = self.spec_width + 2
        self.progress_offset = self.width - 9
        self.bar_offset = self.spec_width + 15
        self.bar_width = self.width - 24 - self.spec_width

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
        index = round((pos + self.shift) * (self.bar_width - 1))
        for ch in sym:
            if ch == ' ':
                index += 1
            elif ch == '\b':
                index -= 1
            else:
                if index in range(self.bar_width):
                    self.chars[self.bar_offset+index] = ch
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

    def __init__(self, info="", audio=None, offset=0.0, tempo=60.0):
        self.info = info
        self.audio = audio
        self.offset = offset
        self.tempo = tempo

        self.definitions = {}
        self.charts = []

        self["x"] = Soft
        self["o"] = Loud
        self["<"] = Incr
        self["%"] = Roll
        self["@"] = Spin
        self["s"] = Sym

    def time(self, beat):
        return self.offset + beat*60/self.tempo

    def beat(self, time):
        return (time - self.offset)*self.tempo/60

    def dtime(self, beat, length):
        return self.time(beat+length) - self.time(beat)

    def __setitem__(self, symbol, builder):
        if symbol == "_" or symbol in self.definitions:
            raise ValueError(f"symbol `{symbol}` is already defined")
        self.definitions[symbol] = NoteType(symbol, builder)

    def __delitem__(self, symbol):
        del self.definitions[symbol]

    def __getitem__(self, symbol):
        return self.definitions[symbol].builder

    def __iadd__(self, chart_str):
        self.charts.append(K_AIKO_STD_FORMAT.read_chart(chart_str, self.definitions))
        return self


    def __enter__(self):
        # audio metadata
        if self.audio is not None:
            with audioread.audio_open(self.audio) as file:
                self.duration = file.duration
                self.samplerate = file.samplerate
                self.channels = file.channels
        else:
            self.duration = 0.0
            self.samplerate = 44100
            self.channels = 1

        # events, hits
        self.events = [event for chart in self.charts for event in chart.build_events(self)]
        self.hits = [event for event in self.events if isinstance(event, HitObject)]
        self.start = min([0.0, *[event.lifespan[0] - self.prepare_time for event in self.events]])
        self.end = max([self.duration, *[event.lifespan[1] + self.prepare_time for event in self.events]])

        # hit state
        self.judging_object = None
        self.hit_index = 0
        self.hit_time = self.start - max(self.hit_decay, self.hit_sustain)*2
        self.hit_strength = 0.0
        self.hit_object = None
        self.draw_index = self.hit_index
        self.draw_time = self.hit_time

        # spectrum show
        self.spectrum = " "*self.spec_width

        return self

    def __exit__(self, type, value, traceback):
        return False

    @property
    def total_score(self):
        return sum(hit.total_score for hit in self.hits)

    @property
    def score(self):
        return sum(hit.score for hit in self.hits)

    @property
    def progress(self):
        if len(self.hits) == 0:
            return 1.0
        return sum(1 for hit in self.hits if hit.finished) / len(self.hits)

    @ra.DataNode.from_generator
    def get_hits_handler(self):
        hits = iter(sorted(self.hits, key=lambda e: e.range))
        hit = next(hits, None)

        time = yield
        while True:
            while hit is not None and (hit.finished or time > hit.range[1]):
                if not hit.finished:
                    hit.finish()
                hit = next(hits, None)

            time = yield (hit if hit is not None and time > hit.range[0] else None)

    @ra.DataNode.from_generator
    def get_knock_handler(self):
        hits_handler = self.get_hits_handler()

        with hits_handler:
            while True:
                time, strength, detected = yield
                time += self.start

                self.judging_object = hits_handler.send(time)

                if not detected:
                    continue

                # hit note
                self.hit_index += 1
                self.hit_time = time
                self.hit_strength = min(1.0, strength)
                self.hit_object = self.judging_object

                if self.judging_object is None:
                    continue

                self.judging_object.hit(time, strength)
                self.judging_object = hits_handler.send(time)

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

    def draw_target(self, bar, time):
        strength = self.hit_strength - (time - self.draw_time) / self.hit_decay
        strength = max(0.0, min(1.0, strength))
        loudness = int(strength * (len(self.target_syms) - 1))
        if abs(time - self.hit_time) < self.hit_sustain:
            loudness = max(1, loudness)
        bar.draw_sym(0.0, self.target_syms[loudness])

    @ra.DataNode.from_generator
    def get_view_handler(self):
        bar_shift = 0.1
        width = int(os.popen("stty size", "r").read().split()[1])
        bar = ScrollingBar(width, bar_shift, self.spec_width)

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
                    bar.clear()
                    events = events_dripper.send(time)
                    for event in sorted(events[::-1], key=lambda e: e.zindex):
                        event.draw(bar, time)

                    # draw target
                    stop_drawing_target = False
                    if not stop_drawing_target and self.judging_object is not None:
                        stop_drawing_target = self.judging_object.draw_judging(bar, time)
                    if not stop_drawing_target and self.hit_object is not None:
                        if abs(time - self.draw_time) < self.hit_sustain:
                            stop_drawing_target = self.hit_object.draw_hitting(bar, time)
                    if not stop_drawing_target:
                        self.draw_target(bar, time)

                    # draw others
                    bar.draw_spectrum(self.spectrum)
                    bar.draw_score(self.score, self.total_score)
                    bar.draw_progress(self.progress)

                    # render
                    print('\r' + str(bar) + '\r', end='', flush=True)

            finally:
                print()


class NoteChart:
    def __init__(self, *, beat=0, length=1, meter=4, hide=False):
        self.beat = beat
        self.length = length
        self.meter = meter
        self.hide = hide
        self.notes = []


    def build_events(self, beatmap):
        events = []

        if self.hide:
            return events

        context = dict()
        for note in self.notes:
            event = note.create(beatmap, context)
            if event is not None:
                events.append(event)

        return events

def NoteType(symbol, builder):
    # builder(beatmap, context, *, beat, **) -> Event | None
    signature = inspect.signature(builder)
    signature = signature.replace(parameters=list(signature.parameters.values())[2:])
    return type(builder.__name__+"Note", (Note,), dict(symbol=symbol, builder=builder, __signature__=signature))

class Note:
    def __init__(self, *posargs, **kwargs):
        self.bound = self.__signature__.bind(*posargs, **kwargs)

    def __str__(self):
        posargs_str = [repr(value) for value in self.bound.args]
        kwargs_str = [key+"="+repr(value) for key, value in self.bound.kwargs.items()]
        args_str = ", ".join([*posargs_str, *kwargs_str])
        return f"{self.symbol}({args_str})"

    def create(self, beatmap, context):
        return self.builder(beatmap, context, *self.bound.args, **self.bound.kwargs)


K_AIKO_STD_FORMAT = K_AIKO_STD(Beatmap, NoteChart)

