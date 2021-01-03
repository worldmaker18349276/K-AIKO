import os
import datetime
from enum import Enum
from typing import List, Tuple, Dict, Optional, Union
from collections import OrderedDict
import queue
import numpy
import audioread
from . import cfg
from . import datanodes as dn


class Event:
    # lifespan
    # __init__(beatmap, *args, **kwargs)
    # register(field)
    is_subject = False
    full_score = 0

class Text(Event):
    def __init__(self, beatmap, text=None, sound=None, beat=None, *, speed=1.0):
        if sound is not None:
            sound = os.path.join(beatmap.path, sound)

        self.time = beatmap.time(beat)
        self.speed = speed
        self.text = text
        self.sound = sound

        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.time + travel_time)
        self.zindex = (-2, -self.time)

    def pos(self, time):
        return (self.time-time) * 0.5 * self.speed

    def register(self, field):
        if self.sound is not None:
            field.play(self.sound, time=self.time)

        if self.text is not None:
            field.draw_content(self.pos, self.text, zindex=self.zindex,
                               start=self.lifespan[0], duration=self.lifespan[1]-self.lifespan[0])

# scripts
class Flip(Event):
    def __init__(self, beatmap, flip=None, beat=None):
        self.time = beatmap.time(beat)
        self.flip = flip
        self.lifespan = (self.time, self.time)

    def register(self, field):
        field.on_before_render(self._node(field))

    @dn.datanode
    def _node(self, field):
        time, screen = yield
        time -= field.start_time

        while time < self.time:
            time, screen = yield
            time -= field.start_time

        if self.flip is None:
            field.bar_flip = not field.bar_flip
        else:
            field.bar_flip = self.flip

        time, screen = yield
        time -= field.start_time

class Shift(Event):
    def __init__(self, beatmap, shift, beat=None, length=None):
        self.time = beatmap.time(beat)
        self.end = beatmap.time(beat+length)
        self.shift = shift
        self.lifespan = (self.time, self.end)

    def register(self, field):
        field.on_before_render(self._node(field))

    @dn.datanode
    def _node(self, field):
        time, screen = yield
        time -= field.start_time

        while time < self.time:
            time, screen = yield
            time -= field.start_time

        shift0 = field.bar_shift
        speed = (self.shift - shift0) / (self.end - self.time) if self.end != self.time else 0

        while time < self.end:
            field.bar_shift = shift0 + speed * (time - self.time)
            time, screen = yield
            time -= field.start_time

        field.bar_shift = self.shift

        time, screen = yield
        time -= field.start_time

def set_context(beatmap, *, context, **kw):
    context.update(**kw)

# targets
class Target(Event):
    # lifespan, range, is_finished
    # __init__(beatmap, *args, **kwargs)
    # approach(field)
    # hit(field, time, strength)
    # finish(field)
    is_subject = True

    @dn.datanode
    def listen(self, field):
        try:
            while True:
                time, strength = yield
                self.hit(field, time, strength)
                if self.is_finished:
                    break
        except GeneratorExit:
            if not self.is_finished:
                self.finish(field)
        finally:
            field.add_finished()

    def zindex(self):
        return (0, not self.is_finished, -self.range[0])

    def register(self, field):
        self.approach(field)
        field.listen(self.listen(field), start=self.range[0], duration=self.range[1]-self.range[0])

class PerformanceGrade(Enum):
    MISS               = (None, None)
    PERFECT            = ( 0, False)
    LATE_GOOD          = (+1, False)
    EARLY_GOOD         = (-1, False)
    LATE_BAD           = (+2, False)
    EARLY_BAD          = (-2, False)
    LATE_FAILED        = (+3, False)
    EARLY_FAILED       = (-3, False)
    PERFECT_WRONG      = ( 0,  True)
    LATE_GOOD_WRONG    = (+1,  True)
    EARLY_GOOD_WRONG   = (-1,  True)
    LATE_BAD_WRONG     = (+2,  True)
    EARLY_BAD_WRONG    = (-2,  True)
    LATE_FAILED_WRONG  = (+3,  True)
    EARLY_FAILED_WRONG = (-3,  True)

    def __init__(self, shift, is_wrong):
        self.shift = shift
        self.is_wrong = is_wrong

    def __repr__(self):
        return f"PerformanceGrade.{self.name}"

class Performance:
    def __init__(self, grade, time, err):
        self.grade = grade
        self.time = time
        self.err = err

    @staticmethod
    def judge(tol, time, hit_time=None, is_correct_key=True):
        if hit_time is None:
            return Performance(PerformanceGrade((None, None)), time, None)

        is_wrong = not is_correct_key
        err = hit_time - time
        shift = next((i for i in range(3) if abs(err) < tol*(2*i+1)), 3)
        if err < 0:
            shift = -shift

        return Performance(PerformanceGrade((shift, is_wrong)), time, err)

    @property
    def shift(self):
        return self.grade.shift

    @property
    def is_wrong(self):
        return self.grade.is_wrong

    @property
    def is_miss(self):
        return self.grade == PerformanceGrade.MISS

    discriptions = {
        PerformanceGrade.MISS               : "Miss"                      ,
        PerformanceGrade.PERFECT            : "Perfect"                   ,
        PerformanceGrade.LATE_GOOD          : "Late Good"                 ,
        PerformanceGrade.EARLY_GOOD         : "Early Good"                ,
        PerformanceGrade.LATE_BAD           : "Late Bad"                  ,
        PerformanceGrade.EARLY_BAD          : "Early Bad"                 ,
        PerformanceGrade.LATE_FAILED        : "Late Failed"               ,
        PerformanceGrade.EARLY_FAILED       : "Early Failed"              ,
        PerformanceGrade.PERFECT_WRONG      : "Perfect but Wrong Key"     ,
        PerformanceGrade.LATE_GOOD_WRONG    : "Late Good but Wrong Key"   ,
        PerformanceGrade.EARLY_GOOD_WRONG   : "Early Good but Wrong Key"  ,
        PerformanceGrade.LATE_BAD_WRONG     : "Late Bad but Wrong Key"    ,
        PerformanceGrade.EARLY_BAD_WRONG    : "Early Bad but Wrong Key"   ,
        PerformanceGrade.LATE_FAILED_WRONG  : "Late Failed but Wrong Key" ,
        PerformanceGrade.EARLY_FAILED_WRONG : "Early Failed but Wrong Key",
    }

    @property
    def description(self):
        return self.discriptions[self.grade]

class OneshotTarget(Target):
    # time, speed, volume, perf, sound
    # approach_appearance, wrong_appearance
    # hit(field, time, strength)

    def __init__(self, beatmap, beat=None, *, speed=1.0, volume=0.0):
        self.performance_tolerance = beatmap.settings.performance_tolerance

        self.time = beatmap.time(beat)
        self.speed = speed
        self.volume = volume
        self.perf = None

        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.time + travel_time)
        tol = beatmap.settings.failed_tolerance
        self.range = (self.time-tol, self.time+tol)
        self._scores = beatmap.settings.performances_scores
        self.full_score = beatmap.settings.performances_max_score

    def pos(self, time):
        return (self.time-time) * 0.5 * self.speed

    def appearance(self, time):
        if not self.is_finished:
            return self.approach_appearance
        elif self.perf.is_miss:
            return self.approach_appearance
        elif self.perf.is_wrong:
            return self.wrong_appearance
        else:
            return ""

    @property
    def score(self):
        return self._scores[self.perf.grade] if self.perf is not None else 0

    @property
    def is_finished(self):
        return self.perf is not None

    @property
    def perfs(self):
        return (self.perf,) if self.perf is not None else ()

    def approach(self, field):
        if self.sound is not None:
            field.play(self.sound, time=self.time, volume=self.volume)

        field.draw_content(self.pos, self.appearance, zindex=self.zindex,
                           start=self.lifespan[0], duration=self.lifespan[1]-self.lifespan[0])
        field.reset_sight(start=self.range[0])

    def hit(self, field, time, strength, is_correct_key=True):
        perf = Performance.judge(self.performance_tolerance, self.time, time, is_correct_key)
        field.set_perf_hint(perf, self.speed < 0)
        self.finish(field, perf)

    def finish(self, field, perf=None):
        if perf is None:
            perf = Performance.judge(self.performance_tolerance, self.time)
        self.perf = perf
        field.add_full_score(self.full_score)
        field.add_score(self.score)

class Soft(OneshotTarget):
    def __init__(self, beatmap, beat=None, *, speed=1.0, volume=0.0):
        super().__init__(beatmap, beat=beat, speed=speed, volume=volume)
        self.approach_appearance = beatmap.settings.soft_approach_appearance
        self.wrong_appearance = beatmap.settings.soft_wrong_appearance
        self.sound = beatmap.settings.soft_sound
        self.threshold = beatmap.settings.soft_threshold

    def hit(self, field, time, strength):
        super().hit(field, time, strength, strength < self.threshold)

class Loud(OneshotTarget):
    def __init__(self, beatmap, beat=None, *, speed=1.0, volume=0.0):
        super().__init__(beatmap, beat=beat, speed=speed, volume=volume)
        self.approach_appearance = beatmap.settings.loud_approach_appearance
        self.wrong_appearance = beatmap.settings.loud_wrong_appearance
        self.sound = beatmap.settings.loud_sound
        self.threshold = beatmap.settings.loud_threshold

    def hit(self, field, time, strength):
        super().hit(field, time, strength, strength >= self.threshold)

class IncrGroup:
    def __init__(self, threshold=0.0, total=0):
        self.threshold = threshold
        self.total = total
        self.volume = 0.0

    def hit(self, strength):
        self.threshold = max(self.threshold, strength)

class Incr(OneshotTarget):
    def __init__(self, beatmap, group=None, beat=None, *, context, speed=1.0, volume=0.0):
        super().__init__(beatmap, beat=beat, speed=speed)

        self.approach_appearance = beatmap.settings.incr_approach_appearance
        self.wrong_appearance = beatmap.settings.incr_wrong_appearance
        self.sound = beatmap.settings.incr_sound
        self.incr_threshold = beatmap.settings.incr_threshold

        if '_incrs' not in context:
            context['_incrs'] = OrderedDict()
        incrs = context['_incrs']

        group_key = group
        if group_key is None:
            # determine group of incr note according to the context
            for key, (_, last_beat) in reversed(incrs.items()):
                if beat - 1 <= last_beat <= beat:
                    group_key = key
                    break
            else:
                group_key = 0
                while group_key in incrs:
                    group_key += 1

        group, _ = incrs.get(group_key, (IncrGroup(), beat))
        if group_key not in incrs:
            group.volume = volume
        incrs[group_key] = group, beat
        incrs.move_to_end(group_key)

        group.total += 1
        self.count = group.total
        self.group = group

    @property
    def volume(self):
        return self.group.volume + numpy.log10(0.2 + 0.8 * (self.count-1)/self.group.total) * 20

    @volume.setter
    def volume(self, value):
        pass

    def hit(self, field, time, strength):
        threshold = max(0.0, min(1.0, self.group.threshold + self.incr_threshold))
        super().hit(field, time, strength, strength >= threshold)
        self.group.hit(strength)

class Roll(Target):
    def __init__(self, beatmap, density=2, beat=None, length=None, *, speed=1.0, volume=0.0):
        self.performance_tolerance = beatmap.settings.performance_tolerance
        self.tolerance = beatmap.settings.roll_tolerance
        self.rock_appearance = beatmap.settings.roll_rock_appearance
        self.sound = beatmap.settings.roll_rock_sound
        self.rock_score = beatmap.settings.roll_rock_score

        self.time = beatmap.time(beat)
        self.end = beatmap.time(beat+length)
        self.speed = speed
        self.volume = volume
        self.roll = 0
        self.number = int(length * density)
        self.is_finished = False
        self.score = 0

        self.times = [beatmap.time(beat+i/density) for i in range(self.number)]
        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.end + travel_time)
        self.perfs = []
        self.range = (self.time - self.tolerance, self.end - self.tolerance)
        self.full_score = self.number * self.rock_score

    def pos_of(self, index):
        return lambda time: (self.times[index]-time) * 0.5 * self.speed

    def appearance_of(self, index):
        return lambda time: self.rock_appearance if self.roll <= index else ""

    def approach(self, field):
        for i, time in enumerate(self.times):
            if self.sound is not None:
                field.play(self.sound, time=time, volume=self.volume)
            field.draw_content(self.pos_of(i), self.appearance_of(i), zindex=self.zindex,
                               start=self.lifespan[0], duration=self.lifespan[1]-self.lifespan[0])
        field.reset_sight(start=self.range[0])

    def hit(self, field, time, strength):
        self.roll += 1

        if self.roll <= self.number:
            perf = Performance.judge(self.performance_tolerance, self.times[self.roll-1], time, True)
            self.perfs.append(perf)

            field.add_score(self.rock_score)
            self.score += self.rock_score

        elif self.roll <= 2*self.number:
            field.add_score(-self.rock_score)
            self.score -= self.rock_score

    def finish(self, field):
        self.is_finished = True
        field.add_full_score(self.full_score)

        for time in self.times[self.roll:]:
            perf = Performance.judge(self.performance_tolerance, time)
            self.perfs.append(perf)

class Spin(Target):
    def __init__(self, beatmap, density=2, beat=None, length=None, *, speed=1.0, volume=0.0):
        self.tolerance = beatmap.settings.spin_tolerance
        self.disk_appearances = beatmap.settings.spin_disk_appearances
        self.finishing_appearance = beatmap.settings.spin_finishing_appearance
        self.finish_sustain_time = beatmap.settings.spin_finish_sustain_time
        self.sound = beatmap.settings.spin_disk_sound
        self.full_score = beatmap.settings.spin_score

        self.time = beatmap.time(beat)
        self.end = beatmap.time(beat+length)
        self.speed = speed
        self.volume = volume
        self.charge = 0.0
        self.capacity = length * density
        self.is_finished = False
        self.score = 0

        self.times = [beatmap.time(beat+i/density) for i in range(int(self.capacity))]
        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.end + travel_time)
        self.range = (self.time - self.tolerance, self.end + self.tolerance)

    def pos(self, time):
        return (max(0.0, self.time-time) + min(0.0, self.end-time)) * 0.5 * self.speed

    def appearance(self, time):
        return self.disk_appearances[int(self.charge) % len(self.disk_appearances)] if not self.is_finished else ""

    def approach(self, field):
        for time in self.times:
            if self.sound is not None:
                field.play(self.sound, time=time, volume=self.volume)

        field.draw_content(self.pos, self.appearance, zindex=self.zindex,
                           start=self.lifespan[0], duration=self.lifespan[1]-self.lifespan[0])
        field.draw_sight("", start=self.range[0], duration=self.range[1]-self.range[0])

    def hit(self, field, time, strength):
        self.charge = min(self.charge + min(1.0, strength), self.capacity)

        current_score = int(self.full_score * self.charge / self.capacity)
        field.add_score(current_score - self.score)
        self.score = current_score

        if self.charge == self.capacity:
            self.finish(field)

    def finish(self, field):
        self.is_finished = True
        field.add_full_score(self.full_score)

        if self.charge != self.capacity:
            field.add_score(-self.score)
            self.score = 0

        if self.charge != self.capacity:
            return

        appearance = self.finishing_appearance
        if isinstance(appearance, tuple) and self.speed < 0:
            appearance = appearance[::-1]
        field.draw_sight(appearance, duration=self.finish_sustain_time)


# Play Field
def to_slices(segments):
    middle = segments.index(...)
    pre  = segments[:middle:+1]
    post = segments[:middle:-1]

    pre_index  = [sum(pre[:i+1])  for i in range(len(pre))]
    post_index = [sum(post[:i+1]) for i in range(len(post))]

    first_slice  = slice(None, pre_index[0], None)
    last_slice   = slice(-post_index[0], None, None)
    middle_slice = slice(pre_index[-1], -post_index[-1], None)

    pre_slices  = [slice(+a, +b, None) for a, b in zip(pre_index[:-1],  pre_index[1:])]
    post_slices = [slice(-b, -a, None) for a, b in zip(post_index[:-1], post_index[1:])]

    return [first_slice, *pre_slices, middle_slice, *post_slices[::-1], last_slice]

@cfg.configurable
class BeatbarSettings:
    # BeatbarSkin:
    icon_width: int = 8
    header_width: int = 11
    footer_width: int = 12

    icon_formats: List[str] = [""]
    header_formats: List[str] = [""]
    footer_formats: List[str] = [""]

class Beatbar:
    settings : BeatbarSettings = BeatbarSettings()

    def __init__(self):
        self.status = {}

        # layout
        icon_width = self.settings.icon_width
        header_width = self.settings.header_width
        footer_width = self.settings.footer_width
        layout = to_slices((icon_width, 1, header_width, 1, ..., 1, footer_width, 1))
        self.icon_mask, _, self.header_mask, _, self.content_mask, _, self.footer_mask, _ = layout

    def register_handlers(self, console):
        self.console = console

        # register
        self.console.add_drawer(self._status_handler(), zindex=(-3,))

    @dn.datanode
    def _status_handler(self):
        icon_formats = self.settings.icon_formats
        header_formats = self.settings.header_formats
        footer_formats = self.settings.footer_formats

        while True:
            _, screen = yield

            icon_start, icon_stop, _ = self.icon_mask.indices(screen.width)
            for icon_format in icon_formats:
                icon_text = icon_format.format(**self.status)
                ran = screen.range(icon_start, icon_text)
                if ran.start >= icon_start and ran.stop <= icon_stop:
                    break
            screen.addstr(icon_start, icon_text, self.icon_mask)

            header_start, header_stop, _ = self.header_mask.indices(screen.width)
            for header_format in header_formats:
                header_text = header_format.format(**self.status)
                ran = screen.range(header_start, header_text)
                if ran.start >= header_start and ran.stop <= header_stop:
                    break
            screen.addstr(header_start, header_text, self.header_mask)
            screen.addstr(header_start-1, "[")
            screen.addstr(header_stop, "]")

            footer_start, footer_stop, _ = self.footer_mask.indices(screen.width)
            for footer_format in footer_formats:
                footer_text = footer_format.format(**self.status)
                ran = screen.range(footer_start, footer_text)
                if ran.start >= footer_start and ran.stop <= footer_stop:
                    break
            screen.addstr(footer_start, footer_text, self.footer_mask)
            screen.addstr(footer_start-1, "[")
            screen.addstr(footer_stop, "]")

@cfg.configurable
class PlayFieldSettings(BeatbarSettings):
    # PlayFieldSkin:
    icon_width: int = 8
    header_width: int = 11
    footer_width: int = 12

    icon_formats: List[str] = ["{spectrum:^8s}"]
    header_formats: List[str] = ["{score:05d}/{full_score:05d}"]
    footer_formats: List[str] = ["{progress:>6.1%}|{time:%M:%S}"]

    spec_width: int = 7
    spec_decay_time: float = 0.01
    spec_time_res: float = 0.0116099773 # hop_length = 512 if samplerate == 44100
    spec_freq_res: float = 21.5332031 # win_length = 512*4 if samplerate == 44100

    # PerformanceSkin:
    performances_appearances: Dict[PerformanceGrade, Tuple[str, str]] = {
        PerformanceGrade.MISS               : (""   , ""     ),

        PerformanceGrade.LATE_FAILED        : ("\b⟪", "\t\t⟫"),
        PerformanceGrade.LATE_BAD           : ("\b⟨", "\t\t⟩"),
        PerformanceGrade.LATE_GOOD          : ("\b‹", "\t\t›"),
        PerformanceGrade.PERFECT            : (""   , ""     ),
        PerformanceGrade.EARLY_GOOD         : ("\t\t›", "\b‹"),
        PerformanceGrade.EARLY_BAD          : ("\t\t⟩", "\b⟨"),
        PerformanceGrade.EARLY_FAILED       : ("\t\t⟫", "\b⟪"),

        PerformanceGrade.LATE_FAILED_WRONG  : ("\b⟪", "\t\t⟫"),
        PerformanceGrade.LATE_BAD_WRONG     : ("\b⟨", "\t\t⟩"),
        PerformanceGrade.LATE_GOOD_WRONG    : ("\b‹", "\t\t›"),
        PerformanceGrade.PERFECT_WRONG      : (""   , ""     ),
        PerformanceGrade.EARLY_GOOD_WRONG   : ("\t\t›", "\b‹"),
        PerformanceGrade.EARLY_BAD_WRONG    : ("\t\t⟩", "\b⟨"),
        PerformanceGrade.EARLY_FAILED_WRONG : ("\t\t⟫", "\b⟪"),
        }

    performance_sustain_time: float = 0.1

    # ScrollingBarSkin:
    sight_appearances: Union[List[str], List[Tuple[str, str]]] = ["⛶", "🞎", "🞏", "🞐", "🞑", "🞒", "🞓"]
    hit_decay_time: float = 0.4
    hit_sustain_time: float = 0.1
    bar_shift: float = 0.1
    bar_flip: bool = False

class PlayField(Beatbar):
    settings : PlayFieldSettings = PlayFieldSettings()

    def __init__(self):
        super().__init__()

        # state
        self.bar_shift = self.settings.bar_shift
        self.bar_flip = self.settings.bar_flip

        self.full_score = 0
        self.score = 0
        self.total_subjects = 0
        self.finished_subjects = 0
        self.time = datetime.time(0, 0, 0)
        self.spectrum = "\u2800"*self.settings.spec_width

    @dn.datanode
    def _update_status(self):
        while True:
            yield
            self.status['full_score'] = self.full_score
            self.status['score'] = self.score
            self.status['progress'] = self.finished_subjects/self.total_subjects if self.total_subjects>0 else 1.0
            self.status['time'] = self.time
            self.status['spectrum'] = self.spectrum

    def register_handlers(self, console, start_time):
        super().register_handlers(console)

        self.start_time = start_time

        # event queue
        self.hit_queue = queue.Queue()
        self.sight_queue = queue.Queue()
        self.target_queue = queue.Queue()
        self.perf_queue = queue.Queue()

        # register
        self.console.add_effect(self._spec_handler(), zindex=(-1,))
        self.console.add_listener(self._hit_handler())
        self.console.add_drawer(self._update_status(), zindex=())
        self.console.add_drawer(self._sight_handler(), zindex=(2,))

    def _spec_handler(self):
        spec_width = self.settings.spec_width
        samplerate = self.console.settings.output_samplerate
        nchannels = self.console.settings.output_channels
        hop_length = round(samplerate * self.settings.spec_time_res)
        win_length = round(samplerate / self.settings.spec_freq_res)

        df = samplerate/win_length
        n_fft = win_length//2+1
        n = numpy.linspace(1, 88, spec_width*2+1)
        f = 440 * 2**((n-49)/12) # frequency of n-th piano key
        sec = numpy.minimum(n_fft-1, (f/df).round().astype(int))
        slices = [slice(start, stop) for start, stop in zip(sec[:-1], (sec+1)[1:])]

        decay = hop_length / samplerate / self.settings.spec_decay_time / 4
        volume_of = lambda J: dn.power2db(J.mean() * samplerate / 2, scale=(1e-5, 1e6)) / 60.0

        A = numpy.cumsum([0, 2**6, 2**2, 2**1, 2**0])
        B = numpy.cumsum([0, 2**7, 2**5, 2**4, 2**3])
        draw_bar = lambda a, b: chr(0x2800 + A[int(a*4)] + B[int(b*4)])

        node = dn.pipe(dn.frame(win_length, hop_length), dn.power_spectrum(win_length, samplerate=samplerate))

        @dn.datanode
        def draw_spectrum():
            with node:
                vols = [0.0]*(spec_width*2)

                while True:
                    data = yield
                    J = node.send(data)

                    vols = [max(0.0, prev-decay, min(1.0, volume_of(J[slic])))
                            for slic, prev in zip(slices, vols)]
                    self.spectrum = "".join(map(draw_bar, vols[0::2], vols[1::2]))

        return dn.branch(dn.unchunk(draw_spectrum(), (hop_length, nchannels)))

    @dn.datanode
    def _hit_handler(self):
        target, start, duration = None, None, None
        waiting_targets = []

        time, strength, detected = yield
        time -= self.start_time
        strength = min(1.0, strength)
        if detected:
            self.hit_queue.put(strength)

        while True:
            # update waiting targets
            while not self.target_queue.empty():
                item = self.target_queue.get()
                if item[1] is None:
                    item = (item[0], time, item[2])
                waiting_targets.append(item)
            waiting_targets.sort(key=lambda item: item[1])

            while True:
                # find the next target if absent
                if target is None and waiting_targets and waiting_targets[0][1] <= time:
                    target, start, duration = waiting_targets.pop(0)
                    target.__enter__()

                # end listen if expired
                if duration is not None and start + duration <= time:
                    target.__exit__()
                    target, start, duration = None, None, None

                else:
                    # stop the loop for unexpired target or no target
                    break

            # send message to listening target
            if target is not None and detected:
                try:
                    target.send((time, strength))
                except StopIteration:
                    target, start, duration = None, None, None

            # update hit signal
            time, strength, detected = yield
            time -= self.start_time
            strength = min(1.0, strength)
            if detected:
                self.hit_queue.put(strength)

    @dn.datanode
    def _sight_handler(self):
        hit_decay_time = self.settings.hit_decay_time
        hit_sustain_time = self.settings.hit_sustain_time
        perf_sustain_time = self.settings.performance_sustain_time
        perf_appearances = self.settings.performances_appearances
        sight_appearances = self.settings.sight_appearances

        hit_strength, hit_time = None, None
        perf, perf_is_reversed, perf_time = None, None, None
        drawer, start, duration = None, None, None
        waiting_drawers = []

        while True:
            time, screen = yield
            time -= self.start_time

            # update hit hint
            while not self.hit_queue.empty():
                hit_strength = self.hit_queue.get()
                hit_time = time

            if hit_time is not None and time - hit_time >= max(hit_decay_time, hit_sustain_time):
                hit_strength = None
                hit_time = None

            # update perf hint
            while not self.perf_queue.empty():
                perf, perf_is_reversed = self.perf_queue.get()
                perf_time = time

            if perf is not None and time - perf_time >= perf_sustain_time:
                perf = None
                perf_is_reversed = None
                perf_time = None

            # update sight drawers
            while not self.sight_queue.empty():
                item = self.sight_queue.get()
                if item[1] is None:
                    item = (item[0], time, item[2])
                waiting_drawers.append(item)
            waiting_drawers.sort(key=lambda item: item[1])

            while waiting_drawers and waiting_drawers[0][1] <= time:
                drawer, start, duration = waiting_drawers.pop(0)

            if duration is not None and start + duration <= time:
                drawer, start, duration = None, None, None

            # draw perf hint
            if perf is not None:
                perf_text = perf_appearances[perf.grade]
                if perf_is_reversed:
                    perf_text = perf_text[::-1]

                self._draw_content(screen, 0, perf_text)

            # draw sight
            if drawer is not None:
                sight_text = drawer(time)

            elif hit_time is not None:
                strength = hit_strength - (time - hit_time) / hit_decay_time
                strength = max(0.0, min(1.0, strength))
                loudness = int(strength * (len(sight_appearances) - 1))
                if time - hit_time < hit_sustain_time:
                    loudness = max(1, loudness)
                sight_text = sight_appearances[loudness]

            else:
                sight_text = sight_appearances[0]

            self._draw_content(screen, 0, sight_text)

    def _draw_content(self, screen, pos, text):
        pos = pos + self.bar_shift
        if self.bar_flip:
            pos = 1 - pos

        content_start, content_end, _ = self.content_mask.indices(screen.width)
        index = round(content_start + pos * max(0, content_end - content_start - 1))

        if isinstance(text, tuple):
            text = text[self.bar_flip]

        screen.addstr(index, text, self.content_mask)

    @dn.datanode
    def _content_node(self, pos, text, start, duration):
        pos_func = pos if hasattr(pos, '__call__') else lambda time: pos
        text_func = text if hasattr(text, '__call__') else lambda time: text

        time, screen = yield
        time -= self.start_time

        if start is None:
            start = time

        while time < start:
            time, screen = yield
            time -= self.start_time

        while duration is None or time < start + duration:
            self._draw_content(screen, pos_func(time), text_func(time))
            time, screen = yield
            time -= self.start_time


    def add_score(self, score):
        self.score += score

    def add_full_score(self, full_score):
        self.full_score += full_score

    def add_finished(self, finished=1):
        self.finished_subjects += finished


    def play(self, node, samplerate=None, channels=None, volume=0.0, start=None, end=None, time=None, zindex=(0,)):
        if time is not None:
            time += self.start_time
        return self.console.play(node, samplerate=samplerate, channels=channels,
                                       volume=volume, start=start, end=end,
                                       time=time, zindex=zindex)

    def listen(self, node, start=None, duration=None):
        self.target_queue.put((node, start, duration))

    def draw_sight(self, text, start=None, duration=None):
        text_func = text if hasattr(text, '__call__') else lambda time: text
        self.sight_queue.put((text_func, start, duration))

    def reset_sight(self, start=None):
        self.sight_queue.put((None, start, None))

    def draw_content(self, pos, text, start=None, duration=None, zindex=(0,)):
        node = self._content_node(pos, text, start, duration)
        return self.console.add_drawer(node, zindex=zindex)

    def set_perf_hint(self, perf, is_reversed):
        self.perf_queue.put((perf, is_reversed))

    def on_before_render(self, node):
        return self.console.add_drawer(node, zindex=())

    def on_after_render(self, node):
        return self.console.add_drawer(node, zindex=(numpy.inf,))


# Game
@cfg.configurable
class BeatmapSettings:
    ## Difficulty:
    performance_tolerance: float = 0.02
    soft_threshold: float = 0.5
    loud_threshold: float = 0.5
    incr_threshold: float = -0.1
    roll_tolerance: float = 0.10
    spin_tolerance: float = 0.10

    perfect_tolerance = property(lambda self: self.performance_tolerance*1)
    good_tolerance    = property(lambda self: self.performance_tolerance*3)
    bad_tolerance     = property(lambda self: self.performance_tolerance*5)
    failed_tolerance  = property(lambda self: self.performance_tolerance*7)

    ## Scores:
    performances_scores: Dict[PerformanceGrade, int] = {
        PerformanceGrade.MISS               : 0,

        PerformanceGrade.LATE_FAILED        : 0,
        PerformanceGrade.LATE_BAD           : 2,
        PerformanceGrade.LATE_GOOD          : 8,
        PerformanceGrade.PERFECT            : 16,
        PerformanceGrade.EARLY_GOOD         : 8,
        PerformanceGrade.EARLY_BAD          : 2,
        PerformanceGrade.EARLY_FAILED       : 0,

        PerformanceGrade.LATE_FAILED_WRONG  : 0,
        PerformanceGrade.LATE_BAD_WRONG     : 1,
        PerformanceGrade.LATE_GOOD_WRONG    : 4,
        PerformanceGrade.PERFECT_WRONG      : 8,
        PerformanceGrade.EARLY_GOOD_WRONG   : 4,
        PerformanceGrade.EARLY_BAD_WRONG    : 1,
        PerformanceGrade.EARLY_FAILED_WRONG : 0,
        }

    performances_max_score = property(lambda self: max(self.performances_scores.values()))

    roll_rock_score: int = 2
    spin_score: int = 16

    ## NoteSkin:
    soft_approach_appearance:  Union[str, Tuple[str, str]] = "□"
    soft_wrong_appearance:     Union[str, Tuple[str, str]] = "⬚"
    soft_sound: str = "samples/soft.wav" # pulse(freq=830.61, decay_time=0.03, amplitude=0.5)
    loud_approach_appearance:  Union[str, Tuple[str, str]] = "■"
    loud_wrong_appearance:     Union[str, Tuple[str, str]] = "⬚"
    loud_sound: str = "samples/loud.wav" # pulse(freq=1661.2, decay_time=0.03, amplitude=1.0)
    incr_approach_appearance:  Union[str, Tuple[str, str]] = "⬒"
    incr_wrong_appearance:     Union[str, Tuple[str, str]] = "⬚"
    incr_sound: str = "samples/incr.wav" # pulse(freq=1661.2, decay_time=0.03, amplitude=1.0)
    roll_rock_appearance:      Union[str, Tuple[str, str]] = "◎"
    roll_rock_sound: str = "samples/rock.wav" # pulse(freq=1661.2, decay_time=0.01, amplitude=0.5)
    spin_disk_appearances:     Union[List[str], List[Tuple[str, str]]] = ["◴", "◵", "◶", "◷"]
    spin_finishing_appearance: Union[str, Tuple[str, str]] = "☺"
    spin_finish_sustain_time: float = 0.1
    spin_disk_sound: str = "samples/disk.wav" # pulse(freq=1661.2, decay_time=0.01, amplitude=1.0)

class Beatmap:
    settings: BeatmapSettings = BeatmapSettings()

    def __init__(self, path=".", info="", audio=None, volume=0.0, offset=0.0, tempo=60.0):
        self.path = path
        self.info = info
        self.audio = audio
        self.volume = volume
        self.offset = offset
        self.tempo = tempo

    def time(self, beat):
        return self.offset + beat*60/self.tempo

    def beat(self, time):
        return (time - self.offset)*self.tempo/60

    def dtime(self, beat, length):
        return self.time(beat+length) - self.time(beat)

    def build_events(self):
        raise NotImplementedError

@cfg.configurable
class GameplaySettings:
    ## Controls:
    leadin_time: float = 1.0
    skip_time: float = 8.0
    tickrate: float = 60.0
    prepare_time: float = 0.1

class KAIKOGame:
    settings: GameplaySettings = GameplaySettings()

    def __init__(self, beatmap, config=None):
        self.beatmap = beatmap

        if config is not None:
            cfg.config_read(open(config, 'r'), main=self.settings)

    @dn.datanode
    def connect(self, console):
        self.console = console
        self.playfield = PlayField()

        # events
        self.events = self.beatmap.build_events()
        self.events.sort(key=lambda e: e.lifespan[0])

        leadin_time = self.settings.leadin_time
        events_start_time = min((event.lifespan[0] - leadin_time for event in self.events), default=0.0)
        events_end_time   = max((event.lifespan[1] + leadin_time for event in self.events), default=0.0)

        self.playfield.total_subjects = sum(event.is_subject for event in self.events)

        if self.beatmap.audio is None:
            audionode = None
            duration = 0.0
            volume = 0.0
        else:
            audiopath = os.path.join(self.beatmap.path, self.beatmap.audio)
            with audioread.audio_open(audiopath) as file:
                duration = file.duration
            audionode = dn.DataNode.wrap(self.console.load_sound(audiopath))
            volume = self.beatmap.volume

        # game loop
        tickrate = self.settings.tickrate
        prepare_time = self.settings.prepare_time
        time_shift = prepare_time + max(-events_start_time, 0.0)

        with dn.tick(1/tickrate, prepare_time, -time_shift) as timer:
            start_time = self.console.time + time_shift

            # music
            if audionode is not None:
                self.console.play(audionode, volume=volume, time=start_time, zindex=(-3,))

            # handlers
            self.playfield.register_handlers(self.console, start_time)

            # register events
            events_iter = iter(self.events)
            event = next(events_iter, None)

            yield
            for time in timer:
                if max(events_end_time, duration) <= time:
                    break

                while event is not None and event.lifespan[0] <= time + prepare_time:
                    event.register(self.playfield)
                    event = next(events_iter, None)

                time = int(max(0.0, time))
                self.playfield.time = datetime.time(time//3600, time%3600//60, time%60)

                yield
