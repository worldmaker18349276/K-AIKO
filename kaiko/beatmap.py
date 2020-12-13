import os
from enum import Enum
import inspect
from typing import List, Tuple, Optional, Union
from collections import OrderedDict
import queue
import numpy
import audioread
from . import cfg
from . import datanodes as dn
from .beatsheet import K_AIKO_STD, OSU


class Event:
    # lifespan
    # __init__(beatmap, context, *args, **kwargs)
    # register(field)
    pass

class Text(Event):
    def __init__(self, beatmap, context, text=None, sound=None, *, beat, speed=None):
        self.time = beatmap.time(beat)
        self.text = text

        if speed is None:
            speed = context.get('speed', 1.0)
        self.speed = speed

        if sound is not None:
            self.sound = os.path.join(beatmap.path, sound)
        else:
            self.sound = None

        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.time + travel_time)

    def register(self, field):
        if self.sound is not None:
            field.play(self.sound, time=self.time)

        if self.text is not None:
            pos = lambda time, width: (self.time-time) * 0.5 * self.speed
            field.draw_text(pos, self.text, start=self.lifespan[0],
                            duration=self.lifespan[1]-self.lifespan[0], zindex=(-2, -self.time))

# scripts
class Flip(Event):
    def __init__(self, beatmap, context, flip=None, *, beat):
        self.time = beatmap.time(beat)
        self.flip = flip
        self.lifespan = (self.time, self.time)

    def register(self, field):
        field.console.add_renderer(self._node(field), zindex=())

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
    def __init__(self, beatmap, context, shift, *, beat, length):
        self.time = beatmap.time(beat)
        self.end = beatmap.time(beat+length)
        self.shift = shift
        self.lifespan = (self.time, self.end)

    def register(self, field):
        field.console.add_renderer(self._node(field), zindex=())

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

class Jiggle(Event):
    def __init__(self, beatmap, context, frequency=10.0, *, beat, length):
        self.time = beatmap.time(beat)
        self.end = beatmap.time(beat+length)
        self.frequency = frequency
        self.lifespan = (self.time, self.end)

    def register(self, field):
        field.console.add_renderer(self._node(field), zindex=())

    @dn.datanode
    def _node(self, field):
        time, screen = yield
        time -= field.start_time

        while time < self.time:
            time, screen = yield
            time -= field.start_time

        shift0 = field.sight_shift

        while time < self.end:
            turn = (time - self.time) * self.frequency
            bar_start, bar_end, _ = field.bar_mask.indices(screen.width)
            field.sight_shift = shift0 + 1/(bar_end - bar_start) * (turn // 0.5 % 2 * 2 - 1)
            time, screen = yield
            time -= field.start_time

        field.sight_shift = shift0

        time, screen = yield
        time -= field.start_time

def set_context(beatmap, context, **kw):
    context.update(**kw)

# targets
class Target(Event):
    # lifespan, range, score, full_score, is_finished
    # __init__(beatmap, context, *args, **kwargs)
    # approach(field)
    # hit(field, time, strength)
    # finish(field)

    def register(self, field):
        self.approach(field)
        field.add_target(self._node(field), start=self.range[0], duration=self.range[1]-self.range[0])

    @dn.datanode
    def _node(self, field):
        try:
            while True:
                time, strength = yield
                self.hit(field, time, strength)
                if self.is_finished:
                    break
        except GeneratorExit:
            if not self.is_finished:
                self.finish(field)


class PerformanceType(Enum):
    MISS               = (None, None)
    GREAT              = ( 0, False)
    LATE_GOOD          = (+1, False)
    EARLY_GOOD         = (-1, False)
    LATE_BAD           = (+2, False)
    EARLY_BAD          = (-2, False)
    LATE_FAILED        = (+3, False)
    EARLY_FAILED       = (-3, False)
    GREAT_WRONG        = ( 0,  True)
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
        return f"PerformanceType.{self.name}"

class Performance:
    def __init__(self, type, time, err):
        self.type = type
        self.time = time
        self.err = err

    @property
    def shift(self):
        return self.type.shift

    @property
    def is_wrong(self):
        return self.type.is_wrong

    @property
    def is_miss(self):
        return self.type == PerformanceType.MISS

    discriptions = {
        PerformanceType.MISS               : "Miss"                      ,
        PerformanceType.GREAT              : "Great"                     ,
        PerformanceType.LATE_GOOD          : "Late Good"                 ,
        PerformanceType.EARLY_GOOD         : "Early Good"                ,
        PerformanceType.LATE_BAD           : "Late Bad"                  ,
        PerformanceType.EARLY_BAD          : "Early Bad"                 ,
        PerformanceType.LATE_FAILED        : "Late Failed"               ,
        PerformanceType.EARLY_FAILED       : "Early Failed"              ,
        PerformanceType.GREAT_WRONG        : "Great but Wrong Key"       ,
        PerformanceType.LATE_GOOD_WRONG    : "Late Good but Wrong Key"   ,
        PerformanceType.EARLY_GOOD_WRONG   : "Early Good but Wrong Key"  ,
        PerformanceType.LATE_BAD_WRONG     : "Late Bad but Wrong Key"    ,
        PerformanceType.EARLY_BAD_WRONG    : "Early Bad but Wrong Key"   ,
        PerformanceType.LATE_FAILED_WRONG  : "Late Failed but Wrong Key" ,
        PerformanceType.EARLY_FAILED_WRONG : "Early Failed but Wrong Key",
    }

    @property
    def description(self):
        return self.discriptions[self.type]

    @staticmethod
    def judge(time, err, is_correct_key, tolerances):
        abs_err = abs(err)

        if abs_err < tolerances[0]: # great
            shift = 0
        elif abs_err < tolerances[1]: # good
            shift = 1
        elif abs_err < tolerances[2]: # bad
            shift = 2
        elif abs_err < tolerances[3]: # failed
            shift = 3
        else: # miss
            return None

        if err < 0:
            shift = -shift

        return Performance(PerformanceType((shift, not is_correct_key)), time, err)

    def render(self, field, is_reversed, appearances, sustain_time):
        appearance = appearances[self.type]
        if is_reversed:
            appearance = appearance[::-1]

        field.draw_text(field.sight_shift, appearance, duration=sustain_time, zindex=(1,), key='perf_hint')

class OneshotTarget(Target):
    # time, speed, volume, perf, sound
    # approach_appearance, wrong_appearance
    # hit(field, time, strength)

    def __init__(self, beatmap, context, *, beat, speed=None, volume=None):
        self.tolerances = (
            beatmap.settings.great_tolerance,
            beatmap.settings.good_tolerance,
            beatmap.settings.bad_tolerance,
            beatmap.settings.failed_tolerance,
            )
        self.performance_appearances = {
            PerformanceType.MISS               : beatmap.settings.miss_appearance,
            PerformanceType.GREAT              : beatmap.settings.great_appearance,
            PerformanceType.LATE_GOOD          : beatmap.settings.late_good_appearance,
            PerformanceType.EARLY_GOOD         : beatmap.settings.early_good_appearance,
            PerformanceType.LATE_BAD           : beatmap.settings.late_bad_appearance,
            PerformanceType.EARLY_BAD          : beatmap.settings.early_bad_appearance,
            PerformanceType.LATE_FAILED        : beatmap.settings.late_failed_appearance,
            PerformanceType.EARLY_FAILED       : beatmap.settings.early_failed_appearance,
            PerformanceType.GREAT_WRONG        : beatmap.settings.great_wrong_appearance,
            PerformanceType.LATE_GOOD_WRONG    : beatmap.settings.late_good_wrong_appearance,
            PerformanceType.EARLY_GOOD_WRONG   : beatmap.settings.early_good_wrong_appearance,
            PerformanceType.LATE_BAD_WRONG     : beatmap.settings.late_bad_wrong_appearance,
            PerformanceType.EARLY_BAD_WRONG    : beatmap.settings.early_bad_wrong_appearance,
            PerformanceType.LATE_FAILED_WRONG  : beatmap.settings.late_failed_wrong_appearance,
            PerformanceType.EARLY_FAILED_WRONG : beatmap.settings.early_failed_wrong_appearance,
            }
        self.performance_scores = {
            PerformanceType.MISS               : beatmap.settings.miss_score,
            PerformanceType.GREAT              : beatmap.settings.great_score,
            PerformanceType.LATE_GOOD          : beatmap.settings.late_good_score,
            PerformanceType.EARLY_GOOD         : beatmap.settings.early_good_score,
            PerformanceType.LATE_BAD           : beatmap.settings.late_bad_score,
            PerformanceType.EARLY_BAD          : beatmap.settings.early_bad_score,
            PerformanceType.LATE_FAILED        : beatmap.settings.late_failed_score,
            PerformanceType.EARLY_FAILED       : beatmap.settings.early_failed_score,
            PerformanceType.GREAT_WRONG        : beatmap.settings.great_wrong_score,
            PerformanceType.LATE_GOOD_WRONG    : beatmap.settings.late_good_wrong_score,
            PerformanceType.EARLY_GOOD_WRONG   : beatmap.settings.early_good_wrong_score,
            PerformanceType.LATE_BAD_WRONG     : beatmap.settings.late_bad_wrong_score,
            PerformanceType.EARLY_BAD_WRONG    : beatmap.settings.early_bad_wrong_score,
            PerformanceType.LATE_FAILED_WRONG  : beatmap.settings.late_failed_wrong_score,
            PerformanceType.EARLY_FAILED_WRONG : beatmap.settings.early_failed_wrong_score,
            }
        self.performance_sustain_time = beatmap.settings.performance_sustain_time

        self.full_score = max(self.performance_scores.values())

        if speed is None:
            speed = context.get('speed', 1.0)
        if volume is None:
            volume = context.get('volume', 0.0)

        self.time = beatmap.time(beat)
        self.speed = speed
        self.volume = volume
        self.perf = None

        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.time + travel_time)

    @property
    def pos(self):
        return lambda time, width: (self.time-time) * 0.5 * self.speed

    @property
    def range(self):
        return (self.time - self.tolerances[3], self.time + self.tolerances[3])

    @property
    def score(self):
        return self.performance_scores[self.perf.type] if self.perf is not None else 0

    @property
    def is_finished(self):
        return self.perf is not None

    @property
    def perfs(self):
        return (self.perf,) if self.perf is not None else ()

    def approach(self, field):
        if self.sound is not None:
            field.play(self.sound, time=self.time, volume=self.volume)

        field.draw_target(self, self.pos, self.approach_appearance,
                          start=self.lifespan[0], duration=self.lifespan[1]-self.lifespan[0], key=self)
        field.reset_sight(start=self.range[0])

    def hit(self, field, time, strength, is_correct_key=True):
        perf = Performance.judge(self.time, time - self.time, is_correct_key, self.tolerances)
        if perf is not None:
            perf.render(field, self.speed < 0, self.performance_appearances, self.performance_sustain_time)
            self.finish(field, perf)

    def finish(self, field, perf=None):
        if perf is None:
            perf = Performance(PerformanceType.MISS, self.time, None)
        self.perf = perf

        if self.perf.type == PerformanceType.MISS:
            pass

        elif self.perf.is_wrong: # wrong key
            field.draw_target(self, self.pos, self.wrong_appearance,
                              start=self.lifespan[0], duration=self.lifespan[1]-self.lifespan[0], key=self)

        else: # correct key
            field.remove_target(key=self)

class Soft(OneshotTarget):
    def __init__(self, beatmap, context, *, beat, speed=None, volume=None):
        super().__init__(beatmap, context, beat=beat, speed=speed, volume=volume)
        self.approach_appearance = beatmap.settings.soft_approach_appearance
        self.wrong_appearance = beatmap.settings.soft_wrong_appearance
        self.sound = beatmap.settings.soft_sound
        self.threshold = beatmap.settings.soft_threshold

    def hit(self, field, time, strength):
        super().hit(field, time, strength, strength < self.threshold)

class Loud(OneshotTarget):
    def __init__(self, beatmap, context, *, beat, speed=None, volume=None):
        super().__init__(beatmap, context, beat=beat, speed=speed, volume=volume)
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

    def hit(self, strength):
        self.threshold = max(self.threshold, strength)

class Incr(OneshotTarget):
    def __init__(self, beatmap, context, group=None, *, beat, speed=None, volume=None):
        super().__init__(beatmap, context, beat=beat, speed=speed, volume=volume)

        self.approach_appearance = beatmap.settings.incr_approach_appearance
        self.wrong_appearance = beatmap.settings.incr_wrong_appearance
        self.sound = beatmap.settings.incr_sound
        self.incr_threshold = beatmap.settings.incr_threshold

        if 'incrs' not in context:
            context['incrs'] = OrderedDict()

        group_key = group
        if group_key is None:
            # determine group of incr note according to the context
            for key, (_, last_beat) in reversed(context['incrs'].items()):
                if beat - 1 <= last_beat <= beat:
                    group_key = key
                    break
            else:
                group_key = 0
                while group_key in context['incrs']:
                    group_key += 1

        group, _ = context['incrs'].get(group_key, (IncrGroup(), beat))
        context['incrs'][group_key] = group, beat
        context['incrs'].move_to_end(group_key)

        group.total += 1
        self.count = group.total
        self.group = group

    @property
    def volume(self):
        return self._volume + numpy.log10(0.2 + 0.8 * (self.count-1)/self.group.total) * 20

    @volume.setter
    def volume(self, value):
        self._volume = value

    def hit(self, field, time, strength):
        super().hit(field, time, strength, strength >= min(1.0, self.group.threshold + self.incr_threshold))
        self.group.hit(strength)

class Roll(Target):
    def __init__(self, beatmap, context, density=2, *, beat, length, speed=None, volume=None):
        self.tolerance = beatmap.settings.roll_tolerance
        self.rock_appearance = beatmap.settings.roll_rock_appearance
        self.sound = beatmap.settings.rock_sound

        self.time = beatmap.time(beat)

        self.number = int(length * density)
        self.end = beatmap.time(beat+length)
        self.times = [beatmap.time(beat+i/density) for i in range(self.number)]

        if speed is None:
            speed = context.get('speed', 1.0)
        if volume is None:
            volume = context.get('volume', 0.0)

        self.speed = speed
        self.volume = volume
        self.roll = 0
        self.is_finished = False

        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.end + travel_time)

    def pos(self, index):
        return lambda time, width: (self.times[index]-time) * 0.5 * self.speed

    @property
    def range(self):
        return (self.time - self.tolerance, self.end - self.tolerance)

    @property
    def full_score(self):
        return self.number * 2

    @property
    def score(self):
        if self.roll < self.number:
            return self.roll * 2
        elif self.roll < 2*self.number:
            return (2*self.number - self.roll) * 2
        else:
            return 0

    def approach(self, field):
        for i, time in enumerate(self.times):
            if self.sound is not None:
                field.play(self.sound, time=time, volume=self.volume)
            field.draw_target(self, self.pos(i), self.rock_appearance,
                              start=self.lifespan[0], duration=self.lifespan[1]-self.lifespan[0], key=(self, i))
        field.reset_sight(start=self.range[0])

    def hit(self, field, time, strength):
        self.roll += 1
        if self.roll <= self.number:
            field.remove_target(key=(self, self.roll-1))

    def finish(self, field):
        self.is_finished = True

class Spin(Target):
    full_score = 10

    def __init__(self, beatmap, context, density=2, *, beat, length, speed=None, volume=None):
        self.tolerance = beatmap.settings.spin_tolerance
        self.disk_appearances = beatmap.settings.spin_disk_appearances
        self.finishing_appearance = beatmap.settings.spin_finishing_appearance
        self.finish_sustain_time = beatmap.settings.spin_finish_sustain_time
        self.sound = beatmap.settings.disk_sound

        self.time = beatmap.time(beat)

        self.capacity = length * density
        self.end = beatmap.time(beat+length)
        self.times = [beatmap.time(beat+i/density) for i in range(int(self.capacity))]

        if speed is None:
            speed = context.get('speed', 1.0)
        if volume is None:
            volume = context.get('volume', 0.0)

        self.speed = speed
        self.volume = volume
        self.charge = 0.0
        self.is_finished = False

        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.end + travel_time)

    @property
    def pos(self):
        return lambda time, width: (max(0.0, self.time-time) + min(0.0, self.end-time)) * 0.5 * self.speed

    @property
    def range(self):
        return (self.time - self.tolerance, self.end + self.tolerance)

    @property
    def score(self):
        return self.full_score if self.charge == self.capacity else 0

    def approach(self, field):
        for time in self.times:
            if self.sound is not None:
                field.play(self.sound, time=time, volume=self.volume)

        appearance = lambda: self.disk_appearances[int(self.charge) % len(self.disk_appearances)]
        field.draw_target(self, self.pos, appearance,
                          start=self.lifespan[0], duration=self.lifespan[1]-self.lifespan[0], key=self)
        field.draw_sight("", start=self.range[0], duration=self.range[1]-self.range[0])

    def hit(self, field, time, strength):
        self.charge = min(self.charge + min(1.0, strength), self.capacity)
        if self.charge == self.capacity:
            self.finish(field)

    def finish(self, field):
        self.is_finished = True

        if self.charge != self.capacity:
            return

        field.remove_target(key=self)

        appearance = self.finishing_appearance
        if isinstance(appearance, tuple) and self.speed < 0:
            appearance = appearance[::-1]
        field.draw_sight(appearance, duration=self.finish_sustain_time)


# K-AIKO
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
class KAIKOSettings:
    ## Controls:
    leadin_time: float = 1.0
    skip_time: float = 8.0
    tickrate: float = 60.0
    prepare_time: float = 0.1

    ## PlayFieldSkin:
    spec_width: int = 5
    score_width: int = 13
    progress_width: int = 8
    spec_decay_time: float = 0.01
    spec_time_res: float = 0.0116099773 # hop_length = 512 if samplerate == 44100
    spec_freq_res: float = 21.5332031 # win_length = 512*4 if samplerate == 44100

    ## ScrollingBarSkin:
    sight_appearances: Union[List[str], List[Tuple[str, str]]] = ["⛶", "🞎", "🞏", "🞐", "🞑", "🞒", "🞓"]
    hit_decay_time: float = 0.4
    hit_sustain_time: float = 0.1
    bar_shift: float = 0.1
    sight_shift: float = 0.0
    bar_flip: bool = False

class KAIKO:
    settings: KAIKOSettings = KAIKOSettings()

    def __init__(self, beatmap, config=None):
        self.beatmap = beatmap

        if config is not None:
            cfg.config_read(open(config, 'r'), main=self.settings)

    def get_total_score(self):
        return sum(getattr(event, 'full_score', 0) for event in self.events)

    def get_score(self):
        return sum(getattr(event, 'score', 0) for event in self.events)

    def get_progress(self):
        total = len([event for event in self.events if hasattr(event, 'is_finished')])
        if total == 0:
            return 1.0
        return sum(getattr(event, 'is_finished', False) for event in self.events) / total

    @dn.datanode
    def connect(self, console):
        self.console = console

        self.bar_shift = self.settings.bar_shift
        self.sight_shift = self.settings.sight_shift
        self.bar_flip = self.settings.bar_flip

        # events
        leadin_time = self.settings.leadin_time
        self.events = [event for chart in self.beatmap.charts for event in chart.build_events(self.beatmap)]
        self.events.sort(key=lambda e: e.lifespan[0])
        events_start_time = min((event.lifespan[0] - leadin_time for event in self.events), default=0.0)
        events_end_time   = max((event.lifespan[1] + leadin_time for event in self.events), default=0.0)

        self.total_score = self.get_total_score()
        self.score = self.get_score()
        self.progress = self.get_progress()

        if self.beatmap.audio is None:
            audionode = None
            duration = 0.0
        else:
            audiopath = os.path.join(self.beatmap.path, self.beatmap.audio)
            with audioread.audio_open(audiopath) as file:
                duration = file.duration
            audionode = dn.DataNode.wrap(self.console.load_sound(audiopath))

        # playfield
        spec_width = self.settings.spec_width
        score_width = self.settings.score_width
        progress_width = self.settings.progress_width
        layout = to_slices((1, spec_width, 1, score_width, ..., progress_width, 1))
        _, self.spec_mask, _, self.score_mask, self.bar_mask, self.progress_mask, _ = layout

        self.spectrum = "\u2800"*spec_width
        score_width_ = max(0, score_width-3)
        self.score_format = "[{score:>%dd}/{total_score:>%dd}]" % (score_width_-score_width_//2, score_width_//2)
        self.progress_format = "[{progress_pc:>%d.%df}%%]" % (max(0, progress_width-3), max(0, progress_width-7))

        # game loop
        tickrate = self.settings.tickrate
        prepare_time = self.settings.prepare_time
        time_shift = prepare_time + max(-events_start_time, 0.0)

        with dn.interval(1/tickrate, first=prepare_time) as timer:
            self.start_time = self.console.time + time_shift

            # music
            if audionode is not None:
                self.console.play(audionode, time=self.start_time, zindex=-3)

            # handlers
            self.hit_queue = queue.Queue()
            self.sight_queue = queue.Queue()
            self.target_queue = queue.Queue()

            self.console.add_effect(self._spec_handler(), zindex=-1)
            self.console.add_listener(self._target_handler())
            self.console.add_listener(self._hit_handler())
            self.console.add_renderer(self._status_handler(), zindex=(-3,), key='status')
            self.console.add_renderer(self._sight_handler(), zindex=(2,), key='sight')

            # register events
            events_iter = iter(self.events)
            event = next(events_iter, None)

            yield
            for time, _ in timer:
                time -= time_shift
                if max(events_end_time, duration) <= time:
                    break

                while event is not None and event.lifespan[0] <= time + prepare_time:
                    event.register(self)
                    event = next(events_iter, None)

                self.score = self.get_score()
                self.progress = self.get_progress()

                yield


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

        # braille patterns encoding:
        #   [0] [3]
        #   [1] [4]
        #   [2] [5]
        #   [6] [7]

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
        while True:
            time, strength, detected = yield
            if detected:
                self.hit_queue.put(min(1.0, strength))

    @dn.datanode
    def _target_handler(self):
        target, start, duration = None, None, None
        waiting_targets = []

        time, strength, detected = yield
        time -= self.start_time
        while True:
            while not self.target_queue.empty():
                item = self.target_queue.get()
                if item[1] is None:
                    item = (item[0], time, item[2])
                waiting_targets.append(item)
            waiting_targets.sort(key=lambda item: item[1])

            if target is None and waiting_targets and waiting_targets[0][1] <= time:
                target, start, duration = waiting_targets.pop(0)
                target.__enter__()

            if duration is not None and start + duration <= time:
                target.__exit__()
                target, start, duration = None, None, None
                continue

            if target is not None and detected:
                try:
                    target.send((time, min(1.0, strength)))
                except StopIteration:
                    target, start, duration = None, None, None

            time, strength, detected = yield
            time -= self.start_time

    @dn.datanode
    def _status_handler(self):
        while True:
            _, screen = yield

            spec_text = self.spectrum
            spec_start, _, _ = self.spec_mask.indices(screen.width)
            screen.addstr(spec_start, spec_text, self.spec_mask)

            score_text = self.score_format.format(score=self.score, total_score=self.total_score)
            score_start, _, _ = self.score_mask.indices(screen.width)
            screen.addstr(score_start, score_text, self.score_mask)

            progress_text = self.progress_format.format(progress=self.progress, progress_pc=self.progress*100)
            progress_start, _, _ = self.progress_mask.indices(screen.width)
            screen.addstr(progress_start, progress_text, self.progress_mask)

    @dn.datanode
    def _sight_handler(self):
        hit_decay_time = self.settings.hit_decay_time
        hit_sustain_time = self.settings.hit_sustain_time
        sight_appearances = self.settings.sight_appearances

        hit_strength = None
        hit_time = None
        drawer, start, duration = None, None, None
        waiting_drawers = []

        while True:
            time, screen = yield
            time -= self.start_time

            while not self.hit_queue.empty():
                hit_strength = self.hit_queue.get()
                hit_time = time

            if hit_time is not None and time - hit_time >= max(hit_decay_time, hit_sustain_time):
                hit_strength = None
                hit_time = None

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

            if drawer is not None:
                text = drawer(time, screen.width)

            elif hit_time is not None:
                strength = hit_strength - (time - hit_time) / hit_decay_time
                strength = max(0.0, min(1.0, strength))
                loudness = int(strength * (len(sight_appearances) - 1))
                if time - hit_time < hit_sustain_time:
                    loudness = max(1, loudness)
                text = sight_appearances[loudness]

            else:
                text = sight_appearances[0]

            self._bar_draw(screen, self.sight_shift, text)

    def _bar_draw(self, screen, pos, text):
        pos = pos + self.bar_shift
        if self.bar_flip:
            pos = 1 - pos

        bar_start, bar_end, _ = self.bar_mask.indices(screen.width)
        index = bar_start + pos * max(0, bar_end - bar_start - 1)

        if isinstance(text, tuple):
            text = text[self.bar_flip]

        screen.addstr(index, text, self.bar_mask)

    @dn.datanode
    def _bar_node(self, pos, text, start, duration):
        pos_func = pos if hasattr(pos, '__call__') else lambda time, width: pos
        text_func = text if hasattr(text, '__call__') else lambda time, width: text

        time, screen = yield
        time -= self.start_time

        if start is None:
            start = time

        while time < start:
            time, screen = yield
            time -= self.start_time

        while duration is None or time < start + duration:
            self._bar_draw(screen, pos_func(time, screen.width), text_func(time, screen.width))
            time, screen = yield
            time -= self.start_time


    def play(self, node, samplerate=None, channels=None, volume=0.0, start=None, end=None, time=None, zindex=0, key=None):
        if time is not None:
            time += self.start_time
        return self.console.play(node, samplerate=samplerate, channels=channels,
                                       volume=volume, start=start, end=end,
                                       time=time, zindex=zindex, key=key)

    def add_target(self, target, start=None, duration=None):
        self.target_queue.put((target, start, duration))

    def draw_sight(self, text, start=None, duration=None):
        text_func = text if hasattr(text, '__call__') else lambda time, width: text
        self.sight_queue.put((text_func, start, duration))

    def reset_sight(self, start=None):
        self.sight_queue.put((None, start, None))

    def draw_text(self, pos, text, start=None, duration=None, zindex=(0,), key=None):
        if key is None:
            key = object()
        node = self._bar_node(pos, text, start, duration)
        self.console.add_renderer(node, zindex=zindex, key=('text', key))
        return key

    def remove_text(self, key):
        self.console.remove_renderer(key=('text', key))

    def draw_target(self, target, pos, text, start=None, duration=None, key=None):
        if key is None:
            key = object()
        node = self._bar_node(pos, text, start, duration)
        zindex = lambda: (0, not target.is_finished, -target.range[0])
        self.console.add_renderer(node, zindex=zindex, key=('target', key))
        return key

    def remove_target(self, key):
        self.console.remove_renderer(key=('target', key))


# Beatmap
@cfg.configurable
class BeatmapSettings:
    ## Difficulty:
    great_tolerance:  float = 0.02
    good_tolerance:   float = 0.06
    bad_tolerance:    float = 0.10
    failed_tolerance: float = 0.14
    soft_threshold: float = 0.5
    loud_threshold: float = 0.5
    incr_threshold: float = -0.1
    roll_tolerance: float = 0.10
    spin_tolerance: float = 0.10

    ## Scores:
    miss_score:               int = 0

    late_failed_score:        int = 0
    late_bad_score:           int = 3
    late_good_score:          int = 5
    great_score:              int = 10
    early_good_score:         int = 5
    early_bad_score:          int = 3
    early_failed_score:       int = 0

    late_failed_wrong_score:  int = 0
    late_bad_wrong_score:     int = 1
    late_good_wrong_score:    int = 3
    great_wrong_score:        int = 5
    early_good_wrong_score:   int = 3
    early_bad_wrong_score:    int = 1
    early_failed_wrong_score: int = 0

    ## PerformanceSkin:
    miss_appearance:               Tuple[str, str] = (""   , ""     )

    late_failed_appearance:        Tuple[str, str] = ("\b⟪", "\t\t⟫")
    late_bad_appearance:           Tuple[str, str] = ("\b⟨", "\t\t⟩")
    late_good_appearance:          Tuple[str, str] = ("\b‹", "\t\t›")
    great_appearance:              Tuple[str, str] = (""   , ""     )
    early_good_appearance:         Tuple[str, str] = ("\t\t›", "\b‹")
    early_bad_appearance:          Tuple[str, str] = ("\t\t⟩", "\b⟨")
    early_failed_appearance:       Tuple[str, str] = ("\t\t⟫", "\b⟪")

    late_failed_wrong_appearance:  Tuple[str, str] = ("\b⟪", "\t\t⟫")
    late_bad_wrong_appearance:     Tuple[str, str] = ("\b⟨", "\t\t⟩")
    late_good_wrong_appearance:    Tuple[str, str] = ("\b‹", "\t\t›")
    great_wrong_appearance:        Tuple[str, str] = (""   , ""     )
    early_good_wrong_appearance:   Tuple[str, str] = ("\t\t›", "\b‹")
    early_bad_wrong_appearance:    Tuple[str, str] = ("\t\t⟩", "\b⟨")
    early_failed_wrong_appearance: Tuple[str, str] = ("\t\t⟫", "\b⟪")

    performance_sustain_time: float = 0.1

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
    rock_sound: str = "samples/rock.wav" # pulse(freq=1661.2, decay_time=0.01, amplitude=0.5)
    spin_disk_appearances:     Union[List[str], List[Tuple[str, str]]] = ["◴", "◵", "◶", "◷"]
    spin_finishing_appearance: Union[str, Tuple[str, str]] = "☺"
    spin_finish_sustain_time: float = 0.1
    disk_sound: str = "samples/disk.wav" # pulse(freq=1661.2, decay_time=0.01, amplitude=1.0)

class Beatmap:
    settings: BeatmapSettings = BeatmapSettings()

    def __init__(self, path=".", info="", audio=None, offset=0.0, tempo=60.0):
        self.path = path
        self.info = info
        self.audio = audio
        self.offset = offset
        self.tempo = tempo

        self.definitions = {}
        self.charts = []

        self['x'] = Soft
        self['o'] = Loud
        self['<'] = Incr
        self['%'] = Roll
        self['@'] = Spin
        self['TEXT'] = Text
        self['CONTEXT'] = set_context
        self['FLIP'] = Flip
        self['SHIFT'] = Shift
        self['JIGGLE'] = Jiggle

    def time(self, beat):
        return self.offset + beat*60/self.tempo

    def beat(self, time):
        return (time - self.offset)*self.tempo/60

    def dtime(self, beat, length):
        return self.time(beat+length) - self.time(beat)

    def __setitem__(self, symbol, builder):
        if any(c in symbol for c in " \b\t\n\r\f\v()[]{}\'\"\\#|~") or symbol == '_':
            raise ValueError(f"invalid symbol `{symbol}`")
        if symbol in self.definitions:
            raise ValueError(f"symbol `{symbol}` is already defined")
        self.definitions[symbol] = NoteType(symbol, builder)

    def __delitem__(self, symbol):
        del self.definitions[symbol]

    def __getitem__(self, symbol):
        return self.definitions[symbol].builder

    def __iadd__(self, chart_str):
        self.charts.append(K_AIKO_STD_FORMAT.read_chart(chart_str, self.definitions))
        return self

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
    return type(builder.__name__+"Note", (Note,),
                dict(symbol=symbol, builder=staticmethod(builder), __signature__=signature))

class Note:
    def __init__(self, *psargs, **kwargs):
        self.bound = self.__signature__.bind(*psargs, **kwargs)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        psargs_str = [repr(value) for value in self.bound.args]
        kwargs_str = [key+"="+repr(value) for key, value in self.bound.kwargs.items()]
        args_str = ", ".join([*psargs_str, *kwargs_str])
        return f"{self.symbol}({args_str})"

    def create(self, beatmap, context):
        return self.builder(beatmap, context, *self.bound.args, **self.bound.kwargs)


K_AIKO_STD_FORMAT = K_AIKO_STD(Beatmap, NoteChart)
OSU_FORMAT = OSU(Beatmap, NoteChart)

