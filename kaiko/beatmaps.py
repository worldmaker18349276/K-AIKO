import os
import contextlib
from enum import Enum
from dataclasses import dataclass, replace
from typing import List, Tuple, Dict, Optional, Union
from collections import OrderedDict
from fractions import Fraction
import numpy
import audioread
from .engines import Mixer, Detector, Renderer
from .beatbar import PerformanceGrade, Performance, Beatbar, BeatbarSettings, WidgetManager, WidgetSettings
from . import config as cfg
from . import datanodes as dn
from . import wcbuffers as wcb


@dataclass
class UpdateContext:
    r"""An pseudo-event to update context in preparation phase.

    Attributes
    ----------
    update : dict
        The updated fields in the context.
    """

    update: Dict[str, Union[None, bool, int, Fraction, float, str]]

    def prepare(self, beatmap, context):
        context.update(**self.update)

@dataclass
class Event:
    r"""An event of beatmap.  Event represents an effect and action that occur
    within a specific time span.

    The tracks, which are sequences of events, can be drawn as a timeline diagram

    ..code::

        time    ____00_____________________01_____________________02_____________________03_____
               |                                                                                |
        track1 |                     [event1]    [===event2====]         []                     |
        track2 |                              [========event4========]    [event5]              |
               |________________________________________________________________________________|

    Where `track1` contains three events, and `track2` contains two events.  Like
    `event3`, some events have no length (determined by attribute `has_length`).
    The square bracket is an interval `(beat, beat+length)`, which define the
    timespan of action of this event, just like hit object and hold object in
    others rhythm game.  The interval should be ordered and exclusive in each
    track, but there is no limit to events between tracks.
    The attribute `lifespan` of events can be drawn as a timeline diagram

    ..code::

        time    ____00_____________________01_____________________02_____________________03_____
               |                                                                                |
        event1 |                |<---[======]--->|                                              |
        event2 |          |<---------------------[=============]->|                             |
        event3 |                                            |<-----------[]----------->|        |
        event4 |                         |<---[======================]--->|                     |
        event5 |                                                |<--------[======]-------->|    |
               |________________________________________________________________________________|

    Where the region in the bar lines is an interval `lifespan`, which define
    when this event occurred; it also includes the process that hit object runs
    in and out of the view.  Just like `event1` and `event2`, the lifespan of
    events can be unordered and overlapping.

    Event is a dataclass described by some (immutable) fields, which is
    documented in the section `Fields`.  These fields determine how this event
    occurred.  Event also contains some attributes related to runtime conditions,
    which is documented in the section `Attributes`.  These attributes are
    determined by method `Event.prepare` and may change during execution.
    One can use `dataclasses.replace` to copy the event, which will not copy
    runtime attributes.

    Fields
    ------
    beat, length : Fraction
        The start time and sustain time of action of this event.  The actual
        meaning in each event is different. `beat` is the time in the unit
        defined by beatmap.  `length` is the time difference started from `beat`
        in the unit defined by beatmap.  If `has_length` is False, the attribute
        `length` can be dropped.

    Attributes
    ----------
    lifespan : tuple of float and float
        The start time and end time (in seconds) of this event.  In general, the
        lifespan is determined by attributes `beat` and `length`.
    is_subject : bool, optional
        True if this event is an action.  To increase the value of progress bar,
        use `state.add_finished`.
    full_score : int, optional
        The full score of this event.  To increase score counter and full score
        counter, use `state.add_score` and `state.add_full_score`.
    has_length : bool, optional
        True if the attribute `length` is meaningful.

    Methods
    -------
    prepare(beatmap, context)
        Prepare resources for this event in the given context.  The context is a
        mutable dictionary, which can be used to transfer parameters between events.
        The context of each track is different, so event cannot affect each others
        between tracks.
    register(state, playfield)
        Schedule handlers for this event.  `state` is the game state of beatmap,
        `playfield` is an instance of `BeatmapPlayer`, which controls the whole
        gameplay.
    """

    beat: Fraction = Fraction(0, 1)
    length: Fraction = Fraction(1, 1)

    is_subject = False
    full_score = 0
    has_length = True


# scripts
@dataclass
class Text(Event):
    r"""An event that displays text on the playfield.  The text will move to the
    left at a constant speed.

    Fields
    ------
    beat, length : Fraction
        `beat` is the time the text pass through the position 0.0, `length` is
        meaningless.
    text : str, optional
        The text to show, or None for no text.
    speed : float, optional
        The speed of the text (unit: half bar per second).  Default speed will be
        determined by context value `speed`, or 1.0 if absence.
    """

    has_length = False

    text: Optional[str] = None
    speed: Optional[float] = None

    def prepare(self, beatmap, context):
        self.time = beatmap.time(self.beat)
        if self.speed is None:
            self.speed = context.get('speed', 1.0)

        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.time + travel_time)
        self.zindex = (-2, -self.time)

    def pos(self, time):
        return (self.time-time) * 0.5 * self.speed

    def register(self, state, field):
        if self.text is not None:
            field.draw_content(self.pos, self.text, zindex=self.zindex,
                               start=self.lifespan[0], duration=self.lifespan[1]-self.lifespan[0])

@dataclass
class Title(Event):
    r"""An event that displays title on the playfield.  The text will be placed
    at the specific position.

    Fields
    ------
    beat, length : Fraction
        `beat` is the display time of the text, `length` is the display duration.
    text : str, optional
        The text to show, or None for no text.
    pos : float, optional
        The position to show, default is 0.5.
    """

    has_length = True

    text: Optional[str] = None
    pos: float = 0.5

    def prepare(self, beatmap, context):
        self.time = beatmap.time(self.beat)
        self.end = beatmap.time(self.beat + self.length)

        self.lifespan = (self.time, self.end)
        self.zindex = (10, -self.time)

    def register(self, state, field):
        if self.text is not None:
            field.draw_title(self.pos, self.text, zindex=self.zindex,
                             start=self.lifespan[0], duration=self.lifespan[1]-self.lifespan[0])

@dataclass
class Flip(Event):
    r"""An event that flips the scrolling bar of the playfield.

    Fields
    ------
    beat, length : Fraction
        `beat` is the time to flip, `length` is meaningless.
    flip : bool, optional
        The value of `bar_flip` of the scrolling bar after flipping, or None for
        changing the direction of the scrolling bar.
    """

    has_length = False

    flip: Optional[bool] = None

    def prepare(self, beatmap, context):
        self.time = beatmap.time(self.beat)
        self.lifespan = (self.time, self.time)

    def register(self, state, field):
        field.on_before_render(self._node(field))

    @dn.datanode
    def _node(self, field):
        time, width = yield

        while time < self.time:
            time, width = yield

        if self.flip is None:
            field.bar_flip = not field.bar_flip
        else:
            field.bar_flip = self.flip

        time, width = yield

@dataclass
class Shift(Event):
    r"""An event that shifts the scrolling bar of the playfield.

    Fields
    ------
    beat, length : Fraction
        `beat` is the time to start shifting, `length` is meaningless.
    shift : float, optional
        The value of `bar_shift` of the scrolling bar after shifting, default is 0.0.
    span : int or float or Fraction, optional
        the duration of shifting, default is 0.
    """

    has_length = False

    shift: float = 0.0
    span: Union[int, Fraction, float] = 0

    def prepare(self, beatmap, context):
        self.time = beatmap.time(self.beat)
        self.end = beatmap.time(self.beat+self.span)
        self.lifespan = (self.time, self.end)

    def register(self, state, field):
        field.on_before_render(self._node(field))

    @dn.datanode
    def _node(self, field):
        time, width = yield

        while time < self.time:
            time, width = yield

        shift0 = field.bar_shift
        speed = (self.shift - shift0) / (self.end - self.time) if self.end != self.time else 0

        while time < self.end:
            field.bar_shift = shift0 + speed * (time - self.time)
            time, width = yield

        field.bar_shift = self.shift

        time, width = yield

# targets
class Target(Event):
    r"""A target to hit.  Target will be counted as an action, and recorded by
    the progress bar during the gameplay.  Players will be asked to do some
    actions to accomplish this target, such as hitting a note or keeping hitting
    within a timespan.

    Fields
    ------
    beat, length : Fraction
        `beat` is the time this target should be hit, `length` is the duration of
        this target.

    Attributes
    ----------
    range : tuple of float and float
        The range of time this target listen to.  It is often slightly larger than
        the interval `(beat, beat+length)` to includes the hit tolerance.
    is_finished : bool
        Whether this target is finished.  The progress bar will increase by 1
        after completion.

    Methods
    -------
    approach(state, playfield)
        Register handlers for approaching effect of this target.  The hit handler
        and increasing progress bar will be managed automatically.
    hit(state, playfield, time, strength)
        Deal with the hit event on this target.
    finish(state, playfield)
        Finish this target.
    """

    is_subject = True

    @dn.datanode
    def listen(self, state, field):
        try:
            while True:
                time, strength = yield
                self.hit(state, field, time, strength)
                if self.is_finished:
                    break
        except GeneratorExit:
            if not self.is_finished:
                self.finish(state, field)
        finally:
            state.add_finished()

    def zindex(self):
        return (0, not self.is_finished, -self.range[0])

    def register(self, state, field):
        self.approach(state, field)
        field.listen(self.listen(state, field), start=self.range[0], duration=self.range[1]-self.range[0])

@dataclass
class OneshotTarget(Target):
    r"""A target to hit with one shot.  The target will move in a constant speed.
    The score, range and hit tolerance are determined by settings. Only appearances
    of targets, sound of target and rule of hitting target are different.

    Fields
    ------
    beat, length : Fraction
        `beat` is the time this target should be hit, `length` is meaningless.
    speed : float, optional
        The speed of this target (unit: half bar per second).  Default speed will
        be determined by context value `speed`, or 1.0 if absence.
    volume : float, optional
        The relative volume of sound of this target (unit: dB).  Default volume
        will be determined by context value `volume`, or 0.0 if absence.
    nofeedback : bool, optional
        Whether to make a visual cue for the action of hitting the target.
        Default value will be determined by context value `nofeedback`, or False
        if absence.

    Attributes
    ----------
    sound : DataNode or None
        The sound of the auditory cue of this target.
    approach_appearance : str
        The appearance of approaching target.
    wrong_appearance : str
        The appearance of wrong-shot target.

    Methods
    -------
    hit(state, playfield, time, strength)
    """

    has_length = False

    def prepare(self, beatmap, context):
        self.performance_tolerance = beatmap.settings.difficulty.performance_tolerance

        self.time = beatmap.time(self.beat)
        self.perf = None

        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.time + travel_time)
        tol = beatmap.settings.difficulty.failed_tolerance
        self.range = (self.time-tol, self.time+tol)
        self._scores = beatmap.settings.scores.performances_scores
        self.full_score = beatmap.settings.scores.performances_max_score

    def pos(self, time):
        return (self.time-time) * 0.5 * self.speed

    def appearance(self, time):
        if self.nofeedback or not self.is_finished:
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

    def approach(self, state, field):
        if self.sound is not None:
            field.play(self.sound, time=self.time, volume=self.volume)

        field.draw_content(self.pos, self.appearance, zindex=self.zindex,
                           start=self.lifespan[0], duration=self.lifespan[1]-self.lifespan[0])
        field.reset_sight(start=self.range[0])

    def hit(self, state, field, time, strength, is_correct_key=True):
        perf = Performance.judge(self.performance_tolerance, self.time, time, is_correct_key)
        state.add_perf(perf, self.speed < 0)
        if not self.nofeedback:
            field.set_perf(perf, self.speed < 0)
        self.finish(state, field, perf)

    def finish(self, state, field, perf=None):
        if perf is None:
            perf = Performance.judge(self.performance_tolerance, self.time)
        self.perf = perf
        state.add_full_score(self.full_score)
        state.add_score(self.score)

@dataclass
class Soft(OneshotTarget):
    r"""A target to hit softly.  Player should hit this target with a volume
    below a certain threshold, otherwise it will be counted as a wrong-shot target.

    Fields
    ------
    beat, length : Fraction
        `beat` is the time this target should be hit, `length` is meaningless.
    speed : float, optional
        The speed of this target (unit: half bar per second).  Default speed will
        be determined by context value `speed`, or 1.0 if absence.
    volume : float, optional
        The relative volume of sound of this target (unit: dB).  Default volume
        will be determined by context value `volume`, or 0.0 if absence.
    nofeedback : bool, optional
        Whether to make a visual cue for the action of hitting the target.
        Default value will be determined by context value `nofeedback`, or False
        if absence.
    """

    speed: Optional[float] = None
    volume: Optional[float] = None
    nofeedback: Optional[bool] = None

    def prepare(self, beatmap, context):
        self.approach_appearance = beatmap.settings.notes.soft_approach_appearance
        self.wrong_appearance = beatmap.settings.notes.soft_wrong_appearance
        sound = beatmap.resources.get(beatmap.settings.notes.soft_sound, None)
        self.sound = dn.DataNode.wrap(sound) if sound is not None else None
        self.threshold = beatmap.settings.difficulty.soft_threshold

        if self.speed is None:
            self.speed = context.get('speed', 1.0)
        if self.volume is None:
            self.volume = context.get('volume', 0.0)
        if self.nofeedback is None:
            self.nofeedback = context.get('nofeedback', False)

        super().prepare(beatmap, context)

    def hit(self, state, field, time, strength):
        super().hit(state, field, time, strength, strength < self.threshold)

@dataclass
class Loud(OneshotTarget):
    r"""A target to hit loudly.  Player should hit this target with a volume
    above a certain threshold, otherwise it will be counted as a wrong-shot target.

    Fields
    ------
    beat, length : Fraction
        `beat` is the time this target should be hit, `length` is meaningless.
    speed : float, optional
        The speed of this target (unit: half bar per second).  Default speed will
        be determined by context value `speed`, or 1.0 if absence.
    volume : float, optional
        The relative volume of sound of this target (unit: dB).  Default volume
        will be determined by context value `volume`, or 0.0 if absence.
    nofeedback : bool, optional
        Whether to make a visual cue for the action of hitting the target.
        Default value will be determined by context value `nofeedback`, or False
        if absence.
    """

    speed: Optional[float] = None
    volume: Optional[float] = None
    nofeedback: Optional[bool] = None

    def prepare(self, beatmap, context):
        self.approach_appearance = beatmap.settings.notes.loud_approach_appearance
        self.wrong_appearance = beatmap.settings.notes.loud_wrong_appearance
        sound = beatmap.resources.get(beatmap.settings.notes.loud_sound, None)
        self.sound = dn.DataNode.wrap(sound) if sound is not None else None
        self.threshold = beatmap.settings.difficulty.loud_threshold

        if self.speed is None:
            self.speed = context.get('speed', 1.0)
        if self.volume is None:
            self.volume = context.get('volume', 0.0)
        if self.nofeedback is None:
            self.nofeedback = context.get('nofeedback', False)

        super().prepare(beatmap, context)

    def hit(self, state, field, time, strength):
        super().hit(state, field, time, strength, strength >= self.threshold)

class IncrGroup:
    def __init__(self, threshold=0.0, total=0):
        self.threshold = threshold
        self.total = total
        self.volume = 0.0
        self.last_beat = None

    def hit(self, strength):
        self.threshold = max(self.threshold, strength)

@dataclass
class Incr(OneshotTarget):
    r"""A target to hit louder and louder.  Player should hit the target with a
    volume louder than the previous target, otherwise it will be counted as a
    wrong-shot target.

    Fields
    ------
    beat, length : Fraction
        `beat` is the time this target should be hit, `length` is meaningless.
    group : str, optional
        The group name this target belongs to, or None for automatically determine
        The group by context.  The threshold will be dynamically changed by
        hitting the target under the same group.
    speed : float, optional
        The speed of this target (unit: half bar per second).  Default speed will
        be determined by context value `speed`, or 1.0 if absence.
    group_volume : float, optional
        The relative group volume of sound of this target (unit: dB).  Default
        volume will be determined by context value `volume`, or 0.0 if absence.
        The volume of targets will be louder and louder, which is determined by
        the group volume of the first target in the same group.
    nofeedback : bool, optional
        Whether to make a visual cue for the action of hitting the target.
        Default value will be determined by context value `nofeedback`, or False
        if absence.
    """

    group: Optional[str] = None
    speed: Optional[float] = None
    group_volume: Optional[float] = None
    nofeedback: Optional[float] = None

    def prepare(self, beatmap, context):
        self.approach_appearance = beatmap.settings.notes.incr_approach_appearance
        self.wrong_appearance = beatmap.settings.notes.incr_wrong_appearance
        sound = beatmap.resources.get(beatmap.settings.notes.incr_sound, None)
        self.sound = dn.DataNode.wrap(sound) if sound is not None else None
        self.incr_threshold = beatmap.settings.difficulty.incr_threshold

        if self.speed is None:
            self.speed = context.get('speed', 1.0)
        if self.group_volume is None:
            self.group_volume = context.get('volume', 0.0)
        if self.nofeedback is None:
            self.nofeedback = context.get('nofeedback', False)

        super().prepare(beatmap, context)

        if '<incrs>' not in context:
            context['<incrs>'] = OrderedDict()
        self.groups = context['<incrs>']

        if self.group is None:
            # determine group of incr note according to the context
            for group, group_obj in reversed(self.groups.items()):
                if self.beat - 1 <= group_obj.last_beat <= self.beat:
                    self.group = group
                    break
            else:
                group_num = 0
                while f"#{group_num}" in self.groups:
                    group_num += 1
                self.group = f"#{group_num}"

        if self.group not in self.groups:
            group_obj = IncrGroup()
            group_obj.volume = self.group_volume
            self.groups[self.group] = group_obj

        group_obj = self.groups[self.group]
        group_obj.last_beat = self.beat
        self.groups.move_to_end(self.group)

        group_obj.total += 1
        self.count = group_obj.total

    @property
    def volume(self):
        group_obj = self.groups[self.group]
        return group_obj.volume + numpy.log10(0.2 + 0.8 * (self.count-1)/group_obj.total) * 20

    def hit(self, state, field, time, strength):
        group_obj = self.groups[self.group]
        threshold = max(0.0, min(1.0, group_obj.threshold + self.incr_threshold))
        super().hit(state, field, time, strength, strength >= threshold)
        group_obj.hit(strength)

@dataclass
class Roll(Target):
    r"""A target to hit multiple times within a certain timespan.

    Fields
    ------
    beat, length : Fraction
        `beat` is the time this target start rolling, `length` is the duration
        of rolling.
    density : int or float or Fraction, optional
        The density of rolling (unit: hit per beat).  Default value is Fraction(2).
    speed : float, optional
        The speed of this target (unit: half bar per second).  Default speed will
        be determined by context value `speed`, or 1.0 if absence.
    volume : float, optional
        The relative volume of sound of this target (unit: dB).  Default volume
        will be determined by context value `volume`, or 0.0 if absence.
    nofeedback : bool, optional
        Whether to make a visual cue for the action of hitting the target.
        Default value will be determined by context value `nofeedback`, or False
        if absence.
    """

    density: Union[int, Fraction, float] = Fraction(2)
    speed: Optional[float] = None
    volume: Optional[float] = None
    nofeedback: Optional[bool] = None

    def prepare(self, beatmap, context):
        self.performance_tolerance = beatmap.settings.difficulty.performance_tolerance
        self.tolerance = beatmap.settings.difficulty.roll_tolerance
        self.rock_appearance = beatmap.settings.notes.roll_rock_appearance
        sound = beatmap.resources.get(beatmap.settings.notes.roll_rock_sound, None)
        self.sound = dn.DataNode.wrap(sound) if sound is not None else None
        self.rock_score = beatmap.settings.scores.roll_rock_score

        if self.speed is None:
            self.speed = context.get('speed', 1.0)
        if self.volume is None:
            self.volume = context.get('volume', 0.0)
        if self.nofeedback is None:
            self.nofeedback = context.get('nofeedback', False)

        self.time = beatmap.time(self.beat)
        self.end = beatmap.time(self.beat+self.length)
        self.roll = 0
        self.number = max(int(self.length * self.density // -1 * -1), 1)
        self.is_finished = False
        self.score = 0

        self.times = [beatmap.time(self.beat+i/self.density) for i in range(self.number)]
        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.end + travel_time)
        self.range = (self.time - self.tolerance, self.end + self.tolerance)
        self.full_score = self.number * self.rock_score

    def pos_of(self, index):
        return lambda time: (self.times[index]-time) * 0.5 * self.speed

    def appearance_of(self, index):
        return lambda time: self.rock_appearance if self.nofeedback or self.roll <= index else ""

    def approach(self, state, field):
        for i, time in enumerate(self.times):
            if self.sound is not None:
                field.play(self.sound, time=time, volume=self.volume)
            field.draw_content(self.pos_of(i), self.appearance_of(i), zindex=self.zindex,
                               start=self.lifespan[0], duration=self.lifespan[1]-self.lifespan[0])
        field.reset_sight(start=self.range[0])

    def hit(self, state, field, time, strength):
        self.roll += 1

        if self.roll <= self.number:
            perf = Performance.judge(self.performance_tolerance, self.times[self.roll-1], time, True)
            state.add_perf(perf)

            state.add_score(self.rock_score)
            self.score += self.rock_score

        if self.roll == self.number:
            self.finish(state, field)

    def finish(self, state, field):
        self.is_finished = True
        state.add_full_score(self.full_score)

        for time in self.times[self.roll:]:
            perf = Performance.judge(self.performance_tolerance, time)
            state.add_perf(perf)

@dataclass
class Spin(Target):
    r"""A target to accumulate hitting volume within a certain timespan.

    Fields
    ------
    beat, length : Fraction
        `beat` is the time this target start spinning, `length` is the duration
        of spinning.
    density : int or float or Fraction, optional
        The density of spinning (unit: hit per beat).  Default value is 2.0.
    speed : float, optional
        The speed of this target (unit: half bar per second).  Default speed will
        be determined by context value `speed`, or 1.0 if absence.
    volume : float, optional
        The relative volume of sound of this target (unit: dB).  Default volume
        will be determined by context value `volume`, or 0.0 if absence.
    nofeedback : bool, optional
        Whether to make a visual cue for the action of hitting the target.
        Default value will be determined by context value `nofeedback`, or False
        if absence.
    """

    density: Union[int, Fraction, float] = 2.0
    speed: Optional[float] = None
    volume: Optional[float] = None
    nofeedback: Optional[bool] = None

    def prepare(self, beatmap, context):
        self.tolerance = beatmap.settings.difficulty.spin_tolerance
        self.disk_appearances = beatmap.settings.notes.spin_disk_appearances
        self.finishing_appearance = beatmap.settings.notes.spin_finishing_appearance
        self.finish_sustain_time = beatmap.settings.notes.spin_finish_sustain_time
        sound = beatmap.resources.get(beatmap.settings.notes.spin_disk_sound, None)
        self.sound = dn.DataNode.wrap(sound) if sound is not None else None
        self.full_score = beatmap.settings.scores.spin_score

        if self.speed is None:
            self.speed = context.get('speed', 1.0)
        if self.volume is None:
            self.volume = context.get('volume', 0.0)
        if self.nofeedback is None:
            self.nofeedback = context.get('nofeedback', False)

        self.time = beatmap.time(self.beat)
        self.end = beatmap.time(self.beat+self.length)
        self.charge = 0.0
        self.capacity = float(self.length * self.density)
        self.is_finished = False
        self.score = 0

        self.times = [beatmap.time(self.beat+i/self.density) for i in range(int(self.capacity))]
        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.end + travel_time)
        self.range = (self.time - self.tolerance, self.end + self.tolerance)

    def pos(self, time):
        return (max(0.0, self.time-time) + min(0.0, self.end-time)) * 0.5 * self.speed

    def appearance(self, time):
        if self.nofeedback or not self.is_finished:
            return self.disk_appearances[int(self.charge) % len(self.disk_appearances)]
        else:
            return ""

    def approach(self, state, field):
        for time in self.times:
            if self.sound is not None:
                field.play(self.sound, time=time, volume=self.volume)

        field.draw_content(self.pos, self.appearance, zindex=self.zindex,
                           start=self.lifespan[0], duration=self.lifespan[1]-self.lifespan[0])
        field.draw_sight("", start=self.range[0], duration=self.range[1]-self.range[0])

    def hit(self, state, field, time, strength):
        self.charge = min(self.charge + min(1.0, strength), self.capacity)

        current_score = int(self.full_score * self.charge / self.capacity)
        state.add_score(current_score - self.score)
        self.score = current_score

        if self.charge == self.capacity:
            self.finish(state, field)

    def finish(self, state, field):
        self.is_finished = True
        state.add_full_score(self.full_score)

        if self.charge != self.capacity:
            state.add_score(-self.score)
            self.score = 0

        if self.charge != self.capacity:
            return

        if not self.nofeedback:
            appearance = self.finishing_appearance
            if isinstance(appearance, tuple) and self.speed < 0:
                appearance = appearance[::-1]
            field.draw_sight(appearance, duration=self.finish_sustain_time)

# beatmap
class BeatmapSettings(cfg.Configurable):
    class difficulty(cfg.Configurable):
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

    class scores(cfg.Configurable):
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

    class notes(cfg.Configurable):
        soft_approach_appearance:  Union[str, Tuple[str, str]] = "\x1b[96m□\x1b[m"
        soft_wrong_appearance:     Union[str, Tuple[str, str]] = "\x1b[96m⬚\x1b[m"
        soft_sound: str = 'soft'
        loud_approach_appearance:  Union[str, Tuple[str, str]] = "\x1b[94m■\x1b[m"
        loud_wrong_appearance:     Union[str, Tuple[str, str]] = "\x1b[94m⬚\x1b[m"
        loud_sound: str = 'loud'
        incr_approach_appearance:  Union[str, Tuple[str, str]] = "\x1b[94m⬒\x1b[m"
        incr_wrong_appearance:     Union[str, Tuple[str, str]] = "\x1b[94m⬚\x1b[m"
        incr_sound: str = 'incr'
        roll_rock_appearance:      Union[str, Tuple[str, str]] = "\x1b[96m◎\x1b[m"
        roll_rock_sound: str = 'rock'
        spin_disk_appearances:     Union[List[str], List[Tuple[str, str]]] = ["\x1b[94m◴\x1b[m",
                                                                              "\x1b[94m◵\x1b[m",
                                                                              "\x1b[94m◶\x1b[m",
                                                                              "\x1b[94m◷\x1b[m"]
        spin_finishing_appearance: Union[str, Tuple[str, str]] = "\x1b[94m☺\x1b[m"
        spin_finish_sustain_time: float = 0.1
        spin_disk_sound: str = 'disk'
        event_leadin_time: float = 1.0

    resources: Dict[str, str] = {
        'soft': "samples/soft.wav", # pulse(freq=830.61, decay_time=0.03, amplitude=0.5)
        'loud': "samples/loud.wav", # pulse(freq=1661.2, decay_time=0.03, amplitude=1.0)
        'incr': "samples/incr.wav", # pulse(freq=1661.2, decay_time=0.03, amplitude=1.0)
        'rock': "samples/rock.wav", # pulse(freq=1661.2, decay_time=0.01, amplitude=0.5)
        'disk': "samples/disk.wav", # pulse(freq=1661.2, decay_time=0.01, amplitude=1.0)
    }

class GameplaySettings(cfg.Configurable):
    beatbar = BeatbarSettings

    class controls(cfg.Configurable):
        skip_time: float = 8.0
        load_time: float = 0.5
        prepare_time: float = 0.1
        tickrate: float = 60.0

    widgets = WidgetSettings

class BeatmapScore:
    def __init__(self):
        self.total_subjects = 0
        self.finished_subjects = 0
        self.full_score = 0
        self.score = 0
        self.perfs = []
        self.time = 0.0

    def set_total_subjects(self, total_subjects):
        self.total_subjects = total_subjects

    def add_score(self, score):
        self.score += score

    def add_full_score(self, full_score):
        self.full_score += full_score

    def add_finished(self, finished=1):
        self.finished_subjects += finished

    def add_perf(self, perf, is_reversed=False):
        self.perfs.append(perf)

class Beatmap:
    def __init__(self, root=".", audio=None, volume=0.0,
                 offset=0.0, tempo=120.0,
                 info="", preview=0.0,
                 bar_shift=0.1, bar_flip=False,
                 event_sequences=None,
                 settings=None):
        self.root = root
        self.audio = audio
        self.volume = volume
        self.offset = offset
        self.info = info
        self.preview = preview
        self.tempo = tempo
        self.bar_shift = bar_shift
        self.bar_flip = bar_flip
        self.event_sequences = event_sequences or []

        self.settings = settings or BeatmapSettings()

        self.audionode = None
        self.resources = {}

    def time(self, beat):
        r"""Convert beat to time (in seconds).

        Parameters
        ----------
        beat : int or Fraction or float

        Returns
        -------
        time : float
        """
        return self.offset + beat*60/self.tempo

    def beat(self, time):
        r"""Convert time (in seconds) to beat.

        Parameters
        ----------
        time : float

        Returns
        -------
        beat : float
        """
        return (time - self.offset)*self.tempo/60

    def dtime(self, beat, length):
        r"""Convert length to time difference (in seconds).

        Parameters
        ----------
        beat : int or Fraction or float
        length : int or Fraction or float

        Returns
        -------
        dtime : float
        """
        return self.time(beat+length) - self.time(beat)

    @dn.datanode
    def play(self, manager, data_dir, devices_settings, gameplay_settings=None):
        gameplay_settings = gameplay_settings or GameplaySettings()

        samplerate = devices_settings.mixer.output_samplerate
        nchannels = devices_settings.mixer.output_channels
        load_time = gameplay_settings.controls.load_time
        tickrate = gameplay_settings.controls.tickrate
        prepare_time = gameplay_settings.controls.prepare_time

        # prepare
        with self.load_resources(samplerate, nchannels, data_dir) as task:
            yield from task.join((yield))
        with self.prepare_events() as task:
            yield from task.join((yield))
            total_subjects, start_time, end_time, events = task.result
        self.total_subjects = total_subjects

        ref_time = load_time + abs(start_time)
        mixer_task, mixer = Mixer.create(devices_settings.mixer, manager, ref_time)
        detector_task, detector = Detector.create(devices_settings.detector, manager, ref_time)
        renderer_task, renderer = Renderer.create(devices_settings.renderer, ref_time)

        beatbar = Beatbar(mixer, detector, renderer, self.bar_shift, self.bar_flip, gameplay_settings.beatbar)

        score = BeatmapScore()
        score.set_total_subjects(self.total_subjects)

        # play music
        if self.audionode is not None:
            beatbar.mixer.play(self.audionode, time=0.0, zindex=(-3,))

        # install widgets
        for widget in gameplay_settings.widgets.use:
            WidgetManager.use_widget(widget, score, beatbar, devices_settings, gameplay_settings.widgets)

        # game loop
        updater = self.update_events(events, score, beatbar, start_time, end_time, tickrate, prepare_time)
        event_task = dn.interval(consumer=updater, dt=1/tickrate)

        with dn.pipe(event_task, mixer_task, detector_task, renderer_task) as task:
            yield from task.join((yield))

        return score

    def load_resources(self, output_samplerate, output_nchannels, data_dir):
        r"""Load resources asynchronously.

        Parameters
        ----------
        output_samplerate : int
        output_channels : int
        data_dir : Path
        """
        return dn.create_task(lambda stop_event: self._load_resources(output_samplerate,
                                                                      output_nchannels,
                                                                      data_dir,
                                                                      stop_event))

    def _load_resources(self, output_samplerate, output_nchannels, data_dir, stop_event):
        if self.audio is not None:
            audio_path = os.path.join(self.root, self.audio)
            self.audionode = dn.DataNode.wrap(dn.load_sound(audio_path,
                                              samplerate=output_samplerate,
                                              channels=output_nchannels,
                                              volume=self.volume,
                                              stop_event=stop_event))

        for name, path in self.settings.resources.items():
            sound_path = os.path.join(data_dir, path)
            self.resources[name] = dn.load_sound(sound_path,
                                                 samplerate=output_samplerate,
                                                 channels=output_nchannels,
                                                 stop_event=stop_event)

    def prepare_events(self):
        r"""Prepare events asynchronously.

        Returns
        -------
        total_subjects: int
        start_time: float
        end_time: float
        events: list of Event
        """
        return dn.create_task(lambda stop_event: self._prepare_events(stop_event))

    def _prepare_events(self, stop_event):
        events = []
        for sequence in self.event_sequences:
            context = {}
            for event in sequence:
                if stop_event.is_set():
                    raise RuntimeError("The operation has been cancelled.")
                event = replace(event)
                event.prepare(self, context)
                if isinstance(event, Event):
                    events.append(event)

        events = sorted(events, key=lambda e: e.lifespan[0])

        duration = 0.0
        if self.audio is not None:
            with audioread.audio_open(os.path.join(self.root, self.audio)) as file:
                duration = file.duration

        event_leadin_time = self.settings.notes.event_leadin_time
        total_subjects = sum([1 for event in events if event.is_subject], 0)
        start_time = min([0.0, *[event.lifespan[0] - event_leadin_time for event in events]])
        end_time = max([duration, *[event.lifespan[1] + event_leadin_time for event in events]])

        return total_subjects, start_time, end_time, events

    @dn.datanode
    def update_events(self, events, state, beatbar, start_time, end_time, tickrate, prepare_time):
        # register events
        events_iter = iter(events)
        event = next(events_iter, None)

        yield
        index = 0

        while True:
            time = index / tickrate + start_time

            if end_time <= time:
                return

            while event is not None and event.lifespan[0] - prepare_time <= time:
                event.register(state, beatbar)
                event = next(events_iter, None)

            state.time = time

            yield
            index += 1
