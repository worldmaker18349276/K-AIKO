import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from collections import OrderedDict
from fractions import Fraction
import threading
import numpy
from ..utils import config as cfg
from ..utils import datanodes as dn
from ..utils import markups as mu
from ..devices import audios as aud
from ..devices import engines
from .beatbar import PerformanceGrade, Performance, Beatbar, Sight, BeatbarSettings, BeatbarWidgetSettings, BeatbarWidgetBuilder


@dataclass
class UpdateContext:
    r"""An pseudo-event to update context in preparation phase.

    Attributes
    ----------
    update : dict
        The updated fields in the context.
    """

    update: Dict[str, Union[None, bool, int, Fraction, float, str]]

    def prepare(self, beatmap, rich, context):
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
    prepare(beatmap, rich, context)
        Prepare resources for this event in the given context.  The context is a
        mutable dictionary, which can be used to transfer parameters between events.
        The context of each track is different, so event cannot affect each others
        between tracks.
    register(state, playfield)
        Schedule handlers for this event.  `state` is the game state of beatmap,
        `playfield` is an instance of `Beatmap`, which controls the whole
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

    def prepare(self, beatmap, rich, context):
        self.time = beatmap.metronome.time(self.beat)
        if self.speed is None:
            self.speed = context.get('speed', 1.0)

        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.time + travel_time)
        self.zindex = (-2, -self.time)

    def pos(self, time):
        return (self.time-time) * 0.5 * self.speed

    def register(self, state, field):
        if self.text is not None:
            field.draw_content(self.pos, mu.Text(self.text), zindex=self.zindex,
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

    def prepare(self, beatmap, rich, context):
        self.time = beatmap.metronome.time(self.beat)
        self.end = beatmap.metronome.time(self.beat + self.length)

        self.lifespan = (self.time, self.end)
        self.zindex = (10, -self.time)

    def register(self, state, field):
        if self.text is not None:
            field.draw_title(self.pos, mu.Text(self.text), zindex=self.zindex,
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

    def prepare(self, beatmap, rich, context):
        self.time = beatmap.metronome.time(self.beat)
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

    def prepare(self, beatmap, rich, context):
        self.time = beatmap.metronome.time(self.beat)
        self.end = beatmap.metronome.time(self.beat+self.span)
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
    sound : datanodes.DataNode or None
        The sound of the auditory cue of this target.
    approach_appearance : tuple of str and str
        The appearance of approaching target.
    wrong_appearance : tuple of str and str
        The appearance of wrong-shot target.

    Methods
    -------
    hit(state, playfield, time, strength)
    """

    has_length = False

    def prepare(self, beatmap, rich, context):
        self.performance_tolerance = beatmap.settings.difficulty.performance_tolerance

        self.time = beatmap.metronome.time(self.beat)
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
            return (mu.Text(""), mu.Text(""))

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

    def prepare(self, beatmap, rich, context):
        self.approach_appearance = (
            rich.parse(beatmap.settings.notes.soft_approach_appearance[0]),
            rich.parse(beatmap.settings.notes.soft_approach_appearance[1]),
        )
        self.wrong_appearance = (
            rich.parse(beatmap.settings.notes.soft_wrong_appearance[0]),
            rich.parse(beatmap.settings.notes.soft_wrong_appearance[1]),
        )
        sound = beatmap.resources.get(beatmap.settings.notes.soft_sound, None)
        self.sound = dn.DataNode.wrap(sound) if sound is not None else None
        self.threshold = beatmap.settings.difficulty.soft_threshold

        if self.speed is None:
            self.speed = context.get('speed', 1.0)
        if self.volume is None:
            self.volume = context.get('volume', 0.0)
        if self.nofeedback is None:
            self.nofeedback = context.get('nofeedback', False)

        super().prepare(beatmap, rich, context)

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

    def prepare(self, beatmap, rich, context):
        self.approach_appearance = (
            rich.parse(beatmap.settings.notes.loud_approach_appearance[0]),
            rich.parse(beatmap.settings.notes.loud_approach_appearance[1]),
        )
        self.wrong_appearance = (
            rich.parse(beatmap.settings.notes.loud_wrong_appearance[0]),
            rich.parse(beatmap.settings.notes.loud_wrong_appearance[1]),
        )
        sound = beatmap.resources.get(beatmap.settings.notes.loud_sound, None)
        self.sound = dn.DataNode.wrap(sound) if sound is not None else None
        self.threshold = beatmap.settings.difficulty.loud_threshold

        if self.speed is None:
            self.speed = context.get('speed', 1.0)
        if self.volume is None:
            self.volume = context.get('volume', 0.0)
        if self.nofeedback is None:
            self.nofeedback = context.get('nofeedback', False)

        super().prepare(beatmap, rich, context)

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

    def prepare(self, beatmap, rich, context):
        self.approach_appearance = (
            rich.parse(beatmap.settings.notes.incr_approach_appearance[0]),
            rich.parse(beatmap.settings.notes.incr_approach_appearance[1]),
        )
        self.wrong_appearance = (
            rich.parse(beatmap.settings.notes.incr_wrong_appearance[0]),
            rich.parse(beatmap.settings.notes.incr_wrong_appearance[1]),
        )
        sound = beatmap.resources.get(beatmap.settings.notes.incr_sound, None)
        self.sound = dn.DataNode.wrap(sound) if sound is not None else None
        self.incr_threshold = beatmap.settings.difficulty.incr_threshold

        if self.speed is None:
            self.speed = context.get('speed', 1.0)
        if self.group_volume is None:
            self.group_volume = context.get('volume', 0.0)
        if self.nofeedback is None:
            self.nofeedback = context.get('nofeedback', False)

        super().prepare(beatmap, rich, context)

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

    def prepare(self, beatmap, rich, context):
        self.performance_tolerance = beatmap.settings.difficulty.performance_tolerance
        self.tolerance = beatmap.settings.difficulty.roll_tolerance
        self.rock_appearance = (
            rich.parse(beatmap.settings.notes.roll_rock_appearance[0]),
            rich.parse(beatmap.settings.notes.roll_rock_appearance[1]),
        )
        sound = beatmap.resources.get(beatmap.settings.notes.roll_rock_sound, None)
        self.sound = sound
        self.rock_score = beatmap.settings.scores.roll_rock_score

        if self.speed is None:
            self.speed = context.get('speed', 1.0)
        if self.volume is None:
            self.volume = context.get('volume', 0.0)
        if self.nofeedback is None:
            self.nofeedback = context.get('nofeedback', False)

        self.time = beatmap.metronome.time(self.beat)
        self.end = beatmap.metronome.time(self.beat+self.length)
        self.roll = 0
        self.number = max(int(self.length * self.density // -1 * -1), 1)
        self.is_finished = False
        self.score = 0

        self.times = [beatmap.metronome.time(self.beat+i/self.density) for i in range(self.number)]
        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.end + travel_time)
        self.range = (self.time - self.tolerance, self.end + self.tolerance)
        self.full_score = self.number * self.rock_score

    def pos_of(self, index):
        return lambda time: (self.times[index]-time) * 0.5 * self.speed

    def appearance_of(self, index):
        return lambda time: self.rock_appearance if self.nofeedback or self.roll <= index else (mu.Text(""), mu.Text(""))

    def approach(self, state, field):
        for i, time in enumerate(self.times):
            if self.sound is not None:
                field.play(dn.DataNode.wrap(self.sound), time=time, volume=self.volume)
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

    def prepare(self, beatmap, rich, context):
        self.tolerance = beatmap.settings.difficulty.spin_tolerance
        self.disk_appearances = [(rich.parse(spin_disk_appearance[0]), rich.parse(spin_disk_appearance[1]))
                                 for spin_disk_appearance in beatmap.settings.notes.spin_disk_appearances]
        self.finishing_appearance = (
            rich.parse(beatmap.settings.notes.spin_finishing_appearance[0]),
            rich.parse(beatmap.settings.notes.spin_finishing_appearance[1]),
        )
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

        self.time = beatmap.metronome.time(self.beat)
        self.end = beatmap.metronome.time(self.beat+self.length)
        self.charge = 0.0
        self.capacity = float(self.length * self.density)
        self.is_finished = False
        self.score = 0

        self.times = [beatmap.metronome.time(self.beat+i/self.density) for i in range(int(self.capacity))]
        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.end + travel_time)
        self.range = (self.time - self.tolerance, self.end + self.tolerance)

    def pos(self, time):
        return (max(0.0, self.time-time) + min(0.0, self.end-time)) * 0.5 * self.speed

    def appearance(self, time):
        if self.nofeedback or not self.is_finished:
            return self.disk_appearances[int(self.charge) % len(self.disk_appearances)]
        else:
            return (mu.Text(""), mu.Text(""))

    def approach(self, state, field):
        for time in self.times:
            if self.sound is not None:
                field.play(self.sound, time=time, volume=self.volume)

        field.draw_content(self.pos, self.appearance, zindex=self.zindex,
                           start=self.lifespan[0], duration=self.lifespan[1]-self.lifespan[0])
        field.draw_sight((mu.Text(""), mu.Text("")), start=self.range[0], duration=self.range[1]-self.range[0])

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
            if self.speed < 0:
                appearance = appearance[::-1]
            field.draw_sight(appearance, duration=self.finish_sustain_time)

# beatmap
class BeatmapSettings(cfg.Configurable):
    r"""
    Fields
    ------
    resources : dict from str to str
        The resource name and file path.
    """

    @cfg.subconfig
    class difficulty(cfg.Configurable):
        r"""
        Fields
        ------
        performance_tolerance : float
            The minimal timing tolerance to judge performance.
        soft_threshold : float
            The maximum strength to succeed soft note.
        loud_threshold : float
            The minimum strength to succeed loud note.
        incr_threshold : float
            The threshold of increasing strength to succeed incr note.
        roll_tolerance : float
            The timing tolerance to succeed roll note.
        spin_tolerance : float
            The timing tolerance to succeed spin note.
        """
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

    @cfg.subconfig
    class scores(cfg.Configurable):
        r"""
        Fields
        ------
        performances_scores : dict from PerformanceGrade to int
            The grades of different performance.
        roll_rock_score : int
            The score of each rock in the roll note.
        spin_score : int
            The score of sping note.
        """
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

    @cfg.subconfig
    class notes(cfg.Configurable):
        r"""
        Fields
        ------
        soft_approach_appearance : tuple of str and str
            The appearance of approaching soft note.
        soft_wrong_appearance : tuple of str and str
            The appearance of wrong soft note.
        soft_sound : str
            The name of sound of soft note.

        loud_approach_appearance : tuple of str and str
            The appearance of approaching loud note.
        loud_wrong_appearance : tuple of str and str
            The appearance of wrong loud note.
        loud_sound : str
            The name of sound of loud note.

        incr_approach_appearance : tuple of str and str
            The appearance of approaching incr note.
        incr_wrong_appearance : tuple of str and str
            The appearance of wrong incr note.
        incr_sound : str
            The name of sound of incr note.

        roll_rock_approach_appearance : tuple of str and str
            The appearance of approaching roll in the rock note.
        roll_rock_sound : str
            The name of sound of roll in the rock note.

        spin_disk_appearances : list of tuple of str and str
            The spinning appearance of spin note.
        spin_finishing_appearance : tuple of str and str
            The finishing appearance of spin note.
        spin_finish_sustain_time : float
            The sustain time for the finishing spin note.
        spin_disk_sound : str
            The name of sound of spin note.

        event_leadin_time : float
            The minimum time of silence before and after the gameplay.
        """
        soft_approach_appearance:  Tuple[str, str] = ("[color=bright_cyan]□[/]", "[color=bright_cyan]□[/]")
        soft_wrong_appearance:     Tuple[str, str] = ("[color=bright_cyan]⬚[/]", "[color=bright_cyan]⬚[/]")
        soft_sound: str = 'soft'
        loud_approach_appearance:  Tuple[str, str] = ("[color=bright_blue]■[/]", "[color=bright_blue]■[/]")
        loud_wrong_appearance:     Tuple[str, str] = ("[color=bright_blue]⬚[/]", "[color=bright_blue]⬚[/]")
        loud_sound: str = 'loud'
        incr_approach_appearance:  Tuple[str, str] = ("[color=bright_blue]⬒[/]", "[color=bright_blue]⬒[/]")
        incr_wrong_appearance:     Tuple[str, str] = ("[color=bright_blue]⬚[/]", "[color=bright_blue]⬚[/]")
        incr_sound: str = 'incr'
        roll_rock_appearance:      Tuple[str, str] = ("[color=bright_cyan]◎[/]", "[color=bright_cyan]◎[/]")
        roll_rock_sound: str = 'rock'
        spin_disk_appearances:     List[Tuple[str, str]] = [("[color=bright_blue]◴[/]", "[color=bright_blue]◴[/]"),
                                                            ("[color=bright_blue]◵[/]", "[color=bright_blue]◵[/]"),
                                                            ("[color=bright_blue]◶[/]", "[color=bright_blue]◶[/]"),
                                                            ("[color=bright_blue]◷[/]", "[color=bright_blue]◷[/]")]
        spin_finishing_appearance: Tuple[str, str] = ("[color=bright_blue]☺[/]", "[color=bright_blue]☺[/]")
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
    debug_monitor: bool = False

    beatbar = cfg.subconfig(BeatbarSettings)

    @cfg.subconfig
    class controls(cfg.Configurable):
        r"""
        Fields
        ------
        skip_time : float
        load_time : float
            The minimum time before the first event.
        prepare_time : float
            The time between preparing the event and the lifespan of the event.
        tickrate : float
            The event updating rate.

        stop_key : str
            The key to stop the game.
        sound_delay_adjust_keys : tuple of str and str
            The keys to adjust click sound delay.
            The first/second string is the key to adjust faster/slower.
        sound_delay_adjust_step : float
            The adjustment interval of click sound delay.
        display_delay_adjust_keys : tuple of str and str
            The keys to adjust display delay.
            The first/second string is the key to adjust faster/slower.
        display_delay_adjust_step : float
            The adjustment interval of display delay.
        knock_delay_adjust_keys : tuple of str and str
            The keys to adjust knock delay.
            The first/second string is the key to adjust slower/faster.
        knock_delay_adjust_step : float
            The adjustment interval of knock delay.
        knock_energy_adjust_keys : tuple of str and str
            The keys to adjust knock energy.
            The first/second string is the key to adjust softer/louder.
        knock_energy_adjust_step : float
            The adjustment interval of knock energy.
        """
        skip_time: float = 8.0
        load_time: float = 0.5
        prepare_time: float = 0.1
        tickrate: float = 60.0
        stop_key: str = 'Esc'

        sound_delay_adjust_keys: Tuple[str, str] = ('Shift_Left', 'Shift_Right')
        display_delay_adjust_keys: Tuple[str, str] = ('Ctrl_Left', 'Ctrl_Right')
        knock_delay_adjust_keys: Tuple[str, str] = ('Left', 'Right')
        knock_energy_adjust_keys: Tuple[str, str] = ('Down', 'Up')
        sound_delay_adjust_step: float = 0.001
        display_delay_adjust_step: float = 0.001
        knock_delay_adjust_step: float = 0.001
        knock_energy_adjust_step: float = 0.0001

    widgets = cfg.subconfig(BeatbarWidgetSettings)

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

@dataclass
class BeatmapAudio:
    path: Optional[str] = None
    volume: float = 0.0
    preview: float = 0.0
    info: str = ""

@dataclass
class BeatbarState:
    bar_shift: float = 0.1
    bar_flip: bool = False

class Beatmap:
    def __init__(
        self, *,
        root=None,
        info=None,
        audio=None,
        metronome=None,
        beatbar_state=None,
        event_sequences=None,
        settings=None,
    ):
        self.root = root if root is not None else Path(".").resolve()
        self.info = info if info is not None else ""
        self.audio = audio if audio is not None else BeatmapAudio()
        self.metronome = metronome if metronome is not None else engines.Metronome(offset=0.0, tempo=120.0)
        self.beatbar_state = beatbar_state if beatbar_state is not None else BeatbarState()
        self.event_sequences = event_sequences if event_sequences is not None else []
        self.settings = settings if settings is not None else BeatmapSettings()

        self.audionode = None
        self.resources = {}

    @dn.datanode
    def play(self, manager, user, devices_settings, gameplay_settings=None):
        gameplay_settings = gameplay_settings or GameplaySettings()

        samplerate = devices_settings.mixer.output_samplerate
        nchannels = devices_settings.mixer.output_channels
        load_time = gameplay_settings.controls.load_time
        tickrate = gameplay_settings.controls.tickrate
        prepare_time = gameplay_settings.controls.prepare_time
        debug_monitor = gameplay_settings.debug_monitor

        rich = mu.RichParser(devices_settings.terminal.unicode_version, devices_settings.terminal.color_support)

        # prepare
        try:
            yield from self.load_resources(samplerate, nchannels, user.data_dir).join()
        except aud.IOCancelled:
            return

        total_subjects, start_time, end_time, events = yield from self.prepare_events(rich).join()

        score = BeatmapScore()
        score.set_total_subjects(total_subjects)

        # load engines
        mixer_monitor = detector_monitor = renderer_monitor = None
        if debug_monitor:
            mixer_monitor = engines.Monitor(user.cache_dir / "monitor" / "mixer.csv")
            detector_monitor = engines.Monitor(user.cache_dir / "monitor" / "detector.csv")
            renderer_monitor = engines.Monitor(user.cache_dir / "monitor" / "renderer.csv")

        ref_time = load_time + abs(start_time)
        mixer_task, mixer = engines.Mixer.create(devices_settings.mixer, manager, ref_time, mixer_monitor)
        detector_task, detector = engines.Detector.create(devices_settings.detector, manager, ref_time, detector_monitor)
        renderer_task, renderer = engines.Renderer.create(devices_settings.renderer, devices_settings.terminal, ref_time, renderer_monitor)
        controller_task, controller = engines.Controller.create(devices_settings.controller, devices_settings.terminal, ref_time)

        # load widgets
        widget_builder = BeatbarWidgetBuilder(
            state=score,
            rich=rich,
            mixer=mixer,
            detector=detector,
            renderer=renderer,
            controller=controller,
            devices_settings=devices_settings
        )
        icon = yield from widget_builder.create(gameplay_settings.widgets.icon_widget).load().join()
        header = yield from widget_builder.create(gameplay_settings.widgets.header_widget).load().join()
        footer = yield from widget_builder.create(gameplay_settings.widgets.footer_widget).load().join()
        sight = yield from Sight(rich, gameplay_settings.beatbar.sight).load().join()

        # make beatbar
        beatbar = Beatbar(
            mixer, detector, renderer, controller,
            icon, header, footer, sight,
            self.beatbar_state.bar_shift, self.beatbar_state.bar_flip,
            gameplay_settings.beatbar
        )

        yield from beatbar.load().join()

        # handler
        stop_event = threading.Event()
        beatbar.add_handler(lambda _: stop_event.set(), gameplay_settings.controls.stop_key)

        sound_delay_adjust_step = gameplay_settings.controls.sound_delay_adjust_step
        def incr_sound_delay(_): devices_settings.mixer.sound_delay += sound_delay_adjust_step
        def decr_sound_delay(_): devices_settings.mixer.sound_delay -= sound_delay_adjust_step
        beatbar.add_handler(incr_sound_delay, gameplay_settings.controls.sound_delay_adjust_keys[0])
        beatbar.add_handler(decr_sound_delay, gameplay_settings.controls.sound_delay_adjust_keys[1])

        display_delay_adjust_step = gameplay_settings.controls.display_delay_adjust_step
        def incr_display_delay(_): devices_settings.renderer.display_delay += display_delay_adjust_step
        def decr_display_delay(_): devices_settings.renderer.display_delay -= display_delay_adjust_step
        beatbar.add_handler(incr_display_delay, gameplay_settings.controls.display_delay_adjust_keys[0])
        beatbar.add_handler(decr_display_delay, gameplay_settings.controls.display_delay_adjust_keys[1])

        knock_delay_adjust_step = gameplay_settings.controls.knock_delay_adjust_step
        def incr_knock_delay(_): devices_settings.detector.knock_delay += knock_delay_adjust_step
        def decr_knock_delay(_): devices_settings.detector.knock_delay -= knock_delay_adjust_step
        beatbar.add_handler(incr_knock_delay, gameplay_settings.controls.knock_delay_adjust_keys[0])
        beatbar.add_handler(decr_knock_delay, gameplay_settings.controls.knock_delay_adjust_keys[1])

        knock_energy_adjust_step = gameplay_settings.controls.knock_energy_adjust_step
        def incr_knock_energy(_): devices_settings.detector.knock_energy += knock_energy_adjust_step
        def decr_knock_energy(_): devices_settings.detector.knock_energy -= knock_energy_adjust_step
        beatbar.add_handler(incr_knock_energy, gameplay_settings.controls.knock_energy_adjust_keys[0])
        beatbar.add_handler(decr_knock_energy, gameplay_settings.controls.knock_energy_adjust_keys[1])

        # play music
        if self.audionode is not None:
            beatbar.mixer.play(self.audionode, time=0.0, zindex=(-3,))

        # game loop
        updater = self.update_events(events, score, beatbar, start_time, end_time, tickrate, prepare_time, stop_event)
        event_task = dn.interval(consumer=updater, dt=1/tickrate)

        yield from dn.pipe(event_task, mixer_task, detector_task, renderer_task, controller_task).join()

        if debug_monitor:
            print()
            print("   mixer: " + str(mixer_monitor))
            print("detector: " + str(detector_monitor))
            print("renderer: " + str(renderer_monitor))

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
        if self.audio.path is not None:
            try:
                self.audionode = dn.DataNode.wrap(aud.load_sound(self.audio.path,
                                                                 samplerate=output_samplerate,
                                                                 channels=output_nchannels,
                                                                 volume=self.audio.volume,
                                                                 stop_event=stop_event))
            except Exception as e:
                raise RuntimeError(f"Failed to load song {self.audio.path}") from e

        for name, path in self.settings.resources.items():
            sound_path = os.path.join(data_dir, path)
            try:
                self.resources[name] = aud.load_sound(sound_path,
                                                      samplerate=output_samplerate,
                                                      channels=output_nchannels,
                                                      stop_event=stop_event)
            except Exception as e:
                raise RuntimeError(f"Failed to load resource {name} at {sound_path}") from e

    def prepare_events(self, rich):
        r"""Prepare events asynchronously.

        Parameters
        ----------
        rich : markups.RichParser

        Returns
        -------
        total_subjects: int
        start_time: float
        end_time: float
        events: list of Event
        """
        return dn.create_task(lambda stop_event: self._prepare_events(rich, stop_event))

    def _prepare_events(self, rich, stop_event):
        events = []
        for sequence in self.event_sequences:
            context = {}
            for event in sequence:
                if stop_event.is_set():
                    raise RuntimeError("The operation has been cancelled.")
                event = replace(event)
                event.prepare(self, rich, context)
                if isinstance(event, Event):
                    events.append(event)

        events = sorted(events, key=lambda e: e.lifespan[0])

        duration = 0.0
        if self.audio.path is not None:
            duration = aud.AudioMetadata.read(self.audio.path).duration

        event_leadin_time = self.settings.notes.event_leadin_time
        total_subjects = sum([1 for event in events if event.is_subject], 0)
        start_time = min([0.0, *[event.lifespan[0] - event_leadin_time for event in events]])
        end_time = max([duration, *[event.lifespan[1] + event_leadin_time for event in events]])

        return total_subjects, start_time, end_time, events

    @dn.datanode
    def update_events(self, events, state, beatbar, start_time, end_time, tickrate, prepare_time, stop_event):
        # register events
        events_iter = iter(events)
        event = next(events_iter, None)

        yield
        index = 0

        while True:
            if stop_event.is_set():
                break

            time = index / tickrate + start_time

            if end_time <= time:
                return

            while event is not None and event.lifespan[0] - prepare_time <= time:
                event.register(state, beatbar)
                event = next(events_iter, None)

            state.time = time

            yield
            index += 1

class Loop(Beatmap):
    def __init__(
            self, *,
            metronome=None,
            width=Fraction(0),
            events=None,
            settings=None
        ):
        if metronome is None:
            metronome = engines.Metronome(offset=1.0, tempo=120.0)
        super().__init__(metronome=metronome, event_sequences=[events], settings=settings)
        self.width = width

    def repeat_events(self, rich):
        sequence = self.event_sequences[0]
        width = self.width
        context = {}

        n = 0
        while True:
            events = []

            for event in sequence:
                event = replace(event, beat=event.beat+n*width)
                event.prepare(self, rich, context)
                if isinstance(event, Event):
                    events.append(event)

            events = sorted(events, key=lambda e: e.lifespan[0])
            yield from events
            n += 1

    def _prepare_events(self, rich, stop_event):
        total_subjects = 0
        start_time = 0.0
        end_time = float('inf')
        events = self.repeat_events(rich)

        return total_subjects, start_time, end_time, events

