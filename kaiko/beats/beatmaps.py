import contextlib
import dataclasses
from pathlib import Path
from enum import Enum
from typing import List, Tuple, Dict, Optional, Union, Callable
from collections import OrderedDict
from fractions import Fraction
import threading
import numpy
from ..utils import providers
from ..utils import config as cfg
from ..utils import datanodes as dn
from ..utils import markups as mu
from ..devices import audios as aud
from ..devices import clocks
from ..tui import widgets
from ..tui import beatbars
from . import beatpatterns


@dataclasses.dataclass
class UpdateContext:
    r"""An pseudo-event to update context in preparation phase.

    Fields
    ------
    beat, length : Fraction
        The start time and sustain time.  length has no meaning.
    update : dict
        The updated fields in the context.
    """
    beat: Fraction = Fraction(0, 1)
    length: Fraction = Fraction(0, 1)

    update: Dict[str, Union[None, bool, int, Fraction, float, str]] = dataclasses.field(
        default_factory=dict
    )

    def prepare(self, beatmap, rich, context):
        context.update(**self.update)

    @staticmethod
    def make(beat, length, **contexts):
        return UpdateContext(beat, length, contexts)


@dataclasses.dataclass
class Comment:
    r"""An pseudo-event to annotate patterns in sheet.

    Fields
    ------
    beat, length : Fraction
        The start time and sustain time.  length has no meaning.
    comment : str
    """

    beat: Fraction = Fraction(0, 1)
    length: Fraction = Fraction(0, 1)
    comment: str = ""

    def prepare(self, beatmap, rich, context):
        pass


@dataclasses.dataclass
class Event:
    r"""An event of beatmap. Event represents an effect and action that occur
    within a specific time span.

    The tracks, which are sequences of events, can be drawn as a timeline
    diagram

    ..code::

        time    ____00_____________________01_____________________02_____________________03_____
               |                                                                                |
        track1 |                     [event1]    [===event2====]         []                     |
        track2 |                              [========event4========]    [event5]              |
               |________________________________________________________________________________|

    Where `track1` contains three events, and `track2` contains two events. Like
    `event3`, some events have no length (determined by field `has_length`).
    The square bracket is an interval `(beat, beat+length)`, which define the
    timespan of action of this event, just like hit object and hold object in
    others rhythm game. The interval should be ordered and exclusive in each
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
    in and out of the view. Just like `event1` and `event2`, the lifespan of
    events can be unordered and overlapping.

    Event is a dataclass described by some (immutable) fields, which is
    documented in the section `Fields`. These fields determine how this event
    occurred. Event also contains some attributes related to runtime conditions,
    which is documented in the section `Attributes`. These attributes are
    determined by method `Event.prepare` and may change during execution. One
    can use `dataclasses.replace` to copy the event, which will not copy runtime
    attributes.

    Class Fields
    ------------
    has_length : bool, optional
        True if the field `length` has no effect on the game.
    is_subject : bool, optional
        True if this event is an action. To increase the value of progress bar,
        use `state.add_finished`.

    Fields
    ------
    beat, length : Fraction
        The start time and sustain time of action of this event. The actual
        meaning in each event is different. `beat` is the time in the unit
        defined by beatmap. `length` is the time difference started from `beat`
        in the unit defined by beatmap. If `has_length` is False, the attribute
        `length` can be dropped.

    Attributes
    ----------
    lifespan : tuple of float and float
        The start time and end time (in seconds) of this event. In general, the
        lifespan is determined by attributes `beat` and `length`.
    full_score : int, optional
        The full score of this event. To increase score counter and full score
        counter, use `state.add_score` and `state.add_full_score`.

    Methods
    -------
    prepare(beatmap, rich, context)
        Prepare resources for this event in the given context. The context is a
        mutable dictionary, which can be used to transfer parameters between
        events. The context of each track is different, so event cannot affect
        each others between tracks.
    register(state, beatbar)
        Schedule handlers for this event. `state` is the game state of beatmap,
        `beatbar` is an instance of `beatbars.Beatbar`, which controls the
        scrolling bar of the game.
    """

    beat: Fraction = Fraction(0, 1)
    length: Fraction = Fraction(1, 1)
    has_length = True

    is_subject = False
    full_score = 0


# scripts
@dataclasses.dataclass
class Text(Event):
    r"""An event that displays text on the beatbar. The text will move to the
    left at a constant speed.

    Fields
    ------
    beat, length : Fraction
        `beat` is the time the text pass through the position 0.0, `length` is
        meaningless.
    text : str, optional
        The text to show, or None for no text.
    speed : float, optional
        The speed of the text (unit: half bar per second). Default speed will be
        determined by context value `speed`, or 1.0 if absence.
    """

    has_length = False

    text: Optional[str] = None
    speed: Optional[float] = None

    def prepare(self, beatmap, rich, context):
        self.time = beatmap.beatpoints.time(self.beat)
        if self.speed is None:
            self.speed = context.get("speed", 1.0)

        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.time + travel_time)
        self.zindex = (-2, -self.time)

    def pos(self, time):
        return (self.time - time) * 0.5 * self.speed

    def register(self, state, beatbar):
        if self.text is not None:
            beatbar.draw_content(
                self.pos,
                mu.Text(self.text),
                zindex=self.zindex,
                start=self.lifespan[0],
                duration=self.lifespan[1] - self.lifespan[0],
            )


@dataclasses.dataclass
class Title(Event):
    r"""An event that displays title on the beatbar. The text will be placed
    at the specific position.

    Fields
    ------
    beat, length : Fraction
        `beat` is the display time of the text, `length` is the display
        duration.
    text : str, optional
        The text to show, or None for no text.
    pos : float, optional
        The position to show, default is 0.5.
    """

    has_length = True

    text: Optional[str] = None
    pos: float = 0.5

    def prepare(self, beatmap, rich, context):
        self.time = beatmap.beatpoints.time(self.beat)
        self.end = beatmap.beatpoints.time(self.beat + self.length)

        self.lifespan = (self.time, self.end)
        self.zindex = (10, -self.time)

    def register(self, state, beatbar):
        if self.text is not None:
            beatbar.draw_title(
                self.pos,
                mu.Text(self.text),
                zindex=self.zindex,
                start=self.lifespan[0],
                duration=self.lifespan[1] - self.lifespan[0],
            )


# targets
class Target(Event):
    r"""A target to hit.

    Target will be counted as an action, and recorded by the progress bar during
    the gameplay. Players will be asked to do some actions to accomplish this
    target, such as hitting a note or keeping hitting within a timespan.

    Fields
    ------
    beat, length : Fraction
        `beat` is the time this target should be hit, `length` is the duration
        of this target.

    Attributes
    ----------
    range : tuple of float and float
        The range of time this target listen to. It is often slightly larger
        than the interval `(beat, beat+length)` to includes the hit tolerance.
    is_finished : bool
        Whether this target is finished. The progress bar will increase by 1
        after completion.

    Methods
    -------
    approach(state, beatbar)
        Register handlers for approaching effect of this target. The hit handler
        and increasing progress bar will be managed automatically.
    hit(state, beatbar, time, strength)
        Deal with the hit event on this target.
    finish(state, beatbar)
        Finish this target.
    """

    is_subject = True

    @dn.datanode
    def listen(self, state, beatbar):
        try:
            while True:
                time, strength = yield
                self.hit(state, beatbar, time, strength)
                if self.is_finished:
                    break
        except GeneratorExit:
            if not self.is_finished:
                self.finish(state, beatbar)
        finally:
            state.add_finished()

    def zindex(self):
        return (0, not self.is_finished, -self.range[0])

    def register(self, state, beatbar):
        self.approach(state, beatbar)
        beatbar.listen(
            self.listen(state, beatbar),
            start=self.range[0],
            duration=self.range[1] - self.range[0],
        )


@dataclasses.dataclass
class OneshotTarget(Target):
    r"""A target to hit with one shot.

    The target will move in a constant speed. The score, range and hit tolerance
    are determined by settings. Only appearances of targets, sound of target and
    rule of hitting target are different.

    Fields
    ------
    beat, length : Fraction
        `beat` is the time this target should be hit, `length` is meaningless.
    speed : float, optional
        The speed of this target (unit: half bar per second). Default speed will
        be determined by context value `speed`, or 1.0 if absence.
    volume : float, optional
        The relative volume of sound of this target (unit: dB). Default volume
        will be determined by context value `volume`, or 0.0 if absence.
    nofeedback : bool, optional
        Whether to make a visual cue for the action of hitting the target.
        Default value will be determined by context value `nofeedback`, or False
        if absence.

    Attributes
    ----------
    sound : datanodes.DataNode or None
        The sound of the auditory cue of this target.
    approach_appearance : tuple of markups.Markup and markups.Markup
        The appearance of approaching target.
    wrong_appearance : tuple of markups.Markup and markups.Markup
        The appearance of wrong-shot target.

    Methods
    -------
    hit(state, beatbar, time, strength)
    """

    has_length = False

    def prepare(self, beatmap, rich, context):
        self.performance_tolerance = beatmap.settings.difficulty.performance_tolerance

        self.time = beatmap.beatpoints.time(self.beat)
        self.perf = None

        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.time + travel_time)
        tol = beatmap.settings.difficulty.failed_tolerance
        self.range = (self.time - tol, self.time + tol)
        self._scores = beatmap.settings.scores.performances_scores
        self.full_score = beatmap.settings.scores.performances_max_score

    def pos(self, time):
        return (self.time - time) * 0.5 * self.speed

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

    def approach(self, state, beatbar):
        if self.sound is not None:
            beatbar.play(self.sound, time=self.time, volume=self.volume)

        beatbar.draw_content(
            self.pos,
            self.appearance,
            zindex=self.zindex,
            start=self.lifespan[0],
            duration=self.lifespan[1] - self.lifespan[0],
        )
        beatbar.reset_sight(start=self.range[0])

    def hit(self, state, beatbar, time, strength, is_correct_key=True):
        perf = Performance.judge(
            self.performance_tolerance, self.time, time, is_correct_key
        )
        state.add_perf(perf)
        self.finish(state, beatbar, perf)

    def finish(self, state, beatbar, perf=None):
        if perf is None:
            perf = Performance.judge(self.performance_tolerance, self.time)
        self.perf = perf
        state.add_full_score(self.full_score)
        state.add_score(self.score)


@dataclasses.dataclass
class Soft(OneshotTarget):
    r"""A target to hit softly.

    Player should hit this target with a volume below a certain threshold,
    otherwise it will be counted as a wrong-shot target.

    Fields
    ------
    beat, length : Fraction
        `beat` is the time this target should be hit, `length` is meaningless.
    speed : float, optional
        The speed of this target (unit: half bar per second). Default speed will
        be determined by context value `speed`, or 1.0 if absence.
    volume : float, optional
        The relative volume of sound of this target (unit: dB). Default volume
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
            self.speed = context.get("speed", 1.0)
        if self.volume is None:
            self.volume = context.get("volume", 0.0)
        if self.nofeedback is None:
            self.nofeedback = context.get("nofeedback", False)

        super().prepare(beatmap, rich, context)

    def hit(self, state, beatbar, time, strength):
        super().hit(state, beatbar, time, strength, strength < self.threshold)


@dataclasses.dataclass
class Loud(OneshotTarget):
    r"""A target to hit loudly.

    Player should hit this target with a volume above a certain threshold,
    otherwise it will be counted as a wrong-shot target.

    Fields
    ------
    beat, length : Fraction
        `beat` is the time this target should be hit, `length` is meaningless.
    speed : float, optional
        The speed of this target (unit: half bar per second). Default speed will
        be determined by context value `speed`, or 1.0 if absence.
    volume : float, optional
        The relative volume of sound of this target (unit: dB). Default volume
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
            self.speed = context.get("speed", 1.0)
        if self.volume is None:
            self.volume = context.get("volume", 0.0)
        if self.nofeedback is None:
            self.nofeedback = context.get("nofeedback", False)

        super().prepare(beatmap, rich, context)

    def hit(self, state, beatbar, time, strength):
        super().hit(state, beatbar, time, strength, strength >= self.threshold)


class IncrGroup:
    def __init__(self, threshold=0.0, total=0):
        self.threshold = threshold
        self.total = total
        self.volume = 0.0
        self.last_beat = None

    def hit(self, strength):
        self.threshold = max(self.threshold, strength)


@dataclasses.dataclass
class Incr(OneshotTarget):
    r"""A target to hit louder and louder.

    Player should hit the target with a volume louder than the previous target,
    otherwise it will be counted as a wrong-shot target.

    Fields
    ------
    beat, length : Fraction
        `beat` is the time this target should be hit, `length` is meaningless.
    group : str, optional
        The group name this target belongs to, or None for automatically
        determine The group by context. The threshold will be dynamically
        changed by hitting the target under the same group.
    speed : float, optional
        The speed of this target (unit: half bar per second). Default speed will
        be determined by context value `speed`, or 1.0 if absence.
    group_volume : float, optional
        The relative group volume of sound of this target (unit: dB). Default
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
            self.speed = context.get("speed", 1.0)
        if self.group_volume is None:
            self.group_volume = context.get("volume", 0.0)
        if self.nofeedback is None:
            self.nofeedback = context.get("nofeedback", False)

        super().prepare(beatmap, rich, context)

        if "<incrs>" not in context:
            context["<incrs>"] = OrderedDict()
        self.groups = context["<incrs>"]

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
        return (
            group_obj.volume
            + numpy.log10(0.2 + 0.8 * (self.count - 1) / group_obj.total) * 20
        )

    def hit(self, state, beatbar, time, strength):
        group_obj = self.groups[self.group]
        threshold = max(0.0, min(1.0, group_obj.threshold + self.incr_threshold))
        super().hit(state, beatbar, time, strength, strength >= threshold)
        group_obj.hit(strength)


@dataclasses.dataclass
class Roll(Target):
    r"""A target to hit multiple times within a certain timespan.

    Fields
    ------
    beat, length : Fraction
        `beat` is the time this target start rolling, `length` is the duration
        of rolling.
    density : int or float or Fraction, optional
        The density of rolling (unit: hit per beat). Default value is
        Fraction(2).
    speed : float, optional
        The speed of this target (unit: half bar per second). Default speed will
        be determined by context value `speed`, or 1.0 if absence.
    volume : float, optional
        The relative volume of sound of this target (unit: dB). Default volume
        will be determined by context value `volume`, or 0.0 if absence.
    nofeedback : bool, optional
        Whether to make a visual cue for the action of hitting the target.
        Default value will be determined by context value `nofeedback`, or False
        if absence.
    """

    density: Optional[Union[int, Fraction, float]] = None
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
            self.speed = context.get("speed", 1.0)
        if self.volume is None:
            self.volume = context.get("volume", 0.0)
        if self.nofeedback is None:
            self.nofeedback = context.get("nofeedback", False)

        self.time = beatmap.beatpoints.time(self.beat)
        self.end = beatmap.beatpoints.time(self.beat + self.length)
        self.roll = 0
        self.number = max(int(self.length * self.density // -1 * -1), 1)
        self.is_finished = False
        self.score = 0

        self.times = [
            beatmap.beatpoints.time(self.beat + i / self.density)
            for i in range(self.number)
        ]
        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.end + travel_time)
        self.range = (self.time - self.tolerance, self.end + self.tolerance)
        self.full_score = self.number * self.rock_score

    def pos_of(self, index):
        return lambda time: (self.times[index] - time) * 0.5 * self.speed

    def appearance_of(self, index):
        return (
            lambda time: self.rock_appearance
            if self.nofeedback or self.roll <= index
            else (mu.Text(""), mu.Text(""))
        )

    def approach(self, state, beatbar):
        for i, time in enumerate(self.times):
            if self.sound is not None:
                beatbar.play(
                    dn.DataNode.wrap(self.sound), time=time, volume=self.volume
                )
            beatbar.draw_content(
                self.pos_of(i),
                self.appearance_of(i),
                zindex=self.zindex,
                start=self.lifespan[0],
                duration=self.lifespan[1] - self.lifespan[0],
            )
        beatbar.reset_sight(start=self.range[0])

    def hit(self, state, beatbar, time, strength):
        self.roll += 1

        if self.roll <= self.number:
            perf = Performance.judge(
                self.performance_tolerance, self.times[self.roll - 1], time, True
            )
            state.add_perf(perf)

            state.add_score(self.rock_score)
            self.score += self.rock_score

        if self.roll == self.number:
            self.finish(state, beatbar)

    def finish(self, state, beatbar):
        self.is_finished = True
        state.add_full_score(self.full_score)

        for time in self.times[self.roll :]:
            perf = Performance.judge(self.performance_tolerance, time)
            state.add_perf(perf)


@dataclasses.dataclass
class Spin(Target):
    r"""A target to accumulate hitting volume within a certain timespan.

    Fields
    ------
    beat, length : Fraction
        `beat` is the time this target start spinning, `length` is the duration
        of spinning.
    density : int or float or Fraction, optional
        The density of spinning (unit: hit per beat). Default value is 2.0.
    speed : float, optional
        The speed of this target (unit: half bar per second). Default speed will
        be determined by context value `speed`, or 1.0 if absence.
    volume : float, optional
        The relative volume of sound of this target (unit: dB). Default volume
        will be determined by context value `volume`, or 0.0 if absence.
    nofeedback : bool, optional
        Whether to make a visual cue for the action of hitting the target.
        Default value will be determined by context value `nofeedback`, or False
        if absence.
    """

    density: Optional[Union[int, Fraction, float]] = None
    speed: Optional[float] = None
    volume: Optional[float] = None
    nofeedback: Optional[bool] = None

    def prepare(self, beatmap, rich, context):
        self.tolerance = beatmap.settings.difficulty.spin_tolerance
        self.disk_appearances = [
            (rich.parse(spin_disk_appearance[0]), rich.parse(spin_disk_appearance[1]))
            for spin_disk_appearance in beatmap.settings.notes.spin_disk_appearances
        ]
        self.finishing_appearance = (
            rich.parse(beatmap.settings.notes.spin_finishing_appearance[0]),
            rich.parse(beatmap.settings.notes.spin_finishing_appearance[1]),
        )
        self.finish_sustain_time = beatmap.settings.notes.spin_finish_sustain_time
        self.sound = beatmap.resources.get(beatmap.settings.notes.spin_disk_sound, None)
        self.full_score = beatmap.settings.scores.spin_score

        if self.speed is None:
            self.speed = context.get("speed", 1.0)
        if self.volume is None:
            self.volume = context.get("volume", 0.0)
        if self.nofeedback is None:
            self.nofeedback = context.get("nofeedback", False)

        self.time = beatmap.beatpoints.time(self.beat)
        self.end = beatmap.beatpoints.time(self.beat + self.length)
        self.charge = 0.0
        if self.density is None:
            self.density = 2.0
        self.capacity = float(self.length * self.density)
        self.is_finished = False
        self.score = 0

        self.times = [
            beatmap.beatpoints.time(self.beat + i / self.density)
            for i in range(int(self.capacity))
        ]
        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.end + travel_time)
        self.range = (self.time - self.tolerance, self.end + self.tolerance)

    def pos(self, time):
        return (
            (max(0.0, self.time - time) + min(0.0, self.end - time)) * 0.5 * self.speed
        )

    def appearance(self, time):
        if self.nofeedback or not self.is_finished:
            return self.disk_appearances[int(self.charge) % len(self.disk_appearances)]
        else:
            return (mu.Text(""), mu.Text(""))

    def approach(self, state, beatbar):
        for time in self.times:
            if self.sound is not None:
                beatbar.play(
                    dn.DataNode.wrap(self.sound), time=time, volume=self.volume
                )

        beatbar.draw_content(
            self.pos,
            self.appearance,
            zindex=self.zindex,
            start=self.lifespan[0],
            duration=self.lifespan[1] - self.lifespan[0],
        )
        beatbar.draw_sight(
            (mu.Text(""), mu.Text("")),
            start=self.range[0],
            duration=self.range[1] - self.range[0],
        )

    def hit(self, state, beatbar, time, strength):
        self.charge = min(self.charge + min(1.0, strength), self.capacity)

        current_score = int(self.full_score * self.charge / self.capacity)
        state.add_score(current_score - self.score)
        self.score = current_score

        if self.charge == self.capacity:
            self.finish(state, beatbar)

    def finish(self, state, beatbar):
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
            beatbar.draw_sight(appearance, duration=self.finish_sustain_time)


@dataclasses.dataclass
class Hit:
    offset: Fraction
    beat: Fraction
    is_soft: bool


@dataclasses.dataclass
class FreeStyle(Target):
    r"""A target to play free style within a certain timespan.

    Fields
    ------
    beat, length : Fraction
        `beat` is the time this target start spinning, `length` is the duration
        of free style.
    tempo : int, optional
        The tempo of free style (unit: hits per beat). Default value is 1.
    subtempo : int, optional
        The subtempo of free style, which is the subdivision of tempo,
        indicating the minimal detectable hits. Default value is 4.
    speed : float, optional
        The speed of this target (unit: half bar per second). Default speed will
        be determined by context value `speed`, or 1.0 if absence.
    volume : float, optional
        The relative volume of sound of this target (unit: dB). Default volume
        will be determined by context value `volume`, or 0.0 if absence.
    nofeedback : bool, optional
        Whether to make a visual cue for the action of hitting the target.
        Default value will be determined by context value `nofeedback`, or False
        if absence.
    """
    tempo: Optional[int] = None
    subtempo: Optional[int] = None
    speed: Optional[float] = None
    volume: Optional[float] = None
    nofeedback: Optional[bool] = None

    def prepare(self, beatmap, rich, context):
        self.tolerance = beatmap.settings.difficulty.failed_tolerance
        self.performance_tolerance = beatmap.settings.difficulty.performance_tolerance
        self.soft_threshold = beatmap.settings.difficulty.soft_threshold
        self.freestyle_appearance = (
            rich.parse(beatmap.settings.notes.freestyle_appearances[0]),
            rich.parse(beatmap.settings.notes.freestyle_appearances[1]),
        )
        self.soft_approach_appearance = (
            rich.parse(beatmap.settings.notes.soft_approach_appearance[0]),
            rich.parse(beatmap.settings.notes.soft_approach_appearance[1]),
        )
        self.loud_approach_appearance = (
            rich.parse(beatmap.settings.notes.loud_approach_appearance[0]),
            rich.parse(beatmap.settings.notes.loud_approach_appearance[1]),
        )
        self.sound = beatmap.resources.get(beatmap.settings.notes.freestyle_sound, None)
        self.full_score = beatmap.settings.scores.freestyle_score

        if self.tempo is None:
            self.tempo = context.get("tempo", 1)
        if self.subtempo is None:
            self.subtempo = context.get("subtempo", 4)
        if self.speed is None:
            self.speed = context.get("speed", 1.0)
        if self.volume is None:
            self.volume = context.get("volume", 0.0)
        if self.nofeedback is None:
            self.nofeedback = context.get("nofeedback", False)

        self.is_finished = False
        self.score = 0

        self.time = beatmap.beatpoints.time(self.beat)
        self.end = beatmap.beatpoints.time(self.beat + self.length)
        self.beats_times = [
            (
                Fraction(i, 1) / (self.tempo * self.subtempo),
                beatmap.beatpoints.time(self.beat + i / (self.tempo * self.subtempo)),
            )
            for i in range(int(self.length * self.tempo * self.subtempo) + 1)
        ]
        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.end + travel_time)
        self.range = (self.time - self.tolerance, self.end + self.tolerance)

        self.hits = []

    def approach(self, state, beatbar):
        for _, time in self.beats_times[:: self.subtempo]:
            if self.sound is not None:
                beatbar.play(
                    dn.DataNode.wrap(self.sound), time=time, volume=self.volume
                )

            beatbar.draw_content(
                self.pos_of(time),
                lambda time: self.freestyle_appearance,
                zindex=(-3,),
                start=self.lifespan[0],
                duration=self.lifespan[1] - self.lifespan[0],
            )

    def pos_of(self, target_time):
        return lambda time: (target_time - time) * 0.5 * self.speed

    def appearance_of(self, is_soft):
        if self.nofeedback:
            return lambda time: (mu.Text(""), mu.Text(""))
        elif is_soft:
            return lambda time: self.soft_approach_appearance
        else:
            return lambda time: self.loud_approach_appearance

    def zindex(self):
        return (-1, -self.range[0])

    def hit(self, state, beatbar, time, strength):
        target_beat, target_time = min(self.beats_times, key=lambda t: abs(t[1] - time))
        is_soft = strength < self.soft_threshold

        perf = Performance.judge(self.performance_tolerance, target_time, time, True)
        state.add_perf(perf)
        self.hits.append(Hit(self.beat, target_beat, is_soft))

        beatbar.draw_content(
            self.pos_of(target_time),
            self.appearance_of(is_soft),
            zindex=self.zindex,
            start=self.lifespan[0],
            duration=self.lifespan[1] - self.lifespan[0],
        )

    def finish(self, state, beatbar):
        self.is_finished = True
        state.add_full_score(self.full_score)

        # TODO: judge score
        self.score = self.full_score
        state.add_score(self.score)


# performance
class PerformanceGrade(Enum):
    MISS = (None, None)
    PERFECT = (0, False)
    LATE_GOOD = (+1, False)
    EARLY_GOOD = (-1, False)
    LATE_BAD = (+2, False)
    EARLY_BAD = (-2, False)
    LATE_FAILED = (+3, False)
    EARLY_FAILED = (-3, False)
    PERFECT_WRONG = (0, True)
    LATE_GOOD_WRONG = (+1, True)
    EARLY_GOOD_WRONG = (-1, True)
    LATE_BAD_WRONG = (+2, True)
    EARLY_BAD_WRONG = (-2, True)
    LATE_FAILED_WRONG = (+3, True)
    EARLY_FAILED_WRONG = (-3, True)

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
            return Performance(PerformanceGrade.MISS, time, None)

        is_wrong = not is_correct_key
        err = hit_time - time
        shift = next((i for i in range(3) if abs(err) < tol * (2 * i + 1)), 3)
        if err < 0:
            shift = -shift

        for grade in PerformanceGrade:
            if grade.shift == shift and grade.is_wrong == is_wrong:
                return Performance(grade, time, err)

    @property
    def shift(self):
        return self.grade.shift

    @property
    def is_wrong(self):
        return self.grade.is_wrong

    @property
    def is_miss(self):
        return self.grade == PerformanceGrade.MISS

    descriptions = {
        PerformanceGrade.MISS: "Miss",
        PerformanceGrade.PERFECT: "Perfect",
        PerformanceGrade.LATE_GOOD: "Late Good",
        PerformanceGrade.EARLY_GOOD: "Early Good",
        PerformanceGrade.LATE_BAD: "Late Bad",
        PerformanceGrade.EARLY_BAD: "Early Bad",
        PerformanceGrade.LATE_FAILED: "Late Failed",
        PerformanceGrade.EARLY_FAILED: "Early Failed",
        PerformanceGrade.PERFECT_WRONG: "Perfect but Wrong Key",
        PerformanceGrade.LATE_GOOD_WRONG: "Late Good but Wrong Key",
        PerformanceGrade.EARLY_GOOD_WRONG: "Early Good but Wrong Key",
        PerformanceGrade.LATE_BAD_WRONG: "Late Bad but Wrong Key",
        PerformanceGrade.EARLY_BAD_WRONG: "Early Bad but Wrong Key",
        PerformanceGrade.LATE_FAILED_WRONG: "Late Failed but Wrong Key",
        PerformanceGrade.EARLY_FAILED_WRONG: "Early Failed but Wrong Key",
    }

    @property
    def description(self):
        return self.descriptions[self.grade]


@dataclasses.dataclass
class BeatmapScore:
    total_subjects: int = 0
    finished_subjects: int = 0
    full_score: int = 0
    score: int = 0
    perfs: List[Performance] = dataclasses.field(default_factory=lambda: [])
    time: float = 0.0

    def set_total_subjects(self, total_subjects):
        self.total_subjects = total_subjects

    def add_score(self, score):
        self.score += score

    def add_full_score(self, full_score):
        self.full_score += full_score

    def add_finished(self, finished=1):
        self.finished_subjects += finished

    def add_perf(self, perf):
        self.perfs.append(perf)


@dn.datanode
def observe(stack):
    last = 0
    yield
    while True:
        observed = []
        while len(stack) > last:
            observed.append(stack[last])
            last += 1
        yield observed


# widgets
class BeatbarWidgetFactory:
    spectrum = widgets.SpectrumWidgetSettings
    volume_indicator = widgets.VolumeIndicatorWidgetSettings
    knock_meter = widgets.KnockMeterWidgetSettings
    score = widgets.ScoreWidgetSettings
    progress = widgets.ProgressWidgetSettings
    accuracy_meter = widgets.AccuracyMeterWidgetSettings
    monitor = widgets.MonitorWidgetSettings
    sight = beatbars.SightWidgetSettings

    def __init__(self, state, rich, mixer, detector, renderer):
        self.state = state
        self.rich = rich
        self.mixer = mixer
        self.detector = detector
        self.renderer = renderer

    def create(self, widget_settings):
        with providers.set(self.rich, self.mixer, self.detector, self.renderer):

            if isinstance(widget_settings, BeatbarWidgetFactory.spectrum):
                return widgets.SpectrumWidget(widget_settings).load()
            elif isinstance(widget_settings, BeatbarWidgetFactory.volume_indicator):
                return widgets.VolumeIndicatorWidget(widget_settings).load()
            elif isinstance(widget_settings, BeatbarWidgetFactory.knock_meter):
                return widgets.KnockMeterWidget(widget_settings).load()
            elif isinstance(widget_settings, BeatbarWidgetFactory.accuracy_meter):
                accuracy_getter = dn.pipe(
                    observe(self.state.perfs),
                    lambda perfs: [perf.err for perf in perfs],
                )
                return widgets.AccuracyMeterWidget(
                    accuracy_getter, widget_settings
                ).load()
            elif isinstance(widget_settings, BeatbarWidgetFactory.monitor):
                return widgets.MonitorWidget(widget_settings).load()
            elif isinstance(widget_settings, BeatbarWidgetFactory.score):
                score_getter = lambda _: (self.state.score, self.state.full_score)
                return widgets.ScoreWidget(score_getter, widget_settings).load()
            elif isinstance(widget_settings, BeatbarWidgetFactory.progress):
                progress_getter = lambda _: (
                    self.state.finished_subjects / self.state.total_subjects
                    if self.state.total_subjects > 0
                    else 1.0
                )
                time_getter = lambda _: self.state.time
                return widgets.ProgressWidget(
                    progress_getter, time_getter, widget_settings
                ).load()
            elif isinstance(widget_settings, BeatbarWidgetFactory.sight):
                grade_getter = dn.pipe(
                    observe(self.state.perfs),
                    lambda perfs: [
                        perf.grade.shift
                        for perf in perfs
                        if perf.grade.shift is not None
                    ],
                )
                return beatbars.SightWidget(grade_getter, widget_settings).load()
            else:
                raise TypeError


BeatbarIconWidgetSettings = Union[
    widgets.SpectrumWidgetSettings,
    widgets.VolumeIndicatorWidgetSettings,
    widgets.KnockMeterWidgetSettings,
    widgets.ScoreWidgetSettings,
    widgets.ProgressWidgetSettings,
    widgets.AccuracyMeterWidgetSettings,
    widgets.MonitorWidgetSettings,
]
BeatbarHeaderWidgetSettings = BeatbarIconWidgetSettings
BeatbarFooterWidgetSettings = BeatbarIconWidgetSettings


class BeatbarWidgetSettings(cfg.Configurable):
    r"""
    Fields
    ------
    icon : BeatbarIconWidgetSettings
        The widget on the icon.
    header : BeatbarHeaderWidgetSettings
        The widget on the header.
    footer : BeatbarFooterWidgetSettings
        The widget on the footer.
    """
    icon: BeatbarIconWidgetSettings = BeatbarWidgetFactory.spectrum()
    header: BeatbarHeaderWidgetSettings = BeatbarWidgetFactory.score()
    footer: BeatbarFooterWidgetSettings = BeatbarWidgetFactory.progress()


class BeatbarLayoutSettings(cfg.Configurable):
    """
    Fields
    ------
    icon_width : int
        [rich]The width of icon.

        [color=bright_magenta] ⣠⣴⣤⣿⣤⣦ [/][color=bright_blue][[00000/00400]][/]       [color=bright_cyan]□ [/] [color=bright_magenta]⛶ [/]    [color=bright_cyan]□ [/]                [color=bright_blue]■ [/]  [color=bright_blue][[11.3%|00:09]][/]
        ^^^^^^^^
          here

    header_width : int
        [rich]The width of header.

        [color=bright_magenta] ⣠⣴⣤⣿⣤⣦ [/][color=bright_blue][[00000/00400]][/]       [color=bright_cyan]□ [/] [color=bright_magenta]⛶ [/]    [color=bright_cyan]□ [/]                [color=bright_blue]■ [/]  [color=bright_blue][[11.3%|00:09]][/]
                ^^^^^^^^^^^^^
                    here

    footer_width : int
        [rich]The width of footer.

        [color=bright_magenta] ⣠⣴⣤⣿⣤⣦ [/][color=bright_blue][[00000/00400]][/]       [color=bright_cyan]□ [/] [color=bright_magenta]⛶ [/]    [color=bright_cyan]□ [/]                [color=bright_blue]■ [/]  [color=bright_blue][[11.3%|00:09]][/]
                                                                   ^^^^^^^^^^^^^
                                                                        here

    """

    icon_width: int = 8
    header_width: int = 13
    footer_width: int = 13


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

        perfect_tolerance = property(lambda self: self.performance_tolerance * 1)
        good_tolerance = property(lambda self: self.performance_tolerance * 3)
        bad_tolerance = property(lambda self: self.performance_tolerance * 5)
        failed_tolerance = property(lambda self: self.performance_tolerance * 7)

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
            The score of spin note.
        freestyle_score : int
            The score of freestyle note.
        """
        performances_scores: Dict[PerformanceGrade, int] = {
            PerformanceGrade.MISS: 0,
            PerformanceGrade.LATE_FAILED: 0,
            PerformanceGrade.LATE_BAD: 2,
            PerformanceGrade.LATE_GOOD: 8,
            PerformanceGrade.PERFECT: 16,
            PerformanceGrade.EARLY_GOOD: 8,
            PerformanceGrade.EARLY_BAD: 2,
            PerformanceGrade.EARLY_FAILED: 0,
            PerformanceGrade.LATE_FAILED_WRONG: 0,
            PerformanceGrade.LATE_BAD_WRONG: 1,
            PerformanceGrade.LATE_GOOD_WRONG: 4,
            PerformanceGrade.PERFECT_WRONG: 8,
            PerformanceGrade.EARLY_GOOD_WRONG: 4,
            PerformanceGrade.EARLY_BAD_WRONG: 1,
            PerformanceGrade.EARLY_FAILED_WRONG: 0,
        }

        performances_max_score = property(
            lambda self: max(self.performances_scores.values())
        )

        roll_rock_score: int = 2
        spin_score: int = 16
        freestyle_score: int = 32

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

        freestyle_appearances : tuple of str and str
            The freestyle appearance of freestyle note.
        freestyle_sound : str
            The name of sound of freestyle note.

        event_leadin_time : float
            The minimum time of silence before and after the gameplay.
        """
        soft_approach_appearance: Tuple[str, str] = (
            "[color=bright_cyan]□[/]",
            "[color=bright_cyan]□[/]",
        )
        soft_wrong_appearance: Tuple[str, str] = (
            "[color=bright_cyan]⬚[/]",
            "[color=bright_cyan]⬚[/]",
        )
        soft_sound: str = "soft"
        loud_approach_appearance: Tuple[str, str] = (
            "[color=bright_blue]■[/]",
            "[color=bright_blue]■[/]",
        )
        loud_wrong_appearance: Tuple[str, str] = (
            "[color=bright_blue]⬚[/]",
            "[color=bright_blue]⬚[/]",
        )
        loud_sound: str = "loud"
        incr_approach_appearance: Tuple[str, str] = (
            "[color=bright_blue]⬒[/]",
            "[color=bright_blue]⬒[/]",
        )
        incr_wrong_appearance: Tuple[str, str] = (
            "[color=bright_blue]⬚[/]",
            "[color=bright_blue]⬚[/]",
        )
        incr_sound: str = "incr"
        roll_rock_appearance: Tuple[str, str] = (
            "[color=bright_cyan]◎[/]",
            "[color=bright_cyan]◎[/]",
        )
        roll_rock_sound: str = "rock"
        spin_disk_appearances: List[Tuple[str, str]] = [
            ("[color=bright_blue]◴[/]", "[color=bright_blue]◴[/]"),
            ("[color=bright_blue]◵[/]", "[color=bright_blue]◵[/]"),
            ("[color=bright_blue]◶[/]", "[color=bright_blue]◶[/]"),
            ("[color=bright_blue]◷[/]", "[color=bright_blue]◷[/]"),
        ]
        spin_finishing_appearance: Tuple[str, str] = (
            "[color=bright_blue]☺[/]",
            "[color=bright_blue]☺[/]",
        )
        spin_finish_sustain_time: float = 0.1
        spin_disk_sound: str = "disk"
        freestyle_appearances: Tuple[str, str] = (
            "[color=bright_blue]|[/]",
            "[color=bright_blue]|[/]",
        )
        freestyle_sound: str = "freestyle"

        event_leadin_time: float = 1.0

    resources: Dict[str, Union[Path, dn.Waveform]] = {
        "soft": dn.Waveform("0.5*2**(-t/0.01)*{sine:t*830.61}#tspan:0,0.06"),
        "loud": dn.Waveform("1.0*2**(-t/0.01)*{sine:t*1661.2}#tspan:0,0.06"),
        "incr": dn.Waveform("1.0*2**(-t/0.01)*{sine:t*1661.2}#tspan:0,0.06"),
        "rock": dn.Waveform("0.5*2**(-t/0.005)*{sine:t*1661.2}#tspan:0,0.03"),
        "disk": dn.Waveform("1.0*2**(-t/0.005)*{sine:t*1661.2}#tspan:0,0.03"),
        "freestyle": dn.Waveform("0.5*2**(-t/0.01)*{sine:t*830.61}#tspan:0,0.06"),
    }


class GameplaySettings(cfg.Configurable):
    debug_monitor: bool = False

    @cfg.subconfig
    class playfield(cfg.Configurable):
        layout = cfg.subconfig(BeatbarLayoutSettings)
        sight = cfg.subconfig(beatbars.SightWidgetSettings)
        widgets = cfg.subconfig(BeatbarWidgetSettings)

    @cfg.subconfig
    class controls(cfg.Configurable):
        r"""
        Fields
        ------
        prepare_time : float
            The time between preparing the event and the lifespan of the event.
        tickrate : float
            The event updating rate.

        control_delay : float
            The delay of control action.
        pause_key : str
            The key to pause/resume the game.
        stop_key : str
            The key to stop the game.
        skip_time : float
        skip_key : str
            The key to skip time.

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
        prepare_time: float = 0.1
        tickrate: float = 60.0

        control_delay = 0.1
        pause_key = "Space"
        stop_key: str = "Esc"
        skip_time: float = 0.5
        skip_key = "Tab"

        display_delay_adjust_keys: Tuple[str, str] = ("Ctrl_Left", "Ctrl_Right")
        knock_delay_adjust_keys: Tuple[str, str] = ("Left", "Right")
        knock_energy_adjust_keys: Tuple[str, str] = ("Down", "Up")
        display_delay_adjust_step: float = 0.001
        knock_delay_adjust_step: float = 0.001
        knock_energy_adjust_step: float = 0.0001


@dataclasses.dataclass
class BeatmapAudio:
    path: Optional[str] = None
    volume: float = 0.0
    preview: float = 0.0


@dataclasses.dataclass
class BeatTrack:
    notes: List[beatpatterns.Note]

    @classmethod
    def parse(cls, patterns_str, ret_width=False, notations=None):
        notes, width = beatpatterns.parse_notes(patterns_str)

        track = cls([note for note in notes])

        if ret_width:
            return track, width
        else:
            return track

    def events(self, notations):
        for note in self.notes:
            yield self.to_event(note, notations)

    @staticmethod
    def to_event(note, notations):
        if note.symbol == "#":
            return Comment(
                note.beat,
                note.length,
                *note.arguments.ps,
                **note.arguments.kw,
            )

        if note.symbol == ":":
            return UpdateContext.make(
                note.beat,
                note.length,
                *note.arguments.ps,
                **note.arguments.kw,
            )

        if note.symbol not in notations:
            raise ValueError(f"unknown symbol: {note.symbol}")
        return notations[note.symbol](
            note.beat,
            note.length,
            *note.arguments.ps,
            **note.arguments.kw,
        )

    @staticmethod
    def from_event(event, notations):
        fields = dataclasses.fields(event)
        kwargs = {
            field.name: getattr(event, field.name)
            for field in fields
            if getattr(event, field.name) is not None
        }
        assert "beat" in kwargs and "length" in kwargs
        del kwargs["beat"]
        del kwargs["length"]
        arguments = beatpatterns.Arguments([], kwargs)

        if isinstance(event, Comment):
            return beatpatterns.Note(
                "#",
                event.beat,
                Fraction(0, 1),
                arguments,
            )

        if isinstance(event, UpdateContext):
            return beatpatterns.Note(
                ":",
                event.beat,
                Fraction(0, 1),
                beatpatterns.Arguments([], arguments.kw["update"]),
            )

        symbol = next(
            (symbol for symbol in notations if isinstance(event, notations[symbol])),
            None,
        )
        if symbol is None:
            raise ValueError(f"unknown event: {type(event)}")
        return beatpatterns.Note(
            symbol,
            event.beat,
            event.length if event.has_length else Fraction(0, 1),
            arguments,
        )

    def as_patterns_str(self, notations):
        lengthless = [
            symbol for symbol in notations if not notations[symbol].has_length
        ]
        notes = [self.from_event(event, notations) for event in self.events(notations)]
        return beatpatterns.format_notes(notes, lengthless_symbols=lengthless)


def bisect(list, elem, key):
    length = len(list)

    if length == 0:
        return length

    i, j = 0, length - 1

    if elem < key(list[i]):
        return 0
    if not (elem < key(list[j])):
        return length

    # assert j - i > 0

    while j - i >= 2:
        k = (i + j) // 2
        if elem < key(list[k]):
            j = k
        else:
            i = k
    else:
        return j


@dataclasses.dataclass
class BeatPoint:
    beat: Fraction = Fraction(0, 1)
    length: Fraction = Fraction(1, 1)
    time: float = 0.0
    tempo: Optional[float] = None


@dataclasses.dataclass
class BeatPoints:
    points: List[BeatPoint]

    def is_valid(self):
        length = len(self.points)
        if length == 0:
            return False
        if length == 1 and self.points[0].tempo is None:
            return False
        return True

    def time(self, beat):
        r"""Convert beat to time (in seconds).

        Parameters
        ----------
        beat : int or Fraction or float

        Returns
        -------
        time : float
        """
        length = len(self.points)

        if length == 0:
            raise ValueError("No beat point")

        if length == 1:
            beatpoint = self.points[0]
            if beatpoint.tempo is None:
                raise ValueError("No tempo")
            return beatpoint.time + (beat - beatpoint.beat) * 60.0 / beatpoint.tempo

        i = bisect(self.points, beat, key=lambda beatpoint: beatpoint.beat)

        if i == 0:
            left, right = self.points[:2]
            if left.tempo is not None:
                return left.time + (beat - left.beat) * 60.0 / left.tempo

        elif i == length:
            left, right = self.points[-2:]
            if right.tempo is not None:
                return right.time + (beat - right.beat) * 60.0 / right.tempo

        else:
            left, right = self.points[i - 1 : i + 1]

        s = (beat - left.beat) / (right.beat - left.beat)
        return left.time + s * (right.time - left.time)

    def beat(self, time):
        r"""Convert time (in seconds) to beat.

        Parameters
        ----------
        time : float

        Returns
        -------
        beat : float
        """
        length = len(self.points)

        if length == 0:
            raise ValueError("No beat point")

        if length == 1:
            beatpoint = self.points[0]
            if beatpoint.tempo is None:
                raise ValueError("No tempo")
            return (
                float(beatpoint.beat) + (time - beatpoint.time) * beatpoint.tempo / 60.0
            )

        i = bisect(self.points, time, key=lambda beatpoint: beatpoint.time)

        if i == 0:
            left, right = self.points[:2]
            if left.tempo is not None:
                return float(left.beat) + (time - left.time) * left.tempo / 60.0

        elif i == length:
            left, right = self.points[-2:]
            if right.tempo is not None:
                return float(right.beat) + (time - right.time) * right.tempo / 60.0

        else:
            left, right = self.points[i - 1 : i + 1]

        s = (time - left.time) / (right.time - left.time)
        return float(left.beat) + s * float(right.beat - left.beat)

    def tempo(self, time):
        length = len(self.points)

        if length == 0:
            raise ValueError("No beat point")

        if length == 1:
            beatpoint = self.points[0]
            if beatpoint.tempo is None:
                raise ValueError("No tempo")
            return float(beatpoint.tempo)

        i = bisect(self.points, time, key=lambda beatpoint: beatpoint.time)

        if i == 0:
            left, right = self.points[:2]
            if left.tempo is not None:
                return left.tempo

        elif i == length:
            left, right = self.points[-2:]
            if right.tempo is not None:
                return right.tempo

        else:
            left, right = self.points[i - 1 : i + 1]

        return float(right.beat - left.beat) / (right.time - left.time) * 60.0

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
        return self.time(beat + length) - self.time(beat)

    @staticmethod
    def to_beatpoint(note):
        if note.symbol != "!":
            raise ValueError(f"unknown symbol: {note.symbol}")
        return BeatPoint(
            note.beat,
            note.length,
            *note.arguments.ps,
            **note.arguments.kw,
        )

    @classmethod
    def fixed(cls, offset, tempo):
        return cls([BeatPoint(time=offset, tempo=tempo)])

    @classmethod
    def parse(cls, patterns_str, ret_width=False):
        notes, width = beatpatterns.parse_notes(patterns_str)

        beatpoints = cls([cls.to_beatpoint(note) for note in notes])

        if ret_width:
            return beatpoints, width
        else:
            return beatpoints

    @dn.datanode
    def tempo_node(self):
        if not self.is_valid():
            while True:
                yield None

        time = yield
        beat = self.beat(time)
        tempo = self.tempo(time)
        res = beat, tempo

        for beatpoint in self.points:
            while time < beatpoint.time:
                time = yield res
                res = None
            beat = self.beat(time)
            tempo = self.tempo(time)
            res = beat, tempo

        while True:
            yield None


@dataclasses.dataclass
class Shift:
    beat: Fraction = Fraction(0, 1)
    length: Fraction = Fraction(1, 1)
    shift: Optional[float] = None


@dataclasses.dataclass
class Flip:
    beat: Fraction = Fraction(0, 1)
    length: Fraction = Fraction(1, 1)
    flip: Optional[bool] = None


@dataclasses.dataclass
class BeatState:
    points: List[Union[Shift, Flip]]

    DEFAULT_SHIFT = 0.1
    DEFAULT_FLIP = False

    def get_init_shift(self):
        shift = next(
            (
                statepoint.shift
                for statepoint in self.points
                if isinstance(statepoint, Shift)
            ),
            None,
        )
        if shift is None:
            shift = self.DEFAULT_SHIFT
        return shift

    def get_init_flip(self):
        flip = next(
            (
                statepoint.flip
                for statepoint in self.points
                if isinstance(statepoint, Flip)
            ),
            None,
        )
        if flip is None:
            flip = self.DEFAULT_FLIP
        return flip

    @dn.datanode
    def shift_node(self, beatpoints):
        prev_shift = self.get_init_shift()
        time = yield
        prev_time = time

        for statepoint in self.points:
            if isinstance(statepoint, Shift):
                next_time = beatpoints.time(statepoint.beat)
                speed = (
                    (statepoint.shift - prev_shift) / (next_time - prev_time)
                    if statepoint.shift is not None and next_time != prev_time
                    else 0
                )
                while time < next_time:
                    time = yield prev_shift + speed * (time - prev_time)
                if statepoint.shift is not None:
                    prev_shift = statepoint.shift
                prev_time = next_time

        while True:
            yield prev_shift

    @dn.datanode
    def flip_node(self, beatpoints):
        flip = self.get_init_flip()
        time = yield

        for statepoint in self.points:
            if isinstance(statepoint, Flip):
                next_time = beatpoints.time(statepoint.beat)
                while time < next_time:
                    time = yield flip
                flip = statepoint.flip if statepoint.flip is not None else not flip

        while True:
            yield flip

    @staticmethod
    def to_statepoint(note):
        if note.symbol not in ("Shift", "Flip"):
            raise ValueError(f"unknown symbol: {note.symbol}")
        statepoint_cls = Shift if note.symbol == "Shift" else Flip
        return statepoint_cls(
            note.beat,
            note.length,
            *note.arguments.ps,
            **note.arguments.kw,
        )

    @classmethod
    def parse(cls, patterns_str):
        notes, width = beatpatterns.parse_notes(patterns_str)
        return cls([cls.to_statepoint(note) for note in notes])

    @classmethod
    def fixed(cls, shift=None, flip=None):
        points = []
        if shift is not None:
            points.append(Shift(shift=shift))
        if flip is not None:
            points.append(Flip(flip=flip))
        return cls(points)


DEFAULT_NOTATIONS = {
    "x": Soft,
    "o": Loud,
    "<": Incr,
    "%": Roll,
    "@": Spin,
    "Text": Text,
    "FreeStyle": FreeStyle,
}


@dataclasses.dataclass
class Beatmap:
    path: Optional[str] = None
    info: str = ""
    audio: BeatmapAudio = dataclasses.field(default_factory=BeatmapAudio)
    beatpoints: BeatPoints = dataclasses.field(default_factory=lambda: BeatPoints([]))
    beatstate: BeatState = dataclasses.field(default_factory=lambda: BeatState([]))
    notations: Dict[str, type[Event]] = dataclasses.field(
        default_factory=lambda: dict(**DEFAULT_NOTATIONS)
    )
    tracks: Dict[str, BeatTrack] = dataclasses.field(default_factory=dict)
    settings: BeatmapSettings = dataclasses.field(default_factory=BeatmapSettings)

    @dn.datanode
    def play(self, start_time, gameplay_settings=None):
        self.audionode = None
        self.resources = {}
        self.total_subjects = 0
        self.start_time = 0.0
        self.end_time = float("inf")
        self.events = []

        gameplay_settings = gameplay_settings or GameplaySettings()

        tickrate = gameplay_settings.controls.tickrate
        prepare_time = gameplay_settings.controls.prepare_time

        rich = self.load_rich()

        # prepare
        try:
            yield from dn.create_task(
                dn.chain(self.load_resources(), self.prepare_events(rich)),
            ).join()
        except aud.IOCancelled:
            return

        if start_time is not None:
            self.start_time = start_time

        score = BeatmapScore()
        score.set_total_subjects(self.total_subjects)

        # load engines
        engine_task, engines = self.load_engine(self.start_time)
        mixer, detector, renderer, controller, clock = engines

        Beatmap.register_clock_controller(
            mixer,
            detector,
            renderer,
            controller,
            clock,
            gameplay_settings.controls,
        )

        # build playfield
        widget_factory = BeatbarWidgetFactory(score, rich, mixer, detector, renderer)

        icon = widget_factory.create(gameplay_settings.playfield.widgets.icon)
        header = widget_factory.create(gameplay_settings.playfield.widgets.header)
        footer = widget_factory.create(gameplay_settings.playfield.widgets.footer)
        sight = widget_factory.create(gameplay_settings.playfield.sight)

        beatbar = beatbars.Beatbar(
            mixer,
            detector,
            renderer,
            controller,
            sight,
            self.beatstate.get_init_shift(),
            self.beatstate.get_init_flip(),
        )

        beatbar_node = beatbar.load()

        @dn.datanode
        def state_node(beatbar):
            with self.beatstate.shift_node(self.beatpoints) as shift_node:
                with self.beatstate.flip_node(self.beatpoints) as flip_node:
                    while True:
                        time, ran = yield
                        beatbar.bar_shift = shift_node.send(time)
                        beatbar.bar_flip = flip_node.send(time)

        beatbar.on_before_render(state_node(beatbar))

        # layout
        icon_width = gameplay_settings.playfield.layout.icon_width
        header_width = gameplay_settings.playfield.layout.header_width
        footer_width = gameplay_settings.playfield.layout.footer_width

        [
            icon_mask,
            header_mask,
            content_mask,
            footer_mask,
        ] = widgets.layout([icon_width, header_width, -1, footer_width])

        renderer.add_texts(beatbar_node, xmask=content_mask, zindex=(0,))
        renderer.add_texts(icon, xmask=icon_mask, zindex=(1,))
        renderer.add_texts(header, xmask=header_mask, zindex=(2,))
        renderer.add_texts(footer, xmask=footer_mask, zindex=(3,))

        # play music
        if self.audionode is not None:
            mixer.play(self.audionode, time=0.0, zindex=(-3,))

        # game loop
        event_node = self.update_events(
            self.events,
            score,
            beatbar,
            self.end_time,
            prepare_time,
        )

        with clock.tick(id(self), 0.0) as tick_node:
            event_node = dn.pipe(dn.count(0.0, 1 / tickrate), tick_node, event_node)
            event_task = dn.interval(event_node, dt=1 / tickrate)
            with self.play_mode(mixer, detector, renderer):
                yield from dn.pipe(engine_task, event_task).join()

        self.update_devices_settings(mixer, detector, renderer)

        return score

    @staticmethod
    def register_clock_controller(
        mixer,
        detector,
        renderer,
        controller,
        clock,
        controls_settings,
    ):
        # stop
        stop_key = controls_settings.stop_key
        controller.add_handler(lambda _: clock.stop(), stop_key)

        # display delay
        display_delay_adjust_step = controls_settings.display_delay_adjust_step
        display_delay_adjust_keys = controls_settings.display_delay_adjust_keys

        def incr_display_delay(_):
            renderer.delay(display_delay_adjust_step)
            renderer.add_log(mu.Text(f"display_delay += {display_delay_adjust_step}\n"))

        def decr_display_delay(_):
            renderer.delay(-display_delay_adjust_step)
            renderer.add_log(mu.Text(f"display_delay -= {display_delay_adjust_step}\n"))

        controller.add_handler(incr_display_delay, display_delay_adjust_keys[0])
        controller.add_handler(decr_display_delay, display_delay_adjust_keys[1])

        # knock delay
        knock_delay_adjust_step = controls_settings.knock_delay_adjust_step
        knock_delay_adjust_keys = controls_settings.knock_delay_adjust_keys

        def incr_knock_delay(_):
            detector.delay(knock_delay_adjust_step)
            renderer.add_log(mu.Text(f"knock_delay += {knock_delay_adjust_step}\n"))

        def decr_knock_delay(_):
            detector.delay(-knock_delay_adjust_step)
            renderer.add_log(mu.Text(f"knock_delay -= {knock_delay_adjust_step}\n"))

        controller.add_handler(incr_knock_delay, knock_delay_adjust_keys[0])
        controller.add_handler(decr_knock_delay, knock_delay_adjust_keys[1])

        # knock strength
        knock_energy_adjust_step = controls_settings.knock_energy_adjust_step
        knock_energy_adjust_keys = controls_settings.knock_energy_adjust_keys

        def incr_knock_energy(_):
            detector.increase(knock_energy_adjust_step)
            renderer.add_log(mu.Text(f"knock_energy += {knock_energy_adjust_step}\n"))

        def decr_knock_energy(_):
            detector.increase(-knock_energy_adjust_step)
            renderer.add_log(mu.Text(f"knock_energy -= {knock_energy_adjust_step}\n"))

        controller.add_handler(incr_knock_energy, knock_energy_adjust_keys[0])
        controller.add_handler(decr_knock_energy, knock_energy_adjust_keys[1])

        # pause/resume/skip
        control_delay = controls_settings.control_delay
        pause_key = controls_settings.pause_key
        skip_time = controls_settings.skip_time
        skip_key = controls_settings.skip_key

        @dn.datanode
        def pause_node():
            paused = False
            time_node = dn.time()
            with time_node:
                while True:
                    yield
                    time = time_node.send(None)
                    if paused:
                        clock.speed(time + control_delay, 1.0)
                        paused = False
                    else:
                        clock.speed(time + control_delay, 0.0)
                        paused = True
                        renderer.add_log(mu.Text(f"pause\n"))

        controller.add_handler(pause_node(), pause_key)

        @dn.datanode
        def skip_node():
            time_node = dn.time()
            with time_node:
                while True:
                    yield
                    time = time_node.send(None)
                    clock.skip(time + control_delay, skip_time)
                    renderer.add_log(mu.Text(f"skip\n"))

        controller.add_handler(skip_node(), skip_key)

    @staticmethod
    def load_rich():
        from ..utils import providers
        from ..main.devices import DeviceManager

        return providers.get(DeviceManager).load_rich()

    @staticmethod
    def load_engine(start_time):
        from ..utils import providers
        from ..main.devices import DeviceManager
        from ..main.profiles import ProfileManager

        device_manager = providers.get(DeviceManager)
        profile_manager = providers.get(ProfileManager)
        debug_monitor = profile_manager.current.gameplay.debug_monitor

        clock = clocks.Clock(start_time, 1.0)

        engine_task, engines = device_manager.load_engines(
            "mixer",
            "detector",
            "renderer",
            "controller",
            clock=clock,
            monitoring_session="play" if debug_monitor else None,
        )
        mixer, detector, renderer, controller = engines

        return engine_task, (mixer, detector, renderer, controller, clock)

    @staticmethod
    def update_devices_settings(mixer, detector, renderer):
        from ..main.profiles import ProfileManager

        profile_manager = providers.get(ProfileManager)
        devices_settings = profile_manager.current.devices
        devices_settings.mixer = mixer.settings
        devices_settings.detector = detector.settings
        devices_settings.renderer = renderer.settings
        profile_manager.set_as_changed()

    @staticmethod
    @contextlib.contextmanager
    def play_mode(mixer, detector, renderer):
        from ..main.loggers import Logger

        logger = providers.get(Logger)

        with logger.popup(renderer):
            yield

        if mixer.monitor is not None:
            logger.print()
            logger.print(f"   mixer: {mixer.monitor!s}", markup=False)
            logger.print(f"detector: {detector.monitor!s}", markup=False)
            logger.print(f"renderer: {renderer.monitor!s}", markup=False)

    @staticmethod
    def load_resource_from(src):
        from ..utils import providers
        from ..main.devices import DeviceManager

        return providers.get(DeviceManager).load_sound(src)

    @dn.datanode
    def load_resources(self):
        r"""Load resources to `audionode` and `resources`."""

        if self.path is not None and self.audio.path is not None:
            path = Path(self.path).parent / self.audio.path
            sound = yield from self.load_resource_from(path).join()
            volume = self.audio.volume
            if volume != 0.0:
                sound = dn.pipe(sound, lambda s: s * 10 ** (volume / 20))
            self.audionode = dn.DataNode.wrap(sound)

        for name, src in self.settings.resources.items():
            self.resources[name] = yield from self.load_resource_from(src).join()

    @dn.datanode
    def prepare_events(self, rich):
        r"""Prepare events `total_subjects`, `start_time`, `end_time`, `events`.

        Parameters
        ----------
        rich : markups.RichParser
        """
        events = []
        for track in self.tracks.values():
            context = {}
            for event in track.events(self.notations):
                try:
                    yield
                except GeneratorExit:
                    raise aud.IOCancelled("The operation has been cancelled.")

                event.prepare(self, rich, context)
                if isinstance(event, Event):
                    events.append(event)

        self.events = sorted(events, key=lambda e: e.lifespan[0])

        duration = 0.0
        if self.path is not None and self.audio.path is not None:
            root = Path(self.path).parent
            duration = aud.AudioMetadata.read(root / self.audio.path).duration

        event_leadin_time = self.settings.notes.event_leadin_time
        self.total_subjects = sum([1 for event in self.events if event.is_subject], 0)
        self.start_time = min(
            [0.0, *[event.lifespan[0] - event_leadin_time for event in self.events]]
        )
        self.end_time = max(
            [
                duration,
                *[event.lifespan[1] + event_leadin_time for event in self.events],
            ]
        )

    @dn.datanode
    def update_events(
        self,
        events,
        state,
        beatbar,
        end_time,
        prepare_time,
    ):
        events_iter = iter(events)
        event = next(events_iter, None)

        while True:
            time, ratio = yield

            if end_time <= time:
                return

            while event is not None and event.lifespan[0] - prepare_time <= time:
                event.register(state, beatbar)
                event = next(events_iter, None)

            state.time = time


class Loop(Beatmap):
    def __init__(
        self, *, offset=1.0, tempo=120.0, width=Fraction(0), track=None, settings=None
    ):
        super().__init__()
        if settings is not None:
            self.settings = settings
        self.tracks["main"] = track
        self.beatpoints = BeatPoints.fixed(offset=offset, tempo=tempo)
        self.width = width

    def repeat_events(self, rich):
        track = self.tracks["main"]
        width = self.width
        context = {}

        n = 0
        while True:
            events = []

            for event in track.events(self.notations):
                event = dataclasses.replace(event, beat=event.beat + n * width)
                event.prepare(self, rich, context)
                if isinstance(event, Event):
                    events.append(event)

            events = sorted(events, key=lambda e: e.lifespan[0])
            yield from events
            n += 1

    def prepare_events(self, rich):
        yield
        self.events = self.repeat_events(rich)
