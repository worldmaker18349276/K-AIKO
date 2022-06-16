import dataclasses
from pathlib import Path
from enum import Enum
from typing import List, Tuple, Dict, Optional, Union
from collections import OrderedDict
from fractions import Fraction
import threading
import numpy
from ..utils.providers import Provider
from ..utils import config as cfg
from ..utils import datanodes as dn
from ..utils import markups as mu
from ..devices import audios as aud
from ..devices import engines
from ..tui import widgets
from ..tui import beatbars
from . import beatpatterns


@dataclasses.dataclass
class UpdateContext:
    r"""An pseudo-event to update context in preparation phase.

    Attributes
    ----------
    update : dict
        The updated fields in the context.
    """

    update: Dict[str, Union[None, bool, int, Fraction, float, str]] = dataclasses.field(
        default_factory=dict
    )

    def prepare(self, beatmap, rich, context):
        context.update(**self.update)


def Context(beat, length, **contexts):
    return UpdateContext(contexts)


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
    `event3`, some events have no length (determined by attribute `has_length`).
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
    is_subject : bool, optional
        True if this event is an action. To increase the value of progress bar,
        use `state.add_finished`.
    full_score : int, optional
        The full score of this event. To increase score counter and full score
        counter, use `state.add_score` and `state.add_full_score`.
    has_length : bool, optional
        True if the attribute `length` is meaningful.

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

    is_subject = False
    full_score = 0
    has_length = True


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
        self.time = beatmap.metronome.time(self.beat)
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
        self.time = beatmap.metronome.time(self.beat)
        self.end = beatmap.metronome.time(self.beat + self.length)

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


@dataclasses.dataclass
class Flip(Event):
    r"""An event that flips the direction of the beatbar.

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

    def register(self, state, beatbar):
        beatbar.on_before_render(self._node(beatbar))

    @dn.datanode
    def _node(self, beatbar):
        time, ran = yield

        while time < self.time:
            time, ran = yield

        if self.flip is None:
            beatbar.bar_flip = not beatbar.bar_flip
        else:
            beatbar.bar_flip = self.flip

        time, ran = yield


@dataclasses.dataclass
class Shift(Event):
    r"""An event that shifts the shift of the beatbar.

    Fields
    ------
    beat, length : Fraction
        `beat` is the time to start shifting, `length` is meaningless.
    shift : float, optional
        The value of `bar_shift` of the scrolling bar after shifting, default is
        0.0.
    span : int or float or Fraction, optional
        the duration of shifting, default is 0.
    """

    has_length = False

    shift: float = 0.0
    span: Union[int, Fraction, float] = 0

    def prepare(self, beatmap, rich, context):
        self.time = beatmap.metronome.time(self.beat)
        self.end = beatmap.metronome.time(self.beat + self.span)
        self.lifespan = (self.time, self.end)

    def register(self, state, beatbar):
        beatbar.on_before_render(self._node(beatbar))

    @dn.datanode
    def _node(self, beatbar):
        time, ran = yield

        while time < self.time:
            time, ran = yield

        shift0 = beatbar.bar_shift
        speed = (
            (self.shift - shift0) / (self.end - self.time)
            if self.end != self.time
            else 0
        )

        while time < self.end:
            beatbar.bar_shift = shift0 + speed * (time - self.time)
            time, ran = yield

        beatbar.bar_shift = self.shift

        time, ran = yield


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

        self.time = beatmap.metronome.time(self.beat)
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
            self.speed = context.get("speed", 1.0)
        if self.volume is None:
            self.volume = context.get("volume", 0.0)
        if self.nofeedback is None:
            self.nofeedback = context.get("nofeedback", False)

        self.time = beatmap.metronome.time(self.beat)
        self.end = beatmap.metronome.time(self.beat + self.length)
        self.roll = 0
        self.number = max(int(self.length * self.density // -1 * -1), 1)
        self.is_finished = False
        self.score = 0

        self.times = [
            beatmap.metronome.time(self.beat + i / self.density)
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

    density: Union[int, Fraction, float] = 2.0
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
        sound = beatmap.resources.get(beatmap.settings.notes.spin_disk_sound, None)
        self.sound = dn.DataNode.wrap(sound) if sound is not None else None
        self.full_score = beatmap.settings.scores.spin_score

        if self.speed is None:
            self.speed = context.get("speed", 1.0)
        if self.volume is None:
            self.volume = context.get("volume", 0.0)
        if self.nofeedback is None:
            self.nofeedback = context.get("nofeedback", False)

        self.time = beatmap.metronome.time(self.beat)
        self.end = beatmap.metronome.time(self.beat + self.length)
        self.charge = 0.0
        self.capacity = float(self.length * self.density)
        self.is_finished = False
        self.score = 0

        self.times = [
            beatmap.metronome.time(self.beat + i / self.density)
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
                beatbar.play(self.sound, time=time, volume=self.volume)

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

        self.provider = Provider()
        self.provider.set(rich)
        self.provider.set(mixer)
        self.provider.set(detector)
        self.provider.set(renderer)

    def create(self, widget_settings):
        if isinstance(widget_settings, BeatbarWidgetFactory.spectrum):
            return widgets.SpectrumWidget(widget_settings).load(self.provider)
        elif isinstance(widget_settings, BeatbarWidgetFactory.volume_indicator):
            return widgets.VolumeIndicatorWidget(widget_settings).load(self.provider)
        elif isinstance(widget_settings, BeatbarWidgetFactory.knock_meter):
            return widgets.KnockMeterWidget(widget_settings).load(self.provider)
        elif isinstance(widget_settings, BeatbarWidgetFactory.accuracy_meter):
            accuracy_getter = dn.pipe(
                observe(self.state.perfs), lambda perfs: [perf.err for perf in perfs]
            )
            return widgets.AccuracyMeterWidget(accuracy_getter, widget_settings).load(self.provider)
        elif isinstance(widget_settings, BeatbarWidgetFactory.monitor):
            return widgets.MonitorWidget(widget_settings).load(self.provider)
        elif isinstance(widget_settings, BeatbarWidgetFactory.score):
            score_getter = lambda _: (self.state.score, self.state.full_score)
            return widgets.ScoreWidget(score_getter, widget_settings).load(self.provider)
        elif isinstance(widget_settings, BeatbarWidgetFactory.progress):
            progress_getter = lambda _: (
                self.state.finished_subjects / self.state.total_subjects
                if self.state.total_subjects > 0
                else 1.0
            )
            time_getter = lambda _: self.state.time
            return widgets.ProgressWidget(
                progress_getter, time_getter, widget_settings
            ).load(self.provider)
        elif isinstance(widget_settings, BeatbarWidgetFactory.sight):
            grade_getter = dn.pipe(
                observe(self.state.perfs),
                lambda perfs: [
                    perf.grade.shift for perf in perfs if perf.grade.shift is not None
                ],
            )
            return beatbars.SightWidget(grade_getter, widget_settings).load(self.provider)
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
            The score of sping note.
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

        event_leadin_time: float = 1.0

    resources: Dict[str, Union[Path, dn.Waveform]] = {
        "soft": dn.Waveform("0.5*2**(-t/0.01)*{sine:t*830.61}#tspan:0,0.06"),
        "loud": dn.Waveform("1.0*2**(-t/0.01)*{sine:t*1661.2}#tspan:0,0.06"),
        "incr": dn.Waveform("1.0*2**(-t/0.01)*{sine:t*1661.2}#tspan:0,0.06"),
        "rock": dn.Waveform("0.5*2**(-t/0.005)*{sine:t*1661.2}#tspan:0,0.03"),
        "disk": dn.Waveform("1.0*2**(-t/0.005)*{sine:t*1661.2}#tspan:0,0.03"),
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


class BeatTrack:
    _notations = {
        "x": Soft,
        "o": Loud,
        "<": Incr,
        "%": Roll,
        "@": Spin,
        "Context": Context,
        "Text": Text,
        "Flip": Flip,
        "Shift": Shift,
    }

    def __init__(self, events):
        self.events = events

    def __iter__(self):
        yield from self.events

    @classmethod
    def parse(cls, patterns_str, ret_width=False, notations=None):
        notations = notations if notations is not None else cls._notations
        patterns = beatpatterns.patterns_parser.parse(patterns_str)
        events, width = beatpatterns.to_events(patterns, notations=notations)
        track = cls(events)
        if ret_width:
            return track, width
        else:
            return track


mixer_monitor_file_path = "play_mixer_benchmark.csv"
detector_monitor_file_path = "play_detector_benchmark.csv"
renderer_monitor_file_path = "play_renderer_benchmark.csv"


class Beatmap:
    def __init__(
        self,
        path=None,
        info=None,
        audio=None,
        metronome=None,
        beatbar_state=None,
        tracks=None,
        settings=None,
    ):
        self.path = path
        self.info = info if info is not None else ""
        self.audio = audio if audio is not None else BeatmapAudio()
        self.metronome = (
            metronome
            if metronome is not None
            else engines.Metronome(offset=0.0, tempo=120.0)
        )
        self.beatbar_state = (
            beatbar_state if beatbar_state is not None else beatbars.BeatbarState()
        )
        self.tracks = tracks if tracks is not None else {}
        self.settings = settings if settings is not None else BeatmapSettings()

        self.audionode = None
        self.resources = {}
        self.total_subjects = 0
        self.start_time = 0.0
        self.end_time = float("inf")
        self.events = []

    @dn.datanode
    def play(
        self,
        manager,
        resources_dir,
        cache_dir,
        start_time,
        devices_settings,
        gameplay_settings=None,
    ):
        gameplay_settings = gameplay_settings or GameplaySettings()

        samplerate = devices_settings.mixer.output_samplerate
        nchannels = devices_settings.mixer.output_channels
        tickrate = gameplay_settings.controls.tickrate
        prepare_time = gameplay_settings.controls.prepare_time
        debug_monitor = gameplay_settings.debug_monitor

        rich = mu.RichParser(
            devices_settings.terminal.unicode_version,
            devices_settings.terminal.color_support,
        )

        # prepare
        try:
            yield from dn.create_task(
                dn.chain(
                    self.load_resources(samplerate, nchannels, resources_dir),
                    self.prepare_events(rich),
                ),
            ).join()
        except aud.IOCancelled:
            return

        if start_time is not None:
            self.start_time = start_time

        score = BeatmapScore()
        score.set_total_subjects(self.total_subjects)

        # load engines
        mixer_monitor = detector_monitor = renderer_monitor = None
        if debug_monitor:
            mixer_monitor = engines.Monitor(cache_dir / mixer_monitor_file_path)
            detector_monitor = engines.Monitor(cache_dir / detector_monitor_file_path)
            renderer_monitor = engines.Monitor(cache_dir / renderer_monitor_file_path)

        mixer_task, mixer = engines.Mixer.create(
            devices_settings.mixer, manager, self.start_time, mixer_monitor
        )
        detector_task, detector = engines.Detector.create(
            devices_settings.detector, manager, self.start_time, detector_monitor
        )
        renderer_task, renderer = engines.Renderer.create(
            devices_settings.renderer,
            devices_settings.terminal,
            self.start_time,
            renderer_monitor,
        )
        controller_task, controller = engines.Controller.create(
            devices_settings.controller, devices_settings.terminal, self.start_time
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
            self.beatbar_state,
        )

        beatbar_node = beatbar.load()

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

        # handler
        event_clock = engines.Clock()

        devices_settings = devices_settings.copy()
        settings_changed = Beatmap.register_controllers(
            mixer,
            detector,
            renderer,
            controller,
            event_clock,
            devices_settings,
            gameplay_settings.controls,
        )

        # play music
        if self.audionode is not None:
            mixer.play(self.audionode, time=0.0, zindex=(-3,))

        # game loop
        updater = self.update_events(
            self.events,
            score,
            beatbar,
            self.start_time,
            self.end_time,
            tickrate,
            prepare_time,
            event_clock,
        )
        event_task = dn.interval(updater, dt=1 / tickrate)

        yield from dn.pipe(
            event_task, mixer_task, detector_task, renderer_task, controller_task
        ).join()

        if debug_monitor:
            print()
            print("   mixer: " + str(mixer_monitor))
            print("detector: " + str(detector_monitor))
            print("renderer: " + str(renderer_monitor))

        return score, (devices_settings if settings_changed.is_set() else None)

    @staticmethod
    def register_controllers(
        mixer,
        detector,
        renderer,
        controller,
        event_clock,
        devices_settings,
        controls_settings,
    ):
        settings_changed = threading.Event()

        # stop
        stop_key = controls_settings.stop_key
        controller.add_handler(
            dn.pipe(dn.time(0.0), event_clock.stop), stop_key
        )

        # display delay
        display_delay_adjust_step = controls_settings.display_delay_adjust_step
        display_delay_adjust_keys = controls_settings.display_delay_adjust_keys

        def incr_display_delay(time):
            devices_settings.renderer.display_delay += display_delay_adjust_step
            renderer.clock.skip(time, display_delay_adjust_step)
            renderer.add_log(
                mu.Text(f"display_delay += {display_delay_adjust_step}\n")
            )
            settings_changed.set()

        def decr_display_delay(time):
            devices_settings.renderer.display_delay -= display_delay_adjust_step
            renderer.clock.skip(time, -display_delay_adjust_step)
            renderer.add_log(
                mu.Text(f"display_delay -= {display_delay_adjust_step}\n")
            )
            settings_changed.set()

        controller.add_handler(
            dn.pipe(dn.time(0.0), incr_display_delay), display_delay_adjust_keys[0]
        )
        controller.add_handler(
            dn.pipe(dn.time(0.0), decr_display_delay), display_delay_adjust_keys[1]
        )

        # knock delay
        knock_delay_adjust_step = controls_settings.knock_delay_adjust_step
        knock_delay_adjust_keys = controls_settings.knock_delay_adjust_keys

        def incr_knock_delay(time):
            devices_settings.detector.knock_delay += knock_delay_adjust_step
            detector.clock.skip(time, knock_delay_adjust_step)
            renderer.add_log(
                mu.Text(f"knock_delay += {knock_delay_adjust_step}\n")
            )
            settings_changed.set()

        def decr_knock_delay(time):
            devices_settings.detector.knock_delay -= knock_delay_adjust_step
            detector.clock.skip(time, -knock_delay_adjust_step)
            renderer.add_log(
                mu.Text(f"knock_delay -= {knock_delay_adjust_step}\n")
            )
            settings_changed.set()

        controller.add_handler(
            dn.pipe(dn.time(0.0), incr_knock_delay), knock_delay_adjust_keys[0]
        )
        controller.add_handler(
            dn.pipe(dn.time(0.0), decr_knock_delay), knock_delay_adjust_keys[1]
        )

        # knock strength
        knock_energy_adjust_step = controls_settings.knock_energy_adjust_step
        knock_energy_adjust_keys = controls_settings.knock_energy_adjust_keys

        def incr_knock_energy(_):
            devices_settings.detector.knock_energy += knock_energy_adjust_step
            detector.knock_energy.add(knock_energy_adjust_step)
            renderer.add_log(
                mu.Text(f"knock_energy += {knock_energy_adjust_step}\n")
            )
            settings_changed.set()

        def decr_knock_energy(_):
            devices_settings.detector.knock_energy -= knock_energy_adjust_step
            detector.knock_energy.add(-knock_energy_adjust_step)
            renderer.add_log(
                mu.Text(f"knock_energy -= {knock_energy_adjust_step}\n")
            )
            settings_changed.set()

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
                        mixer.clock.resume(time + control_delay)
                        detector.clock.resume(time + control_delay)
                        renderer.clock.resume(time + control_delay)
                        controller.clock.resume(time + control_delay)
                        event_clock.resume(time + control_delay)
                        paused = False
                    else:
                        mixer.clock.pause(time + control_delay)
                        detector.clock.pause(time + control_delay)
                        renderer.clock.pause(time + control_delay)
                        controller.clock.pause(time + control_delay)
                        event_clock.pause(time + control_delay)
                        paused = True

        controller.add_handler(pause_node(), pause_key)

        @dn.datanode
        def skip_node():
            time_node = dn.time()
            with time_node:
                while True:
                    yield
                    time = time_node.send(None)
                    mixer.clock.skip(time + control_delay, skip_time)
                    detector.clock.skip(time + control_delay, skip_time)
                    renderer.clock.skip(time + control_delay, skip_time)
                    controller.clock.skip(time + control_delay, skip_time)
                    event_clock.skip(time + control_delay, skip_time)

        controller.add_handler(skip_node(), skip_key)

        return settings_changed

    @dn.datanode
    def load_resources(self, output_samplerate, output_nchannels, resources_dir):
        r"""Load resources to `audionode` and `resources`.

        Parameters
        ----------
        output_samplerate : int
        output_channels : int
        resources_dir : Path
        """

        if self.path is not None and self.audio.path is not None:
            root = Path(self.path).parent
            try:
                sound = yield from aud.load_sound(
                    root / self.audio.path,
                    samplerate=output_samplerate,
                    channels=output_nchannels,
                    volume=self.audio.volume,
                ).join()

                self.audionode = dn.DataNode.wrap(sound)

            except aud.IOCancelled:
                raise

            except Exception as e:
                raise RuntimeError(f"Failed to load song {self.audio.path}") from e

        for name, path in self.settings.resources.items():
            if isinstance(path, Path):
                sound_path = resources_dir / path
                try:
                    resource = yield from aud.load_sound(
                        sound_path,
                        samplerate=output_samplerate,
                        channels=output_nchannels,
                    ).join()

                    self.resources[name] = resource

                except aud.IOCancelled:
                    raise

                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load resource {name} at {sound_path}"
                    ) from e

            elif isinstance(path, dn.Waveform):
                waveform_max_time = 30.0
                try:
                    node = path.generate(
                        samplerate=output_samplerate, channels=output_nchannels,
                    )

                    resource = []
                    yield from dn.ensure(
                        dn.pipe(
                            node,
                            dn.tspan(
                                samplerate=output_samplerate, end=waveform_max_time
                            ),
                            resource.append,
                        ),
                        lambda: aud.IOCancelled(
                            f"The operation of generating sound {path!r} has been cancelled."
                        ),
                    ).join()

                    self.resources[name] = resource

                except aud.IOCancelled:
                    raise

                except Exception as e:
                    raise RuntimeError(f"Failed to load resource {name}") from e

            else:
                raise TypeError

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
            for event in track:
                try:
                    yield
                except GeneratorExit:
                    raise aud.IOCancelled("The operation has been cancelled.")

                event = dataclasses.replace(event)
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
        start_time,
        end_time,
        tickrate,
        prepare_time,
        clock,
    ):
        # register events
        events_iter = iter(events)
        event = next(events_iter, None)

        timer = dn.pipe(dn.count(0.0, 1 / tickrate), clock.clock(start_time, 1))

        with timer:
            yield
            while True:
                try:
                    time, ratio = timer.send(None)
                except StopIteration:
                    return

                if end_time <= time:
                    return

                while event is not None and event.lifespan[0] - prepare_time <= time:
                    event.register(state, beatbar)
                    event = next(events_iter, None)

                state.time = time

                yield


class Loop(Beatmap):
    def __init__(
        self, *, offset=1.0, tempo=120.0, width=Fraction(0), track=None, settings=None
    ):
        metronome = engines.Metronome(offset=offset, tempo=tempo)
        super().__init__(metronome=metronome, tracks={"main": track}, settings=settings)
        self.width = width

    def repeat_events(self, rich):
        track = self.tracks["main"]
        width = self.width
        context = {}

        n = 0
        while True:
            events = []

            for event in track:
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
