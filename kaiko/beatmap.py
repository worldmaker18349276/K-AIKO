import os
import datetime
import contextlib
from enum import Enum
from dataclasses import dataclass, replace
from typing import List, Tuple, Dict, Optional, Union
from collections import OrderedDict
from fractions import Fraction
import numpy
import audioread
from .engines import Mixer, MixerSettings, Detector, DetectorSettings, Renderer, RendererSettings
from .beatbar import PerformanceGrade, Performance, Beatbar, BeatbarSettings
from . import config as cfg
from . import datanodes as dn
from . import wcbuffers as wcb


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class UpdateContext:
    update: Dict[str, Union[None, bool, int, Fraction, float, str]]

    def prepare(self, beatmap, context):
        context.update(**self.update)

@dataclass
class Event:
    beat: Fraction = Fraction(0, 1)
    length: Fraction = Fraction(1, 1)

    def prepare(self, beatmap, context):
        raise NotImplementedError

    def register(self, field):
        raise NotImplementedError

    lifespan = (float('-inf'), float('-inf'))
    is_subject = False
    full_score = 0
    has_length = True


# scripts
@dataclass
class Text(Event):
    has_length = False

    text: Optional[str] = None
    sound: Optional[str] = None
    speed: Optional[float] = None

    def prepare(self, beatmap, context):
        self.sound_root = beatmap.root
        self.time = beatmap.time(self.beat)
        if self.speed is None:
            self.speed = context.get('speed', 1.0)

        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.time + travel_time)
        self.zindex = (-2, -self.time)

    def pos(self, time):
        return (self.time-time) * 0.5 * self.speed

    def register(self, field):
        if self.sound is not None:
            sound_path = os.path.join(self.sound_root, self.sound)
            field.play(sound_path, time=self.time)

        if self.text is not None:
            field.draw_content(self.pos, self.text, zindex=self.zindex,
                               start=self.lifespan[0], duration=self.lifespan[1]-self.lifespan[0])

@dataclass
class Flip(Event):
    has_length = False

    flip: Optional[bool] = None

    def prepare(self, beatmap, context):
        self.time = beatmap.time(self.beat)
        self.lifespan = (self.time, self.time)

    def register(self, field):
        field.on_before_render(self._node(field))

    @dn.datanode
    def _node(self, field):
        time, width = yield

        while time < self.time:
            time, width = yield

        if self.flip is None:
            field.beatbar.bar_flip = not field.beatbar.bar_flip
        else:
            field.beatbar.bar_flip = self.flip

        time, width = yield

@dataclass
class Shift(Event):
    shift: float = 0.0
    span: Union[int, Fraction, float] = 0

    def prepare(self, beatmap, context):
        self.time = beatmap.time(self.beat)
        self.end = beatmap.time(self.beat+self.span)
        self.lifespan = (self.time, self.end)

    def register(self, field):
        field.on_before_render(self._node(field))

    @dn.datanode
    def _node(self, field):
        time, width = yield

        while time < self.time:
            time, width = yield

        shift0 = field.beatbar.bar_shift
        speed = (self.shift - shift0) / (self.end - self.time) if self.end != self.time else 0

        while time < self.end:
            field.beatbar.bar_shift = shift0 + speed * (time - self.time)
            time, width = yield

        field.beatbar.bar_shift = self.shift

        time, width = yield

# targets
class Target(Event):
    # lifespan, range, is_finished
    # __init__(*args, **kwargs)
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

@dataclass
class OneshotTarget(Target):
    # time, speed, volume, perf, sound, sound_root
    # approach_appearance, wrong_appearance
    # hit(field, time, strength)

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

    def approach(self, field):
        if self.sound is not None:
            sound_path = os.path.join(self.sound_root, self.sound)
            field.play(sound_path, time=self.time, volume=self.volume)

        field.draw_content(self.pos, self.appearance, zindex=self.zindex,
                           start=self.lifespan[0], duration=self.lifespan[1]-self.lifespan[0])
        field.reset_sight(start=self.range[0])

    def hit(self, field, time, strength, is_correct_key=True):
        perf = Performance.judge(self.performance_tolerance, self.time, time, is_correct_key)
        field.add_perf(perf, True, self.speed < 0)
        self.finish(field, perf)

    def finish(self, field, perf=None):
        if perf is None:
            perf = Performance.judge(self.performance_tolerance, self.time)
        self.perf = perf
        field.add_full_score(self.full_score)
        field.add_score(self.score)

@dataclass
class Soft(OneshotTarget):
    speed: Optional[float] = None
    volume: Optional[float] = None

    def prepare(self, beatmap, context):
        self.approach_appearance = beatmap.settings.notes.soft_approach_appearance
        self.wrong_appearance = beatmap.settings.notes.soft_wrong_appearance
        self.sound = beatmap.settings.notes.soft_sound
        self.sound_root = beatmap.root
        self.threshold = beatmap.settings.difficulty.soft_threshold

        if self.speed is None:
            self.speed = context.get('speed', 1.0)
        if self.volume is None:
            self.volume = context.get('volume', 0.0)

        super().prepare(beatmap, context)

    def hit(self, field, time, strength):
        super().hit(field, time, strength, strength < self.threshold)

@dataclass
class Loud(OneshotTarget):
    speed: Optional[float] = None
    volume: Optional[float] = None

    def prepare(self, beatmap, context):
        self.approach_appearance = beatmap.settings.notes.loud_approach_appearance
        self.wrong_appearance = beatmap.settings.notes.loud_wrong_appearance
        self.sound = beatmap.settings.notes.loud_sound
        self.sound_root = beatmap.root
        self.threshold = beatmap.settings.difficulty.loud_threshold

        if self.speed is None:
            self.speed = context.get('speed', 1.0)
        if self.volume is None:
            self.volume = context.get('volume', 0.0)

        super().prepare(beatmap, context)

    def hit(self, field, time, strength):
        super().hit(field, time, strength, strength >= self.threshold)

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
    group: Optional[str] = None
    speed: Optional[float] = None
    group_volume: Optional[float] = None

    def prepare(self, beatmap, context):
        self.approach_appearance = beatmap.settings.notes.incr_approach_appearance
        self.wrong_appearance = beatmap.settings.notes.incr_wrong_appearance
        self.sound = beatmap.settings.notes.incr_sound
        self.sound_root = beatmap.root
        self.incr_threshold = beatmap.settings.difficulty.incr_threshold

        if self.speed is None:
            self.speed = context.get('speed', 1.0)
        if self.group_volume is None:
            self.group_volume = context.get('volume', 0.0)

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

    def hit(self, field, time, strength):
        group_obj = self.groups[self.group]
        threshold = max(0.0, min(1.0, group_obj.threshold + self.incr_threshold))
        super().hit(field, time, strength, strength >= threshold)
        group_obj.hit(strength)

@dataclass
class Roll(Target):
    density: Union[int, Fraction, float] = 2
    speed: Optional[float] = None
    volume: Optional[float] = None

    def prepare(self, beatmap, context):
        self.performance_tolerance = beatmap.settings.difficulty.performance_tolerance
        self.tolerance = beatmap.settings.difficulty.roll_tolerance
        self.rock_appearance = beatmap.settings.notes.roll_rock_appearance
        self.sound = beatmap.settings.notes.roll_rock_sound
        self.sound_root = beatmap.root
        self.rock_score = beatmap.settings.scores.roll_rock_score

        if self.speed is None:
            self.speed = context.get('speed', 1.0)
        if self.volume is None:
            self.volume = context.get('volume', 0.0)

        self.time = beatmap.time(self.beat)
        self.end = beatmap.time(self.beat+self.length)
        self.roll = 0
        self.number = max(int(self.length * self.density), 1)
        self.is_finished = False
        self.score = 0

        self.times = [beatmap.time(self.beat+i/self.density) for i in range(self.number)]
        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.end + travel_time)
        self.range = (self.time - self.tolerance, self.end - self.tolerance)
        self.full_score = self.number * self.rock_score

    def pos_of(self, index):
        return lambda time: (self.times[index]-time) * 0.5 * self.speed

    def appearance_of(self, index):
        return lambda time: self.rock_appearance if self.roll <= index else ""

    def approach(self, field):
        for i, time in enumerate(self.times):
            if self.sound is not None:
                sound_path = os.path.join(self.sound_root, self.sound)
                field.play(sound_path, time=time, volume=self.volume)
            field.draw_content(self.pos_of(i), self.appearance_of(i), zindex=self.zindex,
                               start=self.lifespan[0], duration=self.lifespan[1]-self.lifespan[0])
        field.reset_sight(start=self.range[0])

    def hit(self, field, time, strength):
        self.roll += 1

        if self.roll <= self.number:
            perf = Performance.judge(self.performance_tolerance, self.times[self.roll-1], time, True)
            field.add_perf(perf, False)

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
            field.add_perf(perf, False)

@dataclass
class Spin(Target):
    density: Union[int, Fraction, float] = 2
    speed: Optional[float] = None
    volume: Optional[float] = None

    def prepare(self, beatmap, context):
        self.tolerance = beatmap.settings.difficulty.spin_tolerance
        self.disk_appearances = beatmap.settings.notes.spin_disk_appearances
        self.finishing_appearance = beatmap.settings.notes.spin_finishing_appearance
        self.finish_sustain_time = beatmap.settings.notes.spin_finish_sustain_time
        self.sound = beatmap.settings.notes.spin_disk_sound
        self.sound_root = beatmap.root
        self.full_score = beatmap.settings.scores.spin_score

        if self.speed is None:
            self.speed = context.get('speed', 1.0)
        if self.volume is None:
            self.volume = context.get('volume', 0.0)

        self.time = beatmap.time(self.beat)
        self.end = beatmap.time(self.beat+self.length)
        self.charge = 0.0
        self.capacity = self.length * self.density
        self.is_finished = False
        self.score = 0

        self.times = [beatmap.time(self.beat+i/self.density) for i in range(int(self.capacity))]
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
                sound_path = os.path.join(self.sound_root, self.sound)
                field.play(sound_path, time=time, volume=self.volume)

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


# Game
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
        soft_sound: str = f"{BASE_DIR}/samples/soft.wav" # pulse(freq=830.61, decay_time=0.03, amplitude=0.5)
        loud_approach_appearance:  Union[str, Tuple[str, str]] = "\x1b[94m■\x1b[m"
        loud_wrong_appearance:     Union[str, Tuple[str, str]] = "\x1b[94m⬚\x1b[m"
        loud_sound: str = f"{BASE_DIR}/samples/loud.wav" # pulse(freq=1661.2, decay_time=0.03, amplitude=1.0)
        incr_approach_appearance:  Union[str, Tuple[str, str]] = "\x1b[94m⬒\x1b[m"
        incr_wrong_appearance:     Union[str, Tuple[str, str]] = "\x1b[94m⬚\x1b[m"
        incr_sound: str = f"{BASE_DIR}/samples/incr.wav" # pulse(freq=1661.2, decay_time=0.03, amplitude=1.0)
        roll_rock_appearance:      Union[str, Tuple[str, str]] = "\x1b[96m◎\x1b[m"
        roll_rock_sound: str = f"{BASE_DIR}/samples/rock.wav" # pulse(freq=1661.2, decay_time=0.01, amplitude=0.5)
        spin_disk_appearances:     Union[List[str], List[Tuple[str, str]]] = ["\x1b[94m◴\x1b[m",
                                                                              "\x1b[94m◵\x1b[m",
                                                                              "\x1b[94m◶\x1b[m",
                                                                              "\x1b[94m◷\x1b[m"]
        spin_finishing_appearance: Union[str, Tuple[str, str]] = "\x1b[94m☺\x1b[m"
        spin_finish_sustain_time: float = 0.1
        spin_disk_sound: str = f"{BASE_DIR}/samples/disk.wav" # pulse(freq=1661.2, decay_time=0.01, amplitude=1.0)
        event_leadin_time: float = 1.0

class Playable:
    # total_subjects: int
    # start_time: float
    # end_time: float

    def get_audionode(self, output_samplerate, output_nchannels):
        raise NotImplementedError

    def prepare_events(self):
        raise NotImplementedError

class Beatmap(Playable):
    def __init__(self, root=".", audio=None, volume=0.0,
                 info="", preview=None,
                 offset=0.0, tempo=60.0,
                 bar_shift=0.1, bar_flip=False,
                 settings=None):
        self.root = root
        self.audio = audio
        self.volume = volume
        self.info = info
        self.preview = preview
        self.offset = offset
        self.tempo = tempo
        self.bar_shift = bar_shift
        self.bar_flip = bar_flip

        self.settings = settings or BeatmapSettings()

        self.events_start_time = None
        self.events_end_time = None
        self.total_subjects = 0
        self.event_sequences = []

    @property
    def duration(self):
        if self.audio is None:
            return 0.0

        with audioread.audio_open(os.path.join(self.root, self.audio)) as file:
            return file.duration

    def time(self, beat):
        return self.offset + beat*60/self.tempo

    def beat(self, time):
        return (time - self.offset)*self.tempo/60

    def dtime(self, beat, length):
        return self.time(beat+length) - self.time(beat)

    def get_audionode(self, output_samplerate, output_nchannels):
        if self.audio is None:
            return None

        else:
            audio_path = os.path.join(self.root, self.audio)
            audionode = dn.DataNode.wrap(dn.load_sound(audio_path,
                                                       samplerate=output_samplerate,
                                                       channels=output_nchannels,
                                                       volume=self.volume))
            return audionode

    def prepare_events(self):
        events = []
        for sequence in self.event_sequences:
            context = {}
            for event in sequence:
                event = replace(event)
                event.prepare(self, context)
                if not isinstance(event, Event):
                    continue
                events.append(event)

        events = sorted(events, key=lambda e: e.lifespan[0])

        event_leadin_time = self.settings.notes.event_leadin_time
        self.total_subjects = sum([1 for event in events if event.is_subject], 0)
        self.start_time = min([event.lifespan[0] - event_leadin_time for event in events], default=0.0)
        self.end_time = max([event.lifespan[1] + event_leadin_time for event in events], default=self.duration)

        return events


def uint_format(value, width, zero_padded=False):
    scales = "KMGTPEZY"
    pad = "0" if zero_padded else " "

    if width == 0:
        return ""
    if width == 1:
        return str(value) if value < 10 else "+"

    if width == 2 and value < 1000:
        return f"{value:{pad}{width}d}" if value < 10 else "9+"
    elif value < 10**width:
        return f"{value:{pad}{width}d}"

    for scale, symbol in enumerate(scales):
        if value < 1000**(scale+2):
            if width == 2:
                return symbol + "+"

            value_ = value // 1000**(scale+1)
            eff = f"{value_:{pad}{width-2}d}" if value_ < 10**(width-2) else str(10**(width-2)-1)
            return eff + symbol + "+"

    else:
        return str(10**(width-2)-1) + scales[-1] + "+"

def time_format(value, width):
    if width < 4:
        return uint_format(value, width, True)
    else:
        return f"{uint_format(value//60, width-3, True)}:{value%60:02d}"

def pc_format(value, width):
    if width == 0:
        return ""
    if width == 1:
        return "1" if value == 1 else "0"
    if width == 2:
        return f"1." if value == 1 else "." + str(int(value*10))
    if width == 3:
        return f"1.0" if value == 1 else f"{value:>{width}.0%}"
    if width >= 4:
        return f"{value:>{width}.0%}" if value == 1 else f"{value:>{width}.{width-4}%}"

class Widget(Enum):
    spectrum = "spectrum"
    volume_indicator = "volume_indicator"
    score = "score"
    progress = "progress"
    bounce = "bounce"
    accuracy_meter = "accuracy_meter"

class WidgetManager:
    @staticmethod
    def use_widget(name, field):
        func = getattr(WidgetManager, name.value, None)
        if func is None:
            raise ValueError("no such widget: " + name)
        func(field)

    @staticmethod
    def spectrum(field):
        attr = field.settings.widgets.spectrum.attr
        spec_width = field.settings.widgets.spectrum.spec_width
        samplerate = field.settings.mixer.output_samplerate
        nchannels = field.settings.mixer.output_channels
        hop_length = round(samplerate * field.settings.widgets.spectrum.spec_time_res)
        win_length = round(samplerate / field.settings.widgets.spectrum.spec_freq_res)
        spec_decay_time = field.settings.widgets.spectrum.spec_decay_time

        df = samplerate/win_length
        n_fft = win_length//2+1
        n = numpy.linspace(1, 88, spec_width*2+1)
        f = 440 * 2**((n-49)/12) # frequency of n-th piano key
        sec = numpy.minimum(n_fft-1, (f/df).round().astype(int))
        slices = [slice(start, stop) for start, stop in zip(sec[:-1], (sec+1)[1:])]

        decay = hop_length / samplerate / spec_decay_time / 4
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
                    try:
                        J = node.send(data)
                    except StopIteration:
                        return

                    vols = [max(0.0, prev-decay, min(1.0, volume_of(J[slic])))
                            for slic, prev in zip(slices, vols)]
                    field.spectrum = "".join(map(draw_bar, vols[0::2], vols[1::2]))

        handler = dn.pipe(lambda a:a[0], dn.branch(dn.unchunk(draw_spectrum(), (hop_length, nchannels))))
        field.spectrum = "\u2800"*spec_width
        field.beatbar.mixer.add_effect(handler, zindex=(-1,))

        def widget_func(time, ran):
            spectrum = field.spectrum
            width = ran.stop - ran.start
            return f"\x1b[{attr}m{spectrum:^{width}.{width}s}\x1b[m"

        field.beatbar.current_icon.set(widget_func)

    @staticmethod
    def volume_indicator(field):
        attr = field.settings.widgets.volume_indicator.attr
        vol_decay_time = field.settings.widgets.volume_indicator.vol_decay_time
        buffer_length = field.settings.mixer.output_buffer_length
        samplerate = field.settings.mixer.output_samplerate

        decay = buffer_length / samplerate / vol_decay_time

        volume_of = lambda x: dn.power2db((x**2).mean(), scale=(1e-5, 1e6)) / 60.0

        @dn.datanode
        def volume_indicator():
            vol = 0.0

            while True:
                data = yield
                vol = max(0.0, vol-decay, min(1.0, volume_of(data)))
                field.volume_indicator = vol

        handler = dn.pipe(lambda a:a[0], dn.branch(volume_indicator()))
        field.volume_indicator = 0.0
        field.beatbar.mixer.add_effect(handler, zindex=(-1,))

        def widget_func(time, ran):
            volume_indicator = field.volume_indicator
            width = ran.stop - ran.start
            return f"\x1b[{attr}m" + "▮" * int(volume_indicator * width) + "\x1b[m"

        field.beatbar.current_icon.set(widget_func)

    @staticmethod
    def score(field):
        attr = field.settings.widgets.score.attr
        def widget_func(time, ran):
            score = field.score
            full_score = field.full_score
            width = ran.stop - ran.start

            if width == 0:
                return ""
            if width == 1:
                return f"\x1b[{attr};1m|\x1b[m"
            if width == 2:
                return f"\x1b[{attr};1m[]\x1b[m"
            if width <= 7:
                score_str = uint_format(score, width-2, True)
                return f"\x1b[{attr};1m[\x1b[22m{score_str}\x1b[1m]\x1b[m"

            w1 = max((width-3)//2, 5)
            w2 = (width-3) - w1
            score_str = uint_format(score, w1, True)
            full_score_str = uint_format(full_score, w2, True)
            return f"\x1b[{attr};1m[\x1b[22m{score_str}\x1b[1m/\x1b[22m{full_score_str}\x1b[1m]\x1b[m"

        field.beatbar.current_header.set(widget_func)

    @staticmethod
    def progress(field):
        attr = field.settings.widgets.progress.attr
        def widget_func(time, ran):
            progress = field.finished_subjects/field.total_subjects if field.total_subjects>0 else 1.0
            time = int(max(0.0, field.time))
            width = ran.stop - ran.start

            if width == 0:
                return ""
            if width == 1:
                return f"\x1b[{attr};1m|\x1b[m"
            if width == 2:
                return f"\x1b[{attr};1m[]\x1b[m"
            if width <= 7:
                progress_str = pc_format(progress, width-2)
                return f"\x1b[{attr};1m[\x1b[22m{progress_str}\x1b[1m]\x1b[m"

            w1 = max((width-3)//2, 5)
            w2 = (width-3) - w1
            progress_str = pc_format(progress, w1)
            time_str = time_format(time, w2)
            return f"\x1b[{attr};1m[\x1b[22m{progress_str}\x1b[1m|\x1b[22m{time_str}\x1b[1m]\x1b[m"

        field.beatbar.current_footer.set(widget_func)

    @staticmethod
    def bounce(field):
        attr = field.settings.widgets.bounce.attr
        division = field.settings.widgets.bounce.division

        offset = field.beatmap.offset
        period = 60.0 / field.beatmap.tempo / division
        def widget_func(time, ran):
            width = ran.stop - ran.start

            if width == 0:
                return ""
            if width == 1:
                return f"\x1b[{attr};1m|\x1b[m"
            if width == 2:
                return f"\x1b[{attr};1m[]\x1b[m"

            turns = (time - offset) / period
            index = int(turns % 1 * (width-3) // 1)
            dir = int(turns % 2 // 1 * 2 - 1)
            inner = [" "]*(width-2)
            if dir > 0:
                inner[index] = "="
            else:
                inner[-1-index] = "="
            return f"\x1b[{attr};1m[\x1b[22m{''.join(inner)}\x1b[1m]\x1b[m"

        field.beatbar.current_icon.set(widget_func)

    @staticmethod
    def accuracy_meter(field):
        meter_width = field.settings.widgets.accuracy_meter.meter_width
        meter_decay_time = field.settings.widgets.accuracy_meter.meter_decay_time
        meter_tolerance = field.settings.widgets.accuracy_meter.meter_tolerance

        length = meter_width*2
        last_perf = 0
        last_time = float("inf")
        hit = [0.0]*length
        nlevel = 24

        def widget_func(time, ran):
            nonlocal last_perf, last_time

            new_err = []
            while len(field.perfs) > last_perf:
                err = field.perfs[last_perf].err
                index = max(min(int((err-meter_tolerance)/-meter_tolerance/2 * length//1), length-1), 0)
                new_err.append(index)
                last_perf += 1

            decay = max(0.0, time - last_time) / meter_decay_time
            last_time = time

            for i in range(meter_width*2):
                if i in new_err:
                    hit[i] = 1.0
                else:
                    hit[i] = max(0.0, hit[i] - decay)

            return "".join(f"\x1b[48;5;{232+int(i*(nlevel-1))};38;5;{232+int(j*(nlevel-1))}m▐\x1b[m"
                           for i, j in zip(hit[::2], hit[1::2]))

        field.beatbar.current_icon.set(widget_func)


class GameplaySettings(cfg.Configurable):
    mixer = MixerSettings
    detector = DetectorSettings
    renderer = RendererSettings
    beatbar = BeatbarSettings

    class controls(cfg.Configurable):
        skip_time: float = 8.0
        load_time: float = 0.5
        prepare_time: float = 0.1
        tickrate: float = 60.0

    class widgets(cfg.Configurable):
        use: List[Widget] = [Widget.spectrum, Widget.score, Widget.progress]

        class spectrum(cfg.Configurable):
            attr: str = "95"
            spec_width: int = 6
            spec_decay_time: float = 0.01
            spec_time_res: float = 0.0116099773 # hop_length = 512 if samplerate == 44100
            spec_freq_res: float = 21.5332031 # win_length = 512*4 if samplerate == 44100

        class volume_indicator(cfg.Configurable):
            attr: str = "95"
            vol_decay_time: float = 0.01

        class score(cfg.Configurable):
            attr: str = "38;5;93"

        class progress(cfg.Configurable):
            attr: str = "38;5;93"

        class bounce(cfg.Configurable):
            attr: str = "95"
            division: int = 2

        class accuracy_meter(cfg.Configurable):
            meter_width: int = 8
            meter_decay_time: float = 1.5
            meter_tolerance: float = 0.10

class BeatmapPlayer:
    def __init__(self, beatmap, settings=None):
        self.beatmap = beatmap
        self.settings = settings or GameplaySettings()

    def prepare(self, output_samplerate, output_nchannels):
        # prepare music
        self.audionode = self.beatmap.get_audionode(output_samplerate, output_nchannels)

        # prepare events
        self.events = self.beatmap.prepare_events()
        self.start_time = self.beatmap.start_time
        self.end_time = self.beatmap.end_time

        # initialize game state
        self.total_subjects = self.beatmap.total_subjects
        self.finished_subjects = 0
        self.full_score = 0
        self.score = 0

        self.perfs = []
        self.time = datetime.time(0, 0, 0)

        return abs(self.start_time)

    @contextlib.contextmanager
    def execute(self, manager):
        tickrate = self.settings.controls.tickrate
        samplerate = self.settings.mixer.output_samplerate
        nchannels = self.settings.mixer.output_channels
        time_shift = self.prepare(samplerate, nchannels)
        load_time = self.settings.controls.load_time
        ref_time = load_time + time_shift

        bar_shift = self.beatmap.bar_shift
        bar_flip = self.beatmap.bar_flip

        mixer_knot, mixer = Mixer.create(self.settings.mixer, manager, ref_time)
        detector_knot, detector = Detector.create(self.settings.detector, manager, ref_time)
        renderer_knot, renderer = Renderer.create(self.settings.renderer, ref_time)
        beatbar_knot = dn.pipe(mixer_knot, detector_knot, renderer_knot)

        self.beatbar = Beatbar.create(self.settings.beatbar, mixer, detector, renderer, bar_shift, bar_flip)

        # play music
        if self.audionode is not None:
            self.beatbar.mixer.play(self.audionode, time=0.0, zindex=(-3,))

        # use widgets
        for widget in self.settings.widgets.use:
            WidgetManager.use_widget(widget, self)

        # game loop
        event_knot = dn.interval(consumer=self.update_events(), dt=1/tickrate)
        game_knot = dn.pipe(event_knot, beatbar_knot)
        dn.exhaust(game_knot, dt=0.1, interruptible=True)

    @dn.datanode
    def update_events(self):
        # register events
        events_iter = iter(self.events)
        event = next(events_iter, None)

        start_time = self.start_time
        tickrate = self.settings.controls.tickrate
        prepare_time = self.settings.controls.prepare_time

        yield
        index = 0

        while True:
            time = index / tickrate + start_time

            if self.end_time <= time:
                return

            while event is not None and event.lifespan[0] - prepare_time <= time:
                event.register(self)
                event = next(events_iter, None)

            self.time = time

            yield
            index += 1


    def add_score(self, score):
        self.score += score

    def add_full_score(self, full_score):
        self.full_score += full_score

    def add_finished(self, finished=1):
        self.finished_subjects += finished

    def add_perf(self, perf, show=True, is_reversed=False):
        self.perfs.append(perf)
        if show:
            self.beatbar.set_perf(perf, is_reversed)


    def play(self, node, samplerate=None, channels=None, volume=0.0, start=None, end=None, time=None, zindex=(0,)):
        return self.beatbar.mixer.play(node, samplerate=samplerate, channels=channels,
                                              volume=volume, start=start, end=end,
                                              time=time, zindex=zindex)

    def listen(self, node, start=None, duration=None):
        self.beatbar.listen(node, start=start, duration=duration)

    def draw_sight(self, text, start=None, duration=None):
        self.beatbar.draw_sight(text, start=start, duration=duration)

    def reset_sight(self, start=None):
        self.beatbar.reset_sight(start=start)

    def draw_content(self, pos, text, start=None, duration=None, zindex=(0,)):
        return self.beatbar.draw_content(pos, text, start=start, duration=duration, zindex=zindex)

    def on_before_render(self, node):
        node = dn.pipe(dn.branch(lambda a:a[1:], node), lambda a:a[0])
        return self.beatbar.renderer.add_drawer(node, zindex=())

