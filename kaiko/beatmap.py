import os
import datetime
import contextlib
from typing import List, Tuple, Dict, Optional, Union
from collections import OrderedDict
import threading
import numpy
import audioread
from . import beatbar
from .beatbar import PerformanceGrade, Performance, Beatbar, BeatbarSettings
from . import cfg
from . import datanodes as dn
from . import tui


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
        time, width = yield

        while time < self.time:
            time, width = yield

        if self.flip is None:
            field.beatbar.bar_flip = not field.beatbar.bar_flip
        else:
            field.beatbar.bar_flip = self.flip

        time, width = yield

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

    def approach(self, field):
        if self.sound is not None:
            field.play(self.sound, time=self.time, volume=self.volume)

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
        self.number = max(int(length * density), 1)
        self.is_finished = False
        self.score = 0

        self.times = [beatmap.time(beat+i/density) for i in range(self.number)]
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
                field.play(self.sound, time=time, volume=self.volume)
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


# Game
class BeatmapSettings(metaclass=cfg.Configurable):
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
    soft_sound: str = f"{BASE_DIR}/samples/soft.wav" # pulse(freq=830.61, decay_time=0.03, amplitude=0.5)
    loud_approach_appearance:  Union[str, Tuple[str, str]] = "■"
    loud_wrong_appearance:     Union[str, Tuple[str, str]] = "⬚"
    loud_sound: str = f"{BASE_DIR}/samples/loud.wav" # pulse(freq=1661.2, decay_time=0.03, amplitude=1.0)
    incr_approach_appearance:  Union[str, Tuple[str, str]] = "⬒"
    incr_wrong_appearance:     Union[str, Tuple[str, str]] = "⬚"
    incr_sound: str = f"{BASE_DIR}/samples/incr.wav" # pulse(freq=1661.2, decay_time=0.03, amplitude=1.0)
    roll_rock_appearance:      Union[str, Tuple[str, str]] = "◎"
    roll_rock_sound: str = f"{BASE_DIR}/samples/rock.wav" # pulse(freq=1661.2, decay_time=0.01, amplitude=0.5)
    spin_disk_appearances:     Union[List[str], List[Tuple[str, str]]] = ["◴", "◵", "◶", "◷"]
    spin_finishing_appearance: Union[str, Tuple[str, str]] = "☺"
    spin_finish_sustain_time: float = 0.1
    spin_disk_sound: str = f"{BASE_DIR}/samples/disk.wav" # pulse(freq=1661.2, decay_time=0.01, amplitude=1.0)

class Beatmap:
    def __init__(self, path=".", info="", audio=None, volume=0.0, offset=0.0, tempo=60.0):
        self.path = path
        self.info = info
        self.audio = audio
        self.volume = volume
        self.offset = offset
        self.tempo = tempo
        self.settings = BeatmapSettings()

    def time(self, beat):
        return self.offset + beat*60/self.tempo

    def beat(self, time):
        return (time - self.offset)*self.tempo/60

    def dtime(self, beat, length):
        return self.time(beat+length) - self.time(beat)

    def build_events(self):
        raise NotImplementedError

class GameplaySettings(metaclass=cfg.Configurable):
    beatbar: BeatbarSettings

    ## Controls:
    leadin_time: float = 1.0
    skip_time: float = 8.0
    load_time: float = 0.5
    prepare_time: float = 0.1
    tickrate: float = 60.0

    # PlayFieldSkin:
    icon_templates: List[str] = ["{spectrum:^8s}"]
    header_templates: List[str] = ["{score:05d}/{full_score:05d}"]
    footer_templates: List[str] = ["{progress:>6.1%}|{time:%M:%S}"]

    spec_width: int = 6
    spec_decay_time: float = 0.01
    spec_time_res: float = 0.0116099773 # hop_length = 512 if samplerate == 44100
    spec_freq_res: float = 21.5332031 # win_length = 512*4 if samplerate == 44100

class BeatmapPlayer:
    def __init__(self, beatmap, settings=None):
        self.beatmap = beatmap

        self.settings = GameplaySettings()
        if settings is not None:
            cfg.config_read(open(settings, 'r'), main=self.settings)

    def prepare(self, output_samplerate, output_nchannels):
        # prepare events
        self.events = self.beatmap.build_events()
        self.events.sort(key=lambda e: e.lifespan[0])

        leadin_time = self.settings.leadin_time
        events_start_time = min((event.lifespan[0] - leadin_time for event in self.events), default=0.0)
        events_end_time   = max((event.lifespan[1] + leadin_time for event in self.events), default=0.0)

        # prepare music
        if self.beatmap.audio is None:
            self.audionode = None
            self.duration = 0.0
            self.volume = 0.0
        else:
            audiopath = os.path.join(self.beatmap.path, self.beatmap.audio)
            with audioread.audio_open(audiopath) as file:
                self.duration = file.duration
            self.audionode = dn.DataNode.wrap(dn.load_sound(audiopath, samplerate=output_samplerate,
                                                                       channels=output_nchannels))
            self.volume = self.beatmap.volume

        self.start_time = min(events_start_time, 0.0)
        self.end_time = max(events_end_time, self.duration)

        # initialize game state
        self.total_subjects = sum(event.is_subject for event in self.events)
        self.finished_subjects = 0
        self.full_score = 0
        self.score = 0

        self.perfs = []
        self.time = datetime.time(0, 0, 0)
        self.spectrum = "\u2800"*self.settings.spec_width

        # icon/header/footer handlers
        icon_templates = self.settings.icon_templates
        header_templates = self.settings.header_templates
        footer_templates = self.settings.footer_templates

        def fit(templates, ran):
            status = self.get_status()
            for template in templates:
                text = template.format(**status)
                text_ran, _ = tui.textrange1(ran.start, text)
                if ran.start <= text_ran.start and text_ran.stop <= ran.stop:
                    break
            return text

        self.icon_func = lambda time, ran: fit(icon_templates, ran)
        self.header_func = lambda time, ran: fit(header_templates, ran)
        self.footer_func = lambda time, ran: fit(footer_templates, ran)

        return abs(self.start_time)

    @contextlib.contextmanager
    def execute(self, manager):
        tickrate = self.settings.tickrate
        samplerate = self.settings.beatbar.mixer.output_samplerate
        nchannels = self.settings.beatbar.mixer.output_channels
        time_shift = self.prepare(samplerate, nchannels)
        load_time = self.settings.load_time
        ref_time = load_time + time_shift

        beatbar_knot, self.beatbar = Beatbar.create(self.settings.beatbar, manager, ref_time)

        # play music
        if self.audionode is not None:
            self.beatbar.mixer.play(self.audionode, volume=self.volume, time=0.0, zindex=(-3,))

        # register handlers
        self.beatbar.mixer.add_effect(self._spec_handler(), zindex=(-1,))
        self.beatbar.current_icon.set(self.icon_func)
        self.beatbar.current_header.set(self.header_func)
        self.beatbar.current_footer.set(self.footer_func)

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
        tickrate = self.settings.tickrate
        prepare_time = self.settings.prepare_time

        yield
        index = 0

        while True:
            time = index / tickrate + start_time

            if self.end_time <= time:
                return

            while event is not None and event.lifespan[0] - prepare_time <= time:
                event.register(self)
                event = next(events_iter, None)

            time = int(max(0.0, time)) # datetime cannot be negative
            self.time = datetime.time(time//3600, time%3600//60, time%60)

            yield
            index += 1


    def get_status(self):
        return dict(
            full_score=self.full_score,
            score=self.score,
            progress=self.finished_subjects/self.total_subjects if self.total_subjects>0 else 1.0,
            time=self.time,
            spectrum=self.spectrum,
            )

    def _spec_handler(self):
        spec_width = self.settings.spec_width
        samplerate = self.settings.beatbar.mixer.output_samplerate
        nchannels = self.settings.beatbar.mixer.output_channels
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
                    try:
                        J = node.send(data)
                    except StopIteration:
                        return

                    vols = [max(0.0, prev-decay, min(1.0, volume_of(J[slic])))
                            for slic, prev in zip(slices, vols)]
                    self.spectrum = "".join(map(draw_bar, vols[0::2], vols[1::2]))

        return dn.pipe(lambda a:a[0], dn.branch(dn.unchunk(draw_spectrum(), (hop_length, nchannels))))


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

