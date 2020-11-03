import os
import time as timer
from enum import Enum
import functools
import inspect
from typing import List, Tuple, Optional, Union
from collections import OrderedDict
import numpy
import audioread
from . import cfg
from . import realtime_analysis as ra
from .beatsheet import K_AIKO_STD, OSU


# scripts
class Event:
    # lifespan, zindex
    # __init__(beatmap, context, *args, **kwargs)
    # play(mixer, time)
    # draw(field, time)

    @staticmethod
    @ra.DataNode.from_generator
    def drip(events):
        it = iter(sorted(events, key=lambda e: e.lifespan))
        waiting = next(it, None)
        buffer = []

        time = yield

        while True:
            for playing in list(buffer):
                if playing.lifespan[1] < time:
                    buffer.remove(playing)

            while waiting is not None and waiting.lifespan[0] < time:
                buffer.append(waiting)
                waiting = next(it, None)

            time = yield list(buffer)

class Text(Event):
    zindex = -2

    def __init__(self, beatmap, context, text=None, sound=None, *, beat, speed=None):
        self.time = beatmap.time(beat)
        self.text = text

        if speed is None:
            speed = context.get("speed", 1.0)

        self.speed = speed
        if sound is not None:
            self.sound, self.samplerate = beatmap.load_audio(sound, path=beatmap.path)
        else:
            self.sound = None
            self.samplerate = None
        self.is_played = False

    @property
    def lifespan(self):
        cross_time = 1.0 / abs(0.5 * self.speed)
        return (self.time-cross_time, self.time+cross_time)

    def play(self, mixer, time):
        if self.sound is not None and not self.is_played:
            self.is_played = True
            mixer.play(self.sound, self.samplerate, delay=self.time-time)

    def draw(self, field, time):
        if self.text is not None:
            pos = (self.time - time) * 0.5 * self.speed
            field.draw_bar(pos, self.text)

def set_context(beatmap, context, **kw):
    context.update(**kw)

# targets
class Target(Event):
    # lifespan, range, score, full_score, is_finished
    # __init__(beatmap, context, *args, **kwargs)
    # hit(time, strength)
    # finish()
    # play(mixer, time)
    # draw(field, time)
    # draw_judging(field, time)
    # draw_hitting(field, time)

    @property
    def zindex(self):
        return -1 if self.is_finished else 1

    def draw_judging(self, field, time): pass
    def draw_hitting(self, field, time): pass

class Performance(Enum):
    MISS               = ("Miss"                      , False, 0)
    GREAT              = ("Great"                     , False, 10)
    LATE_GOOD          = ("Late Good"                 , False, 5)
    EARLY_GOOD         = ("Early Good"                , False, 5)
    LATE_BAD           = ("Late Bad"                  , False, 3)
    EARLY_BAD          = ("Early Bad"                 , False, 3)
    LATE_FAILED        = ("Late Failed"               , False, 0)
    EARLY_FAILED       = ("Early Failed"              , False, 0)
    GREAT_WRONG        = ("Great but Wrong Key"       , True,  5)
    LATE_GOOD_WRONG    = ("Late Good but Wrong Key"   , True,  3)
    EARLY_GOOD_WRONG   = ("Early Good but Wrong Key"  , True,  3)
    LATE_BAD_WRONG     = ("Late Bad but Wrong Key"    , True,  1)
    EARLY_BAD_WRONG    = ("Early Bad but Wrong Key"   , True,  1)
    LATE_FAILED_WRONG  = ("Late Failed but Wrong Key" , True,  0)
    EARLY_FAILED_WRONG = ("Early Failed but Wrong Key", True,  0)

    @classmethod
    def get_full_score(clz):
        return max(perf.score for perf in clz)

    def __init__(self, description, is_wrong, score):
        self.description = description
        self.is_wrong = is_wrong
        self.score = score

    def __repr__(self):
        return "Performance." + self.name

    @staticmethod
    def judge(err, is_correct_key, tolerances):
        abs_err = abs(err)
        too_late = err > 0

        if abs_err < tolerances[0]: # great
            if is_correct_key:
                perf = Performance.GREAT
            else:
                perf = Performance.GREAT_WRONG

        elif abs_err < tolerances[1]: # good
            if is_correct_key:
                perf = Performance.LATE_GOOD         if too_late else Performance.EARLY_GOOD
            else:
                perf = Performance.LATE_GOOD_WRONG   if too_late else Performance.EARLY_GOOD_WRONG

        elif abs_err < tolerances[2]: # bad
            if is_correct_key:
                perf = Performance.LATE_BAD          if too_late else Performance.EARLY_BAD
            else:
                perf = Performance.LATE_BAD_WRONG    if too_late else Performance.EARLY_BAD_WRONG

        elif abs_err < tolerances[3]: # failed
            if is_correct_key:
                perf = Performance.LATE_FAILED       if too_late else Performance.EARLY_FAILED
            else:
                perf = Performance.LATE_FAILED_WRONG if too_late else Performance.EARLY_FAILED_WRONG

        else: # miss
            perf = None

        return perf

    def draw(self, field, flipped, appearances):
        i = (
            Performance.MISS,
            Performance.GREAT,
            Performance.LATE_GOOD,
            Performance.EARLY_GOOD,
            Performance.LATE_BAD,
            Performance.EARLY_BAD,
            Performance.LATE_FAILED,
            Performance.EARLY_FAILED,
            Performance.GREAT_WRONG,
            Performance.LATE_GOOD_WRONG,
            Performance.EARLY_GOOD_WRONG,
            Performance.LATE_BAD_WRONG,
            Performance.EARLY_BAD_WRONG,
            Performance.LATE_FAILED_WRONG,
            Performance.EARLY_FAILED_WRONG,
            ).index(self)
        if field.is_flipped:
            flipped = not flipped
        j = 0 if not flipped else 1

        field.draw_bar(0.0, appearances[i][j])

class OneshotTarget(Target):
    # time, speed, volume, perf, is_played, sound, samplerate
    # appearances: (approach, wrong)
    # hit(time, strength)

    full_score = Performance.get_full_score()

    def __init__(self, beatmap, context, *, beat, speed=None, volume=None):
        self.tolerances = (
            beatmap.settings.great_tolerance,
            beatmap.settings.good_tolerance,
            beatmap.settings.bad_tolerance,
            beatmap.settings.failed_tolerance,
            )
        self.performance_appearances = (
            beatmap.settings.miss_appearance,
            beatmap.settings.great_appearance,
            beatmap.settings.late_good_appearance,
            beatmap.settings.early_good_appearance,
            beatmap.settings.late_bad_appearance,
            beatmap.settings.early_bad_appearance,
            beatmap.settings.late_failed_appearance,
            beatmap.settings.early_failed_appearance,
            beatmap.settings.great_wrong_appearance,
            beatmap.settings.late_good_wrong_appearance,
            beatmap.settings.early_good_wrong_appearance,
            beatmap.settings.late_bad_wrong_appearance,
            beatmap.settings.early_bad_wrong_appearance,
            beatmap.settings.late_failed_wrong_appearance,
            beatmap.settings.early_failed_wrong_appearance,
            )

        if speed is None:
            speed = context.get("speed", 1.0)
        if volume is None:
            volume = context.get("volume", 0.0)

        self.time = beatmap.time(beat)
        self.speed = speed
        self.volume = volume
        self.perf = None
        self.is_played = False

    @property
    def range(self):
        return (self.time - self.tolerances[3], self.time + self.tolerances[3])

    @property
    def score(self):
        return self.perf.score if self.perf is not None else 0

    @property
    def is_finished(self):
        return self.perf is not None

    def finish(self):
        self.perf = Performance.MISS

    def hit(self, time, strength, is_correct_key):
        self.perf = Performance.judge(time - self.time, is_correct_key, self.tolerances)

    @property
    def lifespan(self):
        cross_time = 1.0 / abs(0.5 * self.speed)
        return (self.time - cross_time, self.time + cross_time)

    def play(self, mixer, time):
        if not self.is_played:
            self.is_played = True
            mixer.play(self.sound, self.samplerate, delay=self.time-time, volume=self.volume)

    def draw(self, field, time):
        if self.perf in (None, Performance.MISS): # approaching or miss
            appearance = self.appearances[0]
            if isinstance(appearance, tuple):
                is_flipped = (self.speed < 0) != field.is_flipped
                appearance = appearance[0] if not is_flipped else appearance[1]

            pos = (self.time - time) * 0.5 * self.speed
            field.draw_bar(pos, appearance)

        elif self.perf.is_wrong: # wrong key
            appearance = self.appearances[1]
            if isinstance(appearance, tuple):
                is_flipped = (self.speed < 0) != field.is_flipped
                appearance = appearance[0] if not is_flipped else appearance[1]

            pos = (self.time - time) * 0.5 * self.speed
            field.draw_bar(pos, appearance)

        else: # correct key
            pass

    def draw_hitting(self, field, time):
        self.perf.draw(field, self.speed < 0, self.performance_appearances)

class Soft(OneshotTarget):
    def __init__(self, beatmap, context, *, beat, speed=None, volume=None):
        super().__init__(beatmap, context, beat=beat, speed=speed, volume=volume)
        self.appearances = (
            beatmap.settings.soft_approach_appearance,
            beatmap.settings.soft_wrong_appearance,
            )
        self.sound, self.samplerate = beatmap.load_audio(beatmap.settings.soft_sound)
        self.threshold = beatmap.settings.soft_threshold

    def hit(self, time, strength):
        super().hit(time, strength, strength < self.threshold)

class Loud(OneshotTarget):
    def __init__(self, beatmap, context, *, beat, speed=None, volume=None):
        super().__init__(beatmap, context, beat=beat, speed=speed, volume=volume)
        self.appearances = (
            beatmap.settings.loud_approach_appearance,
            beatmap.settings.loud_wrong_appearance,
            )
        self.sound, self.samplerate = beatmap.load_audio(beatmap.settings.loud_sound)
        self.threshold = beatmap.settings.loud_threshold

    def hit(self, time, strength):
        super().hit(time, strength, strength >= self.threshold)

class IncrGroup:
    def __init__(self, threshold=0.0, total=0):
        self.threshold = threshold
        self.total = total

    def hit(self, strength):
        self.threshold = max(self.threshold, strength)

class Incr(OneshotTarget):
    def __init__(self, beatmap, context, group=None, *, beat, speed=None, volume=None):
        super().__init__(beatmap, context, beat=beat, speed=speed, volume=volume)
        self.appearances = (
            beatmap.settings.incr_approach_appearance,
            beatmap.settings.incr_wrong_appearance,
            )
        self.sound, self.samplerate = beatmap.load_audio(beatmap.settings.incr_sound)
        self.incr_threshold = beatmap.settings.incr_threshold

        if "incrs" not in context:
            context["incrs"] = OrderedDict()

        group_key = group
        if group_key is None:
            # determine group of incr note according to the context
            for key, (_, last_beat) in reversed(context["incrs"].items()):
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
        super().hit(time, strength, strength >= self.group.threshold + self.incr_threshold)
        self.group.hit(strength)

    def play(self, mixer, time):
        if not self.is_played:
            self.is_played = True
            volume = self.volume + numpy.log10(0.2 + 0.8 * (self.count-1)/self.group.total) * 20
            mixer.play(self.sound, self.samplerate, delay=self.time-time, volume=volume)

class Roll(Target):
    def __init__(self, beatmap, context, density=2, *, beat, length, speed=None, volume=None):
        self.tolerance = beatmap.settings.roll_tolerance
        self.rock_appearance = beatmap.settings.roll_rock_appearance
        self.sound, self.samplerate = beatmap.load_audio(beatmap.settings.rock_sound)

        self.time = beatmap.time(beat)

        self.number = int(length * density)
        self.end = beatmap.time(beat+length)
        self.times = [beatmap.time(beat+i/density) for i in range(self.number)]

        if speed is None:
            speed = context.get("speed", 1.0)
        if volume is None:
            volume = context.get("volume", 0.0)

        self.speed = speed
        self.volume = volume
        self.roll = 0
        self.is_finished = False
        self.is_played = False

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

    def hit(self, time, strength):
        self.roll += 1

    def finish(self):
        self.is_finished = True

    @property
    def lifespan(self):
        cross_time = 1.0 / abs(0.5 * self.speed)
        return (self.time - cross_time, self.end + cross_time)

    def play(self, mixer, time):
        if not self.is_played:
            self.is_played = True

            for t in self.times:
                mixer.play(self.sound, self.samplerate, delay=t-time, volume=self.volume)

    def draw(self, field, time):
        appearance = self.rock_appearance
        if isinstance(appearance, tuple):
            is_flipped = (self.speed < 0) != field.is_flipped
            appearance = appearance[0] if not is_flipped else appearance[1]

        for r, t in enumerate(self.times):
            if r > self.roll-1:
                pos = (t - time) * 0.5 * self.speed
                field.draw_bar(pos, appearance)

class Spin(Target):
    full_score = 10

    def __init__(self, beatmap, context, density=2, *, beat, length, speed=None, volume=None):
        self.tolerance = beatmap.settings.spin_tolerance
        self.disk_appearances = beatmap.settings.spin_disk_appearances
        self.finishing_appearance = beatmap.settings.spin_finishing_appearance
        self.sound, self.samplerate = beatmap.load_audio(beatmap.settings.disk_sound)

        self.time = beatmap.time(beat)

        self.capacity = length * density
        self.end = beatmap.time(beat+length)
        self.times = [beatmap.time(beat+i/density) for i in range(int(self.capacity))]

        if speed is None:
            speed = context.get("speed", 1.0)
        if volume is None:
            volume = context.get("volume", 0.0)

        self.speed = speed
        self.volume = volume
        self.charge = 0.0
        self.is_finished = False
        self.is_played = False

    @property
    def range(self):
        return (self.time - self.tolerance, self.end + self.tolerance)

    @property
    def score(self):
        return self.full_score if self.charge == self.capacity else 0

    def hit(self, time, strength):
        self.charge = min(self.charge + min(1.0, strength), self.capacity)
        if self.charge == self.capacity:
            self.is_finished = True

    def finish(self):
        self.is_finished = True

    @property
    def lifespan(self):
        cross_time = 1.0 / abs(0.5 * self.speed)
        return (self.time - cross_time, self.end + cross_time)

    def play(self, mixer, time):
        if not self.is_played:
            self.is_played = True

            for t in self.times:
                mixer.play(self.sound, self.samplerate, delay=t-time, volume=self.volume)

    def draw(self, field, time):
        if self.charge < self.capacity:
            pos = 0.0
            pos += max(0.0, (self.time - time) * 0.5 * self.speed)
            pos += min(0.0, (self.end - time) * 0.5 * self.speed)
            i = int(self.charge) % len(self.disk_appearances)

            appearance = self.disk_appearances[i]
            if isinstance(appearance, tuple):
                is_flipped = (self.speed < 0) != field.is_flipped
                appearance = appearance[0] if not is_flipped else appearance[1]

            field.draw_bar(pos, appearance)

    def draw_judging(self, field, time):
        return True

    def draw_hitting(self, field, time):
        if self.charge == self.capacity:
            appearance = self.finishing_appearance
            if isinstance(appearance, tuple):
                is_flipped = (self.speed < 0) != field.is_flipped
                appearance = appearance[0] if not is_flipped else appearance[1]

            field.draw_bar(0.0, appearance)
            return True


# beatmap
class PlayField:
    def __init__(self, screen, width, spec_width, shift, is_flipped):
        self.screen = screen
        self.width = min(width, self.screen.width)
        self.spec_width = spec_width
        self.shift = shift
        self.is_flipped = is_flipped

        self.spec_offset = 1
        self.score_offset = self.spec_width + 2
        self.progress_offset = self.width - 9
        self.bar_offset = self.spec_width + 15
        self.bar_width = self.width - 24 - self.spec_width

    def __str__(self):
        return "".join(self.screen)

    def clear(self):
        self.screen.clear()

    def draw_spectrum(self, spectrum):
        self.screen.addstr(self.spec_offset, spectrum)

    def draw_score(self, score, total_score):
        self.screen.addstr(self.score_offset, "[{:>5d}/{:>5d}]".format(score, total_score))

    def draw_progress(self, progress):
        self.screen.addstr(self.progress_offset, "[{:>5.1f}%]".format(progress*100))

    def draw_bar(self, pos, str):
        pos += self.shift
        if self.is_flipped:
            pos = 1 - pos
        index = self.bar_offset + round(pos * (self.bar_width - 1))
        self.screen.addstr(index, str, slice(self.bar_offset, self.bar_offset+self.bar_width))


@cfg.configurable
class BeatmapSettings:
    # Difficulty:
    great_tolerance: float = 0.02
    good_tolerance: float = 0.06
    bad_tolerance: float = 0.10
    failed_tolerance: float = 0.14
    soft_threshold: float = 0.5
    loud_threshold: float = 0.5
    incr_threshold: float = -0.1
    roll_tolerance: float = 0.10
    spin_tolerance: float = 0.10

    # Skin:
    ## NoteSkin:
    soft_approach_appearance: Union[str, Tuple[str, str]] = "□"
    soft_wrong_appearance: Union[str, Tuple[str, str]] = "⬚"
    soft_sound: str = "samples/soft.wav" # pulse(freq=830.61, decay_time=0.03, amplitude=0.5)
    loud_approach_appearance: Union[str, Tuple[str, str]] = "■"
    loud_wrong_appearance: Union[str, Tuple[str, str]] = "⬚"
    loud_sound: str = "samples/loud.wav" # pulse(freq=1661.2, decay_time=0.03, amplitude=1.0)
    incr_approach_appearance: Union[str, Tuple[str, str]] = "⬒"
    incr_wrong_appearance: Union[str, Tuple[str, str]] = "⬚"
    incr_sound: str = "samples/incr.wav" # pulse(freq=1661.2, decay_time=0.03, amplitude=1.0)
    roll_rock_appearance: Union[str, Tuple[str, str]] = "◎"
    rock_sound: str = "samples/rock.wav" # pulse(freq=1661.2, decay_time=0.01, amplitude=0.5)
    spin_disk_appearances: Union[List[str], List[Tuple[str, str]]] = ["◴", "◵", "◶", "◷"]
    spin_finishing_appearance: Union[str, Tuple[str, str]] = "☺"
    disk_sound: str = "samples/disk.wav" # pulse(freq=1661.2, decay_time=0.01, amplitude=1.0)

    ## SightSkin:
    sight_appearances: List[str] = ["⛶", "🞎", "🞏", "🞐", "🞑", "🞒", "🞓"]
    hit_decay_time: float = 0.4
    hit_sustain_time: float = 0.1
    sight_shift: float = 0.1
    sight_flipped: bool = False

    ## PerformanceSkin:
    miss_appearance:               Tuple[str, str] = (""   , ""   )

    late_failed_appearance:        Tuple[str, str] = ("\b⟪", "  ⟫")
    late_bad_appearance:           Tuple[str, str] = ("\b⟨", "  ⟩")
    late_good_appearance:          Tuple[str, str] = ("\b«", "  »")
    great_appearance:              Tuple[str, str] = (""   , ""   )
    early_good_appearance:         Tuple[str, str] = ("  »", "\b«")
    early_bad_appearance:          Tuple[str, str] = ("  ⟩", "\b⟨")
    early_failed_appearance:       Tuple[str, str] = ("  ⟫", "\b⟪")

    late_failed_wrong_appearance:  Tuple[str, str] = ("\b⟪", "  ⟫")
    late_bad_wrong_appearance:     Tuple[str, str] = ("\b⟨", "  ⟩")
    late_good_wrong_appearance:    Tuple[str, str] = ("\b«", "  »")
    great_wrong_appearance:        Tuple[str, str] = (""   , ""   )
    early_good_wrong_appearance:   Tuple[str, str] = ("  »", "\b«")
    early_bad_wrong_appearance:    Tuple[str, str] = ("  ⟩", "\b⟨")
    early_failed_wrong_appearance: Tuple[str, str] = ("  ⟫", "\b⟪")

    ## PlayFieldSkin:
    field_width: Optional[int] = None
    spec_width: int = 5
    spec_decay_time: float = 0.01
    spec_time_res: float = 0.0116099773 # hop_length = 512 if samplerate == 44100
    spec_freq_res: float = 21.5332031 # win_length = 512*4 if samplerate == 44100

    # Gameplay:
    prepare_time: float = 1.0
    skip_time: float = 8.0
    tickrate: float = 60.0

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

        self["x"] = Soft
        self["o"] = Loud
        self["<"] = Incr
        self["%"] = Roll
        self["@"] = Spin
        self["text"] = Text
        self["context"] = set_context

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


    @functools.lru_cache(maxsize=32)
    def load_audio(self, filename, path="."):
        filename = os.path.join(path, filename)
        with audioread.audio_open(filename) as file:
            samplerate = file.samplerate
        with ra.load(filename) as filenode:
            sound = list(filenode)
        return sound, samplerate

    @property
    def total_score(self):
        return sum(target.full_score for target in self.targets)

    @property
    def score(self):
        return sum(target.score for target in self.targets)

    @property
    def progress(self):
        if len(self.targets) == 0:
            return 1.0
        return sum(1 for target in self.targets if target.is_finished) / len(self.targets)


    @ra.DataNode.from_generator
    def connect(self, mixer, detector, screen):
        # events, targets
        self.events = [event for chart in self.charts for event in chart.build_events(self)]
        self.targets = [event for event in self.events if isinstance(event, Target)]
        self.start = min([event.lifespan[0] - self.settings.prepare_time for event in self.events])
        self.end = max([event.lifespan[1] + self.settings.prepare_time for event in self.events])
        audio_offset = min(0.0, self.start)

        # audio
        if self.audio is None:
            duration = 0.0

        else:
            abspath = os.path.join(self.path, self.audio)
            with audioread.audio_open(abspath) as file:
                duration = file.duration
                channels = file.channels
                samplerate = file.samplerate
            mixer.play(ra.load(abspath), samplerate, delay=-audio_offset)

            # add spectrum show
            hop_length = round(samplerate * self.settings.spec_time_res)
            win_length = round(samplerate / self.settings.spec_freq_res)

            Dt = hop_length / samplerate
            spec = ra.pipe(ra.frame(win_length, hop_length),
                           ra.power_spectrum(win_length, samplerate=samplerate),
                           ra.draw_spectrum(self.settings.spec_width,
                                            win_length=win_length,
                                            samplerate=samplerate,
                                            decay=Dt/self.settings.spec_decay_time/4),
                           lambda s: setattr(self, "spectrum", s))
            spec = ra.unchunk(spec, chunk_shape=(hop_length, channels))
            mixer.add_effect(ra.branch(spec))

        # hit state: set virtual hitting initially to simplify coding logic
        self.judging_target = None
        self.hit_time = audio_offset - self.settings.hit_sustain_time*2
        # time - hit_time > hit_sustain_time => prevent to showing virtual hitting initially
        self.hit_strength = 0.0
        self.hit_target = None

        # screen
        self.spectrum = " "*self.settings.spec_width
        field = PlayField(screen, self.settings.field_width or screen.width,
                          self.settings.spec_width,
                          self.settings.sight_shift,
                          self.settings.sight_flipped)

        # loop
        events_dripper = Event.drip(self.events)
        target_handler = self.get_target_handler()

        with events_dripper, target_handler:
            dt = 1/self.settings.tickrate

            yield
            t0 = timer.time()
            time = audio_offset

            while time < max(duration, self.end):
                # hit targets
                while not detector.detected.empty():
                    hit_time, hit_strength = detector.detected.get()
                    hit_time += audio_offset
                    hit_strength = min(1.0, hit_strength)
                    hit_target = target_handler.send(hit_time)

                    self.hit_time = hit_time
                    self.hit_strength = hit_strength
                    self.hit_target = hit_target

                    if hit_target is not None:
                        hit_target.hit(hit_time, hit_strength)

                self.judging_target = target_handler.send(detector.time + audio_offset)

                # draw/play events
                field.clear()
                events = events_dripper.send(time)
                for event in sorted(events[::-1], key=lambda e: e.zindex):
                    event.draw(field, screen.time+audio_offset)
                    event.play(mixer, mixer.time+audio_offset)

                # draw sight
                stop_drawing = False
                if not stop_drawing and self.judging_target is not None:
                    stop_drawing = self.judging_target.draw_judging(field, time)
                if not stop_drawing and self.hit_target is not None:
                    if abs(time - self.hit_time) < self.settings.hit_sustain_time:
                        stop_drawing = self.hit_target.draw_hitting(field, time)
                if not stop_drawing:
                    self.draw_sight(field, time)

                # draw others
                field.draw_spectrum(self.spectrum)
                field.draw_score(self.score, self.total_score)
                field.draw_progress(self.progress)

                snap_time = time + dt - (timer.time() - t0 + audio_offset)
                if snap_time < 0:
                    print("underrun")
                timer.sleep(max(0.0, snap_time))
                yield
                time = timer.time() - t0 + audio_offset

    @ra.DataNode.from_generator
    def get_target_handler(self):
        targets = iter(sorted(self.targets, key=lambda e: e.range))
        target = next(targets, None)

        time = yield
        while True:
            while target is not None and (target.is_finished or time > target.range[1]):
                if not target.is_finished:
                    target.finish()
                target = next(targets, None)

            time = yield (target if target is not None and time > target.range[0] else None)

    def draw_sight(self, field, time):
        strength = self.hit_strength - (time - self.hit_time) / self.settings.hit_decay_time
        strength = max(0.0, min(1.0, strength))
        loudness = int(strength * (len(self.settings.sight_appearances) - 1))
        if abs(time - self.hit_time) < self.settings.hit_sustain_time:
            loudness = max(1, loudness)
        field.draw_bar(0.0, self.settings.sight_appearances[loudness])


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
    def __init__(self, *psargs, **kwargs):
        self.bound = self.__signature__.bind(*psargs, **kwargs)

    def __str__(self):
        psargs_str = [repr(value) for value in self.bound.args]
        kwargs_str = [key+"="+repr(value) for key, value in self.bound.kwargs.items()]
        args_str = ", ".join([*psargs_str, *kwargs_str])
        return f"{self.symbol}({args_str})"

    def create(self, beatmap, context):
        return self.builder(beatmap, context, *self.bound.args, **self.bound.kwargs)


K_AIKO_STD_FORMAT = K_AIKO_STD(Beatmap, NoteChart)
OSU_FORMAT = OSU(Beatmap, NoteChart)

