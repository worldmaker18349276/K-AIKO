import os
import datetime
from enum import Enum
import inspect
from typing import List, Tuple, Dict, Optional, Union
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
    # # selected properties:
    # full_score, score, is_finished, perfs
    pass

class Text(Event):
    def __init__(self, beatmap, context, text=None, sound=None, *, beat, speed=None):
        if speed is None:
            speed = context.get('speed', 1.0)
        if sound is not None:
            sound = os.path.join(beatmap.path, sound)

        self.time = beatmap.time(beat)
        self.speed = speed
        self.text = text
        self.sound = sound

        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.time + travel_time)
        self.pos = lambda time, width: (self.time-time) * 0.5 * self.speed

    def register(self, field):
        if self.sound is not None:
            field.play(self.sound, time=self.time)

        if self.text is not None:
            field.draw_text(self.pos, self.text, start=self.lifespan[0],
                            duration=self.lifespan[1]-self.lifespan[0], zindex=(-2, -self.time))

# scripts
class Flip(Event):
    def __init__(self, beatmap, context, flip=None, *, beat):
        self.time = beatmap.time(beat)
        self.flip = flip
        self.lifespan = (self.time, self.time)

    def register(self, field):
        field.console.add_drawer(self._node(field), zindex=())

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
        field.console.add_drawer(self._node(field), zindex=())

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
        field.console.add_drawer(self._node(field), zindex=())

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

class OneshotTarget(Target):
    # time, speed, volume, perf, sound
    # approach_appearance, wrong_appearance
    # hit(field, time, strength)

    def __init__(self, beatmap, context, *, beat, speed=None, volume=None):
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
        self.pos = lambda time, width: (self.time-time) * 0.5 * self.speed
        tol = beatmap.settings.failed_tolerance
        self.range = (self.time-tol, self.time+tol)
        self._scores = beatmap.settings.performances_scores
        self.full_score = beatmap.settings.performances_max_score

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

        field.draw_target(self, self.pos, self.approach_appearance,
                          start=self.lifespan[0], duration=self.lifespan[1]-self.lifespan[0], key=self)
        field.reset_sight(start=self.range[0])

    def hit(self, field, time, strength, is_correct_key=True):
        perf = field.judger.judge(self.time, time, is_correct_key)
        field.judger.render(perf, field, self.speed < 0)
        self.finish(field, perf)

    def finish(self, field, perf=None):
        if perf is None:
            perf = field.judger.judge(self.time)
        self.perf = perf

        if self.perf.is_miss:
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
        self.volume = 0.0

    def hit(self, strength):
        self.threshold = max(self.threshold, strength)

class Incr(OneshotTarget):
    def __init__(self, beatmap, context, group=None, *, beat, speed=None, volume=None):
        super().__init__(beatmap, context, beat=beat, speed=speed)

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
        if volume is not None:
            group.volume = volume
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
    def __init__(self, beatmap, context, density=2, *, beat, length, speed=None, volume=None):
        if speed is None:
            speed = context.get('speed', 1.0)
        if volume is None:
            volume = context.get('volume', 0.0)

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

        self.times = [beatmap.time(beat+i/density) for i in range(self.number)]
        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.end + travel_time)
        self.perfs = []
        self.range = (self.time - self.tolerance, self.end - self.tolerance)
        self.full_score = self.number * self.rock_score

    def get_pos(self, index):
        return lambda time, width: (self.times[index]-time) * 0.5 * self.speed

    @property
    def score(self):
        if self.roll < self.number:
            return self.roll * self.rock_score
        elif self.roll < 2*self.number:
            return (2*self.number - self.roll) * self.rock_score
        else:
            return 0

    def approach(self, field):
        for i, time in enumerate(self.times):
            if self.sound is not None:
                field.play(self.sound, time=time, volume=self.volume)
            field.draw_target(self, self.get_pos(i), self.rock_appearance,
                              start=self.lifespan[0], duration=self.lifespan[1]-self.lifespan[0], key=(self, i))
        field.reset_sight(start=self.range[0])

    def hit(self, field, time, strength):
        self.roll += 1
        if self.roll <= self.number:
            perf = field.judger.judge(self.times[self.roll-1], time, True)
            self.perfs.append(perf)
            field.remove_target(key=(self, self.roll-1))

    def finish(self, field):
        self.is_finished = True
        for time in self.times[self.roll:]:
            perf = field.judger.judge(time)
            self.perfs.append(perf)

class Spin(Target):
    def __init__(self, beatmap, context, density=2, *, beat, length, speed=None, volume=None):
        if speed is None:
            speed = context.get('speed', 1.0)
        if volume is None:
            volume = context.get('volume', 0.0)

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

        self.times = [beatmap.time(beat+i/density) for i in range(int(self.capacity))]
        travel_time = 1.0 / abs(0.5 * self.speed)
        self.lifespan = (self.time - travel_time, self.end + travel_time)
        self.pos = lambda time, width: (max(0.0, self.time-time) + min(0.0, self.end-time)) * 0.5 * self.speed
        self.range = (self.time - self.tolerance, self.end + self.tolerance)

    @property
    def score(self):
        if not self.is_finished:
            return int(self.full_score * self.charge / self.capacity)
        else:
            return self.full_score if self.charge == self.capacity else 0

    def approach(self, field):
        for time in self.times:
            if self.sound is not None:
                field.play(self.sound, time=time, volume=self.volume)

        appearance = lambda time, width: self.disk_appearances[int(self.charge) % len(self.disk_appearances)]
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
class PlayFieldSettings:
    ## Controls:
    leadin_time: float = 1.0
    skip_time: float = 8.0
    tickrate: float = 60.0
    prepare_time: float = 0.1

    ## PlayFieldSkin:
    spec_width: int = 5
    score_width: int = 13
    progress_width: int = 14
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

class PlayField:
    settings: PlayFieldSettings = PlayFieldSettings()

    def __init__(self, beatmap, config=None):
        self.beatmap = beatmap
        self.judger = PerformanceJudger(beatmap.settings)

        if config is not None:
            cfg.config_read(open(config, 'r'), main=self.settings)

    def get_full_score(self):
        return sum(getattr(event, 'full_score', 0) for event in self.events
                   if getattr(event, 'is_finished', True)) * self.score_scale

    def get_score(self):
        return sum(getattr(event, 'score', 0) for event in self.events) * self.score_scale

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

        total_score = sum(getattr(event, 'full_score', 0) for event in self.events)
        self.score_scale = 65536 / total_score
        self.full_score = self.get_full_score()
        self.score = self.get_score()
        self.progress = self.get_progress()
        self.time = datetime.time(0, 0, 0)

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
        self.score_format = "{score:0%d.0f}" % (score_width_-score_width_//2)
        self.full_score_format = "{full_score:0%d.0f}" % (score_width_//2)
        self.progress_format = "{progress:>%d.%d%%}" % (max(0, progress_width-8), max(0, progress_width-13))
        self.time_format = "{time:%M:%S}"

        # game loop
        tickrate = self.settings.tickrate
        prepare_time = self.settings.prepare_time
        time_shift = prepare_time + max(-events_start_time, 0.0)

        with dn.tick(1/tickrate, prepare_time, -time_shift) as timer:
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
            self.console.add_drawer(self._status_handler(), zindex=(-3,), key='status')
            self.console.add_drawer(self._sight_handler(), zindex=(2,), key='sight')

            # register events
            events_iter = iter(self.events)
            event = next(events_iter, None)

            yield
            for time in timer:
                if max(events_end_time, duration) <= time:
                    break

                while event is not None and event.lifespan[0] <= time + prepare_time:
                    event.register(self)
                    event = next(events_iter, None)

                self.full_score = self.get_full_score()
                self.score = self.get_score()
                self.progress = self.get_progress()
                time = int(max(0.0, time))
                self.time = datetime.time(time//3600, time%3600//60, time%60)

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

            score_text = self.score_format.format(score=self.score)
            full_score_text = self.full_score_format.format(full_score=self.full_score)
            score_start, _, _ = self.score_mask.indices(screen.width)
            screen.addstr(score_start, f"[{score_text}/{full_score_text}]", self.score_mask)

            progress_text = self.progress_format.format(progress=self.progress)
            time_text = self.time_format.format(time=self.time)
            progress_start, _, _ = self.progress_mask.indices(screen.width)
            screen.addstr(progress_start, f"[{progress_text}|{time_text}]", self.progress_mask)

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
        self.console.add_drawer(node, zindex=zindex, key=('text', key))
        return key

    def remove_text(self, key):
        self.console.remove_drawer(key=('text', key))

    def draw_target(self, target, pos, text, start=None, duration=None, key=None):
        if key is None:
            key = object()
        node = self._bar_node(pos, text, start, duration)
        zindex = lambda: (0, not target.is_finished, -target.range[0])
        self.console.add_drawer(node, zindex=zindex, key=('target', key))
        return key

    def remove_target(self, key):
        self.console.remove_drawer(key=('target', key))


# Judger
def braille_scatter(width, height, xy, xlim, ylim):
    dx = (xlim[1] - xlim[0])/(width*2-1)
    dy = (ylim[1] - ylim[0])/(height*4-1)

    graph = numpy.zeros((height*4, width*2), dtype=bool)
    for x, y in xy:
        i = round((y-ylim[0])/dy)
        j = round((x-xlim[0])/dx)
        if i in range(height*4) and j in range(width*2):
            graph[i,j] = True

    graph = graph.reshape(height, 4, width, 2)
    block = 2**numpy.array([0, 3, 1, 4, 2, 5, 6, 7]).reshape(1, 4, 1, 2)
    code = 0x2800 + (graph * block).sum(axis=(1, 3))
    strs = numpy.concatenate((code, [[ord("\n")]]*height), axis=1).astype('i2').tostring().decode('utf-16')

    return strs

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

class PerformanceJudger:
    def __init__(self, settings):
        self.perfect_tolerance = settings.perfect_tolerance
        self.good_tolerance    = settings.good_tolerance
        self.bad_tolerance     = settings.bad_tolerance
        self.failed_tolerance  = settings.failed_tolerance
        self.appearances = settings.performances_appearances
        self.sustain_time = settings.performance_sustain_time

    def judge(self, time, hit_time=None, is_correct_key=True):
        if hit_time is None:
            return Performance(PerformanceGrade((None, None)), time, None)

        err = hit_time - time
        abs_err = abs(err)
        is_wrong = not is_correct_key

        if abs_err < self.perfect_tolerance:
            shift = 0
        elif abs_err < self.good_tolerance:
            shift = 1
        elif abs_err < self.bad_tolerance:
            shift = 2
        else: # failed
            shift = 3

        if err < 0:
            shift = -shift

        return Performance(PerformanceGrade((shift, is_wrong)), time, err)

    def render(self, perf, field, is_reversed):
        appearance = self.appearances[perf.grade]
        if is_reversed:
            appearance = appearance[::-1]
        field.draw_text(field.sight_shift, appearance, duration=self.sustain_time, zindex=(1,), key='perf_hint')

    def show_analyze(self, events):
        width = int(os.popen("stty size", 'r').read().split()[1])
        perfs = [perf for event in events for perf in getattr(event, 'perfs', ())]
        emax = self.failed_tolerance
        start = min((perf.time for perf in perfs), default=0.0)
        end   = max((perf.time for perf in perfs), default=0.0)

        grad_minwidth = 15
        stat_minwidth = 15
        scat_height = 7
        acc_height = 2

        # grades infos
        grades = [perf.grade for perf in perfs if not perf.is_miss]
        miss_count = sum(perf.is_miss for perf in perfs)
        failed_count    = sum(not grade.is_wrong and abs(grade.shift) == 3 for grade in grades)
        bad_count       = sum(not grade.is_wrong and abs(grade.shift) == 2 for grade in grades)
        good_count      = sum(not grade.is_wrong and abs(grade.shift) == 1 for grade in grades)
        perfect_count   = sum(not grade.is_wrong and abs(grade.shift) == 0 for grade in grades)
        failed_wrong_count  = sum(grade.is_wrong and abs(grade.shift) == 3 for grade in grades)
        bad_wrong_count     = sum(grade.is_wrong and abs(grade.shift) == 2 for grade in grades)
        good_wrong_count    = sum(grade.is_wrong and abs(grade.shift) == 1 for grade in grades)
        perfect_wrong_count = sum(grade.is_wrong and abs(grade.shift) == 0 for grade in grades)
        accuracy = sum(2.0**(-abs(grade.shift)) for grade in grades) / len(perfs)
        mistakes = sum(grade.is_wrong for grade in grades) / len(grades)

        grad_infos = [
            f"   miss: {   miss_count}",
            f" failed: { failed_count}+{ failed_wrong_count}",
            f"    bad: {    bad_count}+{    bad_wrong_count}",
            f"   good: {   good_count}+{   good_wrong_count}",
            f"perfect: {perfect_count}+{perfect_wrong_count}",
            "",
            "",
            f"accuracy: {accuracy*100:.1f}%",
            f"mistakes: {mistakes*100:.2f}%",
            "",
            ]

        # statistics infos
        errors = [(perf.time, perf.err) for perf in perfs if not perf.is_miss]
        misses = [perf.time for perf in perfs if perf.is_miss]
        err = sum(abs(err) for _, err in errors) / len(errors)
        ofs = sum(err for _, err in errors) / len(errors)
        dev = (sum((err-ofs)**2 for _, err in errors) / len(errors))**0.5

        stat_infos = [
            f"err={err*1000:.3f} ms",
            f"ofs={ofs*1000:+.3f} ms",
            f"dev={dev*1000:.3f} ms",
            ]

        # timespan
        def minsec(sec):
            sec = round(sec)
            sgn = +1 if sec >= 0 else -1
            min, sec = divmod(abs(sec), 60)
            min *= sgn
            return f"{min}:{sec:02d}"

        timespan = f"╡{minsec(start)} ~ {minsec(end)}╞"

        # layout
        grad_width = max(grad_minwidth, len(timespan), max(len(info_str) for info_str in grad_infos))
        stat_width = max(stat_minwidth, max(len(info_str) for info_str in stat_infos))
        scat_width = width - grad_width - stat_width - 4

        grad_top = "═"*grad_width
        grad_bot = timespan.center(grad_width, "═")
        scat_top = scat_bot = "═"*scat_width
        stat_top = stat_bot = "═"*stat_width
        grad_infos = [info_str.ljust(grad_width) for info_str in grad_infos]
        stat_infos = [info_str.ljust(stat_width) for info_str in stat_infos]

        # discretize data
        dx = (end - start)/(scat_width*2-1)
        dy = 2*emax/(scat_height*4-1)
        data = numpy.zeros((scat_height*4+1, scat_width*2), dtype=int)
        for time, err in errors:
            i = round((err+emax)/dy)
            j = round((time-start)/dx)
            if i in range(scat_height*4) and j in range(scat_width*2):
                data[i,j] += 1
        for time in misses:
            j = round((time-start)/dx)
            if j in range(scat_width*2):
                data[-1,j] += 1

        braille_block = 2**numpy.array([0, 3, 1, 4, 2, 5, 6, 7]).reshape(1, 4, 1, 2)

        # plot scatter
        scat_data = (data[:-1,:] > 0).reshape(scat_height, 4, scat_width, 2)
        scat_code = 0x2800 + (scat_data * braille_block).sum(axis=(1, 3)).astype('i2')
        scat_graph = [line.tostring().decode('utf-16') for line in scat_code]
        miss_data = (data[-1,:] > 0).reshape(scat_width, 2)
        miss_code = (miss_data * [1, 2]).sum(axis=-1)
        miss_graph = "".join("─╾╼━"[code] for code in miss_code)

        # plot statistics
        stat_data = data[:-1,:].sum(axis=1)
        stat_level = numpy.linspace(0, numpy.max(stat_data), stat_width*2, endpoint=False)
        stat_data = (stat_level[None,:] < stat_data[:,None]).reshape(scat_height, 4, stat_width, 2)
        stat_code = 0x2800 + (stat_data * braille_block).sum(axis=(1, 3)).astype('i2')
        stat_graph = [line.tostring().decode('utf-16') for line in stat_code]

        # plot accuracies
        acc_weight = 2.0**numpy.array([-3, -2, -1, 0, -1, -2, -3])
        acc_data = (data[:-1,:].reshape(scat_height, 4, scat_width, 2).sum(axis=(1,3)) * acc_weight[:,None]).sum(axis=0)
        acc_data /= numpy.maximum(1, data.sum(axis=0).reshape(scat_width, 2).sum(axis=1))
        acc_level = numpy.arange(acc_height)*8
        acc_code = 0x2580 + numpy.clip(acc_data[None,:]*acc_height*8 - acc_level[::-1,None], 0, 8).astype('i2')
        acc_code[acc_code==0x2580] = ord(" ")
        acc_graph = [line.tostring().decode('utf-16') for line in acc_code]

        # print
        print("╒" + grad_top      + "╤" + scat_top      + "╤" + stat_top      + "╕")
        print("│" + grad_infos[0] + "│" + scat_graph[0] + "│" + stat_graph[0] + "│")
        print("│" + grad_infos[1] + "│" + scat_graph[1] + "│" + stat_graph[1] + "│")
        print("│" + grad_infos[2] + "│" + scat_graph[2] + "│" + stat_graph[2] + "│")
        print("│" + grad_infos[3] + "│" + scat_graph[3] + "│" + stat_graph[3] + "│")
        print("│" + grad_infos[4] + "│" + scat_graph[4] + "│" + stat_graph[4] + "│")
        print("│" + grad_infos[5] + "│" + scat_graph[5] + "│" + stat_graph[5] + "│")
        print("│" + grad_infos[6] + "│" + scat_graph[6] + "│" + stat_graph[6] + "│")
        print("│" + grad_infos[7] + "├" + miss_graph    + "┤" + stat_infos[0] + "│")
        print("│" + grad_infos[8] + "│" + acc_graph[0]  + "│" + stat_infos[1] + "│")
        print("│" + grad_infos[9] + "│" + acc_graph[1]  + "│" + stat_infos[2] + "│")
        print("╘" + grad_bot      + "╧" + scat_bot      + "╧" + stat_bot      + "╛")


# Beatmap
@cfg.configurable
class BeatmapSettings:
    ## Difficulty:
    performance_tolerance: float = 0.02
    soft_threshold: float = 0.5
    loud_threshold: float = 0.5
    incr_threshold: float = -0.1
    roll_tolerance: float = 0.10
    spin_tolerance: float = 0.10

    @property
    def perfect_tolerance(self): return self.performance_tolerance*1
    @property
    def good_tolerance(self): return self.performance_tolerance*3
    @property
    def bad_tolerance(self): return self.performance_tolerance*5
    @property
    def failed_tolerance(self): return self.performance_tolerance*7

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

    @property
    def performances_max_score(self): return max(self.performances_scores.values())

    roll_rock_score: int = 2
    spin_score: int = 16

    ## PerformanceSkin:
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

