import math
from enum import Enum
import dataclasses
from typing import Any, List, Tuple, Dict, Optional, Union
import queue
import threading
from ..utils import config as cfg
from ..utils import datanodes as dn
from ..utils import markups as mu
from . import beatwidgets


# performance
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
            return Performance(PerformanceGrade.MISS, time, None)

        is_wrong = not is_correct_key
        err = hit_time - time
        shift = next((i for i in range(3) if abs(err) < tol*(2*i+1)), 3)
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
        return self.descriptions[self.grade]


# beatbar
class BeatbarSettings(cfg.Configurable):
    @cfg.subconfig
    class layout(cfg.Configurable):
        r"""
        Fields
        ------
        icon_width : int
            The width of icon.

             ⣠⣴⣤⣿⣤⣦ [[00000/00400]]       □  ⛶     □                 ■   [[11.3%|00:09]]
            ^^^^^^^^
              here

        header_width : int
            The width of header.

             ⣠⣴⣤⣿⣤⣦ [[00000/00400]]       □  ⛶     □                 ■   [[11.3%|00:09]]
                    ^^^^^^^^^^^^^
                        here

        footer_width : int
            The width of footer.

             ⣠⣴⣤⣿⣤⣦ [[00000/00400]]       □  ⛶     □                 ■   [[11.3%|00:09]]
                                                                       ^^^^^^^^^^^^^
                                                                            here

        """
        icon_width: int = 8
        header_width: int = 13
        footer_width: int = 13

    @cfg.subconfig
    class sight(cfg.Configurable):
        r"""
        Fields
        ------
        sight_appearances : list of tuple of str and str
            The appearances of the judgement line.
            The first string is default appearance, and the rest is the appearances
            for the different hitting strength (from soft to loud).
            If the element is tuple of strings, then the first/second string is
            used for right-to-left/left-to-right direction.
        hit_decay_time : float
            The decay time of the hitting strength is displayed on the sight.
        hit_sustain_time : float
            The minimum time of the hitting strength is displayed on the sight.
            If the hitting strength is too soft, the style of the sight will
            sustain until this time.

        performances_appearances : dict from PerformanceGrade to tuple of str and str
            The hint for different performance, which will be drawn on the sight.
            The first/second string is used for right-to-left/left-to-right notes.
        performance_sustain_time : float
            The sustain time of performance hint.
        """
        performances_appearances: Dict[PerformanceGrade, Tuple[str, str]] = {
            PerformanceGrade.MISS               : (""   , ""     ),

            PerformanceGrade.LATE_FAILED        : ("[dx=-1/][color=bright_magenta]⟪[/]", "[dx=2/][color=bright_magenta]⟫[/]"),
            PerformanceGrade.LATE_BAD           : ("[dx=-1/][color=bright_magenta]⟨[/]", "[dx=2/][color=bright_magenta]⟩[/]"),
            PerformanceGrade.LATE_GOOD          : ("[dx=-1/][color=bright_magenta]‹[/]", "[dx=2/][color=bright_magenta]›[/]"),
            PerformanceGrade.PERFECT            : (""   , ""     ),
            PerformanceGrade.EARLY_GOOD         : ("[dx=2/][color=bright_magenta]›[/]", "[dx=-1/][color=bright_magenta]‹[/]"),
            PerformanceGrade.EARLY_BAD          : ("[dx=2/][color=bright_magenta]⟩[/]", "[dx=-1/][color=bright_magenta]⟨[/]"),
            PerformanceGrade.EARLY_FAILED       : ("[dx=2/][color=bright_magenta]⟫[/]", "[dx=-1/][color=bright_magenta]⟪[/]"),

            PerformanceGrade.LATE_FAILED_WRONG  : ("[dx=-1/][color=bright_magenta]⟪[/]", "[dx=2/][color=bright_magenta]⟫[/]"),
            PerformanceGrade.LATE_BAD_WRONG     : ("[dx=-1/][color=bright_magenta]⟨[/]", "[dx=2/][color=bright_magenta]⟩[/]"),
            PerformanceGrade.LATE_GOOD_WRONG    : ("[dx=-1/][color=bright_magenta]‹[/]", "[dx=2/][color=bright_magenta]›[/]"),
            PerformanceGrade.PERFECT_WRONG      : (""   , ""     ),
            PerformanceGrade.EARLY_GOOD_WRONG   : ("[dx=2/][color=bright_magenta]›[/]", "[dx=-1/][color=bright_magenta]‹[/]"),
            PerformanceGrade.EARLY_BAD_WRONG    : ("[dx=2/][color=bright_magenta]⟩[/]", "[dx=-1/][color=bright_magenta]⟨[/]"),
            PerformanceGrade.EARLY_FAILED_WRONG : ("[dx=2/][color=bright_magenta]⟫[/]", "[dx=-1/][color=bright_magenta]⟪[/]"),
            }
        performance_sustain_time: float = 0.1

        sight_appearances: List[Tuple[str, str]] = [
            ("[color=ff00ff]⛶[/]", "[color=ff00ff]⛶[/]"),
            ("[color=ff00ff]🞎[/]", "[color=ff00ff]🞎[/]"),
            ("[color=ff00d7]🞏[/]", "[color=ff00d7]🞏[/]"),
            ("[color=ff00af]🞐[/]", "[color=ff00af]🞐[/]"),
            ("[color=ff0087]🞑[/]", "[color=ff0087]🞑[/]"),
            ("[color=ff005f]🞒[/]", "[color=ff005f]🞒[/]"),
            ("[color=ff0000]🞓[/]", "[color=ff0000]🞓[/]"),
        ]
        hit_decay_time: float = 0.4
        hit_sustain_time: float = 0.1

class Beatbar:
    def __init__(self, mixer, detector, renderer, controller, icon, header, footer, sight, bar_shift, bar_flip, settings):
        self.settings = settings

        self.mixer = mixer
        self.detector = detector
        self.renderer = renderer
        self.controller = controller

        # initialize game state
        self.bar_shift = bar_shift
        self.bar_flip = bar_flip

        # layout
        icon_width = settings.layout.icon_width
        header_width = settings.layout.header_width
        footer_width = settings.layout.footer_width

        self.icon_mask = slice(None, icon_width)
        self.header_mask = slice(icon_width, icon_width+header_width)
        self.content_mask = slice(icon_width+header_width, -footer_width if footer_width > 0 else None)
        self.footer_mask = slice(-footer_width, None) if footer_width > 0 else slice(0, 0)

        self.icon_func = icon
        self.header_func = header
        self.footer_func = footer
        self.sight_func = sight

    @dn.datanode
    def load(self):
        # register handlers
        icon_drawer = lambda arg: (0, self.icon_func(arg[0], arg[1]))
        header_drawer = lambda arg: (0, self.header_func(arg[0], arg[1]))
        footer_drawer = lambda arg: (0, self.footer_func(arg[0], arg[1]))

        # sight
        hit_decay_time = self.settings.sight.hit_decay_time
        hit_sustain_time = self.settings.sight.hit_sustain_time
        perf_sustain_time = self.settings.sight.performance_sustain_time

        self.current_hit_hint = TimedVariable(value=None, duration=hit_sustain_time)
        self.current_perf_hint = TimedVariable(value=(None, None), duration=perf_sustain_time)
        self.current_sight = TimedVariable(value=self.sight_func)

        # hit handler
        self.target_queue = queue.Queue()
        hit_handler = Beatbar._hit_handler(self.current_hit_hint, self.target_queue, hit_decay_time, hit_sustain_time)

        self.renderer.add_text(icon_drawer, xmask=self.icon_mask, zindex=(1,))
        self.renderer.add_text(header_drawer, xmask=self.header_mask, zindex=(2,))
        self.renderer.add_text(footer_drawer, xmask=self.footer_mask, zindex=(3,))
        self.detector.add_listener(hit_handler)

        self.draw_content(0.0, self._sight_drawer, zindex=(2,))

        yield
        return

    @dn.datanode
    def _content_node(self, pos_func, text_func, start, duration):
        time, ran = yield

        if start is None:
            start = time

        while time < start:
            time, ran = yield None

        while duration is None or time < start + duration:
            pos = pos_func(time)
            text = text_func(time)
            shift = self.bar_shift
            flip = self.bar_flip

            pos = pos + shift
            if flip:
                pos = 1 - pos

            index = pos * max(0, len(ran)-1)
            if not math.isfinite(index):
                time, ran = yield None
                continue

            index = round(index)
            if isinstance(text, tuple):
                text = text[flip]
            time, ran = yield index, text

    def draw_content(self, pos, text, start=None, duration=None, zindex=(0,)):
        pos_func = pos if hasattr(pos, '__call__') else lambda time: pos
        text_func = text if hasattr(text, '__call__') else lambda time: text

        node = self._content_node(pos_func, text_func, start, duration)
        zindex_ = (lambda: (0, *zindex())) if hasattr(zindex, '__call__') else (0, *zindex)
        return self.renderer.add_text(node, self.content_mask, zindex=zindex_)

    @dn.datanode
    def _title_node(self, pos_func, text_func, start, duration):
        time, ran = yield

        if start is None:
            start = time

        while time < start:
            time, ran = yield None

        while duration is None or time < start + duration:
            pos = pos_func(time)
            text = text_func(time)

            index = pos * max(0, len(ran)-1)
            if not math.isfinite(index):
                time, ran = yield None
                continue

            index = round(index)
            time, ran = yield index, text

    def draw_title(self, pos, text, start=None, duration=None, zindex=(10,)):
        pos_func = pos if hasattr(pos, '__call__') else lambda time: pos
        text_func = text if hasattr(text, '__call__') else lambda time: text

        node = self._title_node(pos_func, text_func, start, duration)
        zindex_ = (lambda: (0, *zindex())) if hasattr(zindex, '__call__') else (0, *zindex)
        return self.renderer.add_text(node, self.content_mask, zindex=zindex_)

    def remove_content_drawer(self, key):
        self.renderer.remove_drawer(key)

    def on_before_render(self, node):
        node = dn.pipe(dn.branch(lambda a:a[1:], node), lambda a:a[0])
        return self.renderer.add_drawer(node, zindex=())


    @staticmethod
    @dn.datanode
    def _hit_handler(current_hit_hint, target_queue, hit_decay_time, hit_sustain_time):
        target, start, duration = None, None, None
        waiting_targets = []

        while True:
            # update hit signal
            _, time, strength, detected = yield

            strength = min(1.0, strength)
            if detected:
                current_hit_hint.set(strength, duration=max(strength * hit_decay_time, hit_sustain_time))

            # update waiting targets
            while not target_queue.empty():
                item = target_queue.get()
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
                if target is not None and duration is not None and start + duration <= time:
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

    def _sight_drawer(self, time):
        # update hit hint, perf hint
        hit_hint = self.current_hit_hint.get(time)
        perf_hint = self.current_perf_hint.get(time)

        # draw sight
        sight_func = self.current_sight.get(time).value
        sight_text = sight_func(time, hit_hint, perf_hint)

        return sight_text

    def listen(self, node, start=None, duration=None):
        self.target_queue.put((node, start, duration))

    def set_perf(self, perf, is_reversed=False):
        self.current_perf_hint.set((perf, is_reversed))

    def draw_sight(self, text, start=None, duration=None):
        text_func = text if hasattr(text, '__call__') else lambda time, hit_hint, perf_hint: text
        self.current_sight.set(text_func, start, duration)

    def reset_sight(self, start=None):
        self.current_sight.reset(start)

    def play(self, node, samplerate=None, channels=None, volume=0.0, start=None, end=None, time=None, zindex=(0,)):
        return self.mixer.play(node, samplerate=samplerate, channels=channels,
                                     volume=volume, start=start, end=end,
                                     time=time, zindex=zindex)

    def add_handler(self, node, keyname=None):
        return self.controller.add_handler(node, keyname)

    def remove_handler(self, key):
        self.controller.remove_handler(key)

class Sight:
    def __init__(self, rich, settings):
        self.rich = rich
        self.settings = settings

    @dn.datanode
    def load(self):
        perf_appearances = {key: (self.rich.parse(f"[restore]{appearance1}[/]"), self.rich.parse(f"[restore]{appearance2}[/]"))
                            for key, (appearance1, appearance2) in self.settings.performances_appearances.items()}
        sight_appearances = [(self.rich.parse(appearance1), self.rich.parse(appearance2))
                             for appearance1, appearance2 in self.settings.sight_appearances]

        def sight_func(time, hit_hint, perf_hint):
            hit_strength = hit_hint.value
            hit_time = hit_hint.start
            hit_duration = hit_hint.duration
            perf, perf_is_reversed = perf_hint.value
            perf_time = perf_hint.start

            # draw perf hint
            perf_ap = perf_appearances[perf.grade] if perf is not None else (mu.Text(""), mu.Text(""))
            if perf_is_reversed:
                perf_ap = perf_ap[::-1]

            # draw sight
            if hit_strength is not None:
                strength = hit_strength - hit_strength * (time - hit_time) / hit_duration
                strength = max(0.0, min(1.0, strength))
                loudness = int(strength * (len(sight_appearances) - 1))
                loudness = max(1, loudness)
                sight_ap = sight_appearances[loudness]

            else:
                sight_ap = sight_appearances[0]

            return (
                mu.Group((perf_ap[0], sight_ap[0])),
                mu.Group((perf_ap[1], sight_ap[1]))
            )

        yield
        return sight_func

@dataclasses.dataclass
class TimedValue:
    value: Any
    start: Optional[float]
    duration: float

class TimedVariable:
    def __init__(self, value=None, duration=float('inf')):
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._scheduled_values = []
        self._default_value = value
        self._default_duration = duration
        self._current_value = TimedValue(value, None, float('inf'))

    def get(self, time):
        with self._lock:
            value = self._current_value
            if value.start is None:
                value.start = time

            while not self._queue.empty():
                item = self._queue.get()
                if item.start is None:
                    item.start = time
                self._scheduled_values.append(item)
            self._scheduled_values.sort(key=lambda item: item.start)

            while self._scheduled_values and self._scheduled_values[0].start <= time:
                value = self._scheduled_values.pop(0)

            if value.start + value.duration <= time:
                value = TimedValue(self._default_value, None, float('inf'))

            self._current_value = value
            return self._current_value

    def set(self, value, start=None, duration=None):
        if duration is None:
            duration = self._default_duration
        self._queue.put(TimedValue(value, start, duration))

    def reset(self, start=None):
        self._queue.put(TimedValue(self._default_value, start, float('inf')))


# widgets
class BeatbarWidgetBuilder:
    spectrum = beatwidgets.SpectrumWidgetSettings
    volume_indicator = beatwidgets.VolumeIndicatorWidgetSettings
    score = beatwidgets.ScoreWidgetSettings
    progress = beatwidgets.ProgressWidgetSettings
    accuracy_meter = beatwidgets.AccuracyMeterWidgetSettings
    monitor = beatwidgets.MonitorWidgetSettings

    def __init__(self, *, state, rich, mixer, detector, renderer, controller, devices_settings):
        self.state = state
        self.rich = rich
        self.mixer = mixer
        self.detector = detector
        self.renderer = renderer
        self.controller = controller
        self.devices_settings = devices_settings

    def create(self, widget_settings):
        if isinstance(widget_settings, BeatbarWidgetBuilder.spectrum):
            return beatwidgets.SpectrumWidget("", self.rich, self.mixer, self.devices_settings.mixer, widget_settings)
        elif isinstance(widget_settings, BeatbarWidgetBuilder.volume_indicator):
            return beatwidgets.VolumeIndicatorWidget(0.0, self.rich, self.mixer, self.devices_settings.mixer, widget_settings)
        elif isinstance(widget_settings, BeatbarWidgetBuilder.accuracy_meter):
            return beatwidgets.AccuracyMeterWidget(0, float("inf"), self.rich, self.state, widget_settings)
        elif isinstance(widget_settings, BeatbarWidgetBuilder.monitor):
            if widget_settings.target == beatwidgets.MonitorTarget.mixer:
                return beatwidgets.MonitorWidget(self.mixer, widget_settings)
            elif widget_settings.target == beatwidgets.MonitorTarget.detector:
                return beatwidgets.MonitorWidget(self.detector, widget_settings)
            elif widget_settings.target == beatwidgets.MonitorTarget.renderer:
                return beatwidgets.MonitorWidget(self.renderer, widget_settings)
            else:
                assert False
        elif isinstance(widget_settings, BeatbarWidgetBuilder.score):
            return beatwidgets.ScoreWidget(self.state, self.rich, widget_settings)
        elif isinstance(widget_settings, BeatbarWidgetBuilder.progress):
            return beatwidgets.ProgressWidget(self.state, self.rich, widget_settings)
        else:
            assert False

BeatbarIconWidgetSettings = Union[
    beatwidgets.SpectrumWidgetSettings,
    beatwidgets.VolumeIndicatorWidgetSettings,
    beatwidgets.ScoreWidgetSettings,
    beatwidgets.ProgressWidgetSettings,
    beatwidgets.AccuracyMeterWidgetSettings,
    beatwidgets.MonitorWidgetSettings,
]
BeatbarHeaderWidgetSettings = BeatbarIconWidgetSettings
BeatbarFooterWidgetSettings = BeatbarIconWidgetSettings

class BeatbarWidgetSettings(cfg.Configurable):
    r"""
    Fields
    ------
    icon_widget : BeatbarIconWidgetSettings
        The widget on the icon.
    header_widget : BeatbarHeaderWidgetSettings
        The widget on the header.
    footer_widget : BeatbarFooterWidgetSettings
        The widget on the footer.
    """
    icon_widget: BeatbarIconWidgetSettings = BeatbarWidgetBuilder.spectrum()
    header_widget: BeatbarHeaderWidgetSettings = BeatbarWidgetBuilder.score()
    footer_widget: BeatbarFooterWidgetSettings = BeatbarWidgetBuilder.progress()

