import math
from typing import List, Tuple, Union
import queue
from ..utils import config as cfg
from ..utils import datanodes as dn
from ..utils import markups as mu
from ..devices import engines
from . import beatwidgets


class SightWidgetSettings(cfg.Configurable):
    r"""
    Fields
    ------
    sight_appearances : list of tuple of str and str
        The appearances of the judgement line. The first string is default
        appearance, and the rest is the appearances for the different
        hitting strength (from soft to loud). If the element is tuple of
        strings, then the first/second string is used for
        right-to-left/left-to-right direction.
    hit_decay_time : float
        The decay time of the hitting strength is displayed on the sight.
    hit_sustain_time : float
        The minimum time of the hitting strength is displayed on the sight.
        If the hitting strength is too soft, the style of the sight will
        sustain until this time.

    performances_appearances : list of tuple of str and str
        The hint for different performances (from early to late), which will be
        drawn on the sight. The first/second string is used for
        right-to-left/left-to-right notes.
    performance_sustain_time : float
        The sustain time of performance hint.
    """
    performances_appearances: List[Tuple[str, str]] = [
        (
            "[dx=2/][color=bright_magenta]⟫[/]",
            "[dx=-1/][color=bright_magenta]⟪[/]",
        ),
        (
            "[dx=2/][color=bright_magenta]⟩[/]",
            "[dx=-1/][color=bright_magenta]⟨[/]",
        ),
        (
            "[dx=2/][color=bright_magenta]›[/]",
            "[dx=-1/][color=bright_magenta]‹[/]",
        ),
        ("", ""),
        (
            "[dx=-1/][color=bright_magenta]‹[/]",
            "[dx=2/][color=bright_magenta]›[/]",
        ),
        (
            "[dx=-1/][color=bright_magenta]⟨[/]",
            "[dx=2/][color=bright_magenta]⟩[/]",
        ),
        (
            "[dx=-1/][color=bright_magenta]⟪[/]",
            "[dx=2/][color=bright_magenta]⟫[/]",
        ),
    ]
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


class SightWidget:
    def __init__(
        self,
        grade_getter,
        rich,
        detector,
        renderer_settings,
        settings,
    ):
        # grade_getter: DataNode[None -> List[int]]
        self.grade_getter = dn.DataNode.wrap(grade_getter)
        self.rich = rich
        self.detector = detector
        self.renderer_settings = renderer_settings
        self.settings = settings

    @dn.datanode
    def load(self):
        perf_appearances = [
            (
                self.rich.parse(f"[restore]{appearance1}[/]"),
                self.rich.parse(f"[restore]{appearance2}[/]"),
            )
            for appearance1, appearance2 in self.settings.performances_appearances
        ]
        sight_appearances = [
            (self.rich.parse(appearance1), self.rich.parse(appearance2))
            for appearance1, appearance2 in self.settings.sight_appearances
        ]

        hit_decay_time = self.settings.hit_decay_time
        hit_sustain_time = self.settings.hit_sustain_time
        perf_sustain_time = self.settings.performance_sustain_time
        framerate = self.renderer_settings.display_framerate

        new_hits = queue.Queue()

        def listen(args):
            _, time, strength, detected = args
            if detected:
                new_hits.put((hit_sustain_time, min(1.0, strength)))

        self.detector.add_listener(listen)

        @dn.datanode
        def sight_node():
            last_grade = (0.0, 0)
            last_hit = (0.0, 0.0)

            with self.grade_getter:
                time = yield
                while True:
                    # draw perf hint
                    try:
                        new_grades = self.grade_getter.send(None)
                    except StopIteration:
                        return

                    if new_grades:
                        last_grade = (perf_sustain_time, new_grades[-1])

                    perf_index = last_grade[1] + len(perf_appearances) // 2
                    if last_grade[0] > 0 and perf_index in range(len(perf_appearances)):
                        perf_ap = perf_appearances[perf_index]

                        last_grade = (
                            last_grade[0] - 1 / framerate,
                            last_grade[1],
                        )

                    else:
                        perf_ap = (mu.Group(()), mu.Group(()))

                    # draw sight
                    while not new_hits.empty():
                        last_hit = new_hits.get()

                    if last_hit[0] > 0 or last_hit[1] > 0:
                        strength = max(0.0, min(1.0, last_hit[1]))
                        loudness = int(strength * (len(sight_appearances) - 1))
                        loudness = max(1, loudness)
                        sight_ap = sight_appearances[loudness]

                        last_hit = (
                            last_hit[0] - 1 / framerate,
                            last_hit[1] - 1 / framerate / hit_decay_time,
                        )

                    else:
                        sight_ap = sight_appearances[0]

                    res = (
                        mu.Group((perf_ap[0], sight_ap[0])),
                        mu.Group((perf_ap[1], sight_ap[1])),
                    )

                    time = yield res

        yield
        return sight_node()


# beatbar
class BeatbarSettings(cfg.Configurable):
    @cfg.subconfig
    class layout(cfg.Configurable):
        r"""
        Fields
        ------
        icon_width : int
            The width of icon.

            ..code::

                 ⣠⣴⣤⣿⣤⣦ [[00000/00400]]       □  ⛶     □                 ■   [[11.3%|00:09]]
                ^^^^^^^^
                  here

        header_width : int
            The width of header.

            ..code::

                 ⣠⣴⣤⣿⣤⣦ [[00000/00400]]       □  ⛶     □                 ■   [[11.3%|00:09]]
                        ^^^^^^^^^^^^^
                            here

        footer_width : int
            The width of footer.

            ..code::

                 ⣠⣴⣤⣿⣤⣦ [[00000/00400]]       □  ⛶     □                 ■   [[11.3%|00:09]]
                                                                           ^^^^^^^^^^^^^
                                                                                here

        """
        icon_width: int = 8
        header_width: int = 13
        footer_width: int = 13

    sight = cfg.subconfig(SightWidgetSettings)


class Beatbar:
    def __init__(
        self,
        mixer,
        detector,
        renderer,
        controller,
        icon,
        header,
        footer,
        sight,
        bar_shift,
        bar_flip,
        settings,
    ):
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
        self.header_mask = slice(icon_width, icon_width + header_width)
        self.content_mask = slice(
            icon_width + header_width, -footer_width if footer_width > 0 else None
        )
        self.footer_mask = (
            slice(-footer_width, None) if footer_width > 0 else slice(0, 0)
        )

        self.icon = icon
        self.header = header
        self.footer = footer
        self.sight_func = sight

    @dn.datanode
    def load(self):
        # hit handler
        self.target_queue = queue.Queue()
        hit_handler = Beatbar._hit_handler(self.target_queue)

        self.detector.add_listener(hit_handler)

        # register handlers
        self.sight_scheduler = Scheduler(default=self.sight_func)

        self.renderer.add_text(self.icon, xmask=self.icon_mask, zindex=(1,))
        self.renderer.add_text(self.header, xmask=self.header_mask, zindex=(2,))
        self.renderer.add_text(self.footer, xmask=self.footer_mask, zindex=(3,))
        self.draw_content(0.0, self.sight_scheduler, zindex=(2,))

        yield
        return

    @dn.datanode
    def _content_node(self, pos_node, text_node, start, duration):
        mask = self.content_mask

        with pos_node, text_node:
            (view, msg, logs), time, width = yield

            if start is None:
                start = time

            while time < start:
                (view, msg, logs), time, width = yield (view, msg, logs)

            while duration is None or time < start + duration:
                ran = engines.to_range(mask.start, mask.stop, width)

                try:
                    pos = pos_node.send(time)
                    text = text_node.send(time)
                except StopIteration:
                    return

                shift = self.bar_shift
                flip = self.bar_flip

                pos = pos + shift
                if flip:
                    pos = 1 - pos

                index = pos * max(0, len(ran) - 1)
                if not math.isfinite(index):
                    (view, msg, logs), time, width = yield (view, msg, logs)
                    continue

                index = round(index)
                if isinstance(text, tuple):
                    text = text[flip]
                if text is not None:
                    view.add_markup(text, mask, shift=index)

                (view, msg, logs), time, width = yield (view, msg, logs)

    def draw_content(self, pos, text, start=None, duration=None, zindex=(0,)):
        pos_node = dn.DataNode.wrap(
            pos if not isinstance(pos, (int, float)) else lambda time: pos
        )
        text_node = dn.DataNode.wrap(
            text if not isinstance(text, mu.Markup) else lambda time: text
        )

        node = self._content_node(pos_node, text_node, start, duration)
        zindex_ = (
            (lambda: (0, *zindex())) if hasattr(zindex, "__call__") else (0, *zindex)
        )
        return self.renderer.add_drawer(node, zindex=zindex_)

    @dn.datanode
    def _title_node(self, pos_node, text_node, start, duration):
        mask = self.content_mask

        with pos_node, text_node:
            (view, msg, logs), time, width = yield

            if start is None:
                start = time

            while time < start:
                (view, msg, logs), time, width = yield (view, msg, logs)

            while duration is None or time < start + duration:
                ran = engines.to_range(mask.start, mask.stop, width)

                try:
                    pos = pos_node.send(time)
                    text = text_node.send(time)
                except StopIteration:
                    return

                index = pos * max(0, len(ran) - 1)
                if not math.isfinite(index):
                    time, ran = yield None
                    continue

                index = round(index)
                if text is not None:
                    view.add_markup(text, mask, shift=index)

                (view, msg, logs), time, width = yield (view, msg, logs)

    def draw_title(self, pos, text, start=None, duration=None, zindex=(10,)):
        pos_node = dn.DataNode.wrap(
            pos if not isinstance(pos, (int, float)) else lambda time: pos
        )
        text_node = dn.DataNode.wrap(
            text if not isinstance(text, mu.Markup) else lambda time: text
        )

        node = self._title_node(pos_node, text_node, start, duration)
        zindex_ = (
            (lambda: (0, *zindex())) if hasattr(zindex, "__call__") else (0, *zindex)
        )
        return self.renderer.add_drawer(node, zindex=zindex_)

    def remove_content_drawer(self, key):
        self.renderer.remove_drawer(key)

    def on_before_render(self, node):
        node = dn.pipe(dn.branch(lambda a: a[1:], node), lambda a: a[0])
        return self.renderer.add_drawer(node, zindex=())

    @staticmethod
    @dn.datanode
    def _hit_handler(target_queue):
        target, start, duration = None, None, None
        waiting_targets = []

        while True:
            # update hit signal
            _, time, strength, detected = yield

            strength = min(1.0, strength)

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
                if (
                    target is not None
                    and duration is not None
                    and start + duration <= time
                ):
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

    def listen(self, node, start=None, duration=None):
        self.target_queue.put((node, start, duration))

    def draw_sight(self, text, start=None, duration=None):
        text_node = dn.DataNode.wrap(
            text if not isinstance(text, tuple) else lambda time: text
        )
        self.sight_scheduler.set(text_node, start, duration)

    def reset_sight(self, start=None):
        self.sight_scheduler.reset(start)

    def play(
        self,
        node,
        samplerate=None,
        channels=None,
        volume=0.0,
        start=None,
        end=None,
        time=None,
        zindex=(0,),
    ):
        return self.mixer.play(
            node,
            samplerate=samplerate,
            channels=channels,
            volume=volume,
            start=start,
            end=end,
            time=time,
            zindex=zindex,
        )

    def add_handler(self, node, keyname=None):
        return self.controller.add_handler(node, keyname)

    def remove_handler(self, key):
        self.controller.remove_handler(key)


class Scheduler(dn.DataNode):
    def __init__(self, default):
        self._queue = queue.Queue()
        super().__init__(self.proxy(default))

    class _TimedNode:
        def __init__(self, value, start, duration):
            self.value = value
            self.start = start
            self.duration = duration

        def __enter__(self):
            if self.value is not None:
                self.value.__enter__()

        def __exit__(self, type=None, value=None, traceback=None):
            if self.value is not None:
                self.value.__exit__(type, value, traceback)

        def send(self, value, fallback):
            if self.value is not None:
                return self.value.send(value)
            else:
                return fallback

    def proxy(self, default):
        default = dn.DataNode.wrap(default)
        current = self._TimedNode(None, None, float("inf"))
        scheduled = []

        def fetch(time):
            while not self._queue.empty():
                item = self._queue.get()
                if item.start is None:
                    item.start = time
                scheduled.append(item)
            scheduled.sort(key=lambda item: item.start)

        with default:
            time = yield
            current.start = time
            fetch(time)
            while True:
                with current:
                    while True:
                        if scheduled and scheduled[0].start <= time:
                            current = scheduled.pop(0)
                            break

                        if current.start + current.duration <= time:
                            current = self._TimedNode(None, time, float("inf"))
                            break

                        try:
                            res = default.send(None)
                            res = current.send(time, fallback=res)
                        except StopIteration:
                            return

                        time = yield res
                        fetch(time)

    def set(self, node, start=None, duration=None):
        if duration is None:
            duration = float("inf")
        self._queue.put(self._TimedNode(node, start, duration))

    def reset(self, start=None):
        self._queue.put(self._TimedNode(None, start, float("inf")))


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
class BeatbarWidgetBuilder:
    spectrum = beatwidgets.SpectrumWidgetSettings
    volume_indicator = beatwidgets.VolumeIndicatorWidgetSettings
    score = beatwidgets.ScoreWidgetSettings
    progress = beatwidgets.ProgressWidgetSettings
    accuracy_meter = beatwidgets.AccuracyMeterWidgetSettings
    monitor = beatwidgets.MonitorWidgetSettings
    sight = SightWidgetSettings

    def __init__(self, *, state, rich, mixer, detector, renderer, controller):
        self.state = state
        self.rich = rich
        self.mixer = mixer
        self.detector = detector
        self.renderer = renderer
        self.controller = controller

    def create(self, widget_settings):
        if isinstance(widget_settings, BeatbarWidgetBuilder.spectrum):
            return beatwidgets.SpectrumWidget(self.rich, self.mixer, widget_settings)
        elif isinstance(widget_settings, BeatbarWidgetBuilder.volume_indicator):
            return beatwidgets.VolumeIndicatorWidget(
                self.rich, self.mixer, widget_settings
            )
        elif isinstance(widget_settings, BeatbarWidgetBuilder.accuracy_meter):
            accuracy_getter = dn.pipe(
                observe(self.state.perfs), lambda perfs: [perf.err for perf in perfs]
            )
            return beatwidgets.AccuracyMeterWidget(
                accuracy_getter,
                self.rich,
                self.renderer.settings,
                widget_settings,
            )
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
            score_getter = lambda _: (self.state.score, self.state.full_score)
            return beatwidgets.ScoreWidget(score_getter, self.rich, widget_settings)
        elif isinstance(widget_settings, BeatbarWidgetBuilder.progress):
            progress_getter = lambda _: (
                self.state.finished_subjects / self.state.total_subjects
                if self.state.total_subjects > 0
                else 1.0
            )
            time_getter = lambda _: self.state.time
            return beatwidgets.ProgressWidget(
                progress_getter, time_getter, self.rich, widget_settings
            )
        elif isinstance(widget_settings, BeatbarWidgetBuilder.sight):
            grade_getter = dn.pipe(
                observe(self.state.perfs),
                lambda perfs: [perf.grade.shift for perf in perfs],
            )
            return SightWidget(
                grade_getter,
                self.rich,
                self.detector,
                self.renderer.settings,
                widget_settings,
            )
        else:
            raise TypeError


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
