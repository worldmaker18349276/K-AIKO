import math
from typing import List, Tuple
import queue
from ..utils import config as cfg
from ..utils import datanodes as dn
from ..utils import markups as mu
from ..devices import engines
from .beatwidgets import layout


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
        ("[dx=2/][color=bright_magenta]⟫[/]", "[dx=-1/][color=bright_magenta]⟪[/]"),
        ("[dx=2/][color=bright_magenta]⟩[/]", "[dx=-1/][color=bright_magenta]⟨[/]"),
        ("[dx=2/][color=bright_magenta]›[/]", "[dx=-1/][color=bright_magenta]‹[/]"),
        ("", ""),
        ("[dx=-1/][color=bright_magenta]‹[/]", "[dx=2/][color=bright_magenta]›[/]"),
        ("[dx=-1/][color=bright_magenta]⟨[/]", "[dx=2/][color=bright_magenta]⟩[/]"),
        ("[dx=-1/][color=bright_magenta]⟪[/]", "[dx=2/][color=bright_magenta]⟫[/]"),
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
    def __init__(self, grade_getter, settings):
        # grade_getter: DataNode[None -> List[int]]
        self.grade_getter = dn.DataNode.wrap(grade_getter)
        self.settings = settings

    def load(self, provider):
        rich = provider.get(mu.RichParser)
        detector = provider.get(engines.Detector)
        renderer = provider.get(engines.Renderer)

        perf_appearances = [
            (
                rich.parse(f"[restore]{appearance1}[/]"),
                rich.parse(f"[restore]{appearance2}[/]"),
            )
            for appearance1, appearance2 in self.settings.performances_appearances
        ]
        sight_appearances = [
            (rich.parse(appearance1), rich.parse(appearance2))
            for appearance1, appearance2 in self.settings.sight_appearances
        ]

        hit_decay_time = self.settings.hit_decay_time
        hit_sustain_time = self.settings.hit_sustain_time
        perf_sustain_time = self.settings.performance_sustain_time
        framerate = renderer.settings.display_framerate

        new_hits = queue.Queue()

        def listen(args):
            _, time, ratio, strength, detected = args
            if detected:
                new_hits.put((hit_sustain_time, min(1.0, strength)))

        detector.add_listener(listen)

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

        return sight_node()


# beatbar
class BeatbarLayoutSettings(cfg.Configurable):
    """
    Fields
    ------
    icon_width : int
        [rich]The width of icon.

        [color=bright_magenta] ⣠⣴⣤⣿⣤⣦ [/][color=bright_blue][[00000/00400]][/]       \
[color=bright_cyan]□ [/] [color=bright_magenta]⛶ [/]    [color=bright_cyan]□ [/]                \
[color=bright_blue]■ [/]  [color=bright_blue][[11.3%|00:09]][/]
        ^^^^^^^^
          here

    header_width : int
        [rich]The width of header.

        [color=bright_magenta] ⣠⣴⣤⣿⣤⣦ [/][color=bright_blue][[00000/00400]][/]       \
[color=bright_cyan]□ [/] [color=bright_magenta]⛶ [/]    [color=bright_cyan]□ [/]                \
[color=bright_blue]■ [/]  [color=bright_blue][[11.3%|00:09]][/]
                ^^^^^^^^^^^^^
                    here

    footer_width : int
        [rich]The width of footer.

        [color=bright_magenta] ⣠⣴⣤⣿⣤⣦ [/][color=bright_blue][[00000/00400]][/]       \
[color=bright_cyan]□ [/] [color=bright_magenta]⛶ [/]    [color=bright_cyan]□ [/]                \
[color=bright_blue]■ [/]  [color=bright_blue][[11.3%|00:09]][/]
                                                                   ^^^^^^^^^^^^^
                                                                        here

    """
    icon_width: int = 8
    header_width: int = 13
    footer_width: int = 13


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
        layout_settings,
    ):
        self.layout_settings = layout_settings

        self.mixer = mixer
        self.detector = detector
        self.renderer = renderer
        self.controller = controller

        # initialize game state
        self.bar_shift = bar_shift
        self.bar_flip = bar_flip

        # layout
        icon_width = layout_settings.icon_width
        header_width = layout_settings.header_width
        footer_width = layout_settings.footer_width

        [
            icon_mask,
            header_mask,
            content_mask,
            footer_mask,
        ] = layout([icon_width, header_width, -1, footer_width])

        self.icon_mask = icon_mask
        self.header_mask = header_mask
        self.content_mask = content_mask
        self.footer_mask = footer_mask

        self.icon = icon
        self.header = header
        self.footer = footer
        self.sight = sight

        self.content_pipeline = dn.DynamicPipeline()

    @dn.datanode
    def load(self):
        # hit handler
        self.target_queue = queue.Queue()
        hit_handler = Beatbar._hit_handler(self.target_queue)

        self.detector.add_listener(hit_handler)

        # register handlers
        self.sight_scheduler = Scheduler(default=self.sight)
        self.draw_content(0.0, self.sight_scheduler, zindex=(2,))

        bar_node = self._bar_node(self.content_pipeline, self.content_mask)
        self.renderer.add_drawer(bar_node, zindex=(0,))
        self.renderer.add_texts(self.icon, xmask=self.icon_mask, zindex=(1,))
        self.renderer.add_texts(self.header, xmask=self.header_mask, zindex=(2,))
        self.renderer.add_texts(self.footer, xmask=self.footer_mask, zindex=(3,))

        yield
        return

    @staticmethod
    @dn.datanode
    def _bar_node(pipeline, xmask):
        with pipeline:
            (view, msg, logs), time, width = yield
            while True:
                xran = engines.to_range(xmask.start, xmask.stop, width)

                try:
                    contents = pipeline.send(([], time, xran))
                except StopIteration:
                    return

                for text, shift in contents:
                    view.add_markup(text, xmask, shift=shift)

                (view, msg, logs), time, width = yield (view, msg, logs)

    @dn.datanode
    def _content_node(self, pos_node, text_node, start, duration):
        with pos_node, text_node:
            contents, time, ran = yield

            if start is None:
                start = time

            while time < start:
                contents, time, ran = yield contents

            while duration is None or time < start + duration:
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
                    contents, time, ran = yield contents
                    continue

                if isinstance(text, tuple):
                    text = text[flip]
                if text is not None:
                    index = round(index)
                    contents.append((text, index))

                contents, time, ran = yield contents

    def draw_content(self, pos, text, start=None, duration=None, zindex=(0,)):
        pos_node = dn.DataNode.wrap(
            pos if not isinstance(pos, (int, float)) else lambda time: pos
        )
        text_node = dn.DataNode.wrap(
            text if not isinstance(text, mu.Markup) else lambda time: text
        )

        node = self._content_node(pos_node, text_node, start, duration)
        return self.content_pipeline.add_node(node, zindex=zindex)

    @dn.datanode
    def _title_node(self, pos_node, text_node, start, duration):
        with pos_node, text_node:
            contents, time, ran = yield

            if start is None:
                start = time

            while time < start:
                contents, time, ran = yield contents

            while duration is None or time < start + duration:
                try:
                    pos = pos_node.send(time)
                    text = text_node.send(time)
                except StopIteration:
                    return

                index = pos * max(0, len(ran) - 1)
                if not math.isfinite(index):
                    contents, time, ran = yield contents
                    continue

                if text is not None:
                    index = round(index)
                    contents.append((text, index))

                contents, time, ran = yield contents

    def draw_title(self, pos, text, start=None, duration=None, zindex=(10,)):
        pos_node = dn.DataNode.wrap(
            pos if not isinstance(pos, (int, float)) else lambda time: pos
        )
        text_node = dn.DataNode.wrap(
            text if not isinstance(text, mu.Markup) else lambda time: text
        )

        node = self._title_node(pos_node, text_node, start, duration)
        return self.content_pipeline.add_node(node, zindex=zindex)

    def remove_content_drawer(self, key):
        self.content_pipeline.remove_node(key)

    def on_before_render(self, node):
        node = dn.pipe(dn.branch(lambda a: a[1:], node), lambda a: a[0])
        return self.content_pipeline.add_node(node, zindex=(0,))

    @staticmethod
    @dn.datanode
    def _hit_handler(target_queue):
        target, start, duration = None, None, None
        waiting_targets = []

        while True:
            # update hit signal
            _, time, ratio, strength, detected = yield

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
            if target is not None and detected and ratio != 0:
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
def cell(node, queue):
    node = dn.DataNode.wrap(node)
    data = None
    while True:
        with node:
            while queue.empty():
                data = yield data
                try:
                    data = node.send(data)
                except StopIteration:
                    return
            node = queue.get()
