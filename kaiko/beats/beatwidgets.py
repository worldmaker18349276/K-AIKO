import dataclasses
from enum import Enum
from typing import Union, List, Tuple
import numpy
from ..utils import config as cfg
from ..utils import datanodes as dn
from ..utils import markups as mu
from ..devices import engines


def uint_format(value, width, zero_padded=False):
    scales = "KMGTPEZY"
    pad = "0" if zero_padded else " "

    if width == 0:
        return ""
    if width == 1:
        return str(value) if value < 10 else "+"

    if width == 2 and value < 1000:
        return f"{value:{pad}{width}d}" if value < 10 else "9+"
    elif value < 10 ** width:
        return f"{value:{pad}{width}d}"

    for scale, symbol in enumerate(scales):
        if value < 1000 ** (scale + 2):
            if width == 2:
                return symbol + "+"

            value_ = value // 1000 ** (scale + 1)
            eff = (
                f"{value_:{pad}{width-2}d}"
                if value_ < 10 ** (width - 2)
                else str(10 ** (width - 2) - 1)
            )
            return eff + symbol + "+"

    else:
        return str(10 ** (width - 2) - 1) + scales[-1] + "+"


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
        return f"1." if value == 1 else "." + str(int(value * 10))
    if width == 3:
        return f"1.0" if value == 1 else f"{value:>{width}.0%}"
    if width >= 4:
        return f"{value:>{width}.0%}" if value == 1 else f"{value:>{width}.{width-4}%}"


def layout(widths):
    if -1 not in widths:
        return layout([*widths, -1])[:-1]

    slices = [slice(0, 0) for _ in widths]

    i = widths.index(-1)

    right = 0
    for j, width in enumerate(widths[:i]):
        left_ = right
        right += width
        slices[j] = slice(left_, right)

    left = 0
    for j, width in enumerate(widths[:i:-1]):
        right_ = left
        left -= width
        slices[-1-j] = slice(left, None if left != right_ == 0 else right_)

    slices[i] = slice(right, None if left == 0 else left)

    return slices


# widgets
@dataclasses.dataclass
class SpectrumWidgetSettings:
    r"""
    Fields
    ------
    template : str
        The template for the spectrum.
    spec_width : int
        The text width of spectrum.
    spec_decay_time : float
        The decay time of pillars on the spectrum.
    spec_time_res : float
        The time resolution of the spectrum.
        The preferred value is `hop_length/samplerate`.
    spec_freq_res : float
        The frequency resolution of the spectrum.
        The preferred value is `samplerate/win_length`.
    """
    template: str = "[color=bright_magenta][slot/][/]"
    spec_width: int = 6
    spec_decay_time: float = 0.01
    spec_time_res: float = 0.0116099773
    spec_freq_res: float = 21.5332031


class SpectrumWidget:
    def __init__(self, settings):
        self.spectrum = ""
        self.settings = settings

    def draw_spectrum(self, samplerate):
        spec_width = self.settings.spec_width
        hop_length = round(samplerate * self.settings.spec_time_res)
        win_length = round(samplerate / self.settings.spec_freq_res)
        spec_decay_time = self.settings.spec_decay_time

        df = samplerate / win_length
        n_fft = win_length // 2 + 1
        n = numpy.linspace(1, 88, spec_width * 2 + 1)
        f = 440 * 2 ** ((n - 49) / 12)  # frequency of n-th piano key
        sec = numpy.minimum(n_fft - 1, (f / df).round().astype(int))
        slices = [slice(start, stop) for start, stop in zip(sec[:-1], (sec + 1)[1:])]

        decay = hop_length / samplerate / spec_decay_time / 4
        volume_of = (
            lambda J: dn.power2db(J.mean() * samplerate / 2, scale=(1e-5, 1e6)) / 60.0
        )

        A = numpy.cumsum([0, 2 ** 6, 2 ** 2, 2 ** 1, 2 ** 0])
        B = numpy.cumsum([0, 2 ** 7, 2 ** 5, 2 ** 4, 2 ** 3])
        draw_bar = lambda a, b: chr(0x2800 + A[int(a * 4)] + B[int(b * 4)])

        node = dn.pipe(
            dn.frame(win_length, hop_length),
            dn.power_spectrum(win_length, samplerate=samplerate),
        )

        @dn.datanode
        def draw():
            with node:
                vols = [0.0] * (spec_width * 2)

                data = yield
                while True:
                    try:
                        J = node.send(data)
                    except StopIteration:
                        return

                    vols = [
                        max(0.0, prev - decay, min(1.0, volume_of(J[slic])))
                        for slic, prev in zip(slices, vols)
                    ]
                    data = yield "".join(map(draw_bar, vols[0::2], vols[1::2]))

        return draw()

    def load(self, provider):
        rich = provider.get(mu.RichParser)
        mixer = provider.get(engines.Mixer)

        spec_width = self.settings.spec_width
        samplerate = mixer.settings.output_samplerate
        nchannels = mixer.settings.output_channels
        hop_length = round(samplerate * self.settings.spec_time_res)

        template = rich.parse(self.settings.template, slotted=True)

        self.spectrum = "\u2800" * spec_width
        draw = dn.pipe(
            self.draw_spectrum(samplerate), lambda v: setattr(self, "spectrum", v)
        )
        handler = dn.pipe(
            lambda a: a[0], dn.branch(dn.unchunk(draw, (hop_length, nchannels)))
        )
        mixer.add_effect(handler, zindex=(-1,))

        def widget_func(arg):
            time, ran = arg
            width = len(ran)
            text = mu.Text(f"{self.spectrum:^{width}.{width}s}")
            res = mu.replace_slot(template, text)
            return [(0, res)]

        return widget_func


@dataclasses.dataclass
class VolumeIndicatorWidgetSettings:
    r"""
    Fields
    ------
    template : str
        The template for the volume indicator.
    vol_decay_time : float
        The decay time of pillar on the volume indicator.
    """
    template: str = "[color=bright_magenta][slot/][/]"
    vol_decay_time: float = 0.01


class VolumeIndicatorWidget:
    def __init__(self, settings):
        self.volume = 0.0
        self.settings = settings

    def load(self, provider):
        rich = provider.get(mu.RichParser)
        mixer = provider.get(engines.Mixer)

        vol_decay_time = self.settings.vol_decay_time
        buffer_length = mixer.settings.output_buffer_length
        samplerate = mixer.settings.output_samplerate

        template = rich.parse(self.settings.template, slotted=True)

        decay = buffer_length / samplerate / vol_decay_time

        volume_of = lambda x: dn.power2db((x ** 2).mean(), scale=(1e-5, 1e6)) / 60.0

        @dn.datanode
        def volume_indicator():
            vol = 0.0

            while True:
                data = yield
                vol = max(0.0, vol - decay, min(1.0, volume_of(data)))
                self.volume = vol

        handler = dn.pipe(lambda a: a[0], dn.branch(volume_indicator()))
        mixer.add_effect(handler, zindex=(-1,))

        def widget_func(arg):
            time, ran = arg
            width = len(ran)
            text = mu.Text("▮" * int(self.volume * width))
            res = mu.replace_slot(template, text)
            return [(0, res)]

        return widget_func


@dataclasses.dataclass
class KnockMeterWidgetSettings:
    r"""
    Fields
    ------
    template : str
        The template for the volume indicator.
    decay_time : float
        The decay time.
    """
    template: str = "[color=bright_magenta][slot/][/]"
    decay_time: float = 0.2


class KnockMeterWidget:
    def __init__(self, settings):
        self.strength = 0.0
        self.settings = settings

    @dn.datanode
    def hit_listener(self):
        while True:
            _, time, strength, detected = yield

            if detected:
                self.strength = strength

    def load(self, provider):
        rich = provider.get(mu.RichParser)
        detector = provider.get(engines.Detector)
        renderer = provider.get(engines.Renderer)

        ticks = " ▏▎▍▌▋▊▉█"
        nticks = len(ticks) - 1
        decay_time = self.settings.decay_time
        display_framerate = renderer.settings.display_framerate
        decay = 1.0 / display_framerate / decay_time

        template = rich.parse(self.settings.template, slotted=True)

        detector.add_listener(self.hit_listener())

        def knock_func(arg):
            length = len(arg[1])
            value = int(max(0.0, self.strength) * length * nticks)
            if self.strength > 0:
                self.strength -= decay
            text = mu.Text(
                "".join(
                    ticks[min(nticks, max(0, value - i * nticks))]
                    for i in range(length)
                )
            )
            res = mu.replace_slot(template, text)
            return [(0, res)]

        return knock_func


@dataclasses.dataclass
class AccuracyMeterWidgetSettings:
    r"""
    Fields
    ------
    meter_width : int
        The text width of the meter.
    meter_decay_time : float
        The decay time of hitting lines on the meter.
    meter_radius : float
        The maximum accuracy error that will be displayed on the meter.
    """
    meter_width: int = 8
    meter_decay_time: float = 1.5
    meter_radius: float = 0.10


class AccuracyMeterWidget:
    def __init__(self, accuracy_getter, settings):
        # accuracy_getter: DataNode[None -> List[float]]
        self.accuracy_getter = dn.DataNode.wrap(accuracy_getter)
        self.hit = []
        self.settings = settings

    @dn.datanode
    def widget_node(self, decay, to_hit, render_heat):
        with self.accuracy_getter:
            time, ran = yield
            while True:
                try:
                    new_errs = self.accuracy_getter.send(None)
                except StopIteration:
                    return

                new_hit = [to_hit(err) for err in new_errs if err is not None]

                for i in range(len(self.hit)):
                    if i in new_hit:
                        self.hit[i] = 1.0
                    else:
                        self.hit[i] = max(0.0, self.hit[i] - decay)

                res = mu.Group(
                    tuple(
                        render_heat(i, j) for i, j in zip(self.hit[::2], self.hit[1::2])
                    )
                )

                time, ran = yield [(0, res)]

    def load(self, provider):
        rich = provider.get(mu.RichParser)
        renderer = provider.get(engines.Renderer)

        meter_width = self.settings.meter_width
        meter_decay_time = self.settings.meter_decay_time
        meter_radius = self.settings.meter_radius
        display_framerate = renderer.settings.display_framerate
        decay = 1 / display_framerate / meter_decay_time

        length = meter_width * 2
        self.hit = [0.0] * length

        def to_hit(err):
            hit = int((err - meter_radius) / -meter_radius / 2 * length // 1)
            return max(min(hit, length - 1), 0)

        colors = [c << 16 | c << 8 | c for c in range(8, 248, 10)]
        nlevel = len(colors)
        prerendered_heat = [
            [rich.parse(f"[bgcolor={a:06x}][color={b:06x}]▐[/][/]") for b in colors]
            for a in colors
        ]

        def render_heat(heat1, heat2):
            i = int(heat1 * (nlevel - 1))
            j = int(heat2 * (nlevel - 1))
            return prerendered_heat[i][j]

        return self.widget_node(decay, to_hit, render_heat)


class MonitorTarget(Enum):
    mixer = "mixer"
    detector = "detector"
    renderer = "renderer"


@dataclasses.dataclass
class MonitorWidgetSettings:
    target: MonitorTarget = MonitorTarget.renderer


class MonitorWidget:
    def __init__(self, settings):
        self.settings = settings

    def load(self, provider):
        if self.settings.target == MonitorTarget.mixer:
            target = provider.get(engines.Mixer)
        elif self.settings.target == MonitorTarget.detector:
            target = provider.get(engines.Detector)
        elif self.settings.target == MonitorTarget.renderer:
            target = provider.get(engines.Renderer)
        else:
            raise TypeError

        ticks = " ▏▎▍▌▋▊▉█"
        ticks_len = len(ticks)
        monitor = target.monitor if target is not None else None

        def widget_func(arg):
            time, ran = arg
            if monitor is None:
                return []
            if monitor.eff is None:
                return []
            width = len(ran)
            level = int(monitor.eff * width * (ticks_len - 1))
            res = mu.Text(
                "".join(
                    ticks[max(0, min(ticks_len - 1, level - i * (ticks_len - 1)))]
                    for i in range(width)
                )
            )
            return [(0, res)]

        return widget_func


@dataclasses.dataclass
class ScoreWidgetSettings:
    r"""
    Fields
    ------
    template : str
        The template for the score indicator.
    """
    template: str = "[color=bright_blue][slot/][/]"


class ScoreWidget:
    def __init__(self, score_getter, settings):
        # score_getter: DataNode[None -> (int, int)]
        self.score_getter = dn.DataNode.wrap(score_getter)
        self.settings = settings

    @dn.datanode
    def widget_node(self, template):
        with self.score_getter:
            time, ran = yield
            while True:
                try:
                    score, full_score = self.score_getter.send(None)
                except StopIteration:
                    return
                width = len(ran)

                if width == 0:
                    res = mu.Text("")
                elif width == 1:
                    res = mu.replace_slot(template, mu.Text("|"))
                elif width == 2:
                    res = mu.replace_slot(template, mu.Text("[]"))
                elif width <= 7:
                    score_str = uint_format(score, width - 2, True)
                    res = mu.replace_slot(template, mu.Text(f"[{score_str}]"))
                else:
                    w1 = max((width - 3) // 2, 5)
                    w2 = (width - 3) - w1
                    score_str = uint_format(score, w1, True)
                    full_score_str = uint_format(full_score, w2, True)
                    res = mu.replace_slot(
                        template, mu.Text(f"[{score_str}/{full_score_str}]")
                    )

                time, ran = yield [(0, res)]

    def load(self, provider):
        rich = provider.get(mu.RichParser)

        template = rich.parse(self.settings.template, slotted=True)

        return self.widget_node(template)


@dataclasses.dataclass
class ProgressWidgetSettings:
    r"""
    Fields
    ------
    template : str
        The template for the progress indicator.
    """
    template: str = "[color=bright_blue][slot/][/]"


class ProgressWidget:
    def __init__(self, progress_getter, time_getter, settings):
        # progress_getter: DataNode[None -> float]
        # time_getter: DataNode[None -> float]
        self.progress_getter = dn.DataNode.wrap(progress_getter)
        self.time_getter = dn.DataNode.wrap(time_getter)
        self.settings = settings

    @dn.datanode
    def widget_node(self, template):
        with self.progress_getter, self.time_getter:
            time, ran = yield
            while True:
                try:
                    progress = self.progress_getter.send(None)
                    time = self.time_getter.send(None)
                except StopIteration:
                    return

                progress = max(min(1.0, progress), 0.0)
                time = int(max(0.0, time))
                width = len(ran)

                if width == 0:
                    res = mu.Text("")
                elif width == 1:
                    res = mu.replace_slot(template, mu.Text("|"))
                elif width == 2:
                    res = mu.replace_slot(template, mu.Text("[]"))
                elif width <= 7:
                    progress_str = pc_format(progress, width - 2)
                    res = mu.replace_slot(template, mu.Text(f"[{progress_str}]"))
                else:
                    w1 = max((width - 3) // 2, 5)
                    w2 = (width - 3) - w1
                    progress_str = pc_format(progress, w1)
                    time_str = time_format(time, w2)
                    res = mu.replace_slot(template, mu.Text(f"[{progress_str}/{time_str}]"))

                time, ran = yield [(0, res)]

    def load(self, provider):
        rich = provider.get(mu.RichParser)

        template = rich.parse(self.settings.template, slotted=True)

        return self.widget_node(template)


@dataclasses.dataclass(frozen=True)
class Caret(mu.Pair):
    name = "caret"


@dataclasses.dataclass
class TextBoxWidgetSettings:
    r"""
    Fields
    ------
    caret_margins : tuple of int and int
        The width of left/right margins of caret.
    overflow_ellipses : tuple of str and str
        Texts to display when overflowing left/right.

    caret_normal_appearance : str
        The markup template of the normal-style caret.
    caret_blinking_appearance : str
        The markup template of the blinking-style caret.
    caret_highlighted_appearance : str
        The markup template of the highlighted-style caret.
    caret_blink_ratio : float
        The ratio to blink.
    """
    caret_margins: Tuple[int, int] = (3, 3)
    overflow_ellipses: Tuple[str, str] = ("…", "…")

    caret_normal_appearance: str = "[slot/]"
    caret_blinking_appearance: str = "[weight=dim][invert][slot/][/][/]"
    caret_highlighted_appearance: str = "[weight=bold][invert][slot/][/][/]"
    caret_blink_ratio: float = 0.3


class TextBox:
    def __init__(self, text_node_getter, settings):
        r"""Constructor.

        Parameters
        ----------
        text_node_getter : function
        settings : TextBoxWidgetSettings
        """
        self.text_node_getter = text_node_getter
        self.settings = settings

        self.text_offset = 0
        self.left_overflow = False
        self.right_overflow = False

    def text_geometry(self, markup, text_width=0, caret_masks=(), *, rich):
        if isinstance(markup, mu.Text):
            w = rich.widthof(markup.string)
            if w == -1:
                raise TypeError(f"invalid text: {markup.string!r}")
            text_width += w
            return text_width, caret_masks

        elif isinstance(markup, (mu.Group, mu.SGR)):
            res = text_width, caret_masks
            for child in markup.children:
                res = self.text_geometry(child, *res, rich=rich)
            return res

        elif isinstance(markup, Caret):
            start = text_width
            res = text_width, caret_masks
            for child in markup.children:
                res = self.text_geometry(child, *res, rich=rich)
            text_width, caret_masks = res
            stop = text_width
            caret_masks = (*caret_masks, slice(start, stop))
            return text_width, caret_masks

        elif isinstance(markup, (mu.CSI, mu.ControlCharacter)):
            raise TypeError(f"invalid markup type: {type(markup)}")

        else:
            raise TypeError(f"unknown markup type: {type(markup)}")

    def shift_text(self, text_width, caret_masks, box_width, *, left_margin, right_margin):
        # trim empty spaces
        if text_width - self.text_offset < box_width - right_margin:
            # from: ......[....I...    ]
            #   to: ...[.......I... ]
            self.text_offset = max(0, text_width - box_width + right_margin)

        # reveal the rightmost caret
        caret_stop = max((caret_slice.stop for caret_slice in caret_masks), default=float("-inf"))
        if caret_stop - self.text_offset > box_width - right_margin:
            # from: ...[............]..I....
            #   to: ........[..........I.]..
            self.text_offset = caret_stop - box_width + right_margin

        # reveal the leftmost caret
        caret_start = min((caret_slice.start for caret_slice in caret_masks), default=float("inf"))
        if caret_start - self.text_offset < left_margin:
            # from: .....I...[............]...
            #   to: ...[.I..........].........
            self.text_offset = max(caret_start - left_margin, 0)

        # determine overflow
        self.left_overflow = self.text_offset > 0
        self.right_overflow = text_width - self.text_offset > box_width

    def draw_ellipses(self, box_width, *, left_ellipsis, right_ellipsis, right_ellipsis_width):
        res = []
        if self.left_overflow:
            res.append((0, left_ellipsis))
        if self.right_overflow:
            res.append((box_width - right_ellipsis_width, right_ellipsis))
        return res

    @dn.datanode
    def adjust_view(self, *, rich):
        caret_margins = self.settings.caret_margins
        overflow_ellipses = self.settings.overflow_ellipses

        left_ellipsis = rich.parse(overflow_ellipses[0])
        left_ellipsis_width = rich.widthof(left_ellipsis)
        right_ellipsis = rich.parse(overflow_ellipses[1])
        right_ellipsis_width = rich.widthof(right_ellipsis)

        if left_ellipsis_width == -1 or right_ellipsis_width == -1:
            raise ValueError(f"invalid ellipsis: {overflow_ellipses!r}")

        left_margin = max(caret_margins[0], left_ellipsis_width)
        right_margin = max(caret_margins[1], right_ellipsis_width)

        text_geometry = dn.starcachemap(self.text_geometry, rich=rich)

        shift_text = dn.starmap(
            self.shift_text,
            left_margin=left_margin,
            right_margin=right_margin,
        )

        draw_ellipses = dn.map(
            self.draw_ellipses,
            left_ellipsis=left_ellipsis,
            right_ellipsis=right_ellipsis,
            right_ellipsis_width=right_ellipsis_width,
        )

        with text_geometry, shift_text, draw_ellipses:
            markup, box_width = yield
            while True:
                text_width, caret_masks = text_geometry.send((markup,))
                shift_text.send((text_width, caret_masks, box_width))
                ellipses_res = draw_ellipses.send(box_width)

                markup, box_width = yield ellipses_res

    @dn.datanode
    def get_caret_template(self, *, rich, metronome):
        caret_blink_ratio = self.settings.caret_blink_ratio
        normal = rich.parse(self.settings.caret_normal_appearance, slotted=True)
        blinking = rich.parse(self.settings.caret_blinking_appearance, slotted=True)
        highlighted = rich.parse(self.settings.caret_highlighted_appearance, slotted=True)

        key_pressed_beat = 0
        time, key_pressed = yield
        while True:
            beat = metronome.beat(time)

            # don't blink while key pressing
            if beat < key_pressed_beat or beat % 1 < caret_blink_ratio:
                if beat % 4 < 1:
                    res = highlighted
                else:
                    res = blinking
            else:
                res = normal

            time, key_pressed = yield res
            if key_pressed:
                key_pressed_beat = metronome.beat(time) // -1 * -1

    @dn.datanode
    def render_caret(self, *, rich, metronome):
        def render_caret_cached(markup, caret_template):
            if caret_template is not None:
                markup = markup.traverse(
                    Caret,
                    lambda m: mu.replace_slot(caret_template, mu.Group(m.children)),
                    strategy=mu.TraverseStrategy.TopDown,
                )
            else:
                markup = markup.traverse(
                    Caret,
                    lambda m: mu.Group(m.children),
                    strategy=mu.TraverseStrategy.TopDown,
                )
            return markup.expand()

        get_caret_template = self.get_caret_template(rich=rich, metronome=metronome)
        render_caret_cached = dn.starcachemap(render_caret_cached)

        with get_caret_template, render_caret_cached:
            markup, time, key_pressed = yield
            while True:
                caret_template = get_caret_template.send((time, key_pressed))
                markup = render_caret_cached.send((markup, caret_template))
                markup, time, key_pressed = yield markup

    @dn.datanode
    def render_textbox(self, *, rich, metronome):
        text_node = self.text_node_getter()
        adjust_view = self.adjust_view(rich=rich)
        render_caret = self.render_caret(rich=rich, metronome=metronome)

        with text_node, adjust_view, render_caret:
            time, ran = yield
            while True:
                markup, key_pressed = text_node.send()
                ellipses = adjust_view.send((markup, len(ran)))
                markup = render_caret.send((markup, time, key_pressed))
                time, ran = yield [(-self.text_offset, markup), *ellipses]

    def load(self, provider):
        rich = provider.get(mu.RichParser)
        metronome = provider.get(engines.Metronome)

        return self.render_textbox(rich=rich, metronome=metronome)