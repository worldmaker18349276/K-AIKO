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

    def load(self, rich, mixer):
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
            return mu.replace_slot(template, text)

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

    def load(self, rich, mixer):
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
            return mu.replace_slot(template, text)

        return widget_func


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

                time, ran = yield res

    def load(self, rich, renderer_settings):
        meter_width = self.settings.meter_width
        meter_decay_time = self.settings.meter_decay_time
        meter_radius = self.settings.meter_radius
        display_framerate = renderer_settings.display_framerate
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
    def __init__(self, target, settings):
        self.target = target
        self.settings = settings

    def load(self):
        ticks = " ▏▎▍▌▋▊▉█"
        ticks_len = len(ticks)
        monitor = self.target.monitor if self.target is not None else None

        def widget_func(arg):
            time, ran = arg
            if monitor is None:
                return mu.Text("")
            if monitor.eff is None:
                return mu.Text("")
            width = len(ran)
            level = int(monitor.eff * width * (ticks_len - 1))
            return mu.Text(
                "".join(
                    ticks[max(0, min(ticks_len - 1, level - i * (ticks_len - 1)))]
                    for i in range(width)
                )
            )

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
                    return mu.Text("")
                if width == 1:
                    return mu.replace_slot(template, mu.Text("|"))
                if width == 2:
                    return mu.replace_slot(template, mu.Text("[]"))
                if width <= 7:
                    score_str = uint_format(score, width - 2, True)
                    return mu.replace_slot(template, mu.Text(f"[{score_str}]"))

                w1 = max((width - 3) // 2, 5)
                w2 = (width - 3) - w1
                score_str = uint_format(score, w1, True)
                full_score_str = uint_format(full_score, w2, True)
                res = mu.replace_slot(
                    template, mu.Text(f"[{score_str}/{full_score_str}]")
                )

                time, ran = yield res

    def load(self, rich):
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
                    return mu.Text("")
                if width == 1:
                    return mu.replace_slot(template, mu.Text("|"))
                if width == 2:
                    return mu.replace_slot(template, mu.Text("[]"))
                if width <= 7:
                    progress_str = pc_format(progress, width - 2)
                    return mu.replace_slot(template, mu.Text(f"[{progress_str}]"))

                w1 = max((width - 3) // 2, 5)
                w2 = (width - 3) - w1
                progress_str = pc_format(progress, w1)
                time_str = time_format(time, w2)
                res = mu.replace_slot(template, mu.Text(f"[{progress_str}/{time_str}]"))

                time, ran = yield res

    def load(self, rich):
        template = rich.parse(self.settings.template, slotted=True)

        return self.widget_node(template)


@dataclasses.dataclass
class PatternsWidgetSettings:
    r"""
    Fields
    ------
    patterns : list of str
        The patterns to loop.
    """
    patterns: List[str] = dataclasses.field(
        default_factory=lambda: [
            "[color=cyan]⠶⠦⣚⠀⠶[/]",
            "[color=cyan]⢎⣀⡛⠀⠶[/]",
            "[color=cyan]⢖⣄⠻⠀⠶[/]",
            "[color=cyan]⠖⠐⡩⠂⠶[/]",
            "[color=cyan]⠶⠀⡭⠲⠶[/]",
            "[color=cyan]⠶⠀⣬⠉⡱[/]",
            "[color=cyan]⠶⠀⣦⠙⠵[/]",
            "[color=cyan]⠶⠠⣊⠄⠴[/]",
        ]
    )


class PatternsWidget:
    def __init__(self, metronome, settings):
        self.metronome = metronome
        self.settings = settings

    def load(self, rich):
        patterns = self.settings.patterns

        markuped_patterns = [rich.parse(pattern) for pattern in patterns]

        def patterns_func(arg):
            time, ran = arg
            beat = self.metronome.beat(time)
            ind = int(beat * len(markuped_patterns) // 1) % len(markuped_patterns)
            return markuped_patterns[ind]

        return patterns_func


@dataclasses.dataclass
class MarkerWidgetSettings:
    r"""
    Fields
    ------
    normal_appearance : str
        The appearance of normal-style markers.
    blinking_appearance : str
        The appearance of blinking-style markers.
    blink_ratio : float
        The ratio to blink.
    """
    normal_appearance: str = "❯ "
    blinking_appearance: str = "[weight=bold]❯ [/]"
    blink_ratio: float = 0.3


class MarkerWidget:
    def __init__(self, metronome, settings):
        self.metronome = metronome
        self.settings = settings

    def load(self, rich):
        blink_ratio = self.settings.blink_ratio
        normal = rich.parse(self.settings.normal_appearance)
        blinking = rich.parse(self.settings.blinking_appearance)

        def marker_func(arg):
            time, ran = arg
            beat = self.metronome.beat(time)
            if beat % 4 < min(1.0, blink_ratio):
                return blinking
            else:
                return normal

        return marker_func
