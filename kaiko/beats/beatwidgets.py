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

@dataclasses.dataclass
class SpectrumWidget:
    spectrum: str
    rich: mu.RichParser
    mixer: engines.Mixer
    mixer_settings: engines.MixerSettings
    settings: SpectrumWidgetSettings

    def draw_spectrum(self):
        spec_width = self.settings.spec_width
        samplerate = self.mixer_settings.output_samplerate
        hop_length = round(samplerate * self.settings.spec_time_res)
        win_length = round(samplerate / self.settings.spec_freq_res)
        spec_decay_time = self.settings.spec_decay_time

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
        def draw():
            with node:
                vols = [0.0]*(spec_width*2)

                data = yield
                while True:
                    try:
                        J = node.send(data)
                    except StopIteration:
                        return

                    vols = [max(0.0, prev-decay, min(1.0, volume_of(J[slic])))
                            for slic, prev in zip(slices, vols)]
                    data = yield "".join(map(draw_bar, vols[0::2], vols[1::2]))

        return draw()

    @dn.datanode
    def load(self):
        spec_width = self.settings.spec_width
        samplerate = self.mixer_settings.output_samplerate
        nchannels = self.mixer_settings.output_channels
        hop_length = round(samplerate * self.settings.spec_time_res)

        template = self.rich.parse(self.settings.template, slotted=True)

        self.spectrum = "\u2800"*spec_width
        draw = dn.pipe(self.draw_spectrum(), lambda v: setattr(self, "spectrum", v))
        handler = dn.pipe(lambda a:a[0], dn.branch(dn.unchunk(draw, (hop_length, nchannels))))
        self.mixer.add_effect(handler, zindex=(-1,))

        def widget_func(time, ran):
            width = len(ran)
            text = mu.Text(f"{self.spectrum:^{width}.{width}s}")
            return mu.replace_slot(template, text)

        yield
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

@dataclasses.dataclass
class VolumeIndicatorWidget:
    volume: float
    rich: mu.RichParser
    mixer: engines.Mixer
    mixer_settings: engines.MixerSettings
    settings: VolumeIndicatorWidgetSettings

    @dn.datanode
    def load(self):
        vol_decay_time = self.settings.vol_decay_time
        buffer_length = self.mixer_settings.output_buffer_length
        samplerate = self.mixer_settings.output_samplerate

        template = self.rich.parse(self.settings.template, slotted=True)

        decay = buffer_length / samplerate / vol_decay_time

        volume_of = lambda x: dn.power2db((x**2).mean(), scale=(1e-5, 1e6)) / 60.0

        @dn.datanode
        def volume_indicator():
            vol = 0.0

            while True:
                data = yield
                vol = max(0.0, vol-decay, min(1.0, volume_of(data)))
                self.volume = vol

        handler = dn.pipe(lambda a:a[0], dn.branch(volume_indicator()))
        self.mixer.add_effect(handler, zindex=(-1,))

        def widget_func(time, ran):
            width = len(ran)
            text = mu.Text("▮" * int(self.volume * width))
            return mu.replace_slot(template, text)

        yield
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

@dataclasses.dataclass
class AccuracyMeterWidget:
    last_perf: int
    last_time: float
    rich: mu.RichParser
    state: object # with property `perfs`
    settings: AccuracyMeterWidgetSettings

    @dn.datanode
    def load(self):
        meter_width = self.settings.meter_width
        meter_decay_time = self.settings.meter_decay_time
        meter_radius = self.settings.meter_radius

        length = meter_width*2
        hit = [0.0]*length

        colors = [c << 16 | c << 8 | c for c in range(8, 248, 10)]
        nlevel = len(colors)
        texts = [[self.rich.parse(f"[bgcolor={a:06x}][color={b:06x}]▐[/][/]") for b in colors] for a in colors]

        def widget_func(time, ran):
            perfs = self.state.perfs

            new_err = []
            while len(perfs) > self.last_perf:
                err = perfs[self.last_perf].err
                if err is not None:
                    new_err.append(max(min(int((err-meter_radius)/-meter_radius/2 * length//1), length-1), 0))
                self.last_perf += 1

            decay = max(0.0, time - self.last_time) / meter_decay_time
            self.last_time = time

            for i in range(meter_width*2):
                if i in new_err:
                    hit[i] = 1.0
                else:
                    hit[i] = max(0.0, hit[i] - decay)

            return mu.Group(tuple(texts[int(i*(nlevel-1))][int(j*(nlevel-1))] for i, j in zip(hit[::2], hit[1::2])))

        yield
        return widget_func


class MonitorTarget(Enum):
    mixer = "mixer"
    detector = "detector"
    renderer = "renderer"

@dataclasses.dataclass
class MonitorWidgetSettings:
    target: MonitorTarget = MonitorTarget.renderer

@dataclasses.dataclass
class MonitorWidget:
    target: Union[engines.Mixer, engines.Detector, engines.Renderer, None]
    settings: MonitorWidgetSettings

    @dn.datanode
    def load(self):
        ticks = " ▏▎▍▌▋▊▉█"
        ticks_len = len(ticks)
        monitor = self.target.monitor if self.target is not None else None

        def widget_func(time, ran):
            if monitor is None:
                return mu.Text("")
            if monitor.eff is None:
                return mu.Text("")
            width = len(ran)
            level = int(monitor.eff * width*(ticks_len-1))
            return mu.Text("".join(ticks[max(0, min(ticks_len-1, level-i*(ticks_len-1)))] for i in range(width)))

        yield
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

@dataclasses.dataclass
class ScoreWidget:
    state: object # with properties `score`, `full_score`
    rich: mu.RichParser
    settings: ScoreWidgetSettings

    @dn.datanode
    def load(self):
        template = self.rich.parse(self.settings.template, slotted=True)

        def widget_func(time, ran):
            score = self.state.score
            full_score = self.state.full_score
            width = len(ran)

            if width == 0:
                return mu.Text("")
            if width == 1:
                return mu.replace_slot(template, mu.Text("|"))
            if width == 2:
                return mu.replace_slot(template, mu.Text("[]"))
            if width <= 7:
                score_str = uint_format(score, width-2, True)
                return mu.replace_slot(template, mu.Text(f"[{score_str}]"))

            w1 = max((width-3)//2, 5)
            w2 = (width-3) - w1
            score_str = uint_format(score, w1, True)
            full_score_str = uint_format(full_score, w2, True)
            return mu.replace_slot(template, mu.Text(f"[{score_str}/{full_score_str}]"))

        yield
        return widget_func


@dataclasses.dataclass
class ProgressWidgetSettings:
    r"""
    Fields
    ------
    template : str
        The template for the progress indicator.
    """
    template: str = "[color=bright_blue][slot/][/]"

@dataclasses.dataclass
class ProgressWidget:
    state: object # with properties `finished_subjects`, `total_subjects`, `time`
    rich: mu.RichParser
    settings: ProgressWidgetSettings

    @dn.datanode
    def load(self):
        template = self.rich.parse(self.settings.template, slotted=True)

        def widget_func(time, ran):
            finished_subjects = self.state.finished_subjects
            total_subjects = self.state.total_subjects
            time = self.state.time

            progress = min(1.0, finished_subjects/total_subjects) if total_subjects>0 else 1.0
            time = int(max(0.0, time))
            width = len(ran)

            if width == 0:
                return mu.Text("")
            if width == 1:
                return mu.replace_slot(template, mu.Text("|"))
            if width == 2:
                return mu.replace_slot(template, mu.Text("[]"))
            if width <= 7:
                progress_str = pc_format(progress, width-2)
                return mu.replace_slot(template, mu.Text(f"[{progress_str}]"))

            w1 = max((width-3)//2, 5)
            w2 = (width-3) - w1
            progress_str = pc_format(progress, w1)
            time_str = time_format(time, w2)
            return mu.replace_slot(template, mu.Text(f"[{progress_str}/{time_str}]"))

        yield
        return widget_func


@dataclasses.dataclass
class PatternsWidgetSettings:
    r"""
    Fields
    ------
    patterns : list of str
        The patterns to loop.
    """
    patterns: List[str] = dataclasses.field(default_factory=lambda: [
        "[color=cyan]⠶⠦⣚⠀⠶[/]",
        "[color=cyan]⢎⣀⡛⠀⠶[/]",
        "[color=cyan]⢖⣄⠻⠀⠶[/]",
        "[color=cyan]⠖⠐⡩⠂⠶[/]",
        "[color=cyan]⠶⠀⡭⠲⠶[/]",
        "[color=cyan]⠶⠀⣬⠉⡱[/]",
        "[color=cyan]⠶⠀⣦⠙⠵[/]",
        "[color=cyan]⠶⠠⣊⠄⠴[/]",
    ])

@dataclasses.dataclass
class PatternsWidget:
    metronome: engines.Metronome
    rich: mu.RichParser
    settings: PatternsWidgetSettings

    @dn.datanode
    def load(self):
        patterns = self.settings.patterns

        markuped_patterns = [self.rich.parse(pattern) for pattern in patterns]

        def patterns_func(time, ran):
            beat = self.metronome.beat(time)
            ind = int(beat * len(markuped_patterns) // 1) % len(markuped_patterns)
            return markuped_patterns[ind]

        yield
        return patterns_func

@dataclasses.dataclass
class MarkerWidgetSettings:
    r"""
    Fields
    ------
    markers : tuple of str and str
        The appearance of normal and blinking-style markers.
    blink_ratio : float
        The ratio to blink.
    """
    markers: Tuple[str, str] = ("❯ ", "[weight=bold]❯ [/]")
    blink_ratio: float = 0.3

@dataclasses.dataclass
class MarkerWidget:
    metronome: engines.Metronome
    rich: mu.RichParser
    settings: MarkerWidgetSettings

    @dn.datanode
    def load(self):
        markers = self.settings.markers
        blink_ratio = self.settings.blink_ratio

        markuped_markers = (
            self.rich.parse(markers[0]),
            self.rich.parse(markers[1]),
        )

        def marker_func(time, ran):
            beat = self.metronome.beat(time)
            if beat % 4 < min(1.0, blink_ratio):
                return markuped_markers[1]
            else:
                return markuped_markers[0]

        yield
        return marker_func

