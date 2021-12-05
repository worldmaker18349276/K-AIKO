import time
import math
import contextlib
from enum import Enum
from typing import List, Tuple, Dict, Optional, Union
import queue
import threading
import numpy
from kaiko.utils import config as cfg
from kaiko.utils import datanodes as dn
from kaiko.utils import markups as mu


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
            return Performance(PerformanceGrade((None, None)), time, None)

        is_wrong = not is_correct_key
        err = hit_time - time
        shift = next((i for i in range(3) if abs(err) < tol*(2*i+1)), 3)
        if err < 0:
            shift = -shift

        return Performance(PerformanceGrade((shift, is_wrong)), time, err)

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
    class layout(cfg.Configurable):
        r"""
        Fields
        ------
        icon_width : int
            The width of icon.

             ⣠⣴⣤⣿⣤⣦ \[00000/00400]       □  ⛶     □                 ■   \[11.3%|00:09]
            ^^^^^^^^
              here

        header_width : int
            The width of header.

             ⣠⣴⣤⣿⣤⣦ \[00000/00400]       □  ⛶     □                 ■   \[11.3%|00:09]
                    ^^^^^^^^^^^^^
                        here

        footer_width : int
            The width of footer.

             ⣠⣴⣤⣿⣤⣦ \[00000/00400]       □  ⛶     □                 ■   \[11.3%|00:09]
                                                                       ^^^^^^^^^^^^^
                                                                            here

        """
        icon_width: int = 8
        header_width: int = 13
        footer_width: int = 13

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
    def __init__(self, mixer, detector, renderer, controller, icon, header, footer, sight, bar_shift, bar_flip, settings=None):
        settings = settings or BeatbarSettings()

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

        self.current_icon = dn.TimedVariable(value=icon)
        self.current_header = dn.TimedVariable(value=header)
        self.current_footer = dn.TimedVariable(value=footer)

        # sight
        hit_decay_time = settings.sight.hit_decay_time
        hit_sustain_time = settings.sight.hit_sustain_time
        perf_appearances = settings.sight.performances_appearances
        sight_appearances = settings.sight.sight_appearances
        perf_sustain_time = settings.sight.performance_sustain_time

        self.current_hit_hint = dn.TimedVariable(value=None, duration=hit_sustain_time)
        self.current_perf_hint = dn.TimedVariable(value=(None, None), duration=perf_sustain_time)
        self.current_sight = dn.TimedVariable(value=sight)

        # hit handler
        self.target_queue = queue.Queue()
        hit_handler = Beatbar._hit_handler(self.current_hit_hint, self.target_queue, hit_decay_time, hit_sustain_time)

        # register handlers
        icon_drawer = lambda arg: (0, self.current_icon.get(arg[0])(arg[0], arg[1]))
        header_drawer = lambda arg: (0, self.current_header.get(arg[0])(arg[0], arg[1]))
        footer_drawer = lambda arg: (0, self.current_footer.get(arg[0])(arg[0], arg[1]))

        renderer.add_text(icon_drawer, xmask=self.icon_mask, zindex=(1,))
        renderer.add_text(header_drawer, xmask=self.header_mask, zindex=(2,))
        renderer.add_text(footer_drawer, xmask=self.footer_mask, zindex=(3,))
        detector.add_listener(hit_handler)

        self.draw_content(0.0, self._sight_drawer, zindex=(2,))

    def set_icon(self, icon, start=None, duration=None):
        icon_func = icon if hasattr(icon, '__call__') else lambda time, ran: icon
        self.current_icon.set(icon_func, start, duration)

    def set_header(self, header, start=None, duration=None):
        header_func = header if hasattr(header, '__call__') else lambda time, ran: header
        self.current_header.set(header_func, start, duration)

    def set_footer(self, footer, start=None, duration=None):
        footer_func = footer if hasattr(footer, '__call__') else lambda time, ran: footer
        self.current_footer.set(footer_func, start, duration)

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
                if duration is not None and start + duration <= time:
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
        hit_hint = self.current_hit_hint.get(time, ret_sched=True)
        perf_hint = self.current_perf_hint.get(time, ret_sched=True)

        # draw sight
        sight_func = self.current_sight.get(time)
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


# widgets
class Sight:
    def __init__(self, rich, settings, **_):
        self.rich = rich
        self.settings = settings

    @dn.datanode
    def load(self):
        perf_appearances = {key: (self.rich.parse(f"[restore]{appearance1}[/]"), self.rich.parse(f"[restore]{appearance2}[/]"))
                            for key, (appearance1, appearance2) in self.settings.performances_appearances.items()}
        sight_appearances = [(self.rich.parse(appearance1), self.rich.parse(appearance2))
                             for appearance1, appearance2 in self.settings.sight_appearances]

        def sight_func(time, hit_hint, perf_hint):
            hit_strength, hit_time, hit_duration = hit_hint
            (perf, perf_is_reversed), perf_time, _ = perf_hint

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

class SpectrumWidget:
    def __init__(self, rich, settings, *, state, mixer, devices_settings, **_):
        self.rich = rich
        self.settings = settings
        self.state = state
        self.mixer = mixer
        self.devices_settings = devices_settings
        self.spectrum = ""

    def draw_spectrum(self):
        spec_width = self.settings.spec_width
        samplerate = self.devices_settings.mixer.output_samplerate
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
        samplerate = self.devices_settings.mixer.output_samplerate
        nchannels = self.devices_settings.mixer.output_channels
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

class VolumeIndicatorWidget:
    def __init__(self, rich, settings, *, mixer, devices_settings, **_):
        self.rich = rich
        self.settings = settings
        self.mixer = mixer
        self.devices_settings = devices_settings
        self.volume = 0.0

    @dn.datanode
    def load(self):
        vol_decay_time = self.settings.vol_decay_time
        buffer_length = self.devices_settings.mixer.output_buffer_length
        samplerate = self.devices_settings.mixer.output_samplerate

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

class AccuracyMeterWidget:
    def __init__(self, rich, settings, *, state, **_):
        self.rich = rich
        self.settings = settings
        self.state = state
        self.last_perf = 0
        self.last_time = float("inf")

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

class MonitorWidget:
    def __init__(self, rich, settings, *, mixer, detector, renderer, **_):
        self.rich = rich
        self.settings = settings
        self.mixer = mixer
        self.detector = detector
        self.renderer = renderer

    @dn.datanode
    def load(self):
        ticks = " ▏▎▍▌▋▊▉█"
        ticks_len = len(ticks)
        monitor_target = self.settings.target
        if monitor_target is MonitorTarget.mixer:
            monitor = self.mixer.monitor
        elif monitor_target is MonitorTarget.detector:
            monitor = self.detector.monitor
        elif monitor_target is MonitorTarget.renderer:
            monitor = self.renderer.monitor
        else:
            monitor = None

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

class ScoreWidget:
    def __init__(self, rich, settings, *, state, **_):
        self.rich = rich
        self.settings = settings
        self.state = state

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

class ProgressWidget:
    def __init__(self, rich, settings, *, state, **_):
        self.rich = rich
        self.settings = settings
        self.state = state

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

class Widget(Enum):
    spectrum = SpectrumWidget
    volume_indicator = VolumeIndicatorWidget
    accuracy_meter = AccuracyMeterWidget
    monitor = MonitorWidget
    score = ScoreWidget
    progress = ProgressWidget

    def __repr__(self):
        return f"Widget.{self.name}"

class WidgetSettings(cfg.Configurable):
    r"""
    Fields
    ------
    icon_widget : Widget
        The widget on the icon.
    header_widget : Widget
        The widget on the header.
    footer_widget : Widget
        The widget on the footer.
    """
    icon_widget: Widget = Widget.spectrum
    header_widget: Widget = Widget.score
    footer_widget: Widget = Widget.progress

    class spectrum(cfg.Configurable):
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

    class volume_indicator(cfg.Configurable):
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

    class score(cfg.Configurable):
        r"""
        Fields
        ------
        template : str
            The template for the score indicator.
        """
        template: str = "[color=bright_blue][slot/][/]"

    class progress(cfg.Configurable):
        r"""
        Fields
        ------
        template : str
            The template for the progress indicator.
        """
        template: str = "[color=bright_blue][slot/][/]"

    class accuracy_meter(cfg.Configurable):
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

    class monitor(cfg.Configurable):
        target: MonitorTarget = MonitorTarget.renderer
