import time
import contextlib
from enum import Enum
from typing import List, Tuple, Dict, Optional, Union
import queue
import threading
from . import cfg
from . import datanodes as dn
from . import tui
from .kerminal import Mixer, MixerSettings, Detector, DetectorSettings, Renderer, RendererSettings


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

class BeatbarSettings(cfg.Configurable):
    mixer = MixerSettings
    detector = DetectorSettings
    renderer = RendererSettings

    class layout(cfg.Configurable):
        icon_width: int = 8
        header_width: int = 11
        footer_width: int = 12

        bar_shift: float = 0.1
        bar_flip: bool = False

        header_quotes: Tuple[str, str] = ("\b\x1b[38;5;93;1m[\x1b[m", "\x1b[38;5;93;1m]\x1b[m")
        footer_quotes: Tuple[str, str] = ("\b\x1b[38;5;93;1m[\x1b[m", "\x1b[38;5;93;1m]\x1b[m")


    class scrollingbar(cfg.Configurable):
        performances_appearances: Dict[PerformanceGrade, Tuple[str, str]] = {
            PerformanceGrade.MISS               : (""   , ""     ),

            PerformanceGrade.LATE_FAILED        : ("\b\x1b[95m⟪\x1b[m", "\t\t\x1b[95m⟫\x1b[m"),
            PerformanceGrade.LATE_BAD           : ("\b\x1b[95m⟨\x1b[m", "\t\t\x1b[95m⟩\x1b[m"),
            PerformanceGrade.LATE_GOOD          : ("\b\x1b[95m‹\x1b[m", "\t\t\x1b[95m›\x1b[m"),
            PerformanceGrade.PERFECT            : (""   , ""     ),
            PerformanceGrade.EARLY_GOOD         : ("\t\t\x1b[95m›\x1b[m", "\b\x1b[95m‹\x1b[m"),
            PerformanceGrade.EARLY_BAD          : ("\t\t\x1b[95m⟩\x1b[m", "\b\x1b[95m⟨\x1b[m"),
            PerformanceGrade.EARLY_FAILED       : ("\t\t\x1b[95m⟫\x1b[m", "\b\x1b[95m⟪\x1b[m"),

            PerformanceGrade.LATE_FAILED_WRONG  : ("\b\x1b[95m⟪\x1b[m", "\t\t\x1b[95m⟫\x1b[m"),
            PerformanceGrade.LATE_BAD_WRONG     : ("\b\x1b[95m⟨\x1b[m", "\t\t\x1b[95m⟩\x1b[m"),
            PerformanceGrade.LATE_GOOD_WRONG    : ("\b\x1b[95m‹\x1b[m", "\t\t\x1b[95m›\x1b[m"),
            PerformanceGrade.PERFECT_WRONG      : (""   , ""     ),
            PerformanceGrade.EARLY_GOOD_WRONG   : ("\t\t\x1b[95m›\x1b[m", "\b\x1b[95m‹\x1b[m"),
            PerformanceGrade.EARLY_BAD_WRONG    : ("\t\t\x1b[95m⟩\x1b[m", "\b\x1b[95m⟨\x1b[m"),
            PerformanceGrade.EARLY_FAILED_WRONG : ("\t\t\x1b[95m⟫\x1b[m", "\b\x1b[95m⟪\x1b[m"),
            }
        performance_sustain_time: float = 0.1

        sight_appearances: Union[List[str], List[Tuple[str, str]]] = ["\x1b[95m⛶\x1b[m",
                                                                      "\x1b[38;5;201m🞎\x1b[m",
                                                                      "\x1b[38;5;200m🞏\x1b[m",
                                                                      "\x1b[38;5;199m🞐\x1b[m",
                                                                      "\x1b[38;5;198m🞑\x1b[m",
                                                                      "\x1b[38;5;197m🞒\x1b[m",
                                                                      "\x1b[38;5;196m🞓\x1b[m"]
        hit_decay_time: float = 0.4
        hit_sustain_time: float = 0.1

class Beatbar:
    def __init__(self, mixer, detector, renderer,
                       icon_mask, header_mask, content_mask, footer_mask, bar_shift, bar_flip,
                       content_scheduler, current_icon, current_header, current_footer,
                       current_hit_hint, current_perf_hint, current_sight, target_queue):
        self.mixer = mixer
        self.detector = detector
        self.renderer = renderer

        self.icon_mask = icon_mask
        self.header_mask = header_mask
        self.content_mask = content_mask
        self.footer_mask = footer_mask

        self.bar_shift = bar_shift
        self.bar_flip = bar_flip

        self.content_scheduler = content_scheduler
        self.current_icon = current_icon
        self.current_header = current_header
        self.current_footer = current_footer

        self.current_hit_hint = current_hit_hint
        self.current_perf_hint = current_perf_hint
        self.current_sight = current_sight
        self.target_queue = target_queue

    @classmethod
    def create(clz, settings, manager, ref_time):
        icon_width = settings.layout.icon_width
        header_width = settings.layout.header_width
        footer_width = settings.layout.footer_width
        bar_shift = settings.layout.bar_shift
        bar_flip = settings.layout.bar_flip
        header_quotes = settings.layout.header_quotes
        footer_quotes = settings.layout.footer_quotes

        icon_mask = slice(None, icon_width)
        header_mask = slice(icon_width+1, icon_width+1+header_width)
        content_mask = slice(icon_width+1+header_width+1, -1-footer_width-1)
        footer_mask = slice(-footer_width-1, -1)

        content_scheduler = dn.Scheduler()
        current_icon = dn.TimedVariable(value=lambda time, ran: "")
        current_header = dn.TimedVariable(value=lambda time, ran: "")
        current_footer = dn.TimedVariable(value=lambda time, ran: "")

        icon_drawer = clz._masked_node(current_icon, icon_mask)
        header_drawer = clz._masked_node(current_header, header_mask, header_quotes)
        footer_drawer = clz._masked_node(current_footer, footer_mask, footer_quotes)

        # sight
        hit_decay_time = settings.scrollingbar.hit_decay_time
        hit_sustain_time = settings.scrollingbar.hit_sustain_time
        perf_appearances = settings.scrollingbar.performances_appearances
        sight_appearances = settings.scrollingbar.sight_appearances
        perf_sustain_time = settings.scrollingbar.performance_sustain_time
        hit_hint_duration = max(hit_decay_time, hit_sustain_time)

        default_sight = clz._get_default_sight(hit_decay_time, hit_sustain_time,
                                               perf_appearances, sight_appearances)

        current_hit_hint = dn.TimedVariable(value=None, duration=hit_hint_duration)
        current_perf_hint = dn.TimedVariable(value=(None, None), duration=perf_sustain_time)
        current_sight = dn.TimedVariable(value=default_sight)

        # hit handler
        target_queue = queue.Queue()
        hit_handler = clz._hit_handler(current_hit_hint, target_queue)

        # build mixer, detector, renderer
        mixer_knot, mixer = Mixer.create(settings.mixer, manager, ref_time)
        detector_knot, detector = Detector.create(settings.detector, manager, ref_time)
        renderer_knot, renderer = Renderer.create(settings.renderer, ref_time)

        beatbar_knot = dn.pipe(mixer_knot, detector_knot, renderer_knot)

        # register handlers
        renderer.add_drawer(content_scheduler, zindex=(0,))
        renderer.add_drawer(icon_drawer, zindex=(1,))
        renderer.add_drawer(header_drawer, zindex=(2,))
        renderer.add_drawer(footer_drawer, zindex=(3,))
        detector.add_listener(hit_handler)

        self = clz(mixer, detector, renderer,
                   icon_mask, header_mask, content_mask, footer_mask, bar_shift, bar_flip,
                   content_scheduler, current_icon, current_header, current_footer,
                   current_hit_hint, current_perf_hint, current_sight, target_queue)

        self.draw_content(0.0, self._sight_drawer, zindex=(2,))

        return beatbar_knot, self

    @staticmethod
    @dn.datanode
    def _masked_node(variable, mask, enclosed_by=None):
        view, time, width = yield

        while True:
            mask_ran = range(width)[mask]
            func = variable.get(time)
            text = func(time, mask_ran)
            start = mask_ran.start

            text_ran, _ = tui.textrange1(start, text)

            view = tui.clear1(view, width, xmask=mask)
            view, _ = tui.addtext1(view, width, start, text, xmask=mask)

            if text_ran.start < mask_ran.start:
                view, _ = tui.addtext1(view, width, mask_ran.start, "…")

            if text_ran.stop > mask_ran.stop:
                view, _ = tui.addtext1(view, width, mask_ran.stop-1, "…")

            if enclosed_by is not None:
                view, _ = tui.addtext1(view, width, mask_ran.start, enclosed_by[0])
                view, _ = tui.addtext1(view, width, mask_ran.stop, enclosed_by[1])

            view, time, width = yield view

    def set_icon(self, icon, start=None, duration=None):
        icon_func = icon if hasattr(icon, '__call__') else lambda time, ran: icon
        self.current_icon.set(icon_func, start, duration)

    def set_header(self, header, start=None, duration=None):
        header_func = header if hasattr(header, '__call__') else lambda time, ran: header
        self.current_header.set(header_func, start, duration)

    def set_footer(self, footer, start=None, duration=None):
        footer_func = footer if hasattr(footer, '__call__') else lambda time, ran: footer
        self.current_footer.set(footer_func, start, duration)

    def add_content_drawer(self, node, zindex=(0,)):
        return self.content_scheduler.add_node(node, zindex=zindex)

    def _draw_content(self, view, width, pos, text):
        mask = self.content_mask

        pos = pos + self.bar_shift
        if self.bar_flip:
            pos = 1 - pos

        content_start, content_end, _ = mask.indices(width)
        index = round(content_start + pos * max(0, content_end - content_start - 1))

        if isinstance(text, tuple):
            text = text[self.bar_flip]

        return tui.addtext1(view, width, index, text, xmask=mask)

    def draw_content(self, pos, text, start=None, duration=None, zindex=(0,)):
        pos_func = pos if hasattr(pos, '__call__') else lambda time: pos
        text_func = text if hasattr(text, '__call__') else lambda time: text

        @dn.datanode
        def _content_node(pos, text, start, duration):
            view, time, width = yield

            if start is None:
                start = time

            while time < start:
                view, time, width = yield view

            while duration is None or time < start + duration:
                view, _ = self._draw_content(view, width, pos_func(time), text_func(time))
                view, time, width = yield view

        node = _content_node(pos, text, start, duration)
        return self.add_content_drawer(node, zindex=zindex)

    def remove_content_drawer(self, key):
        self.content_scheduler.remove_node(key)


    @staticmethod
    @dn.datanode
    def _hit_handler(current_hit_hint, target_queue):
        target, start, duration = None, None, None
        waiting_targets = []

        while True:
            # update hit signal
            _, time, strength, detected = yield

            strength = min(1.0, strength)
            if detected:
                current_hit_hint.set(strength)

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

    @staticmethod
    def _get_default_sight(hit_decay_time, hit_sustain_time, perf_appearances, sight_appearances):
        def _default_sight(time, hit_hint, perf_hint):
            hit_strength, hit_time, _ = hit_hint
            (perf, perf_is_reversed), perf_time, _ = perf_hint

            # draw perf hint
            if perf is not None:
                perf_text = perf_appearances[perf.grade]
                if perf_is_reversed:
                    perf_text = perf_text[::-1]
            else:
                perf_text = ("", "")

            # draw sight
            if hit_strength is not None:
                strength = hit_strength - (time - hit_time) / hit_decay_time
                strength = max(0.0, min(1.0, strength))
                loudness = int(strength * (len(sight_appearances) - 1))
                if time - hit_time < hit_sustain_time:
                    loudness = max(1, loudness)
                sight_text = sight_appearances[loudness]

            else:
                sight_text = sight_appearances[0]

            if isinstance(sight_text, str):
                sight_text = (sight_text, sight_text)

            return (perf_text[0]+"\r"+sight_text[0], perf_text[1]+"\r"+sight_text[1])

        return _default_sight

