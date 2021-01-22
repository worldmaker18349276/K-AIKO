import wcwidth
from enum import Enum
from typing import List, Tuple, Dict, Optional, Union
import threading
import queue
import numpy
from . import cfg
from . import datanodes as dn
from . import tui


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


class TimedVariable:
    def __init__(self, value=None, duration=numpy.inf):
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._scheduled = []
        self._default_value = value
        self._default_duration = duration
        self._item = (value, None, numpy.inf)

    def get(self, time, ret_sched=False):
        with self._lock:
            value, start, duration = self._item
            if start is None:
                start = time

            while not self._queue.empty():
                item = self._queue.get()
                if item[1] is None:
                    item = (item[0], time, item[2])
                self._scheduled.append(item)
            self._scheduled.sort(key=lambda item: item[1])

            while self._scheduled and self._scheduled[0][1] <= time:
                value, start, duration = self._scheduled.pop(0)

            if start + duration <= time:
                value, start, duration = self._default_value, None, numpy.inf

            self._item = (value, start, duration)
            return value if not ret_sched else self._item

    def set(self, value, start=None, duration=None):
        if duration is None:
            duration = self._default_duration
        self._queue.put((value, start, duration))

    def reset(self, start=None):
        self._queue.put((self._default_value, start, numpy.inf))


@cfg.configurable
class BeatbarSettings:
    icon_width: int = 8
    header_width: int = 11
    footer_width: int = 12

class Beatbar:
    settings: BeatbarSettings = BeatbarSettings()

    def __init__(self):
        icon_width = self.settings.icon_width
        header_width = self.settings.header_width
        footer_width = self.settings.footer_width
        layout = to_slices((icon_width, 1, header_width, 1, ..., 1, footer_width, 1))
        self.icon_mask, _, self.header_mask, _, self.content_mask, _, self.footer_mask, _ = layout

        self.content_queue = queue.Queue()
        self.current_icon = TimedVariable(value=lambda time, ran: "")
        self.current_header = TimedVariable(value=lambda time, ran: "")
        self.current_footer = TimedVariable(value=lambda time, ran: "")

    @dn.datanode
    def node(self, start_time):
        self.start_time = start_time

        content_node = dn.schedule(self.content_queue)
        with content_node:
            time, view = yield

            while True:
                time_ = time - self.start_time
                view_range = range(len(view[0]) if view else 0)

                time_, view = content_node.send((time_, view))

                icon_func = self.current_icon.get(time_)
                icon_text = icon_func(time_, view_range[self.icon_mask])
                icon_start = view_range[self.icon_mask].start
                view = self._draw_masked(view, icon_start, icon_text, self.icon_mask)

                header_func = self.current_header.get(time_)
                header_text = header_func(time_, view_range[self.header_mask])
                header_start = view_range[self.header_mask].start
                view = self._draw_masked(view, header_start, header_text, self.header_mask, ("[", "]"))

                footer_func = self.current_footer.get(time_)
                footer_text = footer_func(time_, view_range[self.footer_mask])
                footer_start = view_range[self.footer_mask].start
                view = self._draw_masked(view, footer_start, footer_text, self.footer_mask, ("[", "]"))

                time, view = yield time, view

    def _draw_masked(self, view, start, text, mask, enclosed_by=None):
        mask_ran = range(len(view[0]) if view else 0)[mask]
        _, text_ran, _, _ = tui.textrange(0, start, text)

        view = tui.clear(view, xmask=mask)
        view, _, _ = tui.addtext(view, 0, start, text, xmask=mask)

        if text_ran.start < mask_ran.start:
            view, _, _ = tui.addtext(view, 0, mask_ran.start, "…")

        if text_ran.stop > mask_ran.stop:
            view, _, _ = tui.addtext(view, 0, mask_ran.stop-1, "…")

        if enclosed_by is not None:
            view, _, _ = tui.addtext(view, 0, mask_ran.start-len(enclosed_by[0]), enclosed_by[0])
            view, _, _ = tui.addtext(view, 0, mask_ran.stop, enclosed_by[1])

        return view

    @dn.datanode
    def _bar_drawer(self, variable, mask, enclosed_by=None):
        time, view = yield

        while True:
            time_ = time - self.start_time
            mask_ran = range(len(view[0]) if view else 0)[mask]

            text = variable.get(time_)(time_, mask_ran)
            _, text_ran, _, _ = tui.textrange(0, mask_ran.start, text)

            view = tui.clear(view, xmask=mask)
            view, _, _ = tui.addtext(view, 0, mask_ran.start, text, xmask=mask)

            if text_ran.start < mask_ran.start:
                view, _, _ = tui.addtext(view, 0, mask_ran.start, "…")

            if text_ran.stop > mask_ran.stop:
                view, _, _ = tui.addtext(view, 0, mask_ran.stop-1, "…")

            if enclosed_by is not None:
                view, _, _ = tui.addtext(view, 0, mask_ran.start-wcwidth.wcswidth(enclosed_by[0]), enclosed_by[0])
                view, _, _ = tui.addtext(view, 0, mask_ran.stop, enclosed_by[1])

            time, view = yield time, view

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
        key = object()
        self.content_queue.put((key, node, zindex))
        return key

    def remove_content_drawer(self, key):
        self.content_queue.put((key, None, (0,)))
