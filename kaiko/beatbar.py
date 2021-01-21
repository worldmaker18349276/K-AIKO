import wcwidth
from enum import Enum
from typing import List, Tuple, Dict, Optional, Union
import threading
import queue
import numpy
from . import cfg
from . import datanodes as dn


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

def cover(*rans):
    start = min(ran.start for ran in rans)
    stop = max(ran.stop for ran in rans)
    return range(start, stop)

def clamp(ran, ran_):
    start = min(max(ran.start, ran_.start), ran.stop)
    stop = max(min(ran.stop, ran_.stop), ran.start)
    return range(start, stop)

def addtext(cells, index, text, mask=slice(None, None, None)):
    ran = range(len(cells))

    for ch in text:
        width = wcwidth.wcwidth(ch)

        if ch == "\t":
            index += 1

        elif ch == "\b":
            index -= 1

        elif width == 0:
            index_ = index - 1
            if index_ in ran and cells[index_] == "":
                index_ -= 1
            if index_ in ran[mask]:
                cells[index_] += ch

        elif width == 2:
            index_ = index + 1
            if index in ran[mask] and index_ in ran[mask]:
                if index-1 in ran and cells[index] == "":
                    cells[index-1] = " "
                if index_+1 in ran and cells[index_+1] == "":
                    cells[index_+1] = " "
                cells[index] = ch
                cells[index_] = ""
            index += 2

        elif width == 1:
            if index in ran[mask]:
                if index-1 in ran and cells[index] == "":
                    cells[index-1] = " "
                if index+1 in ran and cells[index+1] == "":
                    cells[index+1] = " "
                cells[index] = ch
            index += 1

        else:
            raise ValueError

    return cells

def addpad(cells, index, pad, mask=slice(None, None, None)):
    ran = range(len(cells))
    indices = clamp(range(index, index+len(pad)), ran[mask])

    if indices:
        if indices[0]-1 in ran and cells[indices[0]] == "":
            cells[indices[0]-1] = " "
        if indices[-1] in ran and cells[indices[-1]] == "":
            cells[indices[-1]] = " "
        for i in indices:
            cells[i] = pad[i]

    return cells

def clear(cells, mask=slice(None, None, None)):
    ran = range(len(cells))
    start, stop, _ = mask.indices(len(cells))

    if start-1 in ran and cells[start] == "":
        cells[start-1] = " "
    if stop in ran and cells[stop] == "":
        cells[stop] = " "
    for i in ran[mask]:
        cells[i] = " "

    return cells

def textrange(index, text):
    start = index
    stop = index

    for ch in text:
        width = wcwidth.wcwidth(ch)

        if ch == "\t":
            index += 1

        elif ch == "\b":
            index -= 1

        elif width == 0:
            pass

        elif width == 2:
            start = min(start, index)
            stop = max(stop, index+2)
            index += 2

        elif width == 1:
            start = min(start, index)
            stop = max(stop, index+1)
            index += 1

    return index, range(start, stop)


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
                view0 = view[0]
                self.width = len(view0)
                cells_range = range(self.width)

                time_, view0 = content_node.send((time_, view0))

                icon = self.current_icon.get(time_)
                icon_text = icon(time_, cells_range[self.icon_mask])
                icon_start = cells_range[self.icon_mask].start
                view0 = self._draw_masked(view0, icon_start, icon_text, self.icon_mask)

                header = self.current_header.get(time_)
                header_text = header(time_, cells_range[self.header_mask])
                header_start = cells_range[self.header_mask].start
                view0 = self._draw_masked(view0, header_start, header_text, self.header_mask, ("[", "]"))

                footer = self.current_footer.get(time_)
                footer_text = footer(time_, cells_range[self.footer_mask])
                footer_start = cells_range[self.footer_mask].start
                view0 = self._draw_masked(view0, footer_start, footer_text, self.footer_mask, ("[", "]"))

                time, view = yield time, view

    def _draw_masked(self, view, start, text, mask, enclosed_by=None):
        mask_ran = range(self.width)[mask]
        _, text_ran = textrange(start, text)

        view = addtext(view, start, text, mask=mask)

        if text_ran.start < mask_ran.start:
            view = addtext(view, mask_ran.start, "…")

        if text_ran.stop > mask_ran.stop:
            view = addtext(view, mask_ran.stop-1, "…")

        if enclosed_by is not None:
            view = addtext(view, mask_ran.start-len(enclosed_by[0]), enclosed_by[0])
            view = addtext(view, mask_ran.stop-1+len(enclosed_by[1]), enclosed_by[1])

        return view

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
