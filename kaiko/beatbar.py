import time
import contextlib
import threading
from . import cfg
from . import datanodes as dn
from . import tui


@cfg.configurable
class BeatbarSettings:
    icon_width: int = 8
    header_width: int = 11
    footer_width: int = 12

    bar_shift: float = 0.1
    bar_flip: bool = False

class Beatbar:
    def __init__(self, icon_mask, header_mask, content_mask, footer_mask, bar_shift, bar_flip,
                       content_scheduler, current_icon, current_header, current_footer,
                       ref_time):
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

        self.ref_time = ref_time

    @classmethod
    @contextlib.contextmanager
    def initialize(clz, kerminal, ref_time=0.0, settings=BeatbarSettings()):
        icon_width = settings.icon_width
        header_width = settings.header_width
        footer_width = settings.footer_width
        bar_shift = settings.bar_shift
        bar_flip = settings.bar_flip

        icon_mask = slice(None, icon_width)
        header_mask = slice(icon_width+1, icon_width+1+header_width)
        content_mask = slice(icon_width+1+header_width+1, -1-footer_width-1)
        footer_mask = slice(-footer_width-1, -1)

        content_scheduler = dn.Scheduler()
        current_icon = dn.TimedVariable(value=lambda time, ran: "")
        current_header = dn.TimedVariable(value=lambda time, ran: "")
        current_footer = dn.TimedVariable(value=lambda time, ran: "")

        content_key = kerminal.renderer.add_drawer(content_scheduler, zindex=(0,))
        icon_key = kerminal.renderer.add_drawer(clz._masked_node(current_icon, icon_mask), zindex=(1,))
        header_key = kerminal.renderer.add_drawer(clz._masked_node(current_header, header_mask, ("\b[", "]")), zindex=(2,))
        footer_key = kerminal.renderer.add_drawer(clz._masked_node(current_footer, footer_mask, ("\b[", "]")), zindex=(3,))

        try:
            yield clz(icon_mask, header_mask, content_mask, footer_mask, bar_shift, bar_flip,
                      content_scheduler, current_icon, current_header, current_footer, ref_time)
        finally:
            kerminal.renderer.remove_drawer(content_key)
            kerminal.renderer.remove_drawer(icon_key)
            kerminal.renderer.remove_drawer(header_key)
            kerminal.renderer.remove_drawer(footer_key)

    @classmethod
    @contextlib.contextmanager
    def subbeatbar(clz, beatbar, ref_time):
        content_scheduler = dn.Scheduler()
        try:
            content_key = beatbar.content_scheduler.add_node(content_scheduler, zindex=(0,))
            yield clz(beatbar.icon_mask, beatbar.header_mask, beatbar.content_mask, beatbar.footer_mask, beatbar.bar_shift, beatbar.bar_flip,
                      content_scheduler, beatbar.current_icon, beatbar.current_header, beatbar.current_footer,
                      beatbar.ref_time + ref_time)
        finally:
            beatbar.current_icon.reset()
            beatbar.current_header.reset()
            beatbar.current_footer.reset()
            beatbar.content_scheduler.remove_node(content_key)

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
        if hasattr(icon, '__call__'):
            icon_func = lambda time, ran: icon(time-self.ref_time, ran)
        elif isinstance(icon, str):
            icon_func = lambda time, ran: icon
        else:
            raise ValueError
        self.current_icon.set(icon_func, start, duration)

    def set_header(self, header, start=None, duration=None):
        if hasattr(header, '__call__'):
            header_func = lambda time, ran: header(time-self.ref_time, ran)
        elif isinstance(header, str):
            header_func = lambda time, ran: header
        else:
            raise ValueError
        self.current_header.set(header_func, start, duration)

    def set_footer(self, footer, start=None, duration=None):
        if hasattr(footer, '__call__'):
            footer_func = lambda time, ran: footer(time-self.ref_time, ran)
        elif isinstance(footer, str):
            footer_func = lambda time, ran: footer
        else:
            raise ValueError
        self.current_footer.set(footer_func, start, duration)

    def add_content_drawer(self, node, zindex=(0,)):
        return self.content_scheduler.add_node(self._shifted_node(node, self.ref_time), zindex=zindex)

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

    @staticmethod
    @dn.datanode
    def _shifted_node(node, ref_time):
        with node:
            view, time, width = yield
            while True:
                view, time, width = yield node.send((view, time-ref_time, width))

    def remove_content_drawer(self, key):
        self.content_scheduler.remove_node(key)

