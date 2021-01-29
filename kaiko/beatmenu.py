import time
import contextlib
import threading
from . import cfg
from . import datanodes as dn
from . import tui
from .beatmap import BeatmapPlayer
from .beatsheet import BeatmapDraft


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


@cfg.configurable
class BeatbarSettings:
    icon_width: int = 8
    header_width: int = 11
    footer_width: int = 12

class Beatbar:
    def __init__(self, icon_mask, header_mask, content_mask, footer_mask,
                       content_scheduler, current_icon, current_header, current_footer,
                       ref_time):
        self.icon_mask = icon_mask
        self.header_mask = header_mask
        self.content_mask = content_mask
        self.footer_mask = footer_mask

        self.content_scheduler = content_scheduler
        self.current_icon = current_icon
        self.current_header = current_header
        self.current_footer = current_footer

        self.ref_time = ref_time

    @classmethod
    def initialize(clz, kerminal, ref_time=0.0, settings=BeatbarSettings()):
        icon_width = settings.icon_width
        header_width = settings.header_width
        footer_width = settings.footer_width
        layout = to_slices((icon_width, 1, header_width, 1, ..., 1, footer_width, 1))
        icon_mask, _, header_mask, _, content_mask, _, footer_mask, _ = layout

        content_scheduler = dn.Scheduler()
        current_icon = dn.TimedVariable(value=lambda time, ran: "")
        current_header = dn.TimedVariable(value=lambda time, ran: "")
        current_footer = dn.TimedVariable(value=lambda time, ran: "")

        kerminal.add_drawer(content_scheduler, zindex=(0,))
        kerminal.add_drawer(clz._masked_node(current_icon, icon_mask), zindex=(1,))
        kerminal.add_drawer(clz._masked_node(current_header, header_mask, ("\b[", "]")), zindex=(2,))
        kerminal.add_drawer(clz._masked_node(current_footer, footer_mask, ("\b[", "]")), zindex=(3,))

        return clz(icon_mask, header_mask, content_mask, footer_mask,
                   content_scheduler, current_icon, current_header, current_footer, ref_time)

    @classmethod
    @contextlib.contextmanager
    def subbeatbar(clz, beatbar, ref_time):
        content_scheduler = dn.Scheduler()
        try:
            content_key = beatbar.content_scheduler.add_node(content_scheduler, zindex=(0,))
            yield clz(beatbar.icon_mask, beatbar.header_mask, beatbar.content_mask, beatbar.footer_mask,
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
        elif isinstance(header, str):
            footer_func = lambda time, ran: footer
        else:
            raise ValueError
        self.current_footer.set(footer_func, start, duration)

    def add_content_drawer(self, node, zindex=(0,)):
        return self.content_scheduler.add_node(self._shifed_node(node, self.ref_time), zindex=zindex)

    @staticmethod
    @dn.datanode
    def _shifed_node(node, ref_time):
        with node:
            view, time, width = yield
            while True:
                view, time, width = yield node.send((view, time-ref_time, width))

    def remove_content_drawer(self, key):
        self.content_scheduler.remove_node(key)


class KAIKO:
    def __init__(self, filename):
        self.filename = filename

    @contextlib.contextmanager
    def connect(self, kerminal):
        self.kerminal = kerminal
        self.beatbar = Beatbar.initialize(kerminal)

        self.beatmap = BeatmapDraft.read(self.filename)
        self.game = BeatmapPlayer(self.beatmap)

        with self.game.connect(self.kerminal, self.beatbar) as main:
            yield main

    def start(self):
        return self.game.start()

    def is_alive(self):
        return self.game.is_alive()

