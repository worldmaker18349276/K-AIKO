import wcwidth
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

        self.content_scheduler = dn.Scheduler()
        self.current_icon = dn.TimedVariable(value=lambda time, ran: "")
        self.current_header = dn.TimedVariable(value=lambda time, ran: "")
        self.current_footer = dn.TimedVariable(value=lambda time, ran: "")

    def register_drawers(self, kerminal):
        kerminal.add_drawer(self.content_scheduler, zindex=(0,))
        kerminal.add_drawer(self._masked_node(self.current_icon, self.icon_mask), zindex=(1,))
        kerminal.add_drawer(self._masked_node(self.current_header, self.header_mask, ("\b[", "]")), zindex=(2,))
        kerminal.add_drawer(self._masked_node(self.current_footer, self.footer_mask, ("\b[", "]")), zindex=(3,))

    @dn.datanode
    def _masked_node(self, variable, mask, enclosed_by=None):
        view, time, height, width = yield

        while True:
            mask_ran = range(width)[mask]
            func = variable.get(time)
            text = func(time, mask_ran)
            start = mask_ran.start

            _, text_ran, _, _ = tui.textrange(0, start, text)

            view = tui.clear(view, height, width, xmask=mask)
            view, _, _ = tui.addtext(view, height, width, 0, start, text, xmask=mask)

            if text_ran.start < mask_ran.start:
                view, _, _ = tui.addtext(view, height, width, 0, mask_ran.start, "…")

            if text_ran.stop > mask_ran.stop:
                view, _, _ = tui.addtext(view, height, width, 0, mask_ran.stop-1, "…")

            if enclosed_by is not None:
                view, _, _ = tui.addtext(view, height, width, 0, mask_ran.start, enclosed_by[0])
                view, _, _ = tui.addtext(view, height, width, 0, mask_ran.stop, enclosed_by[1])

            view, time, height, width = yield view

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

    def remove_content_drawer(self, key):
        self.content_scheduler.remove_node(key)
