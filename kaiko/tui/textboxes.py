import dataclasses
from typing import Tuple
import numpy
from ..utils import datanodes as dn
from ..utils import markups as mu
from ..devices import engines


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
    def __init__(self, text_node, settings):
        r"""Constructor.

        Parameters
        ----------
        text_node : dn.datanode
        settings : TextBoxWidgetSettings
        """
        self.text_node = text_node
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
        normal_template = rich.parse(self.settings.caret_normal_appearance, slotted=True)
        blinking_template = rich.parse(self.settings.caret_blinking_appearance, slotted=True)
        highlighted_template = rich.parse(self.settings.caret_highlighted_appearance, slotted=True)

        key_pressed_beat = 0
        time, key_pressed = yield
        while True:
            beat = metronome.beat(time)

            # don't blink while key pressing
            if beat < key_pressed_beat or beat % 1 < caret_blink_ratio:
                if beat % 4 < 1:
                    res = highlighted_template
                else:
                    res = blinking_template
            else:
                res = normal_template

            time, key_pressed = yield res
            if key_pressed:
                key_pressed_beat = metronome.beat(time) // -1 * -1

    @dn.datanode
    def render_caret(self, *, rich, metronome):
        def render_caret_cached(markup, caret_template):
            if caret_template is not None:
                markup = markup.traverse(
                    Caret,
                    lambda m: caret_template(mu.Group(m.children)),
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
        text_node = self.text_node
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
