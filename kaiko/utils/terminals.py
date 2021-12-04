import sys
import os
import time
import re
import contextlib
import queue
import threading
import signal
import shutil
import termios
import select
import tty
import enum
from typing import Optional, Union
import dataclasses
import wcwidth
from . import markups as mu
from . import datanodes as dn


unicode_version = "latest"

@dn.datanode
def ucs_detect():
    pattern = re.compile(r"\x1b\[(\d*);(\d*)R")
    channel = queue.Queue()

    def get_pos(arg):
        m = pattern.match(arg[1])
        if not m:
            return
        x = int(m.group(2) or "1") - 1
        channel.put(x)

    @dn.datanode
    def query_pos():
        global unicode_version

        old_version = '4.1.0'
        wide_by_version = [
            ('5.1.0', '龼'),
            ('5.2.0', '🈯'),
            ('6.0.0', '🈁'),
            ('8.0.0', '🉐'),
            ('9.0.0', '🐹'),
            ('10.0.0', '🦖'),
            ('11.0.0', '🧪'),
            ('12.0.0', '🪐'),
            ('12.1.0', '㋿'),
            ('13.0.0', '🫕'),
        ]

        yield

        xs = []
        for _, wchar in wide_by_version:
            print(wchar, end="", flush=True)
            print("\x1b[6n", end="", flush=True)

            while True:
                yield
                try:
                    x = channel.get(False)
                except queue.Empty:
                    continue
                else:
                    break

            print(f"\twidth={x}", end="\n", flush=True)
            xs.append(x)

        index = xs.index(1) if 1 in xs else len(wide_by_version)
        version = old_version if index == 0 else wide_by_version[index-1][0]
        unicode_version = version

        return version

    query_task = query_pos()
    with dn.pipe(inkey(get_pos), query_task) as task:
        yield from task.join((yield))
    return query_task.result

@dn.datanode
def terminal_size():
    resize_event = threading.Event()
    def SIGWINCH_handler(sig, frame):
        resize_event.set()
    resize_event.set()
    signal.signal(signal.SIGWINCH, SIGWINCH_handler)

    yield
    while True:
        if resize_event.is_set():
            resize_event.clear()
            size = shutil.get_terminal_size()
        yield size

@contextlib.contextmanager
def inkey_ctxt(stream, raw=False):
    fd = stream.fileno()
    old_attrs = termios.tcgetattr(fd)
    old_blocking = os.get_blocking(fd)

    try:
        tty.setcbreak(fd, termios.TCSANOW)
        if raw:
            tty.setraw(fd, termios.TCSANOW)
        os.set_blocking(fd, False)

        yield

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        os.set_blocking(fd, old_blocking)

@dn.datanode
def inkey(node, stream=None, raw=False):
    node = dn.DataNode.wrap(node)
    dt = 0.01

    if stream is None:
        stream = sys.stdin
    fd = stream.fileno()

    def run(stop_event):
        ref_time = time.perf_counter()
        while True:
            ready, _, _ = select.select([fd], [], [], dt)
            if stop_event.is_set():
                break
            if fd not in ready:
                continue

            data = stream.read()

            try:
                node.send((time.perf_counter()-ref_time, data))
            except StopIteration:
                return

    with inkey_ctxt(stream, raw):
        with node:
            with dn.create_task(run) as task:
                yield from task.join((yield))

@contextlib.contextmanager
def show_ctxt(stream, hide_cursor=False, end="\n"):
    hide_cursor = hide_cursor and stream == sys.stdout

    try:
        if hide_cursor:
            stream.write("\x1b[?25l")

        yield

    finally:
        if hide_cursor:
            stream.write("\x1b[?25h")
        stream.write(end)
        stream.flush()

@dn.datanode
def show(node, dt, t0=0, stream=None, hide_cursor=False, end="\n"):
    node = dn.DataNode.wrap(node)
    if stream is None:
        stream = sys.stdout

    def run(stop_event):
        ref_time = time.perf_counter()

        shown = False
        i = -1
        while True:
            try:
                view = node.send(shown)
            except StopIteration:
                break
            shown = False
            i += 1

            delta = ref_time+t0+i*dt - time.perf_counter()
            if delta < 0:
                continue
            if stop_event.wait(delta):
                break

            stream.write(view)
            stream.flush()
            shown = True

    with show_ctxt(stream, hide_cursor, end):
        with node:
            with dn.create_task(run) as task:
                yield from task.join((yield))


# ctrl code
@dataclasses.dataclass(frozen=True)
class CSI(mu.Single):
    name = "csi"
    code: str

    @classmethod
    def parse(clz, param):
        if param is None:
            raise mu.MarkupParseError(f"missing parameter for tag [{clz.name}/]")
        return param,

    @property
    def param(self):
        return self.code

    @property
    def ansi_code(self):
        return f"\x1b[{self.code}"

    def __str__(self):
        return self.ansi_code

@dataclasses.dataclass(frozen=True)
class Move(mu.Single):
    name = "move"
    x: int
    y: int

    @classmethod
    def parse(clz, param):
        if param is None:
            raise mu.MarkupParseError(f"missing parameter for tag [{clz.name}/]")
        try:
            x, y = tuple(int(n) for n in param.split(","))
        except ValueError:
            raise mu.MarkupParseError(f"invalid parameter for tag [{clz.name}/]: {param}")
        return x, y

    @property
    def param(self):
        return f"{self.x},{self.y}"

    def expand(self):
        res = []
        if self.x > 0:
            res.append(CSI(f"{self.x}C"))
        elif self.x < 0:
            res.append(CSI(f"{-self.x}D"))
        if self.y > 0:
            res.append(CSI(f"{self.y}B"))
        elif self.y < 0:
            res.append(CSI(f"{-self.y}A"))
        return mu.Group(tuple(res))

    def __str__(self):
        res = ""
        if self.x > 0:
            res += f"\x1b[{self.x}C"
        elif self.x < 0:
            res += f"\x1b[{-self.x}D"
        if self.y > 0:
            res += f"\x1b[{self.y}B"
        elif self.y < 0:
            res += f"\x1b[{-self.y}A"
        return res

@dataclasses.dataclass(frozen=True)
class Pos(mu.Single):
    name = "pos"
    x: int
    y: int

    @classmethod
    def parse(clz, param):
        if param is None:
            raise mu.MarkupParseError(f"missing parameter for tag [{clz.name}/]")
        try:
            x, y = tuple(int(n) for n in param.split(","))
        except ValueError:
            raise mu.MarkupParseError(f"invalid parameter for tag [{clz.name}/]: {param}")
        return x, y

    @property
    def param(self):
        return f"{self.x},{self.y}"

    def expand(self):
        return CSI(f"{self.y+1};{self.x+1}H")

    def __str__(self):
        return f"\x1b[{self.y+1};{self.x+1}H"

@dataclasses.dataclass(frozen=True)
class Scroll(mu.Single):
    name = "scroll"
    x: int

    @classmethod
    def parse(clz, param):
        if param is None:
            raise mu.MarkupParseError(f"missing parameter for tag [{clz.name}/]")
        try:
            x = int(param)
        except ValueError:
            raise mu.MarkupParseError(f"invalid parameter for tag [{clz.name}/]: {param}")
        return x,

    @property
    def param(self):
        return str(self.x)

    def expand(self):
        if self.x > 0:
            return CSI(f"{self.x}T")
        elif self.x < 0:
            return CSI(f"{-self.x}S")
        else:
            return mu.Group(())

    def __str__(self):
        if self.x > 0:
            return f"\x1b[{self.x}T"
        elif self.x < 0:
            return f"\x1b[{-self.x}S"
        else:
            return ""

class ClearRegion(enum.Enum):
    to_right = "0K"
    to_left = "1K"
    line = "2K"
    to_end = "0J"
    to_beginning = "1J"
    screen = "2J"

@dataclasses.dataclass(frozen=True)
class Clear(mu.Single):
    name = "clear"
    region: ClearRegion

    @classmethod
    def parse(clz, param):
        if param is None:
            raise mu.MarkupParseError(f"missing parameter for tag [{clz.name}/]")
        if all(param != region.name for region in ClearRegion):
            raise mu.MarkupParseError(f"invalid parameter for tag [{clz.name}/]: {param}")
        region = ClearRegion[param]

        return region,

    @property
    def param(self):
        return self.region.name

    def expand(self):
        return CSI(f"{self.region.value}")

    def __str__(self):
        return f"\x1b[{self.region.value}"


# attr code
@dataclasses.dataclass(frozen=True)
class SGR(mu.Pair):
    name = "sgr"
    attr: tuple

    @classmethod
    def parse(clz, param):
        if param is None:
            return clz((), ())

        try:
            attr = tuple(int(n or "0") for n in param.split(";"))
        except ValueError:
            raise mu.MarkupParseError(f"invalid parameter for tag [{clz.name}]: {param}")
        return attr

    @property
    def param(self):
        if not self.attr:
            return None
        return ';'.join(map(str, self.attr))

    @property
    def ansi_delimiters(self):
        if self.attr:
            return f"\x1b[{';'.join(map(str, self.attr))}m", "\x1b[m"
        else:
            return None, None

    def __str__(self):
        return self.ansi_delimiters[0] or ""

@dataclasses.dataclass(frozen=True)
class SimpleAttr(mu.Pair):
    option: str

    @property
    def param(self):
        return self.option

    @classmethod
    def parse(clz, param):
        if param is None:
            param = next(iter(clz._options.keys()))
        if param not in clz._options:
            raise mu.MarkupParseError(f"invalid parameter for tag [{clz.name}]: {param}")
        return param,

    def expand(self):
        return SGR(tuple(child.expand() for child in self.children), (self._options[self.option],))

    def __str__(self):
        return f"\x1b[{self._options[self.option]}m"

@dataclasses.dataclass(frozen=True)
class Reset(SimpleAttr):
    name = "reset"
    _options = {"on": 0}

@dataclasses.dataclass(frozen=True)
class Weight(SimpleAttr):
    name = "weight"
    _options = {"bold": 1, "dim": 2, "normal": 22}

@dataclasses.dataclass(frozen=True)
class Italic(SimpleAttr):
    name = "italic"
    _options = {"on": 3, "off": 23}

@dataclasses.dataclass(frozen=True)
class Underline(SimpleAttr):
    name = "underline"
    _options = {"on": 4, "double": 21, "off": 24}

@dataclasses.dataclass(frozen=True)
class Strike(SimpleAttr):
    name = "strike"
    _options = {"on": 9, "off": 29}

@dataclasses.dataclass(frozen=True)
class Blink(SimpleAttr):
    name = "blink"
    _options = {"on": 5, "off": 25}

@dataclasses.dataclass(frozen=True)
class Invert(SimpleAttr):
    name = "invert"
    _options = {"on": 7, "off": 27}


# colors
colors_16 = [
    (0x00, 0x00, 0x00),
    (0x80, 0x00, 0x00),
    (0x00, 0x80, 0x00),
    (0x80, 0x80, 0x00),
    (0x00, 0x00, 0x80),
    (0x80, 0x00, 0x80),
    (0x00, 0x80, 0x80),
    (0xc0, 0xc0, 0xc0),
    (0x80, 0x80, 0x80),
    (0xff, 0x00, 0x00),
    (0x00, 0xff, 0x00),
    (0xff, 0xff, 0x00),
    (0x00, 0x00, 0xff),
    (0xff, 0x00, 0xff),
    (0x00, 0xff, 0xff),
    (0xff, 0xff, 0xff),
]
colors_256 = []
for r in (0, *range(95, 256, 40)):
    for g in (0, *range(95, 256, 40)):
        for b in (0, *range(95, 256, 40)):
            colors_256.append((r, g, b))
for gray in range(8, 248, 10):
    colors_256.append((gray, gray, gray))

# TODO: determine them
class ColorSupport(enum.Enum):
    MONO = "mono"
    STANDARD = "standard"
    COLORS256 = "colors256"
    TRUECOLOR = "truecolor"
color_support = ColorSupport.TRUECOLOR

def sRGB(c1, c2):
    r1, g1, b1 = c1
    r2, g2, b2 = c2

    if r1 + r2 < 256:
        return 2*(r1-r2)**2 + 4*(g1-g2)**2 + 3*(b1-b2)**2
    else:
        return 3*(r1-r2)**2 + 4*(g1-g2)**2 + 2*(b1-b2)**2

def find_256color(code):
    code = min(colors_256, key=lambda c: sRGB(c, code))
    return colors_256.index(code) + 16

def find_16color(code):
    code = min(colors_16, key=lambda c: sRGB(c, code))
    return colors_16.index(code)

class Palette(enum.Enum):
    DEFAULT = "default"
    BLACK = "black"
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    BLUE = "blue"
    MAGENTA = "magenta"
    CYAN = "cyan"
    WHITE = "white"
    BRIGHT_BLACK = "bright_black"
    BRIGHT_RED = "bright_red"
    BRIGHT_GREEN = "bright_green"
    BRIGHT_YELLOW = "bright_yellow"
    BRIGHT_BLUE = "bright_blue"
    BRIGHT_MAGENTA = "bright_magenta"
    BRIGHT_CYAN = "bright_cyan"
    BRIGHT_WHITE = "bright_white"

color_names = [color.value for color in Palette]

@dataclasses.dataclass(frozen=True)
class Color(mu.Pair):
    name = "color"
    _palette = {
        Palette.DEFAULT: 39,
        Palette.BLACK: 30,
        Palette.RED: 31,
        Palette.GREEN: 32,
        Palette.YELLOW: 33,
        Palette.BLUE: 34,
        Palette.MAGENTA: 35,
        Palette.CYAN: 36,
        Palette.WHITE: 37,
        Palette.BRIGHT_BLACK: 90,
        Palette.BRIGHT_RED: 91,
        Palette.BRIGHT_GREEN: 92,
        Palette.BRIGHT_YELLOW: 93,
        Palette.BRIGHT_BLUE: 94,
        Palette.BRIGHT_MAGENTA: 95,
        Palette.BRIGHT_CYAN: 96,
        Palette.BRIGHT_WHITE: 97,
    }
    rgb: Union[Palette, int]
    color_support: ColorSupport

    @classmethod
    def parse(clz, param):
        if param is None:
            param = "default"

        try:
            rgb = Palette(param) if param in color_names else int(param, 16)
        except ValueError:
            raise mu.MarkupParseError(f"invalid parameter for tag [{clz.name}]: {param}")
        return rgb,

    @property
    def param(self):
        return self.rgb.value if isinstance(self.rgb, Palette) else f"{self.rgb:06x}"

    def expand(self):
        if isinstance(self.rgb, Palette):
            if self.color_support is not ColorSupport.MONO:
                return SGR(tuple(child.expand() for child in self.children), (self._palette[self.rgb],))
            else:
                return mu.Group(tuple(child.expand() for child in self.children))

        r = (self.rgb & 0xff0000) >> 16
        g = (self.rgb & 0x00ff00) >> 8
        b = (self.rgb & 0x0000ff)
        if self.color_support is ColorSupport.TRUECOLOR:
            return SGR(tuple(child.expand() for child in self.children), (38,2,r,g,b))
        elif self.color_support is ColorSupport.COLORS256:
            c = find_256color((r,g,b))
            return SGR(tuple(child.expand() for child in self.children), (38,5,c))
        elif self.color_support is ColorSupport.STANDARD:
            c = find_16color((r,g,b))
            if c < 8:
                c += 30
            else:
                c += 82
            return SGR(tuple(child.expand() for child in self.children), (c,))
        else:
            return mu.Group(tuple(child.expand() for child in self.children))

@dataclasses.dataclass(frozen=True)
class BgColor(mu.Pair):
    name = "bgcolor"
    _palette = {
        Palette.DEFAULT: 49,
        Palette.BLACK: 40,
        Palette.RED: 41,
        Palette.GREEN: 42,
        Palette.YELLOW: 43,
        Palette.BLUE: 44,
        Palette.MAGENTA: 45,
        Palette.CYAN: 46,
        Palette.WHITE: 47,
        Palette.BRIGHT_BLACK: 100,
        Palette.BRIGHT_RED: 101,
        Palette.BRIGHT_GREEN: 102,
        Palette.BRIGHT_YELLOW: 103,
        Palette.BRIGHT_BLUE: 104,
        Palette.BRIGHT_MAGENTA: 105,
        Palette.BRIGHT_CYAN: 106,
        Palette.BRIGHT_WHITE: 107,
    }
    rgb: Union[Palette, int]
    color_support: ColorSupport

    @classmethod
    def parse(clz, param):
        if param is None:
            param = "default"

        try:
            rgb = Palette(param) if param in color_names else int(param, 16)
        except ValueError:
            raise mu.MarkupParseError(f"invalid parameter for tag [{clz.name}]: {param}")
        return rgb,

    @property
    def param(self):
        return self.rgb.value if isinstance(self.rgb, Palette) else f"{self.rgb:06x}"

    def expand(self):
        if isinstance(self.rgb, Palette):
            if self.color_support is not ColorSupport.MONO:
                return SGR(tuple(child.expand() for child in self.children), (self._palette[self.rgb],))
            else:
                return mu.Group(tuple(child.expand() for child in self.children))

        r = (self.rgb & 0xff0000) >> 16
        g = (self.rgb & 0x00ff00) >> 8
        b = (self.rgb & 0x0000ff)
        if self.color_support is ColorSupport.TRUECOLOR:
            return SGR(tuple(child.expand() for child in self.children), (48,2,r,g,b))
        elif self.color_support is ColorSupport.COLORS256:
            c = find_256color((r,g,b))
            return SGR(tuple(child.expand() for child in self.children), (48,5,c))
        elif self.color_support is ColorSupport.STANDARD:
            c = find_16color((r,g,b))
            if c < 8:
                c += 40
            else:
                c += 92
            return SGR(tuple(child.expand() for child in self.children), (c,))
        else:
            return mu.Group(tuple(child.expand() for child in self.children))


# others
def widthof(text):
    width = 0
    for ch in text:
        w = wcwidth.wcwidth(ch, unicode_version)
        if w == -1:
            return -1
        width += w
    return width

@dataclasses.dataclass(frozen=True)
class Newline(mu.Single):
    name = "nl"

    @classmethod
    def parse(clz, param):
        if param is not None:
            raise mu.MarkupParseError(f"no parameter is needed for tag [{clz.name}/]")
        return ()

    @property
    def param(self):
        return None

    def expand(self):
        return mu.Text("\n")

@dataclasses.dataclass(frozen=True)
class Space(mu.Single):
    name = "sp"

    @classmethod
    def parse(clz, param):
        if param is not None:
            raise mu.MarkupParseError(f"no parameter is needed for tag [{clz.name}/]")
        return ()

    @property
    def param(self):
        return None

    def expand(self):
        return mu.Text(" ")

@dataclasses.dataclass(frozen=True)
class Wide(mu.Single):
    name = "wide"
    char: str
    unicode_version: str

    @classmethod
    def parse(clz, param):
        if param is None:
            raise mu.MarkupParseError(f"missing parameter for tag [{clz.name}/]")
        if len(param) != 1 or not param.isprintable():
            raise mu.MarkupParseError(f"invalid parameter for tag [{clz.name}/]: {param}")
        return param,

    @property
    def param(self):
        return self.char

    def expand(self):
        w = wcwidth.wcwidth(self.char, self.unicode_version)
        if w == 1:
            return mu.Text(self.char+" ")
        else:
            return mu.Text(self.char)


class RichTextRenderer:
    default_tags = {
        Reset.name: Reset,
        Weight.name: Weight,
        Italic.name: Italic,
        Underline.name: Underline,
        Strike.name: Strike,
        Blink.name: Blink,
        Invert.name: Invert,
        Color.name: Color,
        BgColor.name: BgColor,
        Newline.name: Newline,
        Space.name: Space,
        Wide.name: Wide,
    }

    def __init__(self):
        self.tags = dict(RichTextRenderer.default_tags)
        self.unicode_version = unicode_version
        self.color_support = color_support

    @property
    def props(self):
        return {
            Wide.name: (self.unicode_version,),
            Color.name: (self.color_support,),
            BgColor.name: (self.color_support,),
        }

    def parse(self, markup_str, expand=True, slotted=False):
        tags = self.tags if not slotted else dict(self.tags, slot=mu.Slot)
        markup = mu.parse_markup(markup_str, tags, self.props)
        if expand:
            markup = markup.expand()
        return markup

    def add_single_template(self, name, template):
        tag = mu.make_single_template(name, template, self.tags, self.props)
        self.tags[tag.name] = tag
        return tag

    def add_pair_template(self, name, template):
        tag = mu.make_pair_template(name, template, self.tags, self.props)
        self.tags[tag.name] = tag
        return tag

    def _render(self, markup, reopens=()):
        if isinstance(markup, mu.Text):
            yield from markup.string

        elif isinstance(markup, mu.Group):
            for child in markup.children:
                yield from self._render(child, reopens)

        elif isinstance(markup, CSI):
            yield markup.ansi_code

        elif isinstance(markup, SGR):
            open, close = markup.ansi_delimiters

            if open:
                yield open
            for child in markup.children:
                yield from self._render(child, (open, *reopens))
            if close:
                yield close
            for reopen in reopens[::-1]:
                if reopen:
                    yield reopen

        else:
            raise TypeError(f"unknown markup type: {type(markup)}")

    def render(self, markup):
        return "".join(self._render(markup))

    def _render_context(self, markup, printer, reopens=()):
        if isinstance(markup, mu.Text):
            printer(markup.string)

        elif isinstance(markup, CSI):
            printer(markup.ansi_code)

        elif isinstance(markup, mu.Slot):
            yield

        elif isinstance(markup, mu.Group):
            if not markup.children:
                return
            child = markup.children[0]
            try:
                yield from self._render_context(child, printer, reopens)
            finally:
                yield from self._render_context(mu.Group(markup.children[1:]), printer, reopens)

        elif isinstance(markup, SGR):
            open, close = markup.ansi_delimiters

            if open:
                printer(open)

            try:
                yield from self._render_context(mu.Group(markup.children), printer, (open, *reopens))
            finally:
                if close:
                    printer(close)
                for reopen in reopens[::-1]:
                    if reopen:
                        printer(reopen)

        else:
            raise TypeError(f"unknown markup type: {type(markup)}")

    @contextlib.contextmanager
    def render_context(self, markup, printer):
        yield from self._render_context(markup, printer)

    def _less(self, markup, size, pos=(0,0), reopens=(), wrap=True):
        if pos is None:
            return None

        elif isinstance(markup, mu.Text):
            x, y = pos
            for ch in markup.string:
                if ch == "\n":
                    y += 1
                    x = 0

                else:
                    w = wcwidth.wcwidth(ch, self.unicode_version)
                    if w == -1:
                        raise ValueError(f"unprintable character: {repr(ch)}")
                    x += w
                    if wrap and x > size.columns:
                        y += 1
                        x = w

                if y == size.lines:
                    return None
                if x <= size.columns:
                    yield ch
            return x, y

        elif isinstance(markup, mu.Group):
            for child in markup.children:
                pos = yield from self._less(child, size, pos, reopens, wrap=wrap)
            return pos

        elif isinstance(markup, SGR):
            open, close = markup.ansi_delimiters

            if open:
                yield open
            for child in markup.children:
                pos = yield from self._less(child, size, pos, (open, *reopens), wrap=wrap)
            if close:
                yield close
            for reopen in reopens[::-1]:
                if reopen:
                    yield reopen
            return pos

        else:
            raise TypeError(f"unknown markup type: {type(markup)}")

    def render_less(self, markup, size, pos=(0,0), wrap=True, restore=True):
        def _restore_pos(markup, size, pos, wrap):
            x0, y0 = pos
            pos = yield from self._less(markup, size, pos, wrap=wrap)
            x, y = pos or (None, size.lines-1)
            if y > y0:
                yield f"\x1b[{y-y0}A"
            yield "\r"
            if x0 > 0:
                yield f"\x1b[{x0}C"

        if restore:
            markup = _restore_pos(markup, size, pos, wrap)
        return "".join(markup)


# bar position
@dataclasses.dataclass(frozen=True)
class X(mu.Single):
    name = "x"
    x: int

    @classmethod
    def parse(clz, param):
        if param is None:
            raise mu.MarkupParseError(f"missing parameter for tag [{clz.name}]")
        try:
            x = int(param)
        except ValueError:
            raise mu.MarkupParseError(f"invalid parameter for tag [{clz.name}]: {param}")
        return x,

    @property
    def param(self):
        return str(self.x)

@dataclasses.dataclass(frozen=True)
class DX(mu.Single):
    name = "dx"
    dx: int

    @classmethod
    def parse(clz, param):
        if param is None:
            raise mu.MarkupParseError(f"missing parameter for tag [{clz.name}]")
        try:
            dx = int(param)
        except ValueError:
            raise mu.MarkupParseError(f"invalid parameter for tag [{clz.name}]: {param}")
        return dx,

    @property
    def param(self):
        return str(self.dx)

@dataclasses.dataclass(frozen=True)
class Restore(mu.Pair):
    name = "restore"

    @classmethod
    def parse(clz, param):
        if param is not None:
            raise mu.MarkupParseError(f"no parameter is needed for tag [{clz.name}]")
        return ()

    @property
    def param(self):
        return None

@dataclasses.dataclass(frozen=True)
class Mask(mu.Pair):
    name = "mask"
    mask: slice

    @classmethod
    def parse(clz, param):
        try:
            start, stop = [int(p) if p else None for p in (param or ":").split(":")]
        except ValueError:
            raise mu.MarkupParseError(f"invalid parameter for tag [{clz.name}]: {param}")
        return slice(start, stop),

    @property
    def param(self):
        return f"{self.mask.start if self.mask.start is not None else ''}:{self.mask.stop if self.mask.stop is not None else ''}"

def clamp(ran, mask):
    start = min(max(mask.start, ran.start), mask.stop)
    stop = max(min(mask.stop, ran.stop), mask.start)
    return range(start, stop)

class RichBarRenderer:
    default_tags = {
        Reset.name: Reset,
        Weight.name: Weight,
        Italic.name: Italic,
        Underline.name: Underline,
        Strike.name: Strike,
        Blink.name: Blink,
        Invert.name: Invert,
        Color.name: Color,
        BgColor.name: BgColor,
        Newline.name: Newline,
        Space.name: Space,
        Wide.name: Wide,
        X.name: X,
        DX.name: DX,
        Mask.name: Mask,
    }

    def __init__(self):
        self.tags = dict(RichBarRenderer.default_tags)
        self.unicode_version = unicode_version
        self.color_support = color_support

    @property
    def props(self):
        return {
            Wide.name: (self.unicode_version,),
            Color.name: (self.color_support,),
            BgColor.name: (self.color_support,),
        }
    def parse(self, markup_str, expand=True):
        markup = mu.parse_markup(markup_str, self.tags, self.props)
        if expand:
            markup = markup.expand()
        return markup

    def add_single_template(self, name, template):
        tag = mu.make_single_template(name, template, self.tags, self.props)
        self.tags[tag.name] = tag
        return tag

    def add_pair_template(self, name, template):
        tag = mu.make_pair_template(name, template, self.tags, self.props)
        self.tags[tag.name] = tag
        return tag

    def _render_text(self, buffer, string, x, width, xmask, attrs):
        start = xmask.start
        stop = xmask.stop
        for ch in string:
            w = wcwidth.wcwidth(ch, self.unicode_version)
            if w == -1:
                raise ValueError(f"invalid string: {repr(ch)} in {repr(string)}")

            if x+w > stop:
                break

            if x < start:
                x += w
                continue

            # assert start <= x <= x+w <= stop

            if w == 0:
                x_ = x - 1
                if 0 <= x_ and buffer[x_] == "":
                    x_ -= 1
                if start <= x_:
                    buffer[x_] += ch

            elif w == 1:
                if 0 <= x-1 and buffer[x] == "":
                    buffer[x-1] = " "
                if x+1 < width and buffer[x+1] == "":
                    buffer[x+1] = " "
                buffer[x] = ch if not attrs else f"\x1b[{';'.join(map(str, attrs))}m{ch}\x1b[m"
                x += 1

            else:
                x_ = x + 1
                if 0 <= x-1 and buffer[x] == "":
                    buffer[x-1] = " "
                if x_+1 < width and buffer[x_+1] == "":
                    buffer[x_+1] = " "
                buffer[x] = ch if not attrs else f"\x1b[{';'.join(map(str, attrs))}m{ch}\x1b[m"
                buffer[x_] = ""
                x += 2

        return x

    def _render(self, buffer, markup, x, width, xmask, attrs):
        if isinstance(markup, mu.Text):
            return self._render_text(buffer, markup.string, x, width, xmask, attrs)

        elif isinstance(markup, mu.Group):
            for child in markup.children:
                x = self._render(buffer, child, x, width, xmask, attrs)
            return x

        elif isinstance(markup, SGR):
            attrs = (*attrs, *markup.attr)
            for child in markup.children:
                x = self._render(buffer, child, x, width, xmask, attrs)
            return x

        elif isinstance(markup, X):
            return markup.x

        elif isinstance(markup, DX):
            return x+markup.dx

        elif isinstance(markup, Restore):
            x0 = x
            for child in markup.children:
                x = self._render(buffer, child, x, width, xmask, attrs)
            return x0

        elif isinstance(markup, Mask):
            xmask = clamp(range(width)[markup.mask], xmask)
            for child in markup.children:
                x = self._render(buffer, child, x, width, xmask, attrs)
            return x

        else:
            raise TypeError(f"unknown markup type: {type(markup)}")

    def render(self, width, markup):
        buffer = [" "]*width
        self._render(buffer, markup, 0, width, range(width), ())
        return "".join(buffer).rstrip()

