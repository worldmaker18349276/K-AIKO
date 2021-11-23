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
import dataclasses
import wcwidth
from . import markups as mu
from . import datanodes as dn


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
@dataclasses.dataclass
class CSI(mu.Single):
    name = "csi"
    code: str

    @classmethod
    def parse(clz, param):
        if param is None:
            raise mu.MarkupParseError(f"missing parameter for tag [{clz.name}/]")
        return clz(param)

    @property
    def param(self):
        return self.code

    @property
    def ansi_code(self):
        return f"\x1b[{self.code}"

@dataclasses.dataclass
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
        return clz(x, y)

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
        return mu.Group(res)

@dataclasses.dataclass
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
        return clz(x, y)

    @property
    def param(self):
        return f"{self.x},{self.y}"

    def expand(self):
        return CSI(f"{self.y+1};{self.x+1}H")

@dataclasses.dataclass
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
        return clz(x)

    @property
    def param(self):
        return str(self.x)

    def expand(self):
        if self.x > 0:
            return CSI(f"{self.x}T")
        elif self.x < 0:
            return CSI(f"{-self.x}S")
        else:
            return mu.Group([])

class ClearRegion(enum.Enum):
    to_right = "0K"
    to_left = "1K"
    line = "2K"
    to_end = "0J"
    to_beginning = "1J"
    screen = "2J"

@dataclasses.dataclass
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

        return clz(region)

    @property
    def param(self):
        return self.region.name

    def expand(self):
        return CSI(f"{self.region.value}")


# attr code
@dataclasses.dataclass
class SGR(mu.Pair):
    name = "sgr"
    attr: tuple

    @classmethod
    def parse(clz, param):
        if param is None:
            return clz([], ())

        try:
            attr = tuple(int(n or "0") for n in param.split(";"))
        except ValueError:
            raise mu.MarkupParseError(f"invalid parameter for tag [{clz.name}]: {param}")
        return clz([], attr)

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

@dataclasses.dataclass
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
        return clz([], param)

    def expand(self):
        return SGR([child.expand() for child in self.children], (self._options[self.option],))

@dataclasses.dataclass
class Reset(SimpleAttr):
    name = "reset"
    _options = {"on": 0}

@dataclasses.dataclass
class Weight(SimpleAttr):
    name = "weight"
    _options = {"bold": 1, "dim": 2, "normal": 22}

@dataclasses.dataclass
class Italic(SimpleAttr):
    name = "italic"
    _options = {"on": 3, "off": 23}

@dataclasses.dataclass
class Underline(SimpleAttr):
    name = "underline"
    _options = {"on": 4, "double": 21, "off": 24}

@dataclasses.dataclass
class Strike(SimpleAttr):
    name = "strike"
    _options = {"on": 9, "off": 29}

@dataclasses.dataclass
class Blink(SimpleAttr):
    name = "blink"
    _options = {"on": 5, "off": 25}

@dataclasses.dataclass
class Invert(SimpleAttr):
    name = "invert"
    _options = {"on": 7, "off": 27}

@dataclasses.dataclass
class Color(SimpleAttr):
    name = "color"
    _options = {
        "default": 39,
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "white": 37,
        "bright_black": 90,
        "bright_red": 91,
        "bright_green": 92,
        "bright_yellow": 93,
        "bright_blue": 94,
        "bright_magenta": 95,
        "bright_cyan": 96,
        "bright_white": 97,
    }

@dataclasses.dataclass
class BgColor(SimpleAttr):
    name = "bgcolor"
    _options = {
        "default": 49,
        "black": 40,
        "red": 41,
        "green": 42,
        "yellow": 43,
        "blue": 44,
        "magenta": 45,
        "cyan": 46,
        "white": 47,
        "bright_black": 100,
        "bright_red": 101,
        "bright_green": 102,
        "bright_yellow": 103,
        "bright_blue": 104,
        "bright_magenta": 105,
        "bright_cyan": 106,
        "bright_white": 107,
    }


style_tags = [
    Reset,
    Weight,
    Italic,
    Underline,
    Strike,
    Blink,
    Invert,
    Color,
    BgColor,
]

def _render(markup, reopens=()):
    if isinstance(markup, mu.Text):
        yield from markup.string

    elif isinstance(markup, mu.Group):
        for child in markup.children:
            yield from _render(child, reopens)

    elif isinstance(markup, CSI):
        yield markup.ansi_code

    elif isinstance(markup, SGR):
        open, close = markup.ansi_delimiters

        if open:
            yield open
        for child in markup.children:
            yield from _render(child, (open, *reopens))
        if close:
            yield close
        for reopen in reopens[::-1]:
            if reopen:
                yield reopen

    else:
        raise TypeError(f"unknown markup type: {type(markup)}")

def render(markup):
    return "".join(_render(markup.expand()))

def parse(markup_str):
    return mu.parse_markup(markup_str, [SGR, *style_tags])


def addmarkup1(view, width, x, markup, xmask=slice(None,None), x0=None, attrs=()):
    xran = range(width)
    if x0 is None:
        x0 = x

    if isinstance(markup, mu.Text):
        for ch in markup.string:
            w = wcwidth.wcwidth(ch)

            if ch == "\t":
                x += 1

            elif ch == "\b":
                x -= 1

            elif ch == "\r":
                x = x0

            elif ch == "\x00":
                pass

            elif w == 0:
                x_ = x - 1
                if x_ in xran and view[x_] == "":
                    x_ -= 1
                if x_ in xran[xmask]:
                    view[x_] += ch

            elif w == 1:
                if x in xran[xmask]:
                    if x-1 in xran and view[x] == "":
                        view[x-1] = " "
                    if x+1 in xran and view[x+1] == "":
                        view[x+1] = " "
                    view[x] = ch if not attrs else f"\x1b[{';'.join(map(str, attrs))}m{ch}\x1b[m"
                x += 1

            elif w == 2:
                x_ = x + 1
                if x in xran[xmask] and x_ in xran[xmask]:
                    if x-1 in xran and view[x] == "":
                        view[x-1] = " "
                    if x_+1 in xran and view[x_+1] == "":
                        view[x_+1] = " "
                    view[x] = ch if not attrs else f"\x1b[{';'.join(map(str, attrs))}m{ch}\x1b[m"
                    view[x_] = ""
                x += 2

            else:
                raise ValueError(f"invalid string: {repr(ch)} in {repr(markup.string)}")

        return x

    elif isinstance(markup, mu.Group):
        for child in markup.children:
            x = addmarkup1(view, width, x, child, xmask, x0, attrs)
        return x

    elif isinstance(markup, SGR):
        attrs = (*attrs, *markup.attr)
        for child in markup.children:
            x = addmarkup1(view, width, x, child, xmask, x0, attrs)
        return x

    else:
        raise TypeError(f"unknown markup type: {type(markup)}")


def _less(markup, size, pos=(0,0), reopens=(), wrap=True):
    if pos is None:
        return None

    elif isinstance(markup, mu.Text):
        x, y = pos
        for ch in markup.string:
            if ch == "\n":
                y += 1
                x = 0

            else:
                w = wcwidth.wcwidth(ch)
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
            pos = yield from _less(child, size, pos, reopens, wrap=wrap)
        return pos

    elif isinstance(markup, SGR):
        open, close = markup.ansi_delimiters

        if open:
            yield open
        for child in markup.children:
            pos = yield from _less(child, size, pos, (open, *reopens), wrap=wrap)
        if close:
            yield close
        for reopen in reopens[::-1]:
            if reopen:
                yield reopen
        return pos

    else:
        raise TypeError(f"unknown markup type: {type(markup)}")

def less(markup, size, pos=(0,0), wrap=True, restore=True):
    def _restore_pos(markup, size, pos, wrap):
        x0, y0 = pos
        pos = yield from _less(markup, size, pos, wrap=wrap)
        x, y = pos or (None, size.lines-1)
        if y > y0:
            yield f"\x1b[{y-y0}A"
        yield "\r"
        if x0 > 0:
            yield f"\x1b[{x0}C"

    markup = markup.expand()
    if restore:
        markup = _restore_pos(markup, size, pos, wrap)
    return "".join(markup)

