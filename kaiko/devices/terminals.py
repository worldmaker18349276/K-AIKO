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
from typing import Dict
from ..utils import datanodes as dn
from ..utils import markups as mu
from ..utils import config as cfg


@dn.datanode
def ucs_detect():
    pattern = re.compile(r"\x1b\[(\d*);(\d*)R")
    channel = queue.Queue()

    def get_pos(arg):
        if arg[1] is None:
            return
        m = pattern.match(arg[1])
        if not m:
            return
        x = int(m.group(2) or "1") - 1
        channel.put(x)

    @dn.datanode
    def query_pos():
        old_version = "4.1.0"
        wide_by_version = [
            ("5.1.0", "龼"),
            ("5.2.0", "🈯"),
            ("6.0.0", "🈁"),
            ("8.0.0", "🉐"),
            ("9.0.0", "🐹"),
            ("10.0.0", "🦖"),
            ("11.0.0", "🧪"),
            ("12.0.0", "🪐"),
            ("12.1.0", "㋿"),
            ("13.0.0", "🫕"),
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
        version = old_version if index == 0 else wide_by_version[index - 1][0]

        return version

    query_task = query_pos()
    yield from dn.pipe(inkey(get_pos), query_task).join()
    return query_task.result


@dn.datanode
def terminal_size():
    resize_event = threading.Event()

    def SIGWINCH_handler(sig, frame):
        resize_event.set()

    resize_event.set()
    signal.signal(signal.SIGWINCH, SIGWINCH_handler)

    yield
    size = shutil.get_terminal_size()
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
def inkey(node, stream=None, raw=False, dt=0.1):
    node = dn.DataNode.wrap(node)

    if stream is None:
        stream = sys.stdin
    fd = stream.fileno()

    @dn.datanode
    def run():
        ref_time = time.perf_counter()
        yield
        while True:
            ready, _, _ = select.select([fd], [], [], dt)
            yield

            data = stream.read() if fd in ready else None

            try:
                node.send((time.perf_counter() - ref_time, data))
            except StopIteration as e:
                return e.value

    with inkey_ctxt(stream, raw):
        with node:
            result = yield from dn.create_task(run()).join()
            return result


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

    @dn.datanode
    def run(node):
        with node:
            expired = True
            yield
            while True:
                try:
                    view = node.send(not expired)
                except StopIteration as stop:
                    return stop.value
                shown = False

                expired = yield

                if not expired:
                    stream.write(view)
                    stream.flush()

    with show_ctxt(stream, hide_cursor, end):
        result = yield from dn.interval(run(node), dt, t0).join()
        return result


class TerminalSettings(cfg.Configurable):
    r"""
    Fields
    ------
    unicode_version : str
        The unicode version of terminal.
    color_support : markups.ColorSupport
        The color support of terminal.
    keycodes : dict from str to str
        The maps from keycodes to keynames.

    best_screen_size : int
        Recommended screen size. If your screen size is smaller than this size,
        the system will ask you to adjust it.
    adjust_screen_delay : float
        The delay time to complete the screen adjustment.

    editor : str
        The editor to edit text.
    """

    unicode_version: str = "auto"
    color_support: mu.ColorSupport = mu.ColorSupport.TRUECOLOR
    keycodes: Dict[str, str] = {
        "\x1b": "Esc",
        "\x1b\x1b": "Alt_Esc",
        "\n": "Enter",
        "\x1b\n": "Alt_Enter",
        "\x7f": "Backspace",
        "\x08": "Ctrl_Backspace",
        "\x1b\x7f": "Alt_Backspace",
        "\x1b\x08": "Ctrl_Alt_Backspace",
        "\t": "Tab",
        "\x1b[Z": "Shift_Tab",
        "\x1b\t": "Alt_Tab",
        "\x1b\x1b[Z": "Alt_Shift_Tab",
        "\x1b[A": "Up",
        "\x1b[1;2A": "Shift_Up",
        "\x1b[1;3A": "Alt_Up",
        "\x1b[1;4A": "Alt_Shift_Up",
        "\x1b[1;5A": "Ctrl_Up",
        "\x1b[1;6A": "Ctrl_Shift_Up",
        "\x1b[1;7A": "Ctrl_Alt_Up",
        "\x1b[1;8A": "Ctrl_Alt_Shift_Up",
        "\x1b[B": "Down",
        "\x1b[1;2B": "Shift_Down",
        "\x1b[1;3B": "Alt_Down",
        "\x1b[1;4B": "Alt_Shift_Down",
        "\x1b[1;5B": "Ctrl_Down",
        "\x1b[1;6B": "Ctrl_Shift_Down",
        "\x1b[1;7B": "Ctrl_Alt_Down",
        "\x1b[1;8B": "Ctrl_Alt_Shift_Down",
        "\x1b[C": "Right",
        "\x1b[1;2C": "Shift_Right",
        "\x1b[1;3C": "Alt_Right",
        "\x1b[1;4C": "Alt_Shift_Right",
        "\x1b[1;5C": "Ctrl_Right",
        "\x1b[1;6C": "Ctrl_Shift_Right",
        "\x1b[1;7C": "Ctrl_Alt_Right",
        "\x1b[1;8C": "Ctrl_Alt_Shift_Right",
        "\x1b[D": "Left",
        "\x1b[1;2D": "Shift_Left",
        "\x1b[1;3D": "Alt_Left",
        "\x1b[1;4D": "Alt_Shift_Left",
        "\x1b[1;5D": "Ctrl_Left",
        "\x1b[1;6D": "Ctrl_Shift_Left",
        "\x1b[1;7D": "Ctrl_Alt_Left",
        "\x1b[1;8D": "Ctrl_Alt_Shift_Left",
        "\x1b[F": "End",
        "\x1b[1;2F": "Shift_End",
        "\x1b[1;3F": "Alt_End",
        "\x1b[1;4F": "Alt_Shift_End",
        "\x1b[1;5F": "Ctrl_End",
        "\x1b[1;6F": "Ctrl_Shift_End",
        "\x1b[1;7F": "Ctrl_Alt_End",
        "\x1b[1;8F": "Ctrl_Alt_Shift_End",
        "\x1b[H": "Home",
        "\x1b[1;2H": "Shift_Home",
        "\x1b[1;3H": "Alt_Home",
        "\x1b[1;4H": "Alt_Shift_Home",
        "\x1b[1;5H": "Ctrl_Home",
        "\x1b[1;6H": "Ctrl_Shift_Home",
        "\x1b[1;7H": "Ctrl_Alt_Home",
        "\x1b[1;8H": "Ctrl_Alt_Shift_Home",
        "\x1b[2~": "Insert",
        "\x1b[2;2~": "Shift_Insert",
        "\x1b[2;3~": "Alt_Insert",
        "\x1b[2;4~": "Alt_Shift_Insert",
        "\x1b[2;5~": "Ctrl_Insert",
        "\x1b[2;6~": "Ctrl_Shift_Insert",
        "\x1b[2;7~": "Ctrl_Alt_Insert",
        "\x1b[2;8~": "Ctrl_Alt_Shift_Insert",
        "\x1b[3~": "Delete",
        "\x1b[3;2~": "Shift_Delete",
        "\x1b[3;3~": "Alt_Delete",
        "\x1b[3;4~": "Alt_Shift_Delete",
        "\x1b[3;5~": "Ctrl_Delete",
        "\x1b[3;6~": "Ctrl_Shift_Delete",
        "\x1b[3;7~": "Ctrl_Alt_Delete",
        "\x1b[3;8~": "Ctrl_Alt_Shift_Delete",
        "\x1b[5~": "PageUp",
        "\x1b[5;2~": "Shift_PageUp",
        "\x1b[5;3~": "Alt_PageUp",
        "\x1b[5;4~": "Alt_Shift_PageUp",
        "\x1b[5;5~": "Ctrl_PageUp",
        "\x1b[5;6~": "Ctrl_Shift_PageUp",
        "\x1b[5;7~": "Ctrl_Alt_PageUp",
        "\x1b[5;8~": "Ctrl_Alt_Shift_PageUp",
        "\x1b[6~": "PageDown",
        "\x1b[6;2~": "Shift_PageDown",
        "\x1b[6;3~": "Alt_PageDown",
        "\x1b[6;4~": "Alt_Shift_PageDown",
        "\x1b[6;5~": "Ctrl_PageDown",
        "\x1b[6;6~": "Ctrl_Shift_PageDown",
        "\x1b[6;7~": "Ctrl_Alt_PageDown",
        "\x1b[6;8~": "Ctrl_Alt_Shift_PageDown",
    }

    best_screen_size: int = 80
    adjust_screen_delay: float = 1.0

    editor: str = "nano"


printable_ascii_names = {
    " ": "Space",
    "!": "Bang",
    "?": "Question",
    "#": "Hash",
    "$": "Dollar",
    "%": "Percent",
    "&": "Ampersand",
    "@": "At",
    '"': "Quote",
    "'": "Tick",
    "`": "Backtick",
    "*": "Asterisk",
    "+": "Plus",
    "-": "Dash",
    "/": "Slash",
    "\\": "Backslash",
    "|": "Bar",
    ".": "Period",
    ",": "Comma",
    ":": "Colon",
    ";": "Semicolon",
    "^": "Hat",
    "_": "Underscore",
    "~": "Tilde",
    "=": "Equal",
    "<": "LessThan",
    ">": "GreaterThan",
    "(": "LeftParen",
    ")": "RightParen",
    "[": "LeftBracket",
    "]": "RightBracket",
    "{": "LeftBrace",
    "}": "RightBrace",
}

for ch in range(ord("0"), ord("9") + 1):
    printable_ascii_names[chr(ch)] = chr(ch)
for ch in range(ord("A"), ord("Z") + 1):
    printable_ascii_names[chr(ch)] = chr(ch)
for ch in range(ord("a"), ord("z") + 1):
    printable_ascii_names[chr(ch)] = chr(ch)
