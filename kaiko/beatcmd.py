from enum import Enum
import threading
from . import datanodes as dn
from . import tui


class SHLEXER_STATE(Enum):
    SPACED = " "
    PLAIN = "*"
    BACKSLASHED = "\\"
    QUOTED = "'"
    DOUBLE_QUOTED = '"'

def shlexer_parse(raw, partial=False):
    SPACE = " "
    BACKSLASH = "\\"
    QUOTE = "'"
    DOUBLE_QUOTE = '"'

    raw = enumerate(raw)

    while True:
        try:
            index, char = next(raw)
        except StopIteration:
            return SHLEXER_STATE.SPACED

        # guard space
        if char == SPACE:
            continue

        # parse token
        start = index
        token = []
        masked = []
        while True:
            if char == SPACE:
                # end parsing token
                yield "".join(token), slice(start, index), masked
                break

            elif char == BACKSLASH:
                # escape next character
                try:
                    masked.append(index)
                    index, char = next(raw)
                    token.append(char)

                except StopIteration:
                    if not partial:
                        raise ValueError("No escaped character")
                    yield "".join(token), slice(start, None), masked
                    return SHLEXER_STATE.BACKSLASHED

            elif char == QUOTE:
                # start escape string until next quote
                try:
                    masked.append(index)
                    index, char = next(raw)
                    while char != QUOTE:
                        token.append(char)
                        index, char = next(raw)
                    masked.append(index)

                except StopIteration:
                    if not partial:
                        raise ValueError("No closing quotation")
                    yield "".join(token), slice(start, None), masked
                    return SHLEXER_STATE.QUOTED

            elif char == DOUBLE_QUOTE:
                # start escape string until next double quote
                try:
                    masked.append(index)
                    index, char = next(raw)
                    while char != DOUBLE_QUOTE:
                        token.append(char)
                        index, char = next(raw)
                    masked.append(index)

                except StopIteration:
                    if not partial:
                        raise ValueError("No closing double quotation")
                    yield "".join(token), slice(start, None), masked
                    return SHLEXER_STATE.DOUBLE_QUOTED

            else:
                # otherwise, as it is
                token.append(char)

            try:
                index, char = next(raw)
            except StopIteration:
                yield "".join(token), slice(start, None), masked
                return SHLEXER_STATE.PLAIN

class SHLEXER_ESCAPE(Enum):
    MIX = "*"
    BACKSLASH = "\\"
    QUOTE = "'"
    DOUBLE_QUOTE = '"'

def shlexer_escape(token, strategy=SHLEXER_ESCAPE.MIX):
    if strategy == SHLEXER_ESCAPE.MIX:
        if len(token) == 0:
            yield from shlexer_escape(token, SHLEXER_ESCAPE.QUOTE)
        elif " " not in token:
            yield from shlexer_escape(token, SHLEXER_ESCAPE.BACKSLASH)
        elif "'" not in token:
            yield from shlexer_escape(token, SHLEXER_ESCAPE.QUOTE)
        else:
            yield from shlexer_escape(token, SHLEXER_ESCAPE.DOUBLE_QUOTE)

    elif strategy == SHLEXER_ESCAPE.BACKSLASH:
        if len(token) == 0:
            raise ValueError("Unable to escape empty string")

        for ch in token:
            if ch in (" ", "\\", "'", '"'):
                yield "\\"
                yield ch
            else:
                yield ch

    elif strategy == SHLEXER_ESCAPE.QUOTE:
        yield "'"
        for ch in token:
            if ch == "'":
                yield "'"
                yield "\\"
                yield ch
                yield "'"
            else:
                yield ch
        yield "'"

    elif strategy == SHLEXER_ESCAPE.DOUBLE_QUOTE:
        yield '"'
        for ch in token:
            if ch == '"':
                yield '"'
                yield "\\"
                yield ch
                yield '"'
            else:
                yield ch
        yield '"'

    else:
        raise ValueError

def shlexer_complete(raw, index, completer):
    state = SHLEXER_STATE.SPACED
    parser = shlexer_parse(raw[:index], partial=True)
    tokens = []
    try:
        while True:
            token, *_ = next(parser)
            tokens.append(token)
    except StopIteration as e:
        state = e.value

    is_appended = len(raw) == index

    if state == SHLEXER_STATE.SPACED:
        target = ""
        for compreply in completer(tokens, target):
            compreply = list(shlexer_escape(compreply))
            yield [*compreply, " "]

    elif state == SHLEXER_STATE.PLAIN:
        target = tokens[-1]
        tokens = tokens[:-1]
        for compreply in completer(tokens, target):
            compreply = list(shlexer_escape(compreply)) if compreply else []
            yield [*compreply, " "]

    elif state == SHLEXER_STATE.BACKSLASHED:
        target = tokens[-1]
        tokens = tokens[:-1]
        for compreply in completer(tokens, target):
            compreply = list(shlexer_escape(compreply, SHLEXER_ESCAPE.BACKSLASH)) if compreply else ["\b"]
            if compreply.startswith("\\"):
                compreply = compreply[1:]
            if is_appended:
                yield [*compreply, " "]
            else:
                yield [*compreply, " ", "\\"]

    elif state == SHLEXER_STATE.QUOTED:
        target = tokens[-1]
        tokens = tokens[:-1]
        for compreply in completer(tokens, target):
            compreply = list(shlexer_escape(compreply, SHLEXER_ESCAPE.QUOTE))[1:]
            if is_appended:
                yield [*compreply, " "]
            else:
                yield [*compreply, " ", "'"]

    elif state == SHLEXER_STATE.DOUBLE_QUOTED:
        target = tokens[-1]
        tokens = tokens[:-1]
        for compreply in completer(tokens, target):
            compreply = list(shlexer_escape(compreply, SHLEXER_ESCAPE.DOUBLE_QUOTE))[1:]
            if is_appended:
                yield [*compreply, " "]
            else:
                yield [*compreply, " ", '"']


def explore_path(root, partial=False):
    curr = root

    while True:
        token = yield curr

        if curr is None:
            continue

        if not isinstance(curr, dict):
            if not partial:
                raise ValueError(f"no more dict: {dict}")
            curr = None
            continue

        if token not in curr:
            if not partial:
                raise ValueError(f"no such field {token} in {curr}")
            curr = None
            continue

        curr = curr.get(token)

def complete_path(root, tokens, target):
    curr = root
    for token in tokens:
        if token not in curr:
            break

        curr = curr.get(token)

        if not isinstance(curr, dict):
            break

    else:
        for key in curr.keys():
            if key.startswith(target):
                yield key[len(target):]


default_keymap = {
    "Backspace"        : "\x7f",
    "Delete"           : "\x1b[3~",
    "Left"             : "\x1b[D",
    "Right"            : "\x1b[C",
    "Home"             : "\x1b[H",
    "End"              : "\x1b[F",
    "Up"               : "\x1b[A",
    "Down"             : "\x1b[B",
    "PageUp"           : "\x1b[5~",
    "PageDown"         : "\x1b[6~",
    "Tab"              : "\t",
    "Esc"              : "\x1b",
    "Enter"            : "\n",
    "Shift+Tab"        : "\x1b[Z",
    "Ctrl+Backspace"   : "\x08",
    "Ctrl+Delete"      : "\x1b[3;5~",
    "Shift+Left"       : "\x1b[1;2D",
    "Shift+Right"      : "\x1b[1;2C",
    "Alt+Left"         : "\x1b[1;3D",
    "Alt+Right"        : "\x1b[1;3C",
    "Alt+Shift+Left"   : "\x1b[1;4D",
    "Alt+Shift+Right"  : "\x1b[1;4C",
    "Ctrl+Left"        : "\x1b[1;5D",
    "Ctrl+Right"       : "\x1b[1;5C",
    "Ctrl+Shift+Left"  : "\x1b[1;6D",
    "Ctrl+Shift+Right" : "\x1b[1;6C",
    "Alt+Ctrl+Left"    : "\x1b[1;7D",
    "Alt+Ctrl+Right"   : "\x1b[1;7C",
    "Alt+Ctrl+Shift+Left"  : "\x1b[1;8D",
    "Alt+Ctrl+Shift+Right" : "\x1b[1;8C",
}

class BeatText:
    def __init__(self, command=dict()):
        self.buffer = []
        self.pos = 0
        self.command = command

        self.tokens = []
        self.state = SHLEXER_STATE.SPACED

    def parse(self):
        lex_parser = shlexer_parse(self.buffer, partial=True)
        cmd_parser = explore_path(self.command, partial=True)
        next(cmd_parser)

        self.tokens = []
        while True:
            try:
                token, index, masked = next(lex_parser)
            except StopIteration as e:
                self.state = e.value
                break

            cmd = cmd_parser.send(token)
            self.tokens.append((cmd, index, masked))

    def complete(self):
        path_completer = lambda tokens, target: complete_path(self.command, tokens, target)
        lex_completer = shlexer_complete(self.buffer, self.pos, path_completer)

        original_buffer = list(self.buffer)
        original_pos = self.pos

        for compreply in lex_completer:
            while compreply[0] == "\b":
                del compreply[0]
                del self.buffer[self.pos-1]
                self.pos = self.pos-1

            self.buffer[self.pos:self.pos] = compreply
            self.pos = self.pos + len(compreply)
            self.parse()

            yield

            self.buffer = list(original_buffer)
            self.pos = original_pos

        else:
            self.parse()

    def render(self, view, width, ran, cursor=None):
        escape = lambda s: f"\x1b[2m{s}\x1b[m"
        warn = lambda s: f"\x1b[31m{s}\x1b[m"
        whitespace = "\x1b[2m⌴\x1b[m"

        pos = self.pos
        buffer = list(self.buffer)
        for cmd, slic, masked in self.tokens:
            for index in masked:
                buffer[index] = escape(buffer[index])

            for index in range(len(buffer))[slic]:
                if buffer[index] == " ":
                    buffer[index] = whitespace

            if cmd is None and slic.stop is not None:
                for index in range(len(buffer))[slic]:
                    buffer[index] = warn(buffer[index])

        tui.addtext1(view, width, ran.start, "".join(buffer), ran)

        if cursor:
            _, cursor_pos = tui.textrange1(ran.start, "".join(buffer[:pos]))
            cursor_ran = tui.select1(view, width, slice(cursor_pos, cursor_pos+1))
            if hasattr(cursor, '__call__'):
                view[cursor_ran.start] = cursor(view[cursor_ran.start])
            else:
                tui.addtext1(view, width, cursor_ran.start, cursor)

    def input(self, ch):
        self.buffer[self.pos:self.pos] = [ch]
        self.pos += 1
        self.parse()

    def backspace(self):
        if self.pos == 0:
            return
        self.pos -= 1
        del self.buffer[self.pos]
        self.parse()

    def delete(self):
        if self.pos >= len(self.buffer):
            return
        del self.buffer[self.pos]
        self.parse()

    def move(self, offset):
        self.pos = min(max(0, self.pos+offset), len(self.buffer))

    def move_to(self, pos):
        self.pos = min(max(0, pos), len(self.buffer))

    def move_to_end(self):
        self.pos = len(self.buffer)


class INPUT_STATE(Enum):
    EDIT = "edit"
    TAB = "tab"

class BeatStroke:
    def __init__(self, text):
        self.text = text
        self.event = threading.Event()
        self.keymap = default_keymap
        self.state = INPUT_STATE.EDIT

    @dn.datanode
    def input_handler(self):
        while True:
            _, key = yield
            self.event.set()

            # completions
            while key == self.keymap["Tab"]:
                self.state = INPUT_STATE.TAB
                for _ in self.text.complete():
                    _, key = yield
                    if key != self.keymap["Tab"]:
                        self.state = INPUT_STATE.EDIT
                        break
                else:
                    self.state = INPUT_STATE.EDIT
                    _, key = yield

            # edit
            if len(key) == 1 and key.isprintable():
                self.text.input(key)

            elif key == self.keymap["Backspace"]:
                self.text.backspace()

            elif key == self.keymap["Delete"]:
                self.text.delete()

            elif key == self.keymap["Left"]:
                self.text.move(-1)

            elif key == self.keymap["Right"]:
                self.text.move(+1)

            elif key == self.keymap["Home"]:
                self.text.move_to(0)

            elif key == self.keymap["End"]:
                self.text.move_to_end()


BLOCKER_HEADERS = [
    "\x1b[96;1m⠶⠦⣚⠀⠶\x1b[m\x1b[38;5;255m❯\x1b[m ",
    "\x1b[96;1m⢎⣀⡛⠀⠶\x1b[m\x1b[38;5;255m❯\x1b[m ",
    "\x1b[36m⢖⣄⠻⠀⠶\x1b[m\x1b[38;5;254m❯\x1b[m ",
    "\x1b[36m⠖⠐⡩⠂⠶\x1b[m\x1b[38;5;254m❯\x1b[m ",
    "\x1b[96m⠶⠀⡭⠲⠶\x1b[m\x1b[38;5;253m❯\x1b[m ",
    "\x1b[36m⠶⠀⣬⠉⡱\x1b[m\x1b[38;5;253m❯\x1b[m ",
    "\x1b[36m⠶⠀⣦⠙⠵\x1b[m\x1b[38;5;252m❯\x1b[m ",
    "\x1b[36m⠶⠠⣊⠄⠴\x1b[m\x1b[38;5;252m❯\x1b[m ",

    "\x1b[96m⠶⠦⣚⠀⠶\x1b[m\x1b[38;5;251m❯\x1b[m ",
    "\x1b[36m⢎⣀⡛⠀⠶\x1b[m\x1b[38;5;251m❯\x1b[m ",
    "\x1b[36m⢖⣄⠻⠀⠶\x1b[m\x1b[38;5;250m❯\x1b[m ",
    "\x1b[36m⠖⠐⡩⠂⠶\x1b[m\x1b[38;5;250m❯\x1b[m ",
    "\x1b[96m⠶⠀⡭⠲⠶\x1b[m\x1b[38;5;249m❯\x1b[m ",
    "\x1b[36m⠶⠀⣬⠉⡱\x1b[m\x1b[38;5;249m❯\x1b[m ",
    "\x1b[36m⠶⠀⣦⠙⠵\x1b[m\x1b[38;5;248m❯\x1b[m ",
    "\x1b[36m⠶⠠⣊⠄⠴\x1b[m\x1b[38;5;248m❯\x1b[m ",

    "\x1b[96m⠶⠦⣚⠀⠶\x1b[m\x1b[38;5;247m❯\x1b[m ",
    "\x1b[36m⢎⣀⡛⠀⠶\x1b[m\x1b[38;5;247m❯\x1b[m ",
    "\x1b[36m⢖⣄⠻⠀⠶\x1b[m\x1b[38;5;246m❯\x1b[m ",
    "\x1b[36m⠖⠐⡩⠂⠶\x1b[m\x1b[38;5;246m❯\x1b[m ",
    "\x1b[96m⠶⠀⡭⠲⠶\x1b[m\x1b[38;5;245m❯\x1b[m ",
    "\x1b[36m⠶⠀⣬⠉⡱\x1b[m\x1b[38;5;245m❯\x1b[m ",
    "\x1b[36m⠶⠀⣦⠙⠵\x1b[m\x1b[38;5;244m❯\x1b[m ",
    "\x1b[36m⠶⠠⣊⠄⠴\x1b[m\x1b[38;5;244m❯\x1b[m ",

    "\x1b[96m⠶⠦⣚⠀⠶\x1b[m\x1b[38;5;243m❯\x1b[m ",
    "\x1b[36m⢎⣀⡛⠀⠶\x1b[m\x1b[38;5;243m❯\x1b[m ",
    "\x1b[36m⢖⣄⠻⠀⠶\x1b[m\x1b[38;5;242m❯\x1b[m ",
    "\x1b[36m⠖⠐⡩⠂⠶\x1b[m\x1b[38;5;242m❯\x1b[m ",
    "\x1b[96m⠶⠀⡭⠲⠶\x1b[m\x1b[38;5;241m❯\x1b[m ",
    "\x1b[36m⠶⠀⣬⠉⡱\x1b[m\x1b[38;5;241m❯\x1b[m ",
    "\x1b[36m⠶⠀⣦⠙⠵\x1b[m\x1b[38;5;240m❯\x1b[m ",
    "\x1b[36m⠶⠠⣊⠄⠴\x1b[m\x1b[38;5;240m❯\x1b[m ",
]

class BeatPrompt:
    def __init__(self, stroke, text, framerate, offset=0.0, tempo=130.0):
        self.stroke = stroke
        self.text = text
        self.framerate = framerate
        self.offset = offset
        self.tempo = tempo
        self.headers = BLOCKER_HEADERS

    @dn.datanode
    def output_handler(self):
        size_node = dn.terminal_size()

        with size_node:
            yield
            t = self.offset/(60/self.tempo)
            tr = 0
            while True:
                if self.stroke.event.is_set():
                    self.stroke.event.clear()
                    tr = t // -1 * -1

                ind = int(t / 4 * len(self.headers)) % len(self.headers)
                header = self.headers[ind]

                # cursor
                if t-tr < 0 or (t-tr) % 1 < 0.3:
                    if ind == 0 or ind == 1:
                        cursor = lambda s: f"\x1b[7;1m{s}\x1b[m"
                    else:
                        cursor = lambda s: f"\x1b[7;2m{s}\x1b[m"

                    if self.stroke.state == INPUT_STATE.TAB:
                        cursor = cursor("↹ ")
                else:
                    cursor = None

                # size
                try:
                    size = size_node.send(None)
                except StopIteration:
                    return
                width = size.columns

                # draw
                view = tui.newwin1(width)
                view, x = tui.addtext1(view, width, 0, header)
                self.text.render(view, width, slice(x, None), cursor=cursor)

                yield "\r" + "".join(view) + "\r"
                t += 1/self.framerate/(60/self.tempo)


def prompt(command, framerate=60.0):
    text = BeatText(command=command)
    stroke = BeatStroke(text=text)
    input_knot = dn.input(stroke.input_handler())
    prompt = BeatPrompt(stroke, text, framerate)
    display_knot = dn.interval(prompt.output_handler(), dn.show(hide_cursor=True), 1/framerate)

    menu_knot = dn.pipe(input_knot, display_knot)
    dn.exhaust(menu_knot, dt=0.01, interruptible=True)


