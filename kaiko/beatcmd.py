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
            token, _, _ = next(parser)
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


class PromptComplete(Exception):
    pass

class TOKEN_TYPE(Enum):
    COMMAND = "command"
    FUNCTION = "function"
    ARGUMENT = "argument"
    LITERAL = "literal"

class Promptable:
    def __init__(self, root):
        self.root = root

    def _node(self):
        curr = self.root

        try:
            while True:
                token = yield TOKEN_TYPE.COMMAND
                if curr is None:
                    pass
                elif not isinstance(curr, dict):
                    curr = None
                elif token not in curr:
                    curr = None
                else:
                    curr = curr.get(token)

        except PromptComplete as e:
            if curr is not None:
                target, = e.args
                return [key[len(target):] for key in curr.keys() if key.startswith(target)]
            else:
                return []

        except GeneratorExit:
            return curr

    def parse(self, tokens):
        gen = self._node()
        next(gen)

        types = []
        for token in tokens:
            types.append(gen.send(token))

        gen.close()
        return types

    def explore(self, tokens):
        gen = self._node()
        next(gen)

        for token in tokens:
            gen.send(token)

        return gen.close()

    def complete(self, tokens, target):
        if target == "":
            return ""

        gen = self._node()
        next(gen)

        for token in tokens:
            gen.send(token)

        try:
            gen.throw(PromptComplete(target))
        except StopIteration as e:
            return e.value
        else:
            raise ValueError


class BeatInput:
    def __init__(self, promptable):
        self.buffer = []
        self.pos = 0
        self.promptable = promptable
        self.suggestion = ""

        self.tokens = []
        self.state = SHLEXER_STATE.SPACED

    def parse(self):
        lex_parser = shlexer_parse(self.buffer, partial=True)

        tokens = []
        while True:
            try:
                token, slic, masked = next(lex_parser)
            except StopIteration as e:
                self.state = e.value
                break

            tokens.append((token, slic, masked))

        types = self.promptable.parse(token for token, _, _ in tokens)
        self.tokens = [(token, type, slic, masked) for (token, slic, masked), type in zip(tokens, types)]

    def complete(self):
        self.suggestion = ""

        lex_completer = shlexer_complete(self.buffer, self.pos, self.promptable.complete)

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

    def suggest(self):
        lex_completer = shlexer_complete(self.buffer, len(self.buffer), self.promptable.complete)
        compreply = next(lex_completer, None)
        if compreply is None:
            self.suggestion = ""
        elif compreply[0] != "\b":
            self.suggestion = "".join(compreply)

    def input(self, ch):
        self.buffer[self.pos:self.pos] = [ch]
        self.pos += 1
        self.parse()
        if self.pos == len(self.buffer):
            self.suggest()

    def backspace(self):
        if self.pos == 0:
            return
        self.pos -= 1
        del self.buffer[self.pos]
        self.parse()
        self.suggestion = ""

    def delete(self):
        if self.pos >= len(self.buffer):
            return
        del self.buffer[self.pos]
        self.parse()
        self.suggestion = ""

    def move(self, offset):
        self.pos = min(max(0, self.pos+offset), len(self.buffer))
        self.suggestion = ""

    def move_to(self, pos):
        self.pos = min(max(0, pos), len(self.buffer))
        self.suggestion = ""

    def move_to_end(self):
        self.pos = len(self.buffer)
        self.suggestion = ""

    def move_to_token_start(self):
        width = len(self.buffer)
        grid = (slic.indices(width)[0] for _, _, slic, _ in self.tokens[::-1])
        self.pos = next((pos for pos in grid if pos < self.pos), 0)
        self.suggestion = ""

    def move_to_token_end(self):
        width = len(self.buffer)
        grid = (slic.indices(width)[1] for _, _, slic, _ in self.tokens)
        self.pos = next((pos for pos in grid if pos > self.pos), width)
        self.suggestion = ""


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

class INPUT_STATE(Enum):
    EDIT = "edit"
    TAB = "tab"

class BeatStroke:
    def __init__(self, input):
        self.input = input
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
                for _ in self.input.complete():
                    _, key = yield
                    if key != self.keymap["Tab"]:
                        self.state = INPUT_STATE.EDIT
                        break
                else:
                    self.state = INPUT_STATE.EDIT
                    _, key = yield

            # edit
            if len(key) == 1 and key.isprintable():
                self.input.input(key)

            elif key == self.keymap["Backspace"]:
                self.input.backspace()

            elif key == self.keymap["Delete"]:
                self.input.delete()

            elif key == self.keymap["Left"]:
                self.input.move(-1)

            elif key == self.keymap["Right"]:
                self.input.move(+1)

            elif key == self.keymap["Ctrl+Left"]:
                self.input.move_to_token_start()

            elif key == self.keymap["Ctrl+Right"]:
                self.input.move_to_token_end()

            elif key == self.keymap["Home"]:
                self.input.move_to(0)

            elif key == self.keymap["End"]:
                self.input.move_to_end()


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
BLOCKER_HEADER_WIDTH = 7

class BeatPrompt:
    def __init__(self, stroke, input, framerate, t0=0.0, tempo=130.0):
        self.stroke = stroke
        self.input = input
        self.framerate = framerate
        self.t0 = t0
        self.tempo = tempo
        self.headers = BLOCKER_HEADERS
        self.header_width = BLOCKER_HEADER_WIDTH

    def output_handler(self):
        size_node = dn.terminal_size()
        header_node = self.header_node()
        render_node = self.render_node()
        draw_node = self.draw_node()
        return dn.pipe((lambda _: (None,None,None)), dn.pair(header_node, render_node, size_node), draw_node)

    @dn.datanode
    def header_node(self):
        yield
        t = self.t0/(60/self.tempo)
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

            yield header, cursor
            t += 1/self.framerate/(60/self.tempo)

    @dn.datanode
    def render_node(self):
        render_escape = lambda s: f"\x1b[2m{s}\x1b[m"
        render_warn = lambda s: f"\x1b[31m{s}\x1b[m"
        render_suggestion = lambda s: f"\x1b[2m{s}\x1b[m"
        whitespace = "\x1b[2m⌴\x1b[m"

        yield
        while True:
            # render buffer
            rendered_buffer = list(self.input.buffer)
            for token, type, slic, masked in self.input.tokens:
                for index in masked:
                    rendered_buffer[index] = render_escape(rendered_buffer[index])

                for index in range(len(rendered_buffer))[slic]:
                    if rendered_buffer[index] == " ":
                        rendered_buffer[index] = whitespace

                if type is None and slic.stop is not None:
                    for index in range(len(rendered_buffer))[slic]:
                        rendered_buffer[index] = render_warn(rendered_buffer[index])

            rendered_text = "".join(rendered_buffer)
            rendered_suggestion = render_suggestion(self.input.suggestion) if self.input.suggestion else ""
            _, cursor_pos = tui.textrange1(0, "".join(rendered_buffer[:self.input.pos]))

            yield rendered_text, rendered_suggestion, cursor_pos

    @dn.datanode
    def draw_node(self):
        input_offset = 0

        (header, cursor), (rendered_text, rendered_suggestion, cursor_pos), size = yield
        while True:
            width = size.columns

            # draw header
            view = tui.newwin1(width)
            tui.addtext1(view, width, 0, header, slice(0, self.header_width))

            # adjust input offset
            input_ran = slice(self.header_width, None)
            input_width = len(range(width)[input_ran])
            _, text_length = tui.textrange1(0, rendered_text)

            if text_length+1 <= input_width:
                input_offset = 0
            elif cursor_pos - input_offset >= input_width:
                input_offset = cursor_pos - input_width + 1
            elif cursor_pos - input_offset < 0:
                input_offset = cursor_pos

            # draw input
            tui.addtext1(view, width, input_ran.start-input_offset,
                         rendered_text+rendered_suggestion, input_ran)
            if input_offset > 0:
                tui.addtext1(view, width, input_ran.start, "…", input_ran)
            if text_length-input_offset >= input_width:
                tui.addtext1(view, width, input_ran.start+input_width-1, "…", input_ran)

            # draw cursor
            if cursor:
                cursor_x = input_ran.start - input_offset + cursor_pos
                cursor_ran = tui.select1(view, width, slice(cursor_x, cursor_x+1))
                if hasattr(cursor, '__call__'):
                    view[cursor_ran.start] = cursor(view[cursor_ran.start])
                else:
                    tui.addtext1(view, width, cursor_ran.start, cursor)

            (header, cursor), (rendered_text, rendered_suggestion, cursor_pos), size = yield "\r" + "".join(view) + "\r"


def prompt(promptable, framerate=60.0):
    input = BeatInput(promptable=promptable)
    stroke = BeatStroke(input=input)
    input_knot = dn.input(stroke.input_handler())
    prompt = BeatPrompt(stroke, input, framerate)
    display_knot = dn.interval(prompt.output_handler(), dn.show(hide_cursor=True), 1/framerate)

    prompt_knot = dn.pipe(input_knot, display_knot)
    dn.exhaust(prompt_knot, dt=0.01, interruptible=True)


