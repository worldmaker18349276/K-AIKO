from enum import Enum
import functools
import re
import queue
import threading
import inspect
from typing import List, Tuple
from . import datanodes as dn
from . import tui
from . import cfg


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
    is_appended = len(raw) == index
    parser = shlexer_parse(raw[:index], partial=True)
    tokens = []
    try:
        while True:
            token, _, _ = next(parser)
            tokens.append(token)
    except StopIteration as e:
        state = e.value

    if state == SHLEXER_STATE.SPACED:
        # empty target
        target = ""
        for compreply in completer(tokens, target):
            # escape any compreply, including empty compreply
            compreply = list(shlexer_escape(compreply))
            yield [*compreply, " "]

    elif state == SHLEXER_STATE.PLAIN:
        # use the last token as target
        target = tokens[-1]
        tokens = tokens[:-1]
        for compreply in completer(tokens, target):
            # don't escape empty compreply
            compreply = list(shlexer_escape(compreply)) if compreply else []
            yield [*compreply, " "]

    elif state == SHLEXER_STATE.BACKSLASHED:
        target = tokens[-1]
        tokens = tokens[:-1]
        for compreply in completer(tokens, target):
            # delete backslash for empty compreply
            compreply = list(shlexer_escape(compreply, SHLEXER_ESCAPE.BACKSLASH)) if compreply else ["\b"]
            # don't escape the backslash for escaping
            if compreply[0] == "\\":
                compreply = compreply[1:]
            # add back backslash for original escaped character
            yield [*compreply, " "] if is_appended else [*compreply, " ", "\\"]

    elif state == SHLEXER_STATE.QUOTED:
        target = tokens[-1]
        tokens = tokens[:-1]
        for compreply in completer(tokens, target):
            compreply = list(shlexer_escape(compreply, SHLEXER_ESCAPE.QUOTE))
            # remove opening quote
            compreply = compreply[1:]
            # add back quote for original escaped string
            yield [*compreply, " "] if is_appended else [*compreply, " ", "'"]

    elif state == SHLEXER_STATE.DOUBLE_QUOTED:
        target = tokens[-1]
        tokens = tokens[:-1]
        for compreply in completer(tokens, target):
            compreply = list(shlexer_escape(compreply, SHLEXER_ESCAPE.DOUBLE_QUOTE))
            # remove opening double quote
            compreply = compreply[1:]
            # add back double quote for original escaped string
            yield [*compreply, " "] if is_appended else [*compreply, " ", '"']

def echo_str(escaped_str):
    regex = r"\\c.*|\\[\\abefnrtv]|\\0[0-7]{0,3}|\\x[0-9a-fA-F]{1,2}|."

    escaped = {
        r"\\": "\\",
        r"\a": "\a",
        r"\b": "\b",
        "\\e": "\x1b",
        r"\f": "\f",
        r"\n": "\n",
        r"\r": "\r",
        r"\t": "\t",
        r"\v": "\v",
        }

    def repl(match):
        matched = match.group(0)

        if matched.startswith("\\c"):
            return ""
        elif matched in escaped:
            return escaped[matched]
        elif matched.startswith("\\0"):
            return chr(int(matched[2:] or "0", 8))
        elif matched.startswith("\\x"):
            return chr(int(matched[2:], 16))
        else:
            return matched

    return re.sub(regex, repl, escaped_str)


class PromptUnfinishError(Exception):
    def __init__(self, index, suggestion, info):
        self.index = index
        self.suggestion = suggestion
        self.info = info

class PromptParseError(Exception):
    def __init__(self, index, suggestion, info):
        self.index = index
        self.suggestion = suggestion
        self.info = info

class TOKEN_TYPE(Enum):
    COMMAND = "command"
    FUNCTION = "function"
    ARGUMENT = "argument"
    LITERAL = "literal"

class Promptable:
    def __init__(self, root):
        self.root = root
        self._PromptReturn = object()

    def _sig(self, func):
        # parse signature: (a, b, c, d=1, e=2)
        if not hasattr(func, '__call__'):
            raise ValueError(f"Not a function: {curr!r}")

        sig = inspect.signature(func)
        args = list()
        kwargs = dict()
        for param in sig.parameters.values():
            if param.default is param.empty:
                args.append(param)
            else:
                kwargs["--" + param.name] = param

        return args, kwargs

    def parse_lit(self, token, param):
        type = param.annotation
        if isinstance(type, (tuple, list)) and token in type:
            return token

        elif type == bool and re.fullmatch("True|False", token):
            return bool(token)

        elif type == int and re.fullmatch(r"[-+]?(0|[1-9][0-9]*)", token):
            return int(token)

        elif type == float and re.fullmatch(r"[-+]?([0-9]+\.[0-9]+(e[-+]?[0-9]+)?|[0-9]+e[-+]?[0-9]+)", token):
            return float(token)

        elif type == str:
            return token

        else:
            return None

    def suggest_lit(self, target, param):
        type = param.annotation
        if isinstance(type, (tuple, list)):
            return type

        elif type == bool:
            if param.default is param.empty:
                return ["True", "False"]
            else:
                return [str(param.default), str(not param.default)]

        elif type == int:
            if param.default is param.empty:
                return []
            else:
                return [str(param.default)]

        elif type == float:
            if param.default is param.empty:
                return []
            else:
                return [str(param.default)]

        elif type == str:
            if param.default is param.empty:
                return []
            else:
                return [param.default]

        else:
            return []

    def parse(self, tokens):
        tokens = iter(tokens)
        curr = self.root
        types = []

        try:
            # parse command
            while isinstance(curr, dict):
                types.append(TOKEN_TYPE.COMMAND)

                token = next(tokens)

                if token not in curr:
                    while True:
                        types.append(None)
                        token = next(tokens)
                curr = curr.get(token)

            # parse function
            if not hasattr(curr, '__call__'):
                while True:
                    types.append(None)
                    token = next(tokens)
            args, kwargs = self._sig(curr)
            types.append(TOKEN_TYPE.FUNCTION)

            token = next(tokens)

            # parse positional arguments
            for param in args:
                value = self.parse_lit(token, param)
                if value is None:
                    while True:
                        types.append(None)
                        token = next(tokens)
                types.append(TOKEN_TYPE.LITERAL)

                token = next(tokens)

            # parse keyword arguments
            while kwargs:
                # parse argument name
                param = kwargs.pop(token, None)
                if param is None:
                    while True:
                        types.append(None)
                        token = next(tokens)
                types.append(TOKEN_TYPE.ARGUMENT)

                token = next(tokens)

                # parse argument value
                value = self.parse_lit(token, param)
                if value is None:
                    while True:
                        types.append(None)
                        token = next(tokens)
                types.append(TOKEN_TYPE.LITERAL)

                token = next(tokens)

            # rest
            while True:
                types.append(None)
                token = next(tokens)

        except StopIteration:
            pass

        return types[1:]

    def generate(self, tokens, state=SHLEXER_STATE.SPACED):
        if state == SHLEXER_STATE.BACKSLASHED:
            raise PromptParseError(len(tokens)-1, None, "No escaped character")
        if state == SHLEXER_STATE.QUOTED:
            raise PromptParseError(len(tokens)-1, None, "No closing quotation")
        if state == SHLEXER_STATE.DOUBLE_QUOTED:
            raise PromptParseError(len(tokens)-1, None, "No closing double quotation")

        tokens = iter(tokens)
        curr = self.root
        index = -1

        # parse command
        while isinstance(curr, dict):
            token = next(tokens, None)
            index += 1

            if token is None:
                raise PromptUnfinishError(index, list(curr.keys()), "Unfinished command")
            if token not in curr:
                raise PromptParseError(index, list(curr.keys()), "Invalid command")
            curr = curr.get(token)

        # parse function
        if not hasattr(curr, '__call__'):
            raise ValueError(f"Not a function: {curr!r}")
        args, kwargs = self._sig(curr)

        token = next(tokens, None)
        index += 1

        # parse positional arguments
        for param in args:
            if token is None:
                raise PromptUnfinishError(index, param.annotation, "Missing value")
            value = self.parse_lit(token, param)
            if value is None:
                raise PromptParseError(index, param.annotation, "Invalid value")
            curr = functools.partial(curr, value)

            token = next(tokens, None)
            index += 1

        # parse keyword arguments
        while kwargs and token is not None:
            # parse argument name
            param = kwargs.pop(token, None)
            if param is None:
                raise PromptUnfinishError(index, list(kwargs.keys()), "Unkown argument")

            token = next(tokens, None)
            index += 1

            # parse argument value
            if token is None:
                raise PromptUnfinishError(index, param.annotation, "Missing value")
            value = self.parse_lit(token, param)
            if value is None:
                raise PromptParseError(index, param.annotation, "Invalid value")
            curr = functools.partial(curr, **{param.name: value})

            token = next(tokens, None)
            index += 1

        # rest
        if token is not None:
            raise PromptParseError(index, None, "Too many arguments")

        return curr

    def suggest(self, tokens, target):
        tokens = iter(tokens)
        curr = self.root

        # parse command
        while isinstance(curr, dict):
            token = next(tokens, None)

            if token is None:
                return list(curr.keys())
            if token not in curr:
                return []
            curr = curr.get(token)

        # parse function
        if not hasattr(curr, '__call__'):
            return []
        args, kwargs = self._sig(curr)

        token = next(tokens, None)

        # parse positional arguments
        for param in args:
            if token is None:
                return self.suggest_lit(target, param)
            value = self.parse_lit(token, param)
            if value is None:
                return []

            token = next(tokens, None)

        # parse keyword arguments
        while kwargs:
            # parse argument name
            if token is None:
                return list(kwargs.keys())
            param = kwargs.pop(token, None)
            if param is None:
                return []

            token = next(tokens, None)

            # parse argument value
            if token is None:
                return self.suggest_lit(target, param)
            value = self.parse_lit(token, param)
            if value is None:
                return []

            token = next(tokens, None)

        # rest
        return []

    def complete(self, tokens, target):
        return [sugg[len(target):] for sugg in self.suggest(tokens, target) if sugg.startswith(target)]


class InputError:
    def __init__(self, pointto, message):
        self.pointto = pointto
        self.message = message

class InputResult:
    def __init__(self, value):
        self.value = value

class BeatInput:
    def __init__(self, promptable, history=None):
        self.promptable = promptable
        self.history = history if history is not None else []

        self.buffers = [list(history_buffer) for history_buffer in self.history]
        self.buffers.append([])
        self.buffers_index = -1

        self.buffer = self.buffers[self.buffers_index]
        self.pos = len(self.buffer)
        self.typeahead = ""

        self.tokens = []
        self.state = SHLEXER_STATE.SPACED

        self.result = queue.Queue()

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

    def complete(self, action=+1):
        self.typeahead = ""

        compreplies = list(shlexer_complete(self.buffer, self.pos, self.promptable.complete))
        length = len(compreplies)

        original_buffer = list(self.buffer)
        original_pos = self.pos

        if action == +1:
            index = 0
        elif action == -1:
            index = length-1
        else:
            raise ValueError

        while index in range(length):
            self.input(compreplies[index], False)

            action = yield
            if action == +1:
                index += 1
            elif action == -1:
                index -= 1
            elif action == 0:
                index = None

            self.buffer = list(original_buffer)
            self.pos = original_pos

        else:
            self.parse()

    def enter(self):
        if len(self.tokens) == 0:
            self.delete_range(None, None)
            self.result.put(InputError(None, None))
            return False

        try:
            res = self.promptable.generate([token for token, _, _, _ in self.tokens], self.state)

        except PromptUnfinishError as e:
            pointto = slice(len(self.buffer)-1, len(self.buffer))

            if isinstance(e.suggestion, (tuple, list)):
                sugg = e.suggestion[:5] + ["…"] if len(e.suggestion) > 5 else e.suggestion
                sugg = "\n".join("  " + "".join(shlexer_escape(s)) for s in sugg)
                msg = e.info + "\n" + f"It should be followed by:\n{sugg}"
            elif isinstance(e.suggestion, type):
                msg = e.info + "\n" + f"It should be followed by {e.suggestion.__name__} literal"
            elif e.suggestion is not None:
                msg = e.info + "\n" + f"It should be followed by {e.suggestion}"
            else:
                msg = e.info

            self.result.put(InputError(pointto, msg))
            return False

        except PromptParseError as e:
            _, _, pointto, _ = self.tokens[e.index]

            if isinstance(e.suggestion, (tuple, list)):
                sugg = e.suggestion[:5] + ["…"] if len(e.suggestion) > 5 else e.suggestion
                sugg = "\n".join("  " + "".join(shlexer_escape(s)) for s in sugg)
                msg = e.info + "\n" + f"It should be one of:\n{sugg}"
            elif isinstance(e.suggestion, type):
                msg = e.info + "\n" + f"It should be {e.suggestion.__name__} literal"
            elif e.suggestion is not None:
                msg = e.info + "\n" + f"It should be {e.suggestion}"
            else:
                msg = e.info

            self.result.put(InputError(pointto, msg))
            return False

        else:
            self.history.append("".join(self.buffer))
            self.result.put(InputResult(res))
            return True

    def suggest(self, suggest=True):
        if not suggest:
            self.typeahead = ""
            return False

        lex_completer = shlexer_complete(self.buffer, len(self.buffer), self.promptable.complete)
        compreply = next(lex_completer, None)

        if compreply is None:
            self.typeahead = ""
            return False
        elif compreply[0] != "\b":
            self.typeahead = "".join(compreply)
            return True
        else:
            return False

    def input(self, text, suggest=True):
        text = list(text)

        if len(text) == 0:
            return False

        while len(text) > 0 and text[0] == "\b":
            del text[0]
            del self.buffer[self.pos-1]
            self.pos = self.pos-1

        self.buffer[self.pos:self.pos] = text
        self.pos += len(text)
        self.parse()

        self.suggest(suggest and self.pos == len(self.buffer))

        return True

    def backspace(self):
        if self.pos == 0:
            return False

        self.pos -= 1
        del self.buffer[self.pos]
        self.parse()
        self.suggest(False)

        return True

    def delete(self):
        if self.pos >= len(self.buffer):
            return False

        del self.buffer[self.pos]
        self.parse()
        self.suggest(False)

        return True

    def delete_range(self, start, end):
        start = min(max(0, start), len(self.buffer)) if start is not None else 0
        end = min(max(0, end), len(self.buffer)) if end is not None else len(self.buffer)

        if start >= end:
            return False

        del self.buffer[start:end]
        self.pos = start
        self.parse()
        self.suggest(False)

        return True

    def delete_to_word_start(self):
        for match in re.finditer("\w+|.", "".join(self.buffer)):
            if match.end() >= self.pos:
                return self.delete_range(match.start(), self.pos)
        else:
            return self.delete_range(None, self.pos)

    def delete_to_word_end(self):
        for match in re.finditer("\w+|.", "".join(self.buffer)):
            if match.end() > self.pos:
                return self.delete_range(self.pos, match.end())
        else:
            return self.delete_range(self.pos, None)

    def move(self, offset):
        return self.move_to(self.pos+offset)

    def move_left(self):
        return self.move(-1)

    def move_right(self):
        return self.move(+1)

    def move_to(self, pos):
        pos = min(max(0, pos), len(self.buffer)) if pos is not None else len(self.buffer)
        self.suggest(False)

        if self.pos == pos:
            return False

        self.pos = pos
        return True

    def move_to_start(self):
        return self.move_to(0)

    def move_to_end(self):
        return self.move_to(None)

    def move_to_word_start(self):
        for match in re.finditer("\w+|.", "".join(self.buffer)):
            if match.end() >= self.pos:
                return self.move_to(match.start())
        else:
            return self.move_to(0)

    def move_to_word_end(self):
        for match in re.finditer("\w+|.", "".join(self.buffer)):
            if match.end() > self.pos:
                return self.move_to(match.end())
        else:
            return self.move_to(None)

    def prev(self):
        if self.buffers_index == -len(self.buffers):
            return False
        self.buffers_index -= 1

        self.buffer = self.buffers[self.buffers_index]
        self.pos = len(self.buffer)
        self.parse()
        self.suggest(False)

        return True

    def next(self):
        if self.buffers_index == -1:
            return False
        self.buffers_index += 1

        self.buffer = self.buffers[self.buffers_index]
        self.pos = len(self.buffer)
        self.parse()
        self.suggest(False)

        return True


default_keycodes = {
    "Esc"       : "\x1b",
    "Alt+Esc"   : "\x1b\x1b",

    "Enter"     : "\n",
    "Alt+Enter" : "\x1b\n",

    "Backspace"            : "\x7f",
    "Ctrl+Backspace"       : "\x08",
    "Alt+Backspace"        : "\x1b\x7f",
    "Ctrl+Alt+Backspace"   : "\x1b\x08",

    "Tab"                  : "\t",
    "Shift+Tab"            : "\x1b[Z",
    "Alt+Tab"              : "\x1b\t",
    "Alt+Shift+Tab"        : "\x1b\x1b[Z",

    "Up"                   : "\x1b[A",
    "Shift+Up"             : "\x1b[1;2A",
    "Alt+Up"               : "\x1b[1;3A",
    "Alt+Shift+Up"         : "\x1b[1;4A",
    "Ctrl+Up"              : "\x1b[1;5A",
    "Ctrl+Shift+Up"        : "\x1b[1;6A",
    "Ctrl+Alt+Up"          : "\x1b[1;7A",
    "Ctrl+Alt+Shift+Up"    : "\x1b[1;8A",

    "Down"                 : "\x1b[B",
    "Shift+Down"           : "\x1b[1;2B",
    "Alt+Down"             : "\x1b[1;3B",
    "Alt+Shift+Down"       : "\x1b[1;4B",
    "Ctrl+Down"            : "\x1b[1;5B",
    "Ctrl+Shift+Down"      : "\x1b[1;6B",
    "Ctrl+Alt+Down"        : "\x1b[1;7B",
    "Ctrl+Alt+Shift+Down"  : "\x1b[1;8B",

    "Right"                : "\x1b[C",
    "Shift+Right"          : "\x1b[1;2C",
    "Alt+Right"            : "\x1b[1;3C",
    "Alt+Shift+Right"      : "\x1b[1;4C",
    "Ctrl+Right"           : "\x1b[1;5C",
    "Ctrl+Shift+Right"     : "\x1b[1;6C",
    "Ctrl+Alt+Right"       : "\x1b[1;7C",
    "Ctrl+Alt+Shift+Right" : "\x1b[1;8C",

    "Left"                 : "\x1b[D",
    "Shift+Left"           : "\x1b[1;2D",
    "Alt+Left"             : "\x1b[1;3D",
    "Alt+Shift+Left"       : "\x1b[1;4D",
    "Ctrl+Left"            : "\x1b[1;5D",
    "Ctrl+Shift+Left"      : "\x1b[1;6D",
    "Ctrl+Alt+Left"        : "\x1b[1;7D",
    "Ctrl+Alt+Shift+Left"  : "\x1b[1;8D",

    "End"                  : "\x1b[F",
    "Shift+End"            : "\x1b[1;2F",
    "Alt+End"              : "\x1b[1;3F",
    "Alt+Shift+End"        : "\x1b[1;4F",
    "Ctrl+End"             : "\x1b[1;5F",
    "Ctrl+Shift+End"       : "\x1b[1;6F",
    "Ctrl+Alt+End"         : "\x1b[1;7F",
    "Ctrl+Alt+Shift+End"   : "\x1b[1;8F",

    "Home"                 : "\x1b[H",
    "Shift+Home"           : "\x1b[1;2H",
    "Alt+Home"             : "\x1b[1;3H",
    "Alt+Shift+Home"       : "\x1b[1;4H",
    "Ctrl+Home"            : "\x1b[1;5H",
    "Ctrl+Shift+Home"      : "\x1b[1;6H",
    "Ctrl+Alt+Home"        : "\x1b[1;7H",
    "Ctrl+Alt+Shift+Home"  : "\x1b[1;8H",

    "Insert"                 : "\x1b[2~",
    "Shift+Insert"           : "\x1b[2;2~",
    "Alt+Insert"             : "\x1b[2;3~",
    "Alt+Shift+Insert"       : "\x1b[2;4~",
    "Ctrl+Insert"            : "\x1b[2;5~",
    "Ctrl+Shift+Insert"      : "\x1b[2;6~",
    "Ctrl+Alt+Insert"        : "\x1b[2;7~",
    "Ctrl+Alt+Shift+Insert"  : "\x1b[2;8~",

    "Delete"                 : "\x1b[3~",
    "Shift+Delete"           : "\x1b[3;2~",
    "Alt+Delete"             : "\x1b[3;3~",
    "Alt+Shift+Delete"       : "\x1b[3;4~",
    "Ctrl+Delete"            : "\x1b[3;5~",
    "Ctrl+Shift+Delete"      : "\x1b[3;6~",
    "Ctrl+Alt+Delete"        : "\x1b[3;7~",
    "Ctrl+Alt+Shift+Delete"  : "\x1b[3;8~",

    "PageUp"                  : "\x1b[5~",
    "Shift+PageUp"            : "\x1b[5;2~",
    "Alt+PageUp"              : "\x1b[5;3~",
    "Alt+Shift+PageUp"        : "\x1b[5;4~",
    "Ctrl+PageUp"             : "\x1b[5;5~",
    "Ctrl+Shift+PageUp"       : "\x1b[5;6~",
    "Ctrl+Alt+PageUp"         : "\x1b[5;7~",
    "Ctrl+Alt+Shift+PageUp"   : "\x1b[5;8~",

    "PageDown"                : "\x1b[6~",
    "Shift+PageDown"          : "\x1b[6;2~",
    "Alt+PageDown"            : "\x1b[6;3~",
    "Alt+Shift+PageDown"      : "\x1b[6;4~",
    "Ctrl+PageDown"           : "\x1b[6;5~",
    "Ctrl+Shift+PageDown"     : "\x1b[6;6~",
    "Ctrl+Alt+PageDown"       : "\x1b[6;7~",
    "Ctrl+Alt+Shift+PageDown" : "\x1b[6;8~",
}

default_keymap = {
    "Backspace": lambda input: input.backspace(),
       "Delete": lambda input: input.delete(),
         "Left": lambda input: input.move_left(),
        "Right": lambda input: input.move_right(),
           "Up": lambda input: input.prev(),
         "Down": lambda input: input.next(),
         "Home": lambda input: input.move_to_start(),
          "End": lambda input: input.move_to_end(),
        "Enter": lambda input: input.enter(),
         "Ctrl+Left": lambda input: input.move_to_word_start(),
        "Ctrl+Right": lambda input: input.move_to_word_end(),
    "Ctrl+Backspace": lambda input: input.delete_to_word_start(),
       "Ctrl+Delete": lambda input: input.delete_to_word_end(),
}

class INPUT_STATE(Enum):
    EDIT = "edit"
    TAB = "tab"

class BeatStroke:
    def __init__(self, input, keymap, keycodes):
        self.input = input
        self.event = threading.Event()
        self.keymap = keymap
        self.keycodes = keycodes
        self.state = INPUT_STATE.EDIT

    @dn.datanode
    def input_handler(self):
        while True:
            _, key = yield
            self.event.set()

            # completions
            while key == self.keycodes["Tab"] or key == self.keycodes["Shift+Tab"]:
                self.state = INPUT_STATE.TAB

                if key == self.keycodes["Tab"]:
                    complete = self.input.complete(+1)
                elif key == self.keycodes["Shift+Tab"]:
                    complete = self.input.complete(-1)

                try:
                    next(complete)

                    while True:
                        _, key = yield
                        if key == self.keycodes["Tab"]:
                            complete.send(+1)
                        elif key == self.keycodes["Shift+Tab"]:
                            complete.send(-1)
                        elif key == self.keycodes["Esc"]:
                            complete.send(0)
                        else:
                            complete.close()
                            self.state = INPUT_STATE.EDIT
                            break

                except StopIteration:
                    self.state = INPUT_STATE.EDIT
                    _, key = yield

            # registered keystrokes
            for keyname, func in self.keymap.items():
                if key == self.keycodes[keyname]:
                    func(self.input)
                    break

            else:
                # edit
                if key.isprintable():
                    self.input.input(key)


class PromptTheme(metaclass=cfg.Configurable):
    framerate: float = 60.0
    t0: float = 0.0
    tempo: float = 130.0

    headers: List[str] = [
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
    header_width: int = 7

    cursor_attr: Tuple[str, str] = ("7;2", "7;1")
    cursor_blink_ratio: float = 0.3
    cursor_tab: str = "↹ "

    error_message_attr: str = "31"

    escape_attr: str = "2"
    typeahead_attr: str = "2"
    whitespace: str = "\x1b[2m⌴\x1b[m"

    token_invalid_attr: str = "31"
    token_command_attr: str = "94"
    token_function_attr: str = "94"
    token_argument_attr: str = "95"
    token_literal_attr: str = "92"

class BeatPrompt:
    def __init__(self, stroke, input, theme):
        self.stroke = stroke
        self.input = input
        self.theme = theme
        self.result = None

    def output_handler(self):
        size_node = dn.terminal_size()
        result_node = self.result_node()
        header_node = self.header_node()
        render_node = self.render_node()
        draw_node = self.draw_node()
        return dn.pipe((lambda _: (None, None, None, None)),
                       dn.pair(result_node, header_node, render_node, size_node),
                       draw_node)

    @dn.datanode
    def result_node(self):
        yield
        while True:
            result = None
            while not self.input.result.empty():
                result = self.input.result.get()

            yield result

            if isinstance(result, InputResult):
                self.result = result.value
                return

    @dn.datanode
    def header_node(self):
        t0 = self.theme.t0
        tempo = self.theme.tempo
        framerate = self.theme.framerate

        headers = self.theme.headers

        cursor_attr = self.theme.cursor_attr
        cursor_tab = self.theme.cursor_tab
        cursor_blink_ratio = self.theme.cursor_blink_ratio

        yield
        t = t0/(60/tempo)
        tr = 0
        while True:
            # don't blink while key pressing
            if self.stroke.event.is_set():
                self.stroke.event.clear()
                tr = t // -1 * -1

            # render cursor
            if (t-tr < 0 or (t-tr) % 1 < cursor_blink_ratio):
                if t % 4 < cursor_blink_ratio:
                    cursor = lambda s: tui.add_attr(s, cursor_attr[1])
                else:
                    cursor = lambda s: tui.add_attr(s, cursor_attr[0])

                if self.stroke.state == INPUT_STATE.TAB:
                    cursor = cursor(cursor_tab)
            else:
                cursor = None

            # render header
            ind = int(t / 4 * len(headers)) % len(headers)
            header = headers[ind]

            yield header, cursor
            t += 1/framerate/(60/tempo)

    @dn.datanode
    def render_node(self):
        escape_attr     = self.theme.escape_attr
        typeahead_attr = self.theme.typeahead_attr
        whitespace      = self.theme.whitespace

        token_invalid_attr  = self.theme.token_invalid_attr
        token_command_attr  = self.theme.token_command_attr
        token_function_attr = self.theme.token_function_attr
        token_argument_attr = self.theme.token_argument_attr
        token_literal_attr  = self.theme.token_literal_attr

        yield
        while True:
            # render buffer
            rendered_buffer = list(self.input.buffer)

            for token, type, slic, masked in self.input.tokens:
                # render escape
                for index in masked:
                    rendered_buffer[index] = tui.add_attr(rendered_buffer[index], escape_attr)

                # render whitespace
                for index in range(len(rendered_buffer))[slic]:
                    if rendered_buffer[index] == " ":
                        rendered_buffer[index] = whitespace

                # render invalid token except the final one
                if type is None and slic.stop is not None:
                    for index in range(len(rendered_buffer))[slic]:
                        rendered_buffer[index] = tui.add_attr(rendered_buffer[index], token_invalid_attr)

                # render command token
                if type is TOKEN_TYPE.COMMAND:
                    for index in range(len(rendered_buffer))[slic]:
                        rendered_buffer[index] = tui.add_attr(rendered_buffer[index], token_command_attr)

                # render function token
                elif type is TOKEN_TYPE.FUNCTION:
                    for index in range(len(rendered_buffer))[slic]:
                        rendered_buffer[index] = tui.add_attr(rendered_buffer[index], token_function_attr)

                # render argument token
                elif type is TOKEN_TYPE.ARGUMENT:
                    for index in range(len(rendered_buffer))[slic]:
                        rendered_buffer[index] = tui.add_attr(rendered_buffer[index], token_argument_attr)

                # render literal token
                elif type is TOKEN_TYPE.LITERAL:
                    for index in range(len(rendered_buffer))[slic]:
                        rendered_buffer[index] = tui.add_attr(rendered_buffer[index], token_literal_attr)

            rendered_text = "".join(rendered_buffer)

            # render typeahead
            rendered_typeahead = tui.add_attr(self.input.typeahead, typeahead_attr) if self.input.typeahead else ""

            # compute cursor position
            _, cursor_pos = tui.textrange1(0, "".join(rendered_buffer[:self.input.pos]))

            yield rendered_text, rendered_typeahead, cursor_pos

    @dn.datanode
    def draw_node(self):
        header_width = self.theme.header_width
        header_ran = slice(None, header_width)
        input_ran = slice(header_width, None)

        error_message_attr = self.theme.error_message_attr

        input_offset = 0
        output_text = None

        while True:
            result, (header, cursor), (text, typeahead, cursor_pos), size = yield output_text
            width = size.columns
            view = tui.newwin1(width)

            # adjust input offset
            input_width = len(range(width)[input_ran])
            _, text_length = tui.textrange1(0, text)

            if cursor_pos - input_offset >= input_width:
                input_offset = cursor_pos - input_width + 1
            elif cursor_pos - input_offset < 0:
                input_offset = cursor_pos
            elif text_length - input_offset + 1 <= input_width:
                input_offset = max(0, text_length-input_width+1)

            # draw input
            text_ = text + (typeahead if not result else "")
            tui.addtext1(view, width, input_ran.start-input_offset, text_, input_ran)
            if input_offset > 0:
                tui.addtext1(view, width, input_ran.start, "…", input_ran)
            if text_length-input_offset >= input_width:
                tui.addtext1(view, width, input_ran.start+input_width-1, "…", input_ran)

            # draw header
            tui.addtext1(view, width, 0, header, header_ran)

            # draw cursor
            if not result and cursor:
                cursor_x = input_ran.start - input_offset + cursor_pos
                cursor_ran = tui.select1(view, width, slice(cursor_x, cursor_x+1))
                if hasattr(cursor, '__call__'):
                    view[cursor_ran.start] = cursor(view[cursor_ran.start])
                else:
                    tui.addtext1(view, width, cursor_ran.start, cursor, input_ran)

            # print error
            err_text = ""
            if isinstance(result, InputError):
                err_text = "\n"

                if result.message is not None:
                    err_text = "\n" + tui.add_attr(result.message, error_message_attr) + err_text

                if result.pointto is not None:
                    _, pointto_start = tui.textrange1(-input_offset, self.input.buffer[:result.pointto.start])
                    _, pointto_stop = tui.textrange1(-input_offset, self.input.buffer[:result.pointto.stop])
                    pointto_start = max(0, min(input_width-1, pointto_start))
                    pointto_stop = max(1, min(input_width, pointto_stop))

                    padding = " "*(input_ran.start+pointto_start)
                    pointto_text = "^"*(pointto_stop - pointto_start)
                    err_text = "\n" + padding + pointto_text + err_text

            output_text = "\r" + "".join(view) + "\r" + err_text

def prompt(promptable, history=None):
    theme = PromptTheme()

    input = BeatInput(promptable, history)
    stroke = BeatStroke(input, default_keymap, default_keycodes)
    prompt = BeatPrompt(stroke, input, theme)

    input_knot = dn.input(stroke.input_handler())
    display_knot = dn.show(prompt.output_handler(), 1/theme.framerate, hide_cursor=True)

    # `dn.show`, `dn.input` will fight each other...
    @dn.datanode
    def slow(dt=0.1):
        import time
        try:
            yield
            while True:
                yield
        finally:
            time.sleep(dt)

    prompt_knot = dn.pipe(input_knot, slow(), display_knot)
    dn.exhaust(prompt_knot, dt=0.01, interruptible=True)

    return prompt.result()
