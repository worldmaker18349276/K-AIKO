import os
from enum import Enum
import functools
import re
import queue
import threading
import inspect
from pathlib import Path
from typing import List, Tuple
from . import datanodes as dn
from . import tui
from . import cfg


def fit_ratio(target, full):
    if full == "" and target == "":
        return 1.0
    full = full.lower()
    index = 0
    n = 0.0
    try:
        for ch in target:
            index_ = full.index(ch.lower(), index)
            n += (1.0 if full[index_] == ch else 0.75) / (index_-index+1)
            index = index_ + 1

        return n/len(full)
    except ValueError:
        return 0.0

def fit(target, options):
    if target == "":
        return options
    weighted_options = [(fit_ratio(target, opt), opt) for opt in options]
    sorted_options = sorted(weighted_options, reverse=True)
    return [opt for weight, opt in sorted_options if weight != 0.0]

class SHLEXER_STATE(Enum):
    SPACED = " "
    PLAIN = "*"
    BACKSLASHED = "\\"
    QUOTED = "'"

def shlexer_tokenize(raw, partial=False):
    SPACE = " "
    BACKSLASH = "\\"
    QUOTE = "'"

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
        ignored = []
        while True:
            if char == SPACE:
                # end parsing token
                yield "".join(token), slice(start, index), ignored
                break

            elif char == BACKSLASH:
                # escape the next character
                try:
                    ignored.append(index)
                    index, char = next(raw)
                    token.append(char)

                except StopIteration:
                    if not partial:
                        raise ValueError("No escaped character")
                    yield "".join(token), slice(start, None), ignored
                    return SHLEXER_STATE.BACKSLASHED

            elif char == QUOTE:
                # escape the following characters until the next quotation mark
                try:
                    ignored.append(index)
                    index, char = next(raw)
                    while char != QUOTE:
                        token.append(char)
                        index, char = next(raw)
                    ignored.append(index)

                except StopIteration:
                    if not partial:
                        raise ValueError("No closing quotation")
                    yield "".join(token), slice(start, None), ignored
                    return SHLEXER_STATE.QUOTED

            else:
                # otherwise, as it is
                token.append(char)

            try:
                index, char = next(raw)
            except StopIteration:
                yield "".join(token), slice(start, None), ignored
                return SHLEXER_STATE.PLAIN

def shlexer_complete(compreply, partial=False, state=SHLEXER_STATE.SPACED):
    if state == SHLEXER_STATE.PLAIN:
        compreply = re.sub(r"([ \\'])", r"\\\1", compreply)

    elif state == SHLEXER_STATE.SPACED:
        if compreply == "" and not partial:
            # escape empty string
            compreply = "''"
        elif " " not in compreply:
            compreply = re.sub(r"([ \\'])", r"\\\1", compreply)
        else:
            # use quotation
            compreply = compreply.replace("'", r"'\''")
            if not partial:
                compreply = compreply[:-1] if compreply.endswith("'") else compreply + "'"
            compreply = "'" + compreply

    elif state == SHLEXER_STATE.BACKSLASHED:
        if compreply == "":
            return ""
        compreply = re.sub(r"([ \\'])", r"\\\1", compreply)
        # remove opening backslash
        if compreply.startswith("\\"):
            compreply = compreply[1:]

    elif state == SHLEXER_STATE.QUOTED:
        compreply = compreply.replace("'", r"'\''")
        # add closing quotation
        if not partial:
            compreply = compreply[:-1] if compreply.endswith("'") else compreply + "'"

    else:
        raise ValueError

    return compreply if partial else compreply + " "

def echo_str(escaped_str):
    regex = r"\\c.*|\\[\\abefnrtv]|\\0[0-7]{0,3}|\\x[0-9a-fA-F]{1,2}|."

    escaped = {
        r"\\": "\\",
        r"\a": "\a",
        r"\b": "\b",
        r"\e": "\x1b",
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


class TokenUnfinishError(Exception):
    def __init__(self, index, info):
        self.index = index
        self.info = info

class TokenParseError(Exception):
    def __init__(self, index, info):
        self.index = index
        self.info = info

class TOKEN_TYPE(Enum):
    COMMAND = "command"
    ARGUMENT = "argument"
    LITERAL = "literal"


def function_command(proxy=None, **annotations):
    if proxy is None:
        return functools.partial(function_command, **annotations)
    return FunctionCommandDescriptor(proxy, annotations)

def subcommand(proxy):
    return SubCommandDescriptor(proxy)

class CommandDescriptor:
    def __init__(self, proxy):
        self.proxy = proxy

    def __get__(self, instance, owner):
        return self.proxy.__get__(instance, owner)

    def __get_command__(self, instance, owner):
        raise NotImplementedError

class FunctionCommandDescriptor(CommandDescriptor):
    def __init__(self, proxy, annotations):
        super(FunctionCommandDescriptor, self).__init__(proxy)
        self.annotations = annotations

    def __get_command__(self, instance, owner):
        annotations = {key: value.__get__(instance, owner) for key, value in self.annotations.items()}
        return FunctionCommand(self.proxy.__get__(instance, owner), annotations)

class SubCommandDescriptor(CommandDescriptor):
    def __get_command__(self, instance, owner):
        return SubCommand(self.proxy.__get__(instance, owner))

class Command:
    @staticmethod
    def parse_path(token):
        try:
            exists = os.path.exists(token or ".")
        except ValueError:
            exists = False

        if not exists:
            return None

        return Path(token)

    @staticmethod
    def suggest_path(token, default=None):
        suggestions = []

        if default is not None:
            suggestions.append((str(default), False))

        # check path
        try:
            is_dir = os.path.isdir(token or ".")
            is_file = os.path.isfile(token or ".")
        except ValueError:
            return suggestions

        if is_file:
            suggestions.append((token, False))
            return suggestions

        # separate parent and partial name
        if is_dir:
            suggestions.append((os.path.join(token or ".", ""), True))
            target = ""
        else:
            token, target = os.path.split(token)

        # explore directory
        if os.path.isdir(token or "."):
            names = fit(target, [name for name in os.listdir(token or ".") if not name.startswith(".")])
            for name in names:
                subpath = os.path.join(token, name)

                if os.path.isdir(subpath):
                    subpath = os.path.join(subpath, "")
                    suggestions.append((subpath, True))

                elif os.path.isfile(subpath):
                    suggestions.append((subpath, False))

        return suggestions

    @staticmethod
    def parse_lit(token, ann):
        if isinstance(ann, (tuple, list)):
            if token not in ann:
                return None
            return token

        elif ann == bool:
            if not re.fullmatch("True|False", token):
                return None
            return bool(token)

        elif ann == int:
            if not re.fullmatch(r"[-+]?(0|[1-9][0-9]*)", token):
                return None
            return int(token)

        elif ann == float:
            if not re.fullmatch(r"[-+]?([0-9]+\.[0-9]+(e[-+]?[0-9]+)?|[0-9]+e[-+]?[0-9]+)", token):
                return None
            return float(token)

        elif ann == str:
            return token

        elif ann == Path:
            return Command.parse_path(token)

        elif hasattr(ann, 'parse'):
            return ann.parse(token)

        else:
            return None

    @staticmethod
    def suggest_lit(token, ann, default):
        if isinstance(ann, (tuple, list)):
            return [(val, False) for val in fit(token, ann)]

        elif ann == bool:
            if default is inspect.Parameter.empty or default == True:
                return [(val, False) for val in fit(token, ["True", "False"])]
            else:
                return [(val, False) for val in fit(token, ["False", "True"])]

        elif ann == int:
            if default is inspect.Parameter.empty:
                return []
            else:
                return [(val, False) for val in fit(token, [str(default)])]

        elif ann == float:
            if default is inspect.Parameter.empty:
                return []
            else:
                return [(val, False) for val in fit(token, [str(default)])]

        elif ann == str:
            if default is inspect.Parameter.empty:
                return []
            else:
                return [(val, False) for val in fit(token, [default])]

        elif ann == Path:
            if default is inspect.Parameter.empty:
                return Command.suggest_path(token)
            else:
                return Command.suggest_path(token, default)

        elif hasattr(ann, 'suggest'):
            if default is inspect.Parameter.empty:
                return ann.suggest(token)
            else:
                return ann.suggest(token, default)

        else:
            return []

    @staticmethod
    def help_lit(token, ann):
        if isinstance(ann, (tuple, list)):
            return "It should be one of:\n" + "\n".join("  " + shlexer_complete(s) for s in ann)

        elif isinstance(ann, type):
            return f"It should be {ann.__name__} literal"

        elif hasattr(ann, 'help'):
            return ann.help(token)

        else:
            return None

    def parse_command(self, tokens):
        tokens = iter(tokens)
        types = []

        try:
            while True:
                types.append(None)
                token = next(tokens)

        except StopIteration:
            pass

        return types

    def build_command(self, tokens, index=0):
        raise TokenParseError(index, "Too many arguments")

    def suggest_command(self, tokens, target):
        return []

    def help_command(self, tokens, target):
        return None

class FunctionCommand(Command):
    def __init__(self, func, annotations={}):
        self.func = func
        self.annotations = dict(annotations)

    def get_promptable_signature(self):
        sig = inspect.signature(self.func)
        args = list()
        kwargs = dict()
        for param in sig.parameters.values():
            annotation = param.annotation
            if param.name in self.annotations:
                annotation = self.annotations[param.name]

            if param.default is inspect.Parameter.empty:
                args.append((annotation, param.default))
            else:
                kwargs["--" + param.name] = (annotation, param.default)

        return args, kwargs

    def parse_command(self, tokens):
        tokens = iter(tokens)
        types = [TOKEN_TYPE.COMMAND]
        args, kwargs = self.get_promptable_signature()

        try:
            token = next(tokens)

            # parse positional arguments
            for ann, default in args:
                value = self.parse_lit(token, ann)
                if value is None:
                    kwargs = {}
                    break
                types.append(TOKEN_TYPE.LITERAL)

                token = next(tokens)

            # parse keyword arguments
            while kwargs:
                # parse argument name
                ann, default = kwargs.pop(token, (None, None))
                if ann is None:
                    break
                types.append(TOKEN_TYPE.ARGUMENT)

                token = next(tokens)

                # parse argument value
                value = self.parse_lit(token, ann)
                if value is None:
                    break
                types.append(TOKEN_TYPE.LITERAL)

                token = next(tokens)

            # rest
            while True:
                types.append(None)
                token = next(tokens)

        except StopIteration:
            pass

        return types

    def build_command(self, tokens, index=0):
        tokens = iter(tokens)
        func = self.func
        args, kwargs = self.get_promptable_signature()

        token = next(tokens, None)

        # parse positional arguments
        for ann, default in args:
            if token is None:
                help = Command.help_lit(None, ann)
                msg = "Missing value" + ("\n" + help if help is not None else "")
                raise TokenUnfinishError(index, msg)

            value = self.parse_lit(token, ann)

            if value is None:
                help = Command.help_lit(token, ann)
                msg = "Invalid value" + ("\n" + help if help is not None else "")
                raise TokenParseError(index, msg)

            func = functools.partial(func, value)

            token = next(tokens, None)
            index += 1

        # parse keyword arguments
        while kwargs and token is not None:
            # parse argument name
            ann, default = kwargs.pop(token, (None, None))
            name = token[2:]

            if ann is None:
                help = Command.help_lit(None, list(kwargs.keys()))
                msg = f"Unkown argument {name!r}" + "\n" + help
                raise TokenParseError(index, msg)

            token = next(tokens, None)
            index += 1

            # parse argument value
            if token is None:
                help = Command.help_lit(None, ann)
                msg = "Missing value" + ("\n" + help if help is not None else "")
                raise TokenUnfinishError(index, msg)

            value = self.parse_lit(token, ann)

            if value is None:
                help = Command.help_lit(token, ann)
                msg = "Invalid value" + ("\n" + help if help is not None else "")
                raise TokenParseError(index, msg)

            func = functools.partial(func, **{name: value})

            token = next(tokens, None)
            index += 1

        # rest
        if token is not None:
            raise TokenParseError(index, "Too many arguments")

        return func

    def suggest_command(self, tokens, target):
        tokens = iter(tokens)
        args, kwargs = self.get_promptable_signature()

        token = next(tokens, None)

        # parse positional arguments
        for ann, default in args:
            if token is None:
                return self.suggest_lit(target, ann, default)
            value = self.parse_lit(token, ann)
            if value is None:
                return []

            token = next(tokens, None)

        # parse keyword arguments
        while kwargs:
            # parse argument name
            if token is None:
                return [(val, False) for val in fit(target, kwargs.keys())]
            ann, default = kwargs.pop(token, (None, None))
            if ann is None:
                return []

            token = next(tokens, None)

            # parse argument value
            if token is None:
                return self.suggest_lit(target, ann, default)
            value = self.parse_lit(token, ann)
            if value is None:
                return []

            token = next(tokens, None)

        # rest
        return []

    def help_command(self, tokens, target):
        tokens = iter(tokens)
        func = self.func
        args, kwargs = self.get_promptable_signature()

        token = next(tokens, None)

        # parse positional arguments
        for ann, default in args:
            if token is None:
                return Command.help_lit(target, ann)
            value = self.parse_lit(token, ann)
            if value is None:
                return None

            token = next(tokens, None)

        # parse keyword arguments
        while kwargs:
            # parse argument name
            if token is None:
                return Command.help_lit(target, list(kwargs.keys()))
            ann, default = kwargs.pop(token, (None, None))
            if ann is None:
                return None

            token = next(tokens, None)

            # parse argument value
            if token is None:
                return Command.help_lit(target, ann)
            value = self.parse_lit(token, ann)
            if value is None:
                return None

            token = next(tokens, None)

        # rest
        return None

class SubCommand(Command):
    def __init__(self, root):
        self.root = root

    def get_promptable_fields(self):
        return [k for k, v in type(self.root).__dict__.items() if isinstance(v, CommandDescriptor)]

    def get_promptable_field(self, name):
        desc = type(self.root).__dict__[name]
        return desc.__get_command__(self.root, type(self.root))

    def parse_command(self, tokens):
        tokens = iter(tokens)
        type = TOKEN_TYPE.COMMAND

        try:
            token = next(tokens)
        except StopIteration:
            return [type]

        if token not in self.get_promptable_fields():
            return [type, *super().parse_command(tokens)]

        field = self.get_promptable_field(token)
        if not isinstance(field, Command):
            return [type, *super().parse_command(tokens)]

        return [type, *field.parse_command(tokens)]

    def build_command(self, tokens, index=0):
        tokens = iter(tokens)
        token = next(tokens, None)

        fields = self.get_promptable_fields()
        if token is None:
            help = Command.help_lit(None, fields)
            msg = "Unfinished command" + ("\n" + help if help is not None else "")
            raise TokenUnfinishError(index, msg)
        if token not in fields:
            help = Command.help_lit(None, fields)
            msg = "Invalid command" + ("\n" + help if help is not None else "")
            raise TokenParseError(index, "Invalid command")

        field = self.get_promptable_field(token)
        if not isinstance(field, Command):
            raise TokenParseError(index, "Not a command")

        return field.build_command(tokens, index+1)

    def suggest_command(self, tokens, target):
        tokens = iter(tokens)
        token = next(tokens, None)

        fields = self.get_promptable_fields()

        if token is None:
            return [(val, False) for val in fit(target, fields)]
        if token not in fields:
            return []

        field = self.get_promptable_field(token)
        if not isinstance(field, Command):
            return []

        return field.suggest_command(tokens, target)

    def help_command(self, tokens, target):
        tokens = iter(tokens)
        token = next(tokens, None)

        fields = self.get_promptable_fields()

        if token is None:
            return Command.help_lit(target, fields)
        if token not in fields:
            return []

        field = self.get_promptable_field(token)
        if not isinstance(field, Command):
            return []

        return field.help_command(tokens, target)

class RootCommand(SubCommand):
    def get_promptable_fields(self):
        fields = super(RootCommand, self).get_promptable_fields()
        return ["help", *fields]

    def get_promptable_field(self, name):
        if name == "help":
            return HelpCommand(self.root)
        else:
            return super(RootCommand, self).get_promptable_field(name)

class HelpCommand(SubCommand):
    def build_command(self, tokens, index=0):
        tokens = list(tokens)
        if len(tokens) == 0:
            return lambda: "Sorry, can't help."

        super().build_command(tokens[:-1], index)
        help = self.help_command(tokens[:-1], tokens[-1])
        if help is None:
            return lambda: "Sorry, can't help."
        return lambda: help


class InputError:
    def __init__(self, pointto, message):
        self.pointto = pointto
        self.message = message

class InputResult:
    def __init__(self, value):
        self.value = value

class BeatInput:
    def __init__(self, command, history=None):
        self.command = command
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

    def parse_syntax(self):
        tokenizer = shlexer_tokenize(self.buffer, partial=True)

        tokens = []
        while True:
            try:
                token, slic, ignored = next(tokenizer)
            except StopIteration as e:
                self.state = e.value
                break

            tokens.append((token, slic, ignored))

        types = self.command.parse_command(token for token, _, _ in tokens)
        types = types[1:]
        self.tokens = [(token, type, slic, ignored) for (token, slic, ignored), type in zip(tokens, types)]

    def autocomplete(self, action=+1):
        self.typeahead = ""

        # find the token to autocomplete
        tokens = []
        selection = slice(self.pos, self.pos)
        target = ""
        for token, _, slic, _ in self.tokens:
            start, stop, _ = slic.indices(len(self.buffer))
            if stop < self.pos:
                tokens.append(token)
            elif self.pos < start:
                break
            else:
                selection = slic
                target = token

        # generate suggestions
        suggestions = [shlexer_complete(sugg, part) for sugg, part in self.command.suggest_command(tokens, target)]
        length = len(suggestions)

        original_buffer = list(self.buffer)
        original_pos = self.pos

        if action == +1:
            index = 0
        elif action == -1:
            index = length-1
        else:
            raise ValueError

        while index in range(length):
            self.buffer[selection] = suggestions[index]
            self.pos = selection.start + len(suggestions[index])
            self.parse_syntax()

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
            self.parse_syntax()

    def make_typeahead(self, suggest=True):
        if not suggest or self.pos != len(self.buffer):
            self.typeahead = ""
            return False

        if self.state == SHLEXER_STATE.SPACED:
            tokens = [token for token, _, _, _ in self.tokens]
            target = ""
        else:
            tokens = [token for token, _, _, _ in self.tokens[:-1]]
            target, _, _, _ = self.tokens[-1]

        compreply, partial = None, False
        for suggestion, partial_ in self.command.suggest_command(tokens, target):
            if suggestion.startswith(target):
                compreply, partial = suggestion[len(target):], partial_
                break

        if compreply is None:
            self.typeahead = ""
            return False
        else:
            self.typeahead = shlexer_complete(compreply, partial, self.state)
            return True

    def insert_typeahead(self):
        if self.typeahead == "" or self.pos != len(self.buffer):
            return False

        self.buffer[self.pos:self.pos] = self.typeahead
        self.pos += len(self.typeahead)
        self.typeahead = ""
        self.parse_syntax()

        return True

    def enter(self):
        if len(self.tokens) == 0:
            self.delete_range(None, None)
            self.result.put(InputError(None, None))
            return False

        try:
            if self.state == SHLEXER_STATE.BACKSLASHED:
                raise TokenParseError(len(self.tokens)-1, "No escaped character")
            if self.state == SHLEXER_STATE.QUOTED:
                raise TokenParseError(len(self.tokens)-1, "No closing quotation")
            res = self.command.build_command([token for token, _, _, _ in self.tokens])

        except TokenUnfinishError as e:
            if e.index < len(self.tokens):
                _, _, pointto, _ = self.tokens[e.index]
            else:
                pointto = None
            self.result.put(InputError(pointto, e.info))
            return False

        except TokenParseError as e:
            _, _, pointto, _ = self.tokens[e.index]
            self.result.put(InputError(pointto, e.info))
            return False

        else:
            self.history.append("".join(self.buffer))
            self.result.put(InputResult(res))
            return True

    def cancel(self):
        return False

    def input(self, text):
        text = list(text)

        if len(text) == 0:
            return False

        while len(text) > 0 and text[0] == "\b":
            del text[0]
            del self.buffer[self.pos-1]
            self.pos = self.pos-1

        self.buffer[self.pos:self.pos] = text
        self.pos += len(text)
        self.parse_syntax()

        self.make_typeahead(True)

        return True

    def error(self, message):
        self.result.put(InputError(None, message))
        return True

    def backspace(self):
        if self.pos == 0:
            return False

        self.pos -= 1
        del self.buffer[self.pos]
        self.parse_syntax()
        self.make_typeahead(False)

        return True

    def delete(self):
        if self.pos >= len(self.buffer):
            return False

        del self.buffer[self.pos]
        self.parse_syntax()
        self.make_typeahead(False)

        return True

    def delete_range(self, start, end):
        start = min(max(0, start), len(self.buffer)) if start is not None else 0
        end = min(max(0, end), len(self.buffer)) if end is not None else len(self.buffer)

        if start >= end:
            return False

        del self.buffer[start:end]
        self.pos = start
        self.parse_syntax()
        self.make_typeahead(False)

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
        self.make_typeahead(False)

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
        self.parse_syntax()
        self.make_typeahead(False)

        return True

    def next(self):
        if self.buffers_index == -1:
            return False
        self.buffers_index += 1

        self.buffer = self.buffers[self.buffers_index]
        self.pos = len(self.buffer)
        self.parse_syntax()
        self.make_typeahead(False)

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
        "Right": lambda input: input.insert_typeahead() or input.move_right(),
           "Up": lambda input: input.prev(),
         "Down": lambda input: input.next(),
         "Home": lambda input: input.move_to_start(),
          "End": lambda input: input.move_to_end(),
        "Enter": lambda input: input.enter(),
          "Esc": lambda input: input.cancel(),
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
                    selector = self.input.autocomplete(+1)
                elif key == self.keycodes["Shift+Tab"]:
                    selector = self.input.autocomplete(-1)

                try:
                    next(selector)

                    while True:
                        _, key = yield
                        if key == self.keycodes["Tab"]:
                            selector.send(+1)
                        elif key == self.keycodes["Shift+Tab"]:
                            selector.send(-1)
                        elif key == self.keycodes["Esc"]:
                            selector.send(0)
                        else:
                            selector.close()
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
                else:
                    self.input.error(f"Unknown key: {key!r}")


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
    highlight_attr: str = "4"
    whitespace: str = "\x1b[2m⌴\x1b[m"

    token_invalid_attr: str = "31"
    token_command_attr: str = "94"
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
            if not self.input.result.empty():
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
        token_argument_attr = self.theme.token_argument_attr
        token_literal_attr  = self.theme.token_literal_attr

        yield
        while True:
            # render buffer
            rendered_buffer = list(self.input.buffer)

            for token, type, slic, ignored in self.input.tokens:
                # render escape
                for index in ignored:
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
        highlight_attr = self.theme.highlight_attr

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
                    trim_lines = 8
                    msg = "\n".join(result.message.split("\n")[:trim_lines])
                    if result.message.count("\n") >= trim_lines:
                        msg += "\n…"

                    err_text = "\n" + tui.add_attr(msg, error_message_attr) + err_text

                # highlight
                if result.pointto is not None:
                    _, pointto_start = tui.textrange1(-input_offset, self.input.buffer[:result.pointto.start])
                    _, pointto_stop = tui.textrange1(-input_offset, self.input.buffer[:result.pointto.stop])
                    pointto_start = max(0, min(input_width-1, pointto_start))
                    pointto_stop = max(1, min(input_width, pointto_stop))

                    for index in range(input_ran.start+pointto_start, input_ran.start+pointto_stop):
                        if view[index]:
                            view[index] = tui.add_attr(view[index], highlight_attr)

            output_text = "\r" + "".join(view) + "\r" + err_text

def prompt(promptable, history=None):
    theme = PromptTheme()

    command = RootCommand(promptable)
    input = BeatInput(command, history)
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
