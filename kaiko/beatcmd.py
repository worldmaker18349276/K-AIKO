import os
from enum import Enum
from collections import OrderedDict
import functools
import itertools
import re
import queue
import threading
import inspect
from pathlib import Path
from typing import List, Tuple, Dict, Callable
from . import datanodes as dn
from . import tui
from . import cfg


def outdent(doc):
    if doc is None:
        return None
    m = re.search("\\n[ ]*$", doc)
    level = len(m.group(0)[1:]) if m else 0
    return re.sub("\\n[ ]{,%d}"%level, "\\n", doc)

def masks_key(masks):
    return tuple(slic.stop-slic.start for slic in masks), (-masks[-1].stop if masks else 0)

def fit_masks(part, full):
    def substr_treeiter(part, full, start=0):
        if part == "":
            return
        head = part[0]
        tail = part[1:]
        for i in range(start, len(full)):
            if full[i] == head:
                yield i, substr_treeiter(tail, full, i+1)

    def subsec_treeiter(treeiter, index=0, sec=(slice(0,0),)):
        for i, nextiter in treeiter:
            if i == index:
                sec_ = (*sec[:-1], slice(sec[-1].start, i+1))
            else:
                sec_ = (*sec, slice(i, i+1))
            yield sec_, subsec_treeiter(nextiter, i+1, sec_)

    def sorted_subsec_iter(treeiters, depth):
        if depth == 0:
            yield from sorted([sec for sec, _ in treeiters], key=masks_key)
            return
        treeiters_ = [item for _, nextiters in treeiters for item in nextiters]
        treeiters_ = sorted(treeiters_, key=lambda e:masks_key(e[0]), reverse=True)
        for _, nextiters in itertools.groupby(treeiters_, lambda e:masks_key(e[0])):
            yield from sorted_subsec_iter(nextiters, depth-1)

    return sorted_subsec_iter([((), subsec_treeiter(substr_treeiter(part, full)))], len(part))

def fit(part, options):
    if part == "":
        return options

    masked_options = [(next(fit_masks(part.lower(), opt.lower()), ()), opt) for opt in options]
    masked_options = [(m, opt) for m, opt in masked_options if m != ()]

    masked_options = sorted(masked_options, key=lambda e:(masks_key(e[0]), -len(e[1])), reverse=True)
    return [opt for m, opt in masked_options]

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

def shlexer_quoting(compreply, partial=False, state=SHLEXER_STATE.SPACED):
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
    r"""
    It interprets the following backslash-escaped characters like bash's echo:
        \a     alert (bell)
        \b     backspace
        \c     suppress further output
        \e     escape character
        \f     form feed
        \n     new line
        \r     carriage return
        \t     horizontal tab
        \v     vertical tab
        \\     backslash
        \0nnn  the character whose ASCII code is NNN (octal).  NNN can be 0 to 3 octal digits
        \xHH   the eight-bit character whose value is HH (hexadecimal).  HH can be one or two hex digits
    """
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
    pass

class TokenParseError(Exception):
    pass

class TOKEN_TYPE(Enum):
    COMMAND = "command"
    ARGUMENT = "argument"
    LITERAL = "literal"
    UNKNOWN = "unknown"


class LiteralParser:
    @staticmethod
    def wrap(type, default=inspect.Parameter.empty):
        if isinstance(type, (list, tuple)):
            return OptionParser(type, default)
        elif type == bool:
            return BoolParser(default)
        elif type == int:
            return IntParser(default)
        elif type == float:
            return FloatParser(default)
        elif type == str:
            return StrParser(default)
        elif type == Path:
            return PathParser(default)
        else:
            return None

    def parse(self, token):
        help = self.help(token)
        raise TokenParseError("Invalid value" + ("\n" + help if help is not None else ""))

    def suggest(self, token):
        return []

    def help(self, token):
        return None

class OptionParser(LiteralParser):
    def __init__(self, options, default=inspect.Parameter.empty, docs=None):
        self.options = options
        self.default = default
        self.docs = docs

    def parse(self, token):
        if token not in self.options:
            help = self.help(token)
            raise TokenParseError("Invalid value" + ("\n" + help if help is not None else ""))
        return token

    def suggest(self, token):
        return [(val, False) for val in fit(token, self.options)]

    def help(self, token):
        if self.docs:
            return self.docs
        return "It should be one of:\n" + "\n".join("  " + shlexer_quoting(s) for s in self.options)

class BoolParser(LiteralParser):
    def __init__(self, default=inspect.Parameter.empty, docs=None):
        self.default = default
        self.docs = docs

    def parse(self, token):
        if not re.fullmatch("True|False", token):
            help = self.help(token)
            raise TokenParseError("Invalid value" + ("\n" + help if help is not None else ""))
        return token == "True"

    def suggest(self, token):
        if self.default is inspect.Parameter.empty or self.default == True:
            return [(val, False) for val in fit(token, ["True", "False"])]
        else:
            return [(val, False) for val in fit(token, ["False", "True"])]

    def help(self, token):
        if self.docs:
            return self.docs
        return "It should be bool literal"

class IntParser(LiteralParser):
    def __init__(self, default=inspect.Parameter.empty, docs=None):
        self.default = default
        self.docs = docs

    def parse(self, token):
        if not re.fullmatch(r"[-+]?(0|[1-9][0-9]*)", token):
            help = self.help(token)
            raise TokenParseError("Invalid value" + ("\n" + help if help is not None else ""))
        return int(token)

    def suggest(self, token):
        if self.default is inspect.Parameter.empty:
            return []
        else:
            return [(val, False) for val in fit(token, [str(self.default)])]

    def help(self, token):
        if self.docs:
            return self.docs
        return "It should be int literal"

class FloatParser(LiteralParser):
    def __init__(self, default=inspect.Parameter.empty, docs=None):
        self.default = default
        self.docs = docs

    def parse(self, token):
        if not re.fullmatch(r"[-+]?([0-9]+\.[0-9]+(e[-+]?[0-9]+)?|[0-9]+e[-+]?[0-9]+)", token):
            help = self.help(token)
            raise TokenParseError("Invalid value" + ("\n" + help if help is not None else ""))
        return float(token)

    def suggest(self, token):
        if self.default is inspect.Parameter.empty:
            return []
        else:
            return [(val, False) for val in fit(token, [str(self.default)])]

    def help(self, token):
        if self.docs:
            return self.docs
        return "It should be float literal"

class StrParser(LiteralParser):
    def __init__(self, default=inspect.Parameter.empty, docs=None):
        self.default = default
        self.docs = docs

    def parse(self, token):
        return token

    def suggest(self, token):
        if self.default is inspect.Parameter.empty:
            return []
        else:
            return [(val, False) for val in fit(token, [self.default])]

    def help(self, token):
        if self.docs:
            return self.docs
        return "It should be str literal"

class PathParser(LiteralParser):
    def __init__(self, default=inspect.Parameter.empty, docs=None):
        self.default = default
        self.docs = docs

    def parse(self, token):
        try:
            exists = os.path.exists(token or ".")
        except ValueError:
            exists = False

        if not exists:
            help = self.help(token)
            raise TokenParseError("Path does not exist" + ("\n" + help if help is not None else ""))

        return Path(token)

    def suggest(self, token):
        suggestions = []

        if self.default is not inspect.Parameter.empty:
            suggestions.append((str(self.default), False))

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

    def help(self, token):
        if self.docs:
            return self.docs
        return "It should be Path literal"


class CommandDescriptor:
    def __init__(self, proxy):
        self.proxy = proxy

    def __get__(self, instance, owner):
        return self.proxy.__get__(instance, owner)

    def __get_command__(self, instance, owner):
        raise NotImplementedError

class function_command(CommandDescriptor):
    def __init__(self, proxy):
        super(function_command, self).__init__(proxy)
        self.parsers = {}

    def __get_command__(self, instance, owner):
        func = self.proxy.__get__(instance, owner)
        parsers = {key: value.__get__(instance, owner) for key, value in self.parsers.items()}

        sig = inspect.signature(func)
        args = OrderedDict()
        kwargs = OrderedDict()
        for param in sig.parameters.values():
            arg = parsers.get(param.name, LiteralParser.wrap(param.annotation, param.default))

            if param.default is inspect.Parameter.empty:
                args[param.name] = arg
            else:
                kwargs[param.name] = arg

        return FunctionCommand(func, args, kwargs)

    def arg_parser(self, name):
        def arg_parser_dec(parser):
            self.parsers[name] = parser
            return parser
        return arg_parser_dec

class subcommand(CommandDescriptor):
    def __get_command__(self, instance, owner):
        parent = self.proxy.__get__(instance, owner)
        fields = [k for k, v in type(parent).__dict__.items() if isinstance(v, CommandDescriptor)]
        return SubCommand(parent, fields)


class Command:
    @staticmethod
    def help_option(token, options):
        return "It should be one of:\n" + "\n".join("  " + shlexer_quoting(s) for s in options)

    def finish(self):
        raise NotImplementedError

    def parse(self, token):
        raise NotImplementedError

    def suggest(self, token):
        raise NotImplementedError

    def help(self, token):
        raise NotImplementedError

class UnknownCommand(Command):
    def finish(self):
        raise TokenParseError("Unknown command")

    def parse(self, token):
        return TOKEN_TYPE.UNKNOWN, self

    def suggest(self, token):
        return []

    def help(self, token):
        return None

class FunctionCommand(Command):
    def __init__(self, func, args, kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def finish(self):
        if self.args:
            parser = next(iter(self.args.values()))
            help = parser.help(None)
            msg = "Missing value" + ("\n" + help if help is not None else "")
            raise TokenUnfinishError(msg)

        return self.func

    def parse(self, token):
        # parse positional arguments
        if self.args:
            args = OrderedDict(self.args)
            name, parser = args.popitem(False)
            value = parser.parse(token)

            func = functools.partial(self.func, **{name: value})
            return TOKEN_TYPE.LITERAL, FunctionCommand(func, args, self.kwargs)

        # parse keyword arguments
        if self.kwargs:
            kwargs = OrderedDict(self.kwargs)
            name = token[2:] if token.startswith("--") else None
            arg = kwargs.pop(name, None)

            if arg is None:
                help = Command.help_option(None, ["--" + key for key in self.kwargs.keys()])
                msg = f"Unknown argument {token!r}" + "\n" + help
                raise TokenParseError(msg)

            args = OrderedDict([(name, arg)])
            return TOKEN_TYPE.ARGUMENT, FunctionCommand(self.func, args, kwargs)

        # rest
        raise TokenParseError("Too many arguments")

    def suggest(self, token):
        # parse positional arguments
        if self.args:
            parser = next(iter(self.args.values()))
            return parser.suggest(token)

        # parse keyword arguments
        if self.kwargs:
            keys = ["--" + key for key in self.kwargs.keys()]
            return [(key, False) for key in fit(token, keys)]

        # rest
        return []

    def help(self, token):
        # parse positional arguments
        if self.args:
            parser = next(iter(self.args.values()))
            return parser.help(token)

        # parse keyword arguments
        if self.kwargs:
            keys = ["--" + key for key in self.kwargs.keys()]
            return Command.help_option(token, keys)

        # rest
        return None

class SubCommand(Command):
    def __init__(self, parent, fields):
        self.parent = parent
        self.fields = fields

    def get_promptable_field(self, name):
        desc = type(self.parent).__dict__[name]
        return desc.__get_command__(self.parent, type(self.parent))

    def finish(self):
        help = self.help(None)
        raise TokenUnfinishError("Unfinished command" + ("\n" + help if help is not None else ""))

    def parse(self, token):
        if token not in self.fields:
            help = self.help(token)
            msg = "Unknown command" + ("\n" + help if help is not None else "")
            raise TokenParseError(msg)

        field = self.get_promptable_field(token)
        if not isinstance(field, Command):
            raise TokenParseError("Not a command")

        return TOKEN_TYPE.COMMAND, field

    def suggest(self, token):
        return [(val, False) for val in fit(token, self.fields)]

    def help(self, token):
        if token in self.fields:
            desc = type(self.parent).__dict__[token]
            return outdent(desc.proxy.__doc__)

        return Command.help_option(token, self.fields)

class RootCommand(SubCommand):
    def __init__(self, root):
        fields = [k for k, v in type(root).__dict__.items() if isinstance(v, CommandDescriptor)]
        super(RootCommand, self).__init__(root, fields)

    def build(self, tokens):
        _, res, _ = self.parse_command(tokens)

        if isinstance(res, (TokenUnfinishError, TokenParseError)):
            raise res
        else:
            return res

    def parse_command(self, tokens):
        cmd = self
        types = []
        res = None
        index = 0

        for i, token in enumerate(tokens):
            try:
                type, cmd = cmd.parse(token)
            except TokenParseError as err:
                res, index = err, i
                type, cmd = TOKEN_TYPE.UNKNOWN, UnknownCommand()
            types.append(type)

        if res is not None:
            return (types, res, index)

        index = len(types)
        try:
            res = cmd.finish()
        except TokenUnfinishError as err:
            res = err

        return (types, res, index)

    def suggest_command(self, tokens, target):
        cmd = self
        for token in tokens:
            try:
                _, cmd = cmd.parse(token)
            except TokenParseError as err:
                cmd = UnknownCommand()
        return cmd.suggest(target)

    def help_command(self, tokens, target):
        if not tokens and target is None:
            return outdent(self.parent.__doc__)

        cmd = self
        for token in tokens:
            try:
                _, cmd = cmd.parse(token)
            except TokenParseError as err:
                cmd = UnknownCommand()
        return cmd.help(target)


class InputError:
    def __init__(self, index, message):
        self.index = index
        self.message = message

class InputMessage:
    def __init__(self, index, message):
        self.index = index
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

        self.resulted_tokens = None
        self.result = None

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

        types, _, _ = self.command.parse_command(token for token, _, _ in tokens)
        self.tokens = [(token, type, slic, ignored) for (token, slic, ignored), type in zip(tokens, types)]

    def autocomplete(self, action=+1):
        self.typeahead = ""

        # find the token to autocomplete
        tokens = []
        selection = slice(self.pos, self.pos)
        selection_index = len(self.tokens)
        target = ""
        for i, (token, _, slic, _) in enumerate(self.tokens):
            start, stop, _ = slic.indices(len(self.buffer))
            if stop < self.pos:
                tokens.append(token)
            elif self.pos < start:
                break
            else:
                selection = slic
                selection_index = i
                target = token

        # generate suggestions
        suggestions = [shlexer_quoting(sugg, part) for sugg, part in self.command.suggest_command(tokens, target)]
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
            self.help(selection_index)

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
            self.cancel_message()

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
            self.typeahead = shlexer_quoting(compreply, partial, self.state)
            return True

    def insert_typeahead(self):
        if self.typeahead == "" or self.pos != len(self.buffer):
            return False

        self.buffer[self.pos:self.pos] = self.typeahead
        self.pos += len(self.typeahead)
        self.typeahead = ""
        self.parse_syntax()

        return True

    def index(self):
        for index, (_, _, slic, _) in reversed(list(enumerate(self.tokens))):
            if slic.start <= self.pos:
                return index
        return None

    def error(self, message, index=None):
        if index is not None:
            self.resulted_tokens = self.tokens[:index+1]
            self.result = InputError(index, message)
        else:
            self.resulted_tokens = self.tokens[:]
            self.result = InputError(None, message)
        return True

    def message(self, message, index=None):
        if index is not None:
            self.resulted_tokens = self.tokens[:index+1]
            self.result = InputMessage(index, message)
        else:
            self.resulted_tokens = self.tokens[:]
            self.result = InputMessage(None, message)
        return True

    def cancel_message(self):
        self.resulted_tokens = None
        self.result = None
        return True

    def update_message(self):
        if self.resulted_tokens is None:
            return False

        if len(self.resulted_tokens) > len(self.tokens):
            self.resulted_tokens = None
            self.result = None
            return True

        for t1, t2 in zip(self.resulted_tokens, self.tokens):
            if t1[0] != t2[0]:
                self.resulted_tokens = None
                self.result = None
                return True

        return False

    def help(self, index=None):
        self.cancel_message()

        if index is None:
            for index, (_, _, slic, _) in enumerate(self.tokens):
                if slic.stop is None or self.pos <= slic.stop:
                    break
            else:
                index = None

        prefix = [token for token, _, _, _ in self.tokens[:index]]
        target, _, _, _ = self.tokens[index] if index is not None else (None,None,None,None)
        msg = self.command.help_command(prefix, target)
        if msg is None:
            return False
        self.message(msg, index)
        return True

    def enter(self):
        if len(self.tokens) == 0:
            self.delete_range(None, None)
            self.error(None)
            return False

        if self.state == SHLEXER_STATE.BACKSLASHED:
            res, index = TokenParseError("No escaped character"), len(self.tokens)-1
        elif self.state == SHLEXER_STATE.QUOTED:
            res, index = TokenParseError("No closing quotation"), len(self.tokens)-1
        else:
            _, res, index = self.command.parse_command(token for token, _, _, _ in self.tokens)

        if isinstance(res, TokenUnfinishError):
            self.error(res.args[0])
            return False
        elif isinstance(res, TokenParseError):
            self.error(res.args[0], index)
            return False
        else:
            self.history.append("".join(self.buffer))
            self.result = InputResult(res)
            return True

    def cancel(self):
        self.typeahead = ""
        self.cancel_message()
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
        self.update_message()

        return True

    def backspace(self):
        if self.pos == 0:
            return False

        self.pos -= 1
        del self.buffer[self.pos]
        self.parse_syntax()
        self.make_typeahead(False)
        self.update_message()

        return True

    def delete(self):
        if self.pos >= len(self.buffer):
            return False

        del self.buffer[self.pos]
        self.parse_syntax()
        self.make_typeahead(False)
        self.update_message()

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
        self.update_message()

        return True

    def delete_to_word_start(self):
        for match in re.finditer("\w+|\W+", "".join(self.buffer)):
            if match.end() >= self.pos:
                return self.delete_range(match.start(), self.pos)
        else:
            return self.delete_range(None, self.pos)

    def delete_to_word_end(self):
        for match in re.finditer("\w+|\W+", "".join(self.buffer)):
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
        for match in re.finditer("\w+|\W+", "".join(self.buffer)):
            if match.end() >= self.pos:
                return self.move_to(match.start())
        else:
            return self.move_to(0)

    def move_to_word_end(self):
        for match in re.finditer("\w+|\W+", "".join(self.buffer)):
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
        self.cancel_message()

        return True

    def next(self):
        if self.buffers_index == -1:
            return False
        self.buffers_index += 1

        self.buffer = self.buffers[self.buffers_index]
        self.pos = len(self.buffer)
        self.parse_syntax()
        self.make_typeahead(False)
        self.cancel_message()

        return True


class INPUT_STATE(Enum):
    EDIT = "edit"
    TAB = "tab"

class BeatStroke:
    def __init__(self, input, keymap, keycodes):
        self.input = input
        self.event = threading.Event()
        self.queue = queue.Queue()
        self.keymap = keymap
        self.keycodes = keycodes
        self.state = INPUT_STATE.EDIT

    @dn.datanode
    def input_handler(self):
        while True:
            time, key = yield
            self.queue.put((time, key))
            self.event.set()

    @dn.datanode
    def stroke_handler(self):
        while True:
            _, key = yield

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


class BeatPromptSettings(metaclass=cfg.Configurable):
    framerate: float = 60.0
    t0: float = 0.0
    tempo: float = 130.0

    keycodes: Dict[str, str] = {
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
    keymap: Dict[str, Callable] = {
        "Backspace"     : lambda input: input.backspace(),
        "Delete"        : lambda input: input.delete(),
        "Left"          : lambda input: input.move_left(),
        "Right"         : lambda input: input.insert_typeahead() or input.move_right(),
        "Up"            : lambda input: input.prev(),
        "Down"          : lambda input: input.next(),
        "Home"          : lambda input: input.move_to_start(),
        "End"           : lambda input: input.move_to_end(),
        "Enter"         : lambda input: input.enter(),
        "Esc"           : lambda input: input.cancel(),
        "Alt+Enter"     : lambda input: input.help(),
        "Ctrl+Left"     : lambda input: input.move_to_word_start(),
        "Ctrl+Right"    : lambda input: input.move_to_word_end(),
        "Ctrl+Backspace": lambda input: input.delete_to_word_start(),
        "Ctrl+Delete"   : lambda input: input.delete_to_word_end(),
    }

    headers: List[str] = [
        "\x1b[96;1m⠶⠦⣚⠀⠶\x1b[m\x1b[38;5;255m❯ \x1b[m",
        "\x1b[96;1m⢎⣀⡛⠀⠶\x1b[m\x1b[38;5;255m❯ \x1b[m",
        "\x1b[36m⢖⣄⠻⠀⠶\x1b[m\x1b[38;5;254m❯ \x1b[m",
        "\x1b[36m⠖⠐⡩⠂⠶\x1b[m\x1b[38;5;254m❯ \x1b[m",
        "\x1b[96m⠶⠀⡭⠲⠶\x1b[m\x1b[38;5;253m❯ \x1b[m",
        "\x1b[36m⠶⠀⣬⠉⡱\x1b[m\x1b[38;5;253m❯ \x1b[m",
        "\x1b[36m⠶⠀⣦⠙⠵\x1b[m\x1b[38;5;252m❯ \x1b[m",
        "\x1b[36m⠶⠠⣊⠄⠴\x1b[m\x1b[38;5;252m❯ \x1b[m",

        "\x1b[96m⠶⠦⣚⠀⠶\x1b[m\x1b[38;5;251m❯ \x1b[m",
        "\x1b[36m⢎⣀⡛⠀⠶\x1b[m\x1b[38;5;251m❯ \x1b[m",
        "\x1b[36m⢖⣄⠻⠀⠶\x1b[m\x1b[38;5;250m❯ \x1b[m",
        "\x1b[36m⠖⠐⡩⠂⠶\x1b[m\x1b[38;5;250m❯ \x1b[m",
        "\x1b[96m⠶⠀⡭⠲⠶\x1b[m\x1b[38;5;249m❯ \x1b[m",
        "\x1b[36m⠶⠀⣬⠉⡱\x1b[m\x1b[38;5;249m❯ \x1b[m",
        "\x1b[36m⠶⠀⣦⠙⠵\x1b[m\x1b[38;5;248m❯ \x1b[m",
        "\x1b[36m⠶⠠⣊⠄⠴\x1b[m\x1b[38;5;248m❯ \x1b[m",

        "\x1b[96m⠶⠦⣚⠀⠶\x1b[m\x1b[38;5;247m❯ \x1b[m",
        "\x1b[36m⢎⣀⡛⠀⠶\x1b[m\x1b[38;5;247m❯ \x1b[m",
        "\x1b[36m⢖⣄⠻⠀⠶\x1b[m\x1b[38;5;246m❯ \x1b[m",
        "\x1b[36m⠖⠐⡩⠂⠶\x1b[m\x1b[38;5;246m❯ \x1b[m",
        "\x1b[96m⠶⠀⡭⠲⠶\x1b[m\x1b[38;5;245m❯ \x1b[m",
        "\x1b[36m⠶⠀⣬⠉⡱\x1b[m\x1b[38;5;245m❯ \x1b[m",
        "\x1b[36m⠶⠀⣦⠙⠵\x1b[m\x1b[38;5;244m❯ \x1b[m",
        "\x1b[36m⠶⠠⣊⠄⠴\x1b[m\x1b[38;5;244m❯ \x1b[m",

        "\x1b[96m⠶⠦⣚⠀⠶\x1b[m\x1b[38;5;243m❯ \x1b[m",
        "\x1b[36m⢎⣀⡛⠀⠶\x1b[m\x1b[38;5;243m❯ \x1b[m",
        "\x1b[36m⢖⣄⠻⠀⠶\x1b[m\x1b[38;5;242m❯ \x1b[m",
        "\x1b[36m⠖⠐⡩⠂⠶\x1b[m\x1b[38;5;242m❯ \x1b[m",
        "\x1b[96m⠶⠀⡭⠲⠶\x1b[m\x1b[38;5;241m❯ \x1b[m",
        "\x1b[36m⠶⠀⣬⠉⡱\x1b[m\x1b[38;5;241m❯ \x1b[m",
        "\x1b[36m⠶⠀⣦⠙⠵\x1b[m\x1b[38;5;240m❯ \x1b[m",
        "\x1b[36m⠶⠠⣊⠄⠴\x1b[m\x1b[38;5;240m❯ \x1b[m",
    ]
    header_width: int = 7

    cursor_attr: Tuple[str, str] = ("7;2", "7;1")
    cursor_blink_ratio: float = 0.3

    error_message_attr: str = "31"
    info_message_attr: str = "2"
    message_max_lines: int = 16

    escape_attr: str = "2"
    typeahead_attr: str = "2"
    whitespace: str = "\x1b[2m⌴\x1b[m"

    token_unknown_attr: str = "31"
    token_command_attr: str = "94"
    token_argument_attr: str = "95"
    token_literal_attr: str = "92"
    token_highlight_attr: str = "4"

class BeatPrompt:
    def __init__(self, stroke, input, settings):
        self.stroke = stroke
        self.input = input
        self.settings = settings
        self.result = None

    @dn.datanode
    def output_handler(self):
        stroke_handler = self.stroke.stroke_handler()
        size_node = dn.terminal_size()
        header_node = self.header_node()
        text_node = self.text_node()
        message_node = self.message_node()
        render_node = self.render_node()
        with stroke_handler, size_node, header_node, text_node, message_node, render_node:
            yield
            while True:
                # deal with keystrokes
                while not self.stroke.queue.empty():
                    stroke_handler.send(self.stroke.queue.get())
                result = self.input.result

                size = size_node.send()

                # draw message
                msg_data, clean, highlighted = message_node.send(result)

                # draw header
                header_data = header_node.send(clean)

                # draw text
                text_data = text_node.send((clean, highlighted))

                # render
                output_text = render_node.send((result, header_data, text_data, msg_data, size))

                yield output_text

                # end
                if isinstance(result, InputResult):
                    self.result = result.value
                    return

    @dn.datanode
    def header_node(self):
        t0 = self.settings.t0
        tempo = self.settings.tempo
        framerate = self.settings.framerate

        headers = self.settings.headers

        cursor_attr = self.settings.cursor_attr
        cursor_blink_ratio = self.settings.cursor_blink_ratio

        clean = yield
        t = t0/(60/tempo)
        tr = 0
        while True:
            # don't blink while key pressing
            if self.stroke.event.is_set():
                self.stroke.event.clear()
                tr = t // -1 * -1

            # render cursor
            if not clean and (t-tr < 0 or (t-tr) % 1 < cursor_blink_ratio):
                if t % 4 < cursor_blink_ratio:
                    cursor = lambda s: tui.add_attr(s, cursor_attr[1])
                else:
                    cursor = lambda s: tui.add_attr(s, cursor_attr[0])
            else:
                cursor = None

            # render header
            ind = int(t / 4 * len(headers)) % len(headers)
            header = headers[ind]

            clean = yield header, cursor
            t += 1/framerate/(60/tempo)

    @dn.datanode
    def message_node(self):
        message_max_lines = self.settings.message_max_lines
        error_message_attr = self.settings.error_message_attr
        info_message_attr = self.settings.info_message_attr
        clear = False

        result = yield
        while True:
            clean = isinstance(result, (InputError, InputResult))

            if not isinstance(result, (InputError, InputMessage)):
                result = yield ("", clear, False), clean, None
                clear = False
                continue

            # render message
            msg = result.message or ""
            if msg.count("\n") >= message_max_lines:
                msg = "\n".join(msg.split("\n")[:message_max_lines]) + "\x1b[m\n…"
            if isinstance(result, InputError):
                msg = tui.add_attr(msg, error_message_attr)
            elif msg:
                msg = tui.add_attr(msg, info_message_attr)
            msg = "\n" + msg + ("\n" if msg else "")
            moveback = isinstance(result, InputMessage)

            # track changes of the result
            shown_result = result
            result = yield (msg, clear, moveback), clean, shown_result.index
            clear = False
            while shown_result == result:
                result = yield ("", False, False), False, shown_result.index

            if isinstance(shown_result, InputMessage):
                clear = True

    @dn.datanode
    def text_node(self):
        escape_attr     = self.settings.escape_attr
        typeahead_attr  = self.settings.typeahead_attr
        whitespace      = self.settings.whitespace

        token_unknown_attr  = self.settings.token_unknown_attr
        token_command_attr  = self.settings.token_command_attr
        token_argument_attr = self.settings.token_argument_attr
        token_literal_attr  = self.settings.token_literal_attr
        token_highlight_attr = self.settings.token_highlight_attr

        clean, highlighted = yield
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

                # render unknown token
                if type is TOKEN_TYPE.UNKNOWN:
                    if slic.stop is not None or clean:
                        for index in range(len(rendered_buffer))[slic]:
                            rendered_buffer[index] = tui.add_attr(rendered_buffer[index], token_unknown_attr)

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

            if highlighted in range(len(self.input.tokens)):
                # render highlighted token
                _, _, slic, _ = self.input.tokens[highlighted]
                for index in range(len(rendered_buffer))[slic]:
                    rendered_buffer[index] = tui.add_attr(rendered_buffer[index], token_highlight_attr)

            rendered_text = "".join(rendered_buffer)

            # render typeahead
            if self.input.typeahead and not clean:
                rendered_text += tui.add_attr(self.input.typeahead, typeahead_attr)

            # compute cursor position
            _, cursor_pos = tui.textrange1(0, "".join(rendered_buffer[:self.input.pos]))

            clean, highlighted = yield rendered_text, cursor_pos

    @dn.datanode
    def render_node(self):
        header_width = self.settings.header_width
        header_ran = slice(None, header_width)
        input_ran = slice(header_width, None)

        input_offset = 0
        output_text = None

        while True:
            result, (header, cursor), (text, cursor_pos), (msg, clear, moveback), size = yield output_text
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
            tui.addtext1(view, width, input_ran.start-input_offset, text, input_ran)
            if input_offset > 0:
                tui.addtext1(view, width, input_ran.start, "…", input_ran)
            if text_length-input_offset >= input_width:
                tui.addtext1(view, width, input_ran.start+input_width-1, "…", input_ran)

            # draw header
            tui.addtext1(view, width, 0, header, header_ran)

            # draw cursor
            if cursor:
                cursor_x = input_ran.start - input_offset + cursor_pos
                cursor_ran = tui.select1(view, width, slice(cursor_x, cursor_x+1))
                view[cursor_ran.start] = cursor(view[cursor_ran.start])

            # print error
            if moveback:
                _, y = tui.pmove(width, 0, msg)
                if y != 0:
                    msg = msg + f"\x1b[{y}A"
            if clear:
                msg = "\n\x1b[J\x1b[A" + msg

            output_text = "\r\x1b[K" + "".join(view).rstrip() + "\r" + msg

def prompt(promptable, history=None, settings=None):
    if settings is None:
        settings = BeatPromptSettings()

    command = RootCommand(promptable)
    input = BeatInput(command, history)
    stroke = BeatStroke(input, settings.keymap, settings.keycodes)
    prompt = BeatPrompt(stroke, input, settings)

    input_knot = dn.input(stroke.input_handler())
    display_knot = dn.show(prompt.output_handler(), 1/settings.framerate, hide_cursor=True)

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
