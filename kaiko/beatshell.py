import os
from enum import Enum
import dataclasses
from collections import OrderedDict
import functools
import itertools
import re
import queue
import threading
import inspect
from pathlib import Path
from typing import List, Set, Tuple, Dict, Union
import wcwidth
from . import engines
from . import datanodes as dn
from . import biparsers as bp
from . import wcbuffers as wcb
from . import config as cfg


def expected_options(options):
    return "It should be one of:\n" + "\n".join("• " + shlexer_quoting(s + "\000") for s in options)

def suitability(part, full):
    """Compute suitability of a string `full` to the given substring `part`.
    The suitability is defined by substring mask:
    The substring mask of a string `full` is a list of non-empty slices `sections`
    such that `''.join(full[sec] for sec in sections) == part`.  The suitability of
    an option is the greatest tuple `(seclens, -last, -length)` in all possible substring
    masks.  Where `seclens` is a list of section lengths.  `last` is the last index of
    the substring mask.  `length` is the length of string `full`.
    
    Parameters
    ----------
    part : str
        The substring to find.
    full : str
        The string to match.
    
    Returns
    -------
    suitability : tuple
        A value representing suitability of a string `full` to given substring `part`.
        The larger value means more suitable.  The suitability of unmatched string is `()`.
    """
    if part == "":
        return ((), 0, -len(full))

    plen = len(part)
    flen = len(full)
    suitabilities = []
    
    states = [(0, 0, ())]
    while states:
        new_states = []
        for pstart, fstart, seclens in states:
            for flast in range(fstart, flen-plen+pstart+1):
                if full[flast] == part[pstart]:
                    for plast in range(pstart, plen):
                        if full[flast+plast-pstart] != part[plast]:
                            new_states.append((plast, flast+plast-pstart, (*seclens, plast-pstart)))
                            break
                    else:
                        suitabilities.append(((*seclens, plen-pstart), -(flast+plen-pstart), -flen))
        states = new_states
    
    return max(suitabilities, default=())
    
def fit(part, options):
    """Sort options by its suitability.
    It will also filter out the option that has no such substring.
    
    Parameters
    ----------
    part : str
        The substring to match.
    options : list of str
        The options to sort.
    
    Returns
    -------
    options : list of str
        The sorted options.
    """
    if part == "":
        return options
    options = [(suitability(part, opt), opt) for opt in options]
    options = [(m, opt) for m, opt in options if m != ()]
    options = sorted(options, reverse=True)
    options = [opt for _, opt in options]
    return options

class SHLEXER_STATE(Enum):
    SPACED = " "
    PLAIN = "*"
    BACKSLASHED = "\\"
    QUOTED = "'"

class TokenError(Exception):
    pass

def shlexer_tokenize(raw, partial=False):
    """Tokenizer for shell-like grammar.
    The delimiter is just whitespace, and the token is defined as::

        <nonspace-character> ::= /[^ ]/
        <backslashed-character> ::= "\" /./
        <quoted-string> ::= "'" /[^']*/ "'"
        <token> ::= ( <nonspace-character> | <backslashed-character> | <quoted-string> )*
    
    The backslashes and quotation marks used for escaping will be deleted after being interpreted as a string.
    The input string should be printable, so it doesn't contain tab, newline, backspace, etc.
    
    Parameters
    ----------
    raw : str or list of str
        The string to tokenize, which should be printable.
    partial : bool, optional
        True for partially parsing the input string.
    
    Yields
    ------
    token : str
        The tokenized string.
    mask : slice
        The position of this token.
    ignored : list of int
        The indices of all backslashes and quotation marks used for escaping.
        The token is equal to `''.join(raw[i] for i in range(*slice.indices(len(raw))) if i not in ignored)`.
    
    Returns
    -------
    state : SHLEXER_STATE
        The final state of parsing.
    
    Raises
    ------
    TokenError
        If the input string has non-closed escape character.
    """
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
                ignored.append(index)

                try:
                    index, char = next(raw)
                except StopIteration:
                    if not partial:
                        raise TokenError("No escaped character")
                    yield "".join(token), slice(start, None), ignored
                    return SHLEXER_STATE.BACKSLASHED

                token.append(char)
                
            elif char == QUOTE:
                # escape the following characters until the next quotation mark
                ignored.append(index)
                
                while True:
                    try:
                        index, char = next(raw)
                    except StopIteration:
                        if not partial:
                            raise TokenError("No closing quotation")
                        yield "".join(token), slice(start, None), ignored
                        return SHLEXER_STATE.QUOTED

                    if char == QUOTE:
                        ignored.append(index)
                        break
                    else:
                        token.append(char)

            else:
                # otherwise, as it is
                token.append(char)

            try:
                index, char = next(raw)
            except StopIteration:
                yield "".join(token), slice(start, None), ignored
                return SHLEXER_STATE.PLAIN

def shlexer_quoting(compreply, state=SHLEXER_STATE.SPACED):
    partial = not compreply.endswith("\000")
    if not partial:
        compreply = compreply[:-1]

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

def pmove(width, x, text, tabsize=8):
    y = 0

    for ch, w in wcb.parse_attr(text):
        if ch == "\t":
            if tabsize > 0 and x < width:
                x = min((x+1) // -tabsize * -tabsize, width-1)

        elif ch == "\b":
            x = max(min(x, width-1)-1, 0)

        elif ch == "\r":
            x = 0

        elif ch == "\n":
            y += 1
            x = 0

        elif ch == "\v":
            y += 1

        elif ch == "\f":
            y += 1

        elif ch == "\x00":
            pass

        elif ch[0] == "\x1b":
            pass

        else:
            x += w
            if x > width:
                y += 1
                x = w

    return x, y


class TokenUnfinishError(Exception):
    pass

class TokenParseError(Exception):
    pass

class TOKEN_TYPE(Enum):
    COMMAND = "command"
    ARGUMENT = "argument"
    KEYWORD = "keyword"
    UNKNOWN = "unknown"


class ArgumentParser:
    expected = None

    def parse(self, token):
        raise TokenParseError("Invalid value")

    def suggest(self, token):
        return []

    def info(self, token):
        return None

class RawParser(ArgumentParser):
    def __init__(self, default=inspect.Parameter.empty, expected=None):
        self.default = default
        self.expected = expected

    def parse(self, token):
        return token

    def suggest(self, token):
        if self.default is inspect.Parameter.empty:
            return []
        else:
            return [val + "\000" for val in fit(token, [self.default])]

class OptionParser(ArgumentParser):
    def __init__(self, options, default=inspect.Parameter.empty, expected=None):
        self.options = options
        self.default = default

        if expected is None:
            options = list(self.options.keys()) if isinstance(self.options, dict) else self.options
            self.expected = expected_options(options)

    def parse(self, token):
        if token not in self.options:
            expected = self.expected
            raise TokenParseError("Invalid value" + ("\n" + expected if expected is not None else ""))

        if isinstance(self.options, dict):
            return self.options[token]
        else:
            return token

    def suggest(self, token):
        options = list(self.options.keys()) if isinstance(self.options, dict) else self.options
        return [val + "\000" for val in fit(token, options)]

class PathParser(ArgumentParser):
    def __init__(self, root=".", default=inspect.Parameter.empty, expected=None):
        self.root = root
        self.default = default
        self.expected = expected or "It should be a path"

    def parse(self, token):
        try:
            exists = os.path.exists(os.path.join(self.root, token or "."))
        except ValueError:
            exists = False

        if not exists:
            expected = self.expected
            raise TokenParseError("Path does not exist" + ("\n" + expected if expected is not None else ""))

        return Path(token)

    def suggest(self, token):
        if not token:
            token = "."

        suggestions = []

        if self.default is not inspect.Parameter.empty:
            suggestions.append(str(self.default) + "\000")

        # check path
        currpath = os.path.join(self.root, token)
        try:
            is_dir = os.path.isdir(currpath)
            is_file = os.path.isfile(currpath)
        except ValueError:
            return suggestions

        if is_file:
            suggestions.append(token + "\000")
            return suggestions

        # separate parent and partial name
        if is_dir:
            suggestions.append(os.path.join(token, ""))
            target = ""
        else:
            token, target = os.path.split(token)

        # explore directory
        currdir = os.path.join(self.root, token)
        if os.path.isdir(currdir):
            names = fit(target, [name for name in os.listdir(currdir) if not name.startswith(".")])
            for name in names:
                subpath = os.path.join(currdir, name)
                sugg = os.path.join(token, name)

                if os.path.isdir(subpath):
                    sugg = os.path.join(sugg, "")
                    suggestions.append(sugg)

                elif os.path.isfile(subpath):
                    suggestions.append(sugg + "\000")

        return suggestions

class LiteralParser(ArgumentParser):
    def __init__(self, type_hint, default=inspect.Parameter.empty, expected=None):
        self.type_hint = type_hint
        self.biparser = bp.from_type_hint(type_hint)
        self.default = default
        self.expected = expected or f"It should be {type_hint}"

    def parse(self, token):
        try:
            return self.biparser.decode(token)[0]
        except bp.DecodeError:
            expected = self.expected
            raise TokenParseError("Invalid value" + ("\n" + expected if expected is not None else ""))

    def suggest(self, token):
        try:
            self.biparser.decode(token)
        except bp.DecodeError as e:
            sugg = [token[:e.index] + ex for ex in e.expected]
            if token == "" and self.default is not inspect.Parameter.empty:
                default = self.biparser.encode(self.default) + "\000"
                if default in sugg:
                    sugg.remove(default)
                sugg.insert(0, default)
        else:
            sugg = []

        return sugg


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

        def get_parser_func(param):
            if param.name in self.parsers:
                return self.parsers[param.name].__get__(instance, owner)
            elif param.annotation is not inspect.Signature.empty:
                return lambda *_, **__: LiteralParser(param.annotation, param.default)
            else:
                raise ValueError(f"No parser for argument {param.name} in {self.proxy.__name__}")

        sig = inspect.signature(func)
        args = OrderedDict()
        kwargs = OrderedDict()
        for param in sig.parameters.values():
            if param.default is inspect.Parameter.empty:
                args[param.name] = get_parser_func(param)
            else:
                kwargs[param.name] = get_parser_func(param)

        return FunctionCommandParser(func, args, kwargs)

    def arg_parser(self, name):
        def arg_parser_dec(parser):
            self.parsers[name] = parser
            return parser
        return arg_parser_dec

class subcommand(CommandDescriptor):
    def __get_command__(self, instance, owner):
        parent = self.proxy.__get__(instance, owner)
        fields = [k for k, v in type(parent).__dict__.items() if isinstance(v, CommandDescriptor)]
        return SubCommandParser(parent, fields)


class CommandParser:
    def finish(self):
        raise NotImplementedError

    def parse(self, token):
        raise NotImplementedError

    def suggest(self, token):
        raise NotImplementedError

    @property
    def expected(self):
        raise NotImplementedError

    def info(self, token):
        raise NotImplementedError

class UnknownCommandParser(CommandParser):
    def finish(self):
        raise TokenParseError("Unknown command")

    def parse(self, token):
        return TOKEN_TYPE.UNKNOWN, self

    def suggest(self, token):
        return []

    @property
    def expected(self):
        return None

    def info(self, token):
        return None

class FunctionCommandParser(CommandParser):
    def __init__(self, func, args, kwargs, bound=None):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.bound = bound or {}

    def finish(self):
        if self.args:
            parser_func = next(iter(self.args.values()))
            parser = parser_func(**self.bound)
            expected = parser.expected
            msg = "Missing value" + ("\n" + expected if expected is not None else "")
            raise TokenUnfinishError(msg)

        return functools.partial(self.func, **self.bound)

    def parse(self, token):
        # parse positional arguments
        if self.args:
            args = OrderedDict(self.args)
            name, parser_func = args.popitem(False)

            parser = parser_func(**self.bound)
            value = parser.parse(token)
            bound = {**self.bound, name: value}

            return TOKEN_TYPE.ARGUMENT, FunctionCommandParser(self.func, args, self.kwargs, bound)

        # parse keyword arguments
        if self.kwargs:
            kwargs = OrderedDict(self.kwargs)
            name = token[2:] if token.startswith("--") else None
            parser_func = kwargs.pop(name, None)

            if parser_func is None:
                expected = expected_options(["--" + key for key in self.kwargs.keys()])
                msg = f"Unknown argument {token!r}" + "\n" + expected
                raise TokenParseError(msg)

            args = OrderedDict([(name, parser_func)])
            return TOKEN_TYPE.KEYWORD, FunctionCommandParser(self.func, args, kwargs, self.bound)

        # rest
        raise TokenParseError("Too many arguments")

    def suggest(self, token):
        # parse positional arguments
        if self.args:
            parser_func = next(iter(self.args.values()))
            parser = parser_func(**self.bound)
            return parser.suggest(token)

        # parse keyword arguments
        if self.kwargs:
            keys = ["--" + key for key in self.kwargs.keys()]
            return [key + "\000" for key in fit(token, keys)]

        # rest
        return []

    @property
    def expected(self):
        # parse positional arguments
        if self.args:
            parser_func = next(iter(self.args.values()))
            parser = parser_func(**self.bound)
            return parser.expected

        # parse keyword arguments
        if self.kwargs:
            keys = ["--" + key for key in self.kwargs.keys()]
            return expected_options(keys)

        # rest
        return None

    def info(self, token):
        # parse positional arguments
        if self.args:
            parser_func = next(iter(self.args.values()))
            parser = parser_func(**self.bound)
            return parser.info(token)

        # parse keyword arguments
        if self.kwargs:
            return None

        # rest
        return None

class SubCommandParser(CommandParser):
    def __init__(self, parent, fields):
        self.parent = parent
        self.fields = fields

    def get_promptable_field(self, name):
        desc = type(self.parent).__dict__[name]
        return desc.__get_command__(self.parent, type(self.parent))

    def finish(self):
        expected = self.expected
        raise TokenUnfinishError("Unfinished command" + ("\n" + expected if expected is not None else ""))

    def parse(self, token):
        if token not in self.fields:
            expected = self.expected
            msg = "Unknown command" + ("\n" + expected if expected is not None else "")
            raise TokenParseError(msg)

        field = self.get_promptable_field(token)
        if not isinstance(field, CommandParser):
            raise TokenParseError("Not a command")

        return TOKEN_TYPE.COMMAND, field

    def suggest(self, token):
        return [val + "\000" for val in fit(token, self.fields)]

    @property
    def expected(self):
        return expected_options(self.fields)

    def info(self, token):
        # assert token in self.fields
        desc = type(self.parent).__dict__[token]
        doc = desc.proxy.__doc__
        if doc is None:
            return None

        # outdent docstring
        m = re.search("\\n[ ]*", doc)
        level = len(m.group(0)[1:]) if m else 0
        return re.sub("\\n[ ]{,%d}"%level, "\\n", doc)

class RootCommandParser(SubCommandParser):
    def __init__(self, root):
        fields = [k for k, v in type(root).__dict__.items() if isinstance(v, CommandDescriptor)]
        super(RootCommandParser, self).__init__(root, fields)

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
                type, cmd = TOKEN_TYPE.UNKNOWN, UnknownCommandParser()
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
                cmd = UnknownCommandParser()
        return cmd.suggest(target)

    def expected_command(self, tokens):
        cmd = self
        for token in tokens:
            try:
                _, cmd = cmd.parse(token)
            except TokenParseError as err:
                cmd = UnknownCommandParser()
        return cmd.expected

    def info_command(self, tokens, target):
        cmd = self
        for token in tokens:
            try:
                _, cmd = cmd.parse(token)
            except TokenParseError as err:
                cmd = UnknownCommandParser()
        return cmd.info(target)


class InputWarn:
    def __init__(self, message, tokens):
        self.message = message
        self.tokens = tokens

class InputMessage:
    def __init__(self, message, tokens):
        self.message = message
        self.tokens = tokens

class InputError:
    def __init__(self, value):
        self.value = value

class InputComplete:
    def __init__(self, value):
        self.value = value

def onstate(*states):
    def onstate_dec(func):
        @functools.wraps(func)
        def onstate_func(self, *args, **kwargs):
            if self.state not in states:
                return False
            return func(self, *args, **kwargs)
        return onstate_func
    return onstate_dec

def locked(func):
    @functools.wraps(func)
    def locked_func(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)
    return locked_func

class INPUT_STATE(Enum):
    EDIT = "edit"
    TAB = "tab"
    FIN = "fin"

class BeatInput:
    def __init__(self, promptable, history=None):
        self.command = RootCommandParser(promptable)
        self.history = history if history is not None else []
        self.lock = threading.RLock()
        self.state = INPUT_STATE.FIN

        self.new_session(False)

    def prompt(self, settings=None):
        if settings is None:
            settings = BeatShellSettings()

        input_knot, controller = engines.Controller.create(engines.ControllerSettings())
        stroke = BeatStroke(self, settings.input.keymap)
        prompt = BeatPrompt(stroke, self, settings)

        stroke.register(controller)
        display_knot = dn.show(prompt.output_handler(), 1/settings.prompt.framerate, hide_cursor=True)

        # `dn.show`, `dn.input` will fight each other...
        @dn.datanode
        def slow(dt=0.1):
            import time
            try:
                yield
                prompt.ref_time = time.time()
                yield
                while True:
                    yield
            finally:
                time.sleep(dt)

        prompt_knot = dn.pipe(input_knot, slow(), display_knot)
        return prompt_knot, prompt

    @locked
    @onstate(INPUT_STATE.FIN)
    def new_session(self, record_current=True):
        if record_current:
            self.history.append("".join(self.buffer))

        self.buffers = [list(history_buffer) for history_buffer in self.history]
        self.buffers.append([])
        self.buffers_index = -1

        self.buffer = self.buffers[self.buffers_index]
        self.pos = len(self.buffer)
        self.typeahead = ""

        self.tokens = []
        self.lex_state = SHLEXER_STATE.SPACED
        self.highlighted = None

        self.hint = None
        self.result = None
        self.state = INPUT_STATE.EDIT

    @locked
    @onstate(INPUT_STATE.FIN)
    def prev_session(self):
        self.result = None
        self.state = INPUT_STATE.EDIT

    @locked
    @onstate(INPUT_STATE.EDIT)
    def finish(self):
        self.state = INPUT_STATE.FIN
        return True

    @locked
    def parse_syntax(self):
        """Parse syntax.

        Returns
        -------
        succ : bool
        """
        tokenizer = shlexer_tokenize(self.buffer, partial=True)

        tokens = []
        while True:
            try:
                token, slic, ignored = next(tokenizer)
            except StopIteration as e:
                self.lex_state = e.value
                break

            tokens.append((token, slic, ignored))

        types, _, _ = self.command.parse_command(token for token, _, _ in tokens)
        self.tokens = [(token, type, slic, ignored) for (token, slic, ignored), type in zip(tokens, types)]
        return True

    @locked
    def make_typeahead(self):
        """Make typeahead.
        Show the possible command you want to type.  Only work if the caret is at
        the end of buffer.

        Returns
        -------
        succ : bool
            `False` if unable to complete or the caret is not at the end of buffer.
        """
        if self.pos != len(self.buffer):
            self.typeahead = ""
            return False

        if self.lex_state == SHLEXER_STATE.SPACED:
            parents = [token for token, _, _, _ in self.tokens]
            target = ""
        else:
            parents = [token for token, _, _, _ in self.tokens[:-1]]
            target, _, _, _ = self.tokens[-1]

        compreply = None
        for suggestion in self.command.suggest_command(parents, target):
            if suggestion.startswith(target):
                compreply = suggestion[len(target):]
                break

        if compreply is None:
            self.typeahead = ""
            return False
        else:
            self.typeahead = shlexer_quoting(compreply, self.lex_state)
            return True

    @locked
    def cancel_typeahead(self):
        """Cancel typeahead.

        Returns
        -------
        succ : bool
        """
        self.typeahead = ""
        return True

    @locked
    def set_result(self, res_type, value, index=None):
        """Set result.
        Set result of this session.

        Parameters
        ----------
        res_type : InputError or InputComplete
            The type of result.
        value : any
            The value of result.
        index : int or None
            The index of token to highlight.

        Returns
        -------
        succ : bool
        """
        self.highlighted = index
        self.result = res_type(value)
        self.hint = None
        return True

    @locked
    def clear_result(self):
        """Clear result.

        Returns
        -------
        succ : bool
        """
        self.highlighted = None
        self.result = None
        return True

    @locked
    def set_hint(self, hint_type, message, index=None):
        """Set hint.
        Show hint below the prompt.

        Parameters
        ----------
        hint_type : InputWarn or InputMessage
            The type of hint.
        message : str
            The message to show.
        index : int or None
            Index of the token to which the hint is directed, or `None` for nothing.

        Returns
        -------
        succ : bool
        """
        self.highlighted = index
        if index is None:
            msg_tokens = None
        elif hint_type == InputWarn:
            msg_tokens = self.tokens[:index]
        else:
            msg_tokens = self.tokens[:index+1]
        self.hint = hint_type(message, msg_tokens)
        return True

    @locked
    def cancel_hint(self):
        """Cancel hint.
        Remove the hint below the prompt.

        Returns
        -------
        succ : bool
        """
        self.highlighted = None
        self.hint = None
        return True

    @locked
    def update_hint(self):
        """Update hint.
        Remove hint if the target is updated.

        Returns
        -------
        succ : bool
            `False` if there is no hint or the hint isn't removed.
        """
        if self.hint is None:
            return False

        if self.hint.tokens is None:
            return self.cancel_hint()

        if len(self.hint.tokens) > len(self.tokens):
            return self.cancel_hint()

        for t1, t2 in zip(self.hint.tokens, self.tokens):
            if t1[0] != t2[0]:
                return self.cancel_hint()

        return False

    @locked
    @onstate(INPUT_STATE.EDIT)
    def insert_typeahead(self):
        """Insert typeahead.
        Insert the typeahead if the caret is at the end of buffer.

        Returns
        -------
        succ : bool
            `False` if there is no typeahead or the caret is not at the end of buffer.
        """

        if self.typeahead == "" or self.pos != len(self.buffer):
            return False

        self.buffer[self.pos:self.pos] = self.typeahead
        self.pos += len(self.typeahead)
        self.typeahead = ""
        self.parse_syntax()

        return True

    @locked
    @onstate(INPUT_STATE.EDIT)
    def help(self):
        """Help for command.
        Provide some hint for the command before the caret.

        Returns
        -------
        succ : bool
            `False` if there is no hint.
        """
        self.cancel_hint()

        # find the last token before the caret
        for index, (target, token_type, slic, _) in reversed(list(enumerate(self.tokens))):
            if slic.start is None or slic.start <= self.pos:
                break
        else:
            return False

        parents = [token for token, _, _, _ in self.tokens[:index]]

        if token_type == TOKEN_TYPE.UNKNOWN:
            msg = self.command.expected_command(parents)
            hint_type = InputWarn
        else:
            msg = self.command.info_command(parents, target)
            hint_type = InputMessage

        if msg is None:
            return False
        else:
            self.set_hint(hint_type, msg, index)
            return True

    @locked
    @onstate(INPUT_STATE.EDIT)
    def enter(self):
        """Enter.
        Finish the command.

        Returns
        -------
        succ : bool
            `False` if the command is wrong.
        """
        if len(self.tokens) == 0:
            self.set_result(InputComplete, lambda:None)
            self.finish()
            return True

        if self.lex_state == SHLEXER_STATE.BACKSLASHED:
            res, index = TokenParseError("No escaped character"), len(self.tokens)-1
        elif self.lex_state == SHLEXER_STATE.QUOTED:
            res, index = TokenParseError("No closing quotation"), len(self.tokens)-1
        else:
            _, res, index = self.command.parse_command(token for token, _, _, _ in self.tokens)

        if isinstance(res, TokenUnfinishError):
            self.set_result(InputError, res, None)
            self.finish()
            return False
        elif isinstance(res, TokenParseError):
            self.set_result(InputError, res, index)
            self.finish()
            return False
        else:
            self.set_result(InputComplete, res)
            self.finish()
            return True

    @locked
    @onstate(INPUT_STATE.EDIT)
    def input(self, text):
        """Input.
        Insert some text into the buffer.

        Parameters
        ----------
        text : str
            The text to insert.  It shouldn't contain any nongraphic character,
            except for prefix `\\b` which indicate deleting.

        Returns
        -------
        succ : bool
            `False` if buffer isn't changed.
        """
        text = list(text)

        if len(text) == 0:
            return False

        while len(text) > 0 and text[0] == "\b":
            del text[0]
            del self.buffer[self.pos-1]
            self.pos = self.pos-1

        if wcwidth.wcswidth("".join(self.buffer)) == -1:
            raise ValueError("invalid text to insert: " + repr("".join(self.buffer)))

        self.buffer[self.pos:self.pos] = text
        self.pos += len(text)
        self.parse_syntax()

        self.make_typeahead()
        self.update_hint()

        return True

    @locked
    @onstate(INPUT_STATE.EDIT)
    def backspace(self):
        """Backspace.
        Delete one character before the caret if exists.

        Returns
        -------
        succ : bool
        """
        if self.pos == 0:
            return False

        self.pos -= 1
        del self.buffer[self.pos]
        self.parse_syntax()
        self.cancel_typeahead()
        self.update_hint()

        return True

    @locked
    @onstate(INPUT_STATE.EDIT)
    def delete(self):
        """Delete.
        Delete one character after the caret if exists.

        Returns
        -------
        succ : bool
        """
        if self.pos >= len(self.buffer):
            return False

        del self.buffer[self.pos]
        self.parse_syntax()
        self.cancel_typeahead()
        self.update_hint()

        return True

    @locked
    @onstate(INPUT_STATE.EDIT)
    def delete_range(self, start, end):
        """Delete range.

        Parameters
        ----------
        start : int or None
        end : int or None

        Returns
        -------
        succ : bool
        """
        start = min(max(0, start), len(self.buffer)) if start is not None else 0
        end = min(max(0, end), len(self.buffer)) if end is not None else len(self.buffer)

        if start >= end:
            return False

        del self.buffer[start:end]
        self.pos = start
        self.parse_syntax()
        self.cancel_typeahead()
        self.update_hint()

        return True

    @locked
    @onstate(INPUT_STATE.EDIT)
    def delete_to_word_start(self):
        """Delete to the word start.
        The word is defined as `\\w+|\\W+`.

        Returns
        -------
        succ : bool
        """
        for match in re.finditer("\w+|\W+", "".join(self.buffer)):
            if match.end() >= self.pos:
                return self.delete_range(match.start(), self.pos)
        else:
            return self.delete_range(None, self.pos)

    @locked
    @onstate(INPUT_STATE.EDIT)
    def delete_to_word_end(self):
        """Delete to the word end.
        The word is defined as `\\w+|\\W+`.

        Returns
        -------
        succ : bool
        """
        for match in re.finditer("\w+|\W+", "".join(self.buffer)):
            if match.end() > self.pos:
                return self.delete_range(self.pos, match.end())
        else:
            return self.delete_range(self.pos, None)

    @locked
    @onstate(INPUT_STATE.EDIT)
    def move_to(self, pos):
        """Move caret to the specific position.
        Regardless of success or failure, typeahead will be cancelled.

        Parameters
        ----------
        pos : int or None
            Index of buffer, which will be clamped to 0 and length of buffer, or
            `None` for the end of buffer.

        Returns
        -------
        succ : bool
        """
        pos = min(max(0, pos), len(self.buffer)) if pos is not None else len(self.buffer)
        self.cancel_typeahead()

        if self.pos == pos:
            return False

        self.pos = pos
        return True

    @locked
    @onstate(INPUT_STATE.EDIT)
    def move(self, offset):
        """Move caret.

        Parameters
        ----------
        offset : int

        Returns
        -------
        succ : bool
        """
        return self.move_to(self.pos+offset)

    @locked
    @onstate(INPUT_STATE.EDIT)
    def move_left(self):
        """Move caret one character to the left.

        Returns
        -------
        succ : bool
        """
        return self.move(-1)

    @locked
    @onstate(INPUT_STATE.EDIT)
    def move_right(self):
        """Move caret one character to the right.

        Returns
        -------
        succ : bool
        """
        return self.move(+1)

    @locked
    @onstate(INPUT_STATE.EDIT)
    def move_to_start(self):
        """Move caret to the start of buffer.

        Returns
        -------
        succ : bool
        """
        return self.move_to(0)

    @locked
    @onstate(INPUT_STATE.EDIT)
    def move_to_end(self):
        """Move caret to the end of buffer.

        Returns
        -------
        succ : bool
        """
        return self.move_to(None)

    @locked
    @onstate(INPUT_STATE.EDIT)
    def move_to_word_start(self):
        """Move caret to the start of the word.

        Returns
        -------
        succ : bool
        """
        for match in re.finditer("\w+|\W+", "".join(self.buffer)):
            if match.end() >= self.pos:
                return self.move_to(match.start())
        else:
            return self.move_to(0)

    @locked
    @onstate(INPUT_STATE.EDIT)
    def move_to_word_end(self):
        """Move caret to the end of the word.

        Returns
        -------
        succ : bool
        """
        for match in re.finditer("\w+|\W+", "".join(self.buffer)):
            if match.end() > self.pos:
                return self.move_to(match.end())
        else:
            return self.move_to(None)

    @locked
    @onstate(INPUT_STATE.EDIT)
    def prev(self):
        """Previous buffer.

        Returns
        -------
        succ : bool
        """
        if self.buffers_index == -len(self.buffers):
            return False
        self.buffers_index -= 1

        self.buffer = self.buffers[self.buffers_index]
        self.pos = len(self.buffer)
        self.parse_syntax()
        self.cancel_typeahead()
        self.cancel_hint()

        return True

    @locked
    @onstate(INPUT_STATE.EDIT)
    def next(self):
        """Next buffer.

        Returns
        -------
        succ : bool
        """
        if self.buffers_index == -1:
            return False
        self.buffers_index += 1

        self.buffer = self.buffers[self.buffers_index]
        self.pos = len(self.buffer)
        self.parse_syntax()
        self.cancel_typeahead()
        self.cancel_hint()

        return True

    @locked
    @onstate(INPUT_STATE.EDIT, INPUT_STATE.TAB)
    def autocomplete(self, action=+1):
        """Autocomplete.
        Complete the token on the caret, or fill in suggestions if caret is
        located in between.

        Parameters
        ----------
        action : +1 or -1 or 0
            Indicating direction for exploration of suggestions.  `+1` for next
            suggestion; `-1` for previous suggestion; `0` for canceling the process.

        Returns
        -------
        succ : bool
            `False` for canceling the process.
        """

        if self.state == INPUT_STATE.EDIT:
            self.cancel_typeahead()

            # find the token to autocomplete
            parents = []
            target = ""
            selection = slice(self.pos, self.pos)
            for token, _, slic, _ in self.tokens:
                start, stop, _ = slic.indices(len(self.buffer))
                if stop < self.pos:
                    parents.append(token)
                if start <= self.pos <= stop:
                    target = token
                    selection = slic

            # generate suggestions
            self._suggestions = [shlexer_quoting(sugg) for sugg in self.command.suggest_command(parents, target)]
            self._sugg_index = len(self._suggestions) if action == -1 else -1

            # tab state
            self._original_buffer = list(self.buffer)
            self._original_pos = self.pos
            self._selection = selection
            self.state = INPUT_STATE.TAB

        if action == +1:
            self._sugg_index += 1
        elif action == -1:
            self._sugg_index -= 1
        elif action == 0:
            self._sugg_index = None
        else:
            raise ValueError

        self.buffer = list(self._original_buffer)
        self.pos = self._original_pos

        if self._sugg_index in range(len(self._suggestions)):
            # autocomplete selected token
            self.buffer[self._selection] = self._suggestions[self._sugg_index]
            self.pos = self._selection.start + len(self._suggestions[self._sugg_index])
            self.parse_syntax()
            return True

        else:
            # restore state
            del self._original_buffer
            del self._original_pos
            del self._selection
            self.state = INPUT_STATE.EDIT
            self.parse_syntax()
            return False

    @locked
    @onstate(INPUT_STATE.TAB)
    def finish_autocomplete(self):
        del self._original_buffer
        del self._original_pos
        del self._selection
        self.state = INPUT_STATE.EDIT
        return True

class BeatStroke:
    def __init__(self, input, keymap):
        self.input = input
        self.keymap = keymap
        self.key_event = threading.Event()

    def register(self, controller):
        controller.add_handler(self.keypress_handler())
        controller.add_handler(self.finish_autocomplete_handler())
        for key, func in self.keymap.items():
            controller.add_handler(self.action_handler(func), key)
        controller.add_handler(self.printable_handler(), "PRINTABLE")
        controller.add_handler(self.unknown_handler(self.keymap))

    def keypress_handler(self):
        return lambda args: self.key_event.set()

    def finish_autocomplete_handler(self):
        return lambda args: self.input.finish_autocomplete() if args[2] not in ("Tab", "Shift_Tab", "Esc") else False

    def action_handler(self, func):
        return lambda args: func(self.input)

    def printable_handler(self):
        return lambda args: self.input.input(args[3])

    def unknown_handler(self, keymap):
        keys = list(keymap.keys())
        keys.append("PRINTABLE")
        def handler(args):
            _, _, key, _ = args
            if key not in keys:
                with self.input.lock:
                    self.input.set_result(InputError, ValueError(f"Unknown key: " + repr(key or code)), None)
                    self.finish()
        return handler


class BeatShellSettings(cfg.Configurable):
    class input(cfg.Configurable):
        keymap = {
            "Backspace"     : lambda input: input.backspace(),
            "Delete"        : lambda input: input.delete(),
            "Left"          : lambda input: input.move_left(),
            "Right"         : lambda input: input.insert_typeahead() or input.move_right(),
            "Up"            : lambda input: input.prev(),
            "Down"          : lambda input: input.next(),
            "Home"          : lambda input: input.move_to_start(),
            "End"           : lambda input: input.move_to_end(),
            "Enter"         : lambda input: input.enter(),
            "Esc"           : lambda input: (input.cancel_typeahead(), input.cancel_hint(), input.autocomplete(0)),
            "Alt_Enter"     : lambda input: input.help(),
            "Ctrl_Left"     : lambda input: input.move_to_word_start(),
            "Ctrl_Right"    : lambda input: input.move_to_word_end(),
            "Ctrl_Backspace": lambda input: input.delete_to_word_start(),
            "Ctrl_Delete"   : lambda input: input.delete_to_word_end(),
            "Tab"           : lambda input: input.autocomplete(+1),
            "Shift_Tab"     : lambda input: input.autocomplete(-1),
        }

    class prompt(cfg.Configurable):
        framerate: float = 60.0
        t0: float = 0.0
        tempo: float = 130.0

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

        caret_attr: Tuple[str, str] = ("7;2", "7;1")
        caret_blink_ratio: float = 0.3

    class text(cfg.Configurable):
        error_message_attr: str = "31"
        info_message_attr: str = "2"
        message_max_lines: int = 16

        escape_attr: str = "2"
        typeahead_attr: str = "2"
        whitespace: str = "\x1b[2m⌴\x1b[m"

        token_unknown_attr: str = "31"
        token_command_attr: str = "94"
        token_keyword_attr: str = "95"
        token_argument_attr: str = "92"
        token_highlight_attr: str = "4"

class BeatPrompt:
    def __init__(self, stroke, input, settings):
        self.stroke = stroke

        self.input = input
        self.settings = settings
        self.result = None
        self.ref_time = None
        self.t0 = None
        self.tempo = None

    @dn.datanode
    def output_handler(self):
        size_node = dn.terminal_size()
        header_node = self.header_node()
        hint_node = self.hint_node()
        render_node = self.render_node()
        with size_node, header_node, hint_node, render_node:
            yield
            while True:
                # obtain input state
                with self.input.lock:
                    buffer = list(self.input.buffer)
                    tokens = list(self.input.tokens)
                    pos = self.input.pos
                    highlighted = self.input.highlighted

                    typeahead = self.input.typeahead
                    result = self.input.result
                    hint = self.input.hint
                    state = self.input.state

                size = size_node.send()

                # draw hint
                msg_data = hint_node.send(hint)

                # draw header
                clean = result is not None
                header_data = header_node.send(clean)

                # draw text
                text_data = self.render_text(buffer, tokens, typeahead, pos, highlighted, clean)

                # render
                output_text = render_node.send((header_data, text_data, msg_data, size))

                yield output_text

                # end
                if state == INPUT_STATE.FIN:
                    self.result = result.value
                    self.input.clear_result()
                    return

    @dn.datanode
    def header_node(self):
        framerate = self.settings.prompt.framerate

        headers = self.settings.prompt.headers

        caret_attr = self.settings.prompt.caret_attr
        caret_blink_ratio = self.settings.prompt.caret_blink_ratio

        clean = yield
        self.t0 = self.settings.prompt.t0
        self.tempo = self.settings.prompt.tempo

        n = 0
        t = (self.ref_time - self.t0)/(60/self.tempo)
        tr = t // 1
        while True:
            # don't blink while key pressing
            if self.stroke.key_event.is_set():
                self.stroke.key_event.clear()
                tr = t // -1 * -1

            # render caret
            if clean:
                caret = None
            elif t < tr or t % 1 < caret_blink_ratio:
                if t % 4 < caret_blink_ratio:
                    caret = lambda s: wcb.add_attr(s, caret_attr[1])
                else:
                    caret = lambda s: wcb.add_attr(s, caret_attr[0])
            else:
                caret = None

            # render header
            ind = int(t / 4 * len(headers) // 1) % len(headers)
            header = headers[ind]

            clean = yield header, caret
            n += 1
            t = (self.ref_time - self.t0 + n/framerate)/(60/self.tempo)

    @dn.datanode
    def hint_node(self):
        current_hint = None
        hint = yield
        while True:
            # track changes of the hint
            if hint == current_hint:
                hint = yield None
                continue

            current_hint = hint

            # show hint
            hint = yield self.render_hint(hint)

    def render_hint(self, hint):
        if hint is None:
            return ""

        message_max_lines = self.settings.text.message_max_lines
        error_message_attr = self.settings.text.error_message_attr
        info_message_attr = self.settings.text.info_message_attr

        # show hint
        msg = hint.message or ""
        if msg.count("\n") >= message_max_lines:
            msg = "\n".join(msg.split("\n")[:message_max_lines]) + "\x1b[m\n…"
        if msg:
            if isinstance(hint, InputWarn):
                msg = wcb.add_attr(msg, error_message_attr)
            if isinstance(hint, (InputWarn, InputMessage)):
                msg = wcb.add_attr(msg, info_message_attr)
        msg = "\n" + msg + "\n" if msg else "\n"

        return msg

    def render_text(self, rendered_buffer, tokens, typeahead, pos, highlighted, clean):
        escape_attr     = self.settings.text.escape_attr
        typeahead_attr  = self.settings.text.typeahead_attr
        whitespace      = self.settings.text.whitespace

        token_unknown_attr   = self.settings.text.token_unknown_attr
        token_command_attr   = self.settings.text.token_command_attr
        token_keyword_attr   = self.settings.text.token_keyword_attr
        token_argument_attr  = self.settings.text.token_argument_attr
        token_highlight_attr = self.settings.text.token_highlight_attr

        # render buffer
        indices = range(len(rendered_buffer))

        for token, type, slic, ignored in tokens:
            # render whitespace
            for index in indices[slic]:
                if rendered_buffer[index] == " ":
                    rendered_buffer[index] = whitespace

            # render escape
            for index in ignored:
                rendered_buffer[index] = wcb.add_attr(rendered_buffer[index], escape_attr)

            # render unknown token
            if type is TOKEN_TYPE.UNKNOWN:
                if slic.stop is not None or clean:
                    for index in indices[slic]:
                        rendered_buffer[index] = wcb.add_attr(rendered_buffer[index], token_unknown_attr)

            # render command token
            if type is TOKEN_TYPE.COMMAND:
                for index in indices[slic]:
                    rendered_buffer[index] = wcb.add_attr(rendered_buffer[index], token_command_attr)

            # render keyword token
            elif type is TOKEN_TYPE.KEYWORD:
                for index in indices[slic]:
                    rendered_buffer[index] = wcb.add_attr(rendered_buffer[index], token_keyword_attr)

            # render argument token
            elif type is TOKEN_TYPE.ARGUMENT:
                for index in indices[slic]:
                    rendered_buffer[index] = wcb.add_attr(rendered_buffer[index], token_argument_attr)

        if highlighted in range(len(tokens)):
            # render highlighted token
            _, _, slic, _ = tokens[highlighted]
            for index in indices[slic]:
                rendered_buffer[index] = wcb.add_attr(rendered_buffer[index], token_highlight_attr)

        rendered_text = "".join(rendered_buffer)

        # render typeahead
        if typeahead and not clean:
            rendered_typeahead = wcb.add_attr(typeahead, typeahead_attr)
        else:
            rendered_typeahead = ""

        # compute caret position
        _, caret_pos = wcb.textrange1(0, "".join(rendered_buffer[:pos]))

        return rendered_text, rendered_typeahead, caret_pos

    @dn.datanode
    def render_node(self):
        header_width = self.settings.prompt.header_width
        header_ran = slice(None, header_width)
        input_ran = slice(header_width, None)

        input_offset = 0
        output_text = None

        while True:
            (header, caret), (text, typeahead, caret_pos), msg, size = yield output_text
            width = size.columns
            view = wcb.newwin1(width)

            # adjust input offset
            input_width = len(range(width)[input_ran])
            _, solid_text_width = wcb.textrange1(0, text)

            if solid_text_width - input_offset + 1 <= input_width:
                input_offset = max(0, solid_text_width-input_width+1)
            if caret_pos - input_offset >= input_width:
                input_offset = caret_pos - input_width + 1
            elif caret_pos - input_offset < 0:
                input_offset = caret_pos

            # draw input
            _, text_width = wcb.textrange1(0, text+typeahead)
            wcb.addtext1(view, width, input_ran.start-input_offset, text+typeahead, input_ran)
            if input_offset > 0:
                wcb.addtext1(view, width, input_ran.start, "…", input_ran)
            if text_width-input_offset >= input_width:
                wcb.addtext1(view, width, input_ran.start+input_width-1, "…", input_ran)

            # draw header
            wcb.addtext1(view, width, 0, header, header_ran)

            # draw caret
            if caret:
                caret_x = input_ran.start - input_offset + caret_pos
                caret_ran = wcb.select1(view, width, slice(caret_x, caret_x+1))
                view[caret_ran.start] = caret(view[caret_ran.start])

            # print message
            if msg is None:
                output_text = "\r\x1b[K" + "".join(view).rstrip() + "\r"
            elif msg == "":
                output_text = "\r\x1b[J" + "".join(view).rstrip() + "\r"
            else:
                _, y = pmove(width, 0, msg)
                output_text = "\r\x1b[J" + "".join(view).rstrip() + "\r" + msg + f"\x1b[{y}A"
