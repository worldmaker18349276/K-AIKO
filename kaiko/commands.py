import os
import dataclasses
from enum import Enum
from collections import OrderedDict
import functools
import re
import inspect
from pathlib import Path
from . import biparsers as bp


def suitability(part, full):
    r"""Compute suitability of a string `full` to the given substring `part`.
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
    r"""Sort options by its suitability.
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


class TokenUnfinishError(Exception):
    pass

class TokenParseError(Exception):
    pass

class TOKEN_TYPE(Enum):
    COMMAND = "command"
    ARGUMENT = "argument"
    KEYWORD = "keyword"
    UNKNOWN = "unknown"


def desc_options(options):
    return "It should be one of:\n" + "\n".join("• " + s for s in options)

class ArgumentParser:
    def parse(self, token):
        raise TokenParseError("Invalid value")

    def suggest(self, token):
        return []

    def desc(self):
        return None

    def info(self, token):
        return None

class RawParser(ArgumentParser):
    def __init__(self, default=inspect.Parameter.empty, desc=None):
        self.default = default
        self._desc = desc

    def desc(self):
        return self._desc

    def parse(self, token):
        return token

    def suggest(self, token):
        if self.default is inspect.Parameter.empty:
            return []
        else:
            return [val + "\000" for val in fit(token, [self.default])]

class OptionParser(ArgumentParser):
    def __init__(self, options, default=inspect.Parameter.empty, desc=None):
        self.options = options
        self.default = default

        if desc is None:
            options = list(self.options.keys()) if isinstance(self.options, dict) else self.options
            self._desc = desc_options(options)

    def desc(self):
        return self._desc

    def parse(self, token):
        if token not in self.options:
            desc = self._desc
            raise TokenParseError("Invalid value" + ("\n" + desc if desc is not None else ""))

        if isinstance(self.options, dict):
            return self.options[token]
        else:
            return token

    def suggest(self, token):
        options = list(self.options.keys()) if isinstance(self.options, dict) else self.options
        return [val + "\000" for val in fit(token, options)]

class PathParser(ArgumentParser):
    def __init__(self, root=".", default=inspect.Parameter.empty, desc=None):
        self.root = root
        self.default = default
        self._desc = desc or "It should be a path"

    def desc(self):
        return self._desc

    def parse(self, token):
        try:
            exists = os.path.exists(os.path.join(self.root, token or "."))
        except ValueError:
            exists = False

        if not exists:
            desc = self._desc
            raise TokenParseError("Path does not exist" + ("\n" + desc if desc is not None else ""))

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
    def __init__(self, type_hint, default=inspect.Parameter.empty, desc=None):
        self.type_hint = type_hint
        self.biparser = bp.from_type_hint(type_hint)
        self.default = default
        self._desc = desc or f"It should be {type_hint}"

    def desc(self):
        return self._desc

    def parse(self, token):
        try:
            return self.biparser.decode(token)[0]
        except bp.DecodeError:
            desc = self._desc
            raise TokenParseError("Invalid value" + ("\n" + desc if desc is not None else ""))

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

    def desc(self):
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

    def desc(self):
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
            desc = parser.desc()
            msg = "Missing value" + ("\n" + desc if desc is not None else "")
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
                desc = desc_options(["--" + key for key in self.kwargs.keys()])
                msg = f"Unknown argument {token!r}" + "\n" + desc
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

    def desc(self):
        # parse positional arguments
        if self.args:
            parser_func = next(iter(self.args.values()))
            parser = parser_func(**self.bound)
            return parser.desc()

        # parse keyword arguments
        if self.kwargs:
            keys = ["--" + key for key in self.kwargs.keys()]
            return desc_options(keys)

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
        desc = self.desc()
        raise TokenUnfinishError("Unfinished command" + ("\n" + desc if desc is not None else ""))

    def parse(self, token):
        if token not in self.fields:
            desc = self.desc()
            msg = "Unknown command" + ("\n" + desc if desc is not None else "")
            raise TokenParseError(msg)

        field = self.get_promptable_field(token)
        if not isinstance(field, CommandParser):
            raise TokenParseError("Not a command")

        return TOKEN_TYPE.COMMAND, field

    def suggest(self, token):
        return [val + "\000" for val in fit(token, self.fields)]

    def desc(self):
        return desc_options(self.fields)

    def info(self, token):
        # assert token in self.fields
        descriptor = type(self.parent).__dict__[token]
        doc = descriptor.proxy.__doc__
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

    def desc_command(self, tokens):
        cmd = self
        for token in tokens:
            try:
                _, cmd = cmd.parse(token)
            except TokenParseError as err:
                cmd = UnknownCommandParser()
        return cmd.desc()

    def info_command(self, tokens, target):
        cmd = self
        for token in tokens:
            try:
                _, cmd = cmd.parse(token)
            except TokenParseError as err:
                cmd = UnknownCommandParser()
        return cmd.info(target)


