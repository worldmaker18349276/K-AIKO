from enum import Enum
from collections import OrderedDict
import inspect
import functools
from . import parsec as pc
from . import serializers as sz


def suitability(part, full):
    r"""Compute suitability of a string `full` to the given substring `part`.

    The suitability is defined by substring mask: The substring mask of a string
    `full` is a list of non-empty slices `sections` such that `''.join(full[sec]
    for sec in sections) == part`. The suitability of an option is the greatest
    tuple `(seclens, -last, -length)` in all possible substring masks. Where
    `seclens` is a list of section lengths. `last` is the last index of the
    substring mask. `length` is the length of string `full`.

    Parameters
    ----------
    part : str
        The substring to find.
    full : str
        The string to match.

    Returns
    -------
    suitability : tuple
        A value representing suitability of a string `full` to given substring
        `part`. The larger value means more suitable. The suitability of
        unmatched string is `()`.
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
            for flast in range(fstart, flen - plen + pstart + 1):
                if full[flast] == part[pstart]:
                    for plast in range(pstart, plen):
                        if full[flast + plast - pstart] != part[plast]:
                            new_states.append(
                                (
                                    plast,
                                    flast + plast - pstart,
                                    (*seclens, plast - pstart),
                                )
                            )
                            break
                    else:
                        suitabilities.append(
                            ((*seclens, plen - pstart), -(flast + plen - pstart), -flen)
                        )
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


class CommandUnfinishError(Exception):
    pass


class CommandParseError(Exception):
    def __init__(self, msg, token=None, expected=None):
        self.msg = msg
        self.token = token
        self.expected = expected

    def __str__(self):
        if self.token is not None and self.expected is not None:
            desc = do_you_mean(fit(self.token, self.expected))
            if desc:
                return self.msg + "\n" + desc
        return self.msg


class TOKEN_TYPE(Enum):
    COMMAND = "command"
    ARGUMENT = "argument"
    KEYWORD = "keyword"


def do_you_mean(options):
    if not options:
        return None
    return "Do you mean:\n" + "\n".join("• " + s for s in options)


class ArgumentParser:
    r"""Parser for the argument of command."""

    def parse(self, token):
        r"""Parse the token into an argument.

        Parameters
        ----------
        token : str
            The token.

        Returns
        -------
        arg : any
            The argument.

        Raises
        ------
        CommandParseError
            If parsing fails.
        """
        raise CommandParseError("Invalid value")

    def suggest(self, token):
        r"""Suggest some possible tokens related to the given token.

        Parameters
        ----------
        token : str
            The token.

        Returns
        -------
        suggestions : list of str
            A list of possible tokens, where complete tokens have the suffix
            '\000'.
        """
        return []

    def desc(self):
        r"""Describe acceptable tokens.

        Returns
        -------
        description : str or None
            The description.
        """
        return None

    def info(self, token):
        r"""Describe the value of the given token.

        Parameters
        ----------
        token : str
            The token.

        Returns
        -------
        description : str or None
            The description.
        """
        return None


class RawParser(ArgumentParser):
    r"""Parse a raw string."""

    def __init__(self, default=inspect.Parameter.empty, desc=None):
        r"""Contructor.

        Parameters
        ----------
        default : any, optional
            The default value of this argument.
        desc : str, optional
            The description of this argument.
        """
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
    r"""Parse an option."""

    def __init__(self, options, default=inspect.Parameter.empty, desc=None):
        r"""Contructor.

        Parameters
        ----------
        options : list of str or dict
            The list of options, or dictionary that maps option name to its
            value. If it is dictionary, the result of parsing will be its value.
        default : any, optional
            The default value of this argument.
        desc : str, optional
            The description of this argument.
        """
        self.options = options
        self.default = default
        self._desc = desc

    def desc(self):
        return self._desc

    def parse(self, token):
        if token not in self.options:
            desc = self.desc()
            raise CommandParseError(
                "Invalid value" + ("\n" + desc if desc is not None else "")
            )

        if isinstance(self.options, dict):
            return self.options[token]
        else:
            return token

    def suggest(self, token):
        options = (
            list(self.options.keys())
            if isinstance(self.options, dict)
            else self.options
        )
        return [val + "\000" for val in fit(token, options)]


class TreeParser(ArgumentParser):
    r"""Parse something following a tree.

    For example, a tree::

        {
            'abc': {
                'x': lambda _: 'abcx',
                'y': {
                    'z': lambda _: 'abcyz',
                    'w': lambda _: 'abcyw',
                },
            },
            'def': {
                '': lambda _: 'def',
                'g': lambda _: 'defg',
            },
            'ab': {
                'c': lambda _: '<never match>',
                'cx': lambda _: '<never match>',
                'd': lambda _: 'abd',
            },
        }

    will match strings 'abcx', 'abcyz', 'abcyw', 'def', 'defg', 'abd'. In each
    layer, It parsed longer string first, so one should prevent from putting
    ambiguious case.
    """

    def __init__(self, tree, default=inspect.Parameter.empty, desc=None):
        r"""Contructor.

        Parameters
        ----------
        tree : dict from str to parser tree
            The parser tree, the leaf should be a function producing parsed
            result, and the key of node should be a non-empty string.
        default : any, optional
            The default value of this argument.
        desc : str, optional
            The description of this argument.
        """
        self.tree = tree
        self.default = default
        self._desc = desc

    def desc(self):
        return self._desc

    def parse(self, token):
        target = token
        tree = self.tree

        while isinstance(tree, dict):
            if target == "" in tree:
                tree = tree[""]
                break

            for key, subtree in sorted(tree.items(), key=lambda a: a[0], reverse=True):
                if key and target.startswith(key):
                    target = target[len(key) :]
                    tree = subtree
                    break

            else:
                desc = self.desc()
                raise CommandParseError(
                    "Invalid value" + ("\n" + desc if desc is not None else "")
                )

        if target:
            desc = self.desc()
            raise CommandParseError(
                "Invalid value" + ("\n" + desc if desc is not None else "")
            )

        if not hasattr(tree, "__call__"):
            raise ValueError("Not a function.")

        return tree(token)

    def suggest(self, token):
        prefix = ""
        target = token
        tree = self.tree

        while isinstance(tree, dict):
            for key, subtree in sorted(tree.items(), key=lambda a: a[0], reverse=True):
                if target and key and target.startswith(key):
                    prefix = prefix + key
                    target = target[len(key) :]
                    tree = subtree
                    break

            else:
                comp = [
                    key + ("\000" if hasattr(subtree, "__call__") else "")
                    for key, subtree in tree.items()
                ]
                comp = fit(target, comp)
                return [prefix + c for c in comp]

        if not hasattr(tree, "__call__"):
            raise ValueError("Not a function.")

        return [token + "\000"] if not target else []


class LiteralParser(ArgumentParser):
    r"""Parse a Python literal."""

    def __init__(self, type_hint, default=inspect.Parameter.empty, desc=None):
        r"""Contructor.

        Parameters
        ----------
        type_hint : type
            The type of literal to parse.
        default : any, optional
            The default value of this argument.
        desc : str, optional
            The description of this argument.
        """
        self.type_hint = type_hint
        self.default = default
        suggestions = [default] if default is not inspect.Parameter.empty else []
        self.parser = (
            sz.make_serializer_from_type_hint(type_hint, suggestions).parser << pc.eof()
        )
        self._desc = desc or f"It should be {type_hint}"

    def desc(self):
        return self._desc

    def parse(self, token):
        try:
            return self.parser.parse(token)
        except pc.ParseError as e:
            if isinstance(e.__cause__, pc.ParseFailure):
                expected = e.__cause__.expected
                raise CommandParseError(
                    f"Invalid value\nexpecting {expected} at {e.index}"
                ) from e
            else:
                raise CommandParseError(f"Invalid value") from e

    def suggest(self, token):
        try:
            self.parser.parse(token)
        except pc.ParseError as e:
            if isinstance(e.__cause__, pc.ParseFailure):
                sugg = [
                    e.text[: e.index] + sugg for sugg in sz.get_suggestions(e.__cause__)
                ]
                return [*fit(token, sugg), token]
            else:
                return []
        else:
            return [token + "\000"]


class TimeParser(ArgumentParser):
    r"""Parse time."""

    def __init__(self, default=inspect.Parameter.empty):
        r"""Contructor.

        Parameters
        ----------
        default : any, optional
            The default value of this argument.
        """
        self.default = default
        suggestions = [default] if default is not inspect.Parameter.empty else []
        self.parser = (
            pc.concat(
                pc.regex(r"[-+]?").map(lambda a: -1 if a == "-" else 1),
                (pc.regex(r"[0-9]+").map(int) << pc.string(":"))
                .attempt()
                .choice(pc.nothing(0)),
                pc.regex(r"[0-9]+(\.[0-9]+)?").map(float),
            ).starmap(lambda a, m, s: a * (m * 60 + s))
            << pc.eof()
        )

    def desc(self):
        return "It should be in the format `min:sec`"

    def parse(self, token):
        try:
            return self.parser.parse(token)
        except pc.ParseError as e:
            if isinstance(e.__cause__, pc.ParseFailure):
                expected = e.__cause__.expected
                raise CommandParseError(
                    f"Invalid value\nexpecting {expected} at {e.index}"
                ) from e
            else:
                raise CommandParseError(f"Invalid value") from e


class CommandParser:
    r"""Command parser.

    To parse a command with tokens `abc efg 123`, use
    `cmdparser.parse("abc").parse("efg").parse("123").finish()`.
    """

    def finish(self):
        r"""Finish parsing.

        Returns
        -------
        res : any
            The command object, usually is a function.

        Raises
        ------
        CommandUnfinishError
            If this command is unfinished.
        """
        raise NotImplementedError

    def parse(self, token):
        r"""Parse a token.

        Parameters
        ----------
        token : str
            The token to parse.

        Returns
        -------
        toke_type : TOKEN_TYPE
            The type of parsed token.
        next_parser : CommandParser
            The next step of parsing.

        Raises
        ------
        CommandParseError
            If parsing fails.
        """
        raise NotImplementedError

    def suggest(self, token):
        r"""Suggest some possible tokens related to the given token.

        Parameters
        ----------
        token : str
            The token.

        Returns
        -------
        suggestions : list of str
            A list of possible tokens. Use the suffix '\000' to mark token as
            complete.
        """
        raise NotImplementedError

    def desc(self):
        r"""Describe acceptable tokens.

        Returns
        -------
        description : str or None
            The description.
        """
        raise NotImplementedError

    def info(self, token):
        r"""Describe the value of the given token.

        Parameters
        ----------
        token : str
            The token.

        Returns
        -------
        description : str or None
            The description.
        """
        raise NotImplementedError

    def build_command(self, tokens):
        r"""Directly build a command from the given tokens.

        Parameters
        ----------
        tokens : list of str
            The tokens to parse.

        Returns
        -------
        cmd : object
            The command object.

        Raises
        ------
        CommandParseError
        CommandUnfinishError
        """
        parser = self
        for token in tokens:
            _, parser = parser.parse(token)
        res = parser.finish()
        return res

    def parse_command(self, tokens):
        r"""Parse a command.

        Parameters
        ----------
        tokens : list of str
            The tokens to parse.

        Returns
        -------
        types : list of TOKEN_TYPE
            The token types.
        res : object or CommandParseError or CommandUnfinishError
            The command object or the error.
        """
        parser = self
        types = []

        for token in tokens:
            try:
                type, parser = parser.parse(token)
            except CommandParseError as err:
                return types, err
            types.append(type)

        try:
            res = parser.finish()
        except CommandUnfinishError as err:
            return types, err
        else:
            return types, res

    def suggest_command(self, tokens, target):
        r"""Find some suggestions for a command.

        Parameters
        ----------
        tokens : list of str
            The tokens of parent command.
        target : str
            The token to find suggestion.

        Returns
        -------
        suggestion : list of str
        """
        parser = self
        for token in tokens:
            try:
                _, parser = parser.parse(token)
            except CommandParseError:
                return []
        return parser.suggest(target)

    def desc_command(self, tokens):
        r"""Describe a command.

        Parameters
        ----------
        tokens : list of str
            The tokens of command.

        Returns
        -------
        desc : str or None
        """
        parser = self
        for token in tokens:
            try:
                _, parser = parser.parse(token)
            except CommandParseError:
                return None
        return parser.desc()

    def info_command(self, tokens, target):
        r"""Describe a token of a command.

        Parameters
        ----------
        tokens : list of str
            The tokens of parent command.
        target : str
            The token to describe.

        Returns
        -------
        info : str or None
        """
        parser = self
        for token in tokens:
            try:
                _, parser = parser.parse(token)
            except CommandParseError:
                return None
        return parser.info(target)


class FunctionCommandParser(CommandParser):
    r"""Command parser for a function, which will result in partially applied function."""

    def __init__(self, func, args, kwargs, bound=None):
        r"""Constructor.

        Parameters
        ----------
        func : function
            The function to parse. Its signature should be in a simple form,
            like `func(a, b, c, d=1, e=2, f=3)`. The positional arguments are
            required, and keyword arguments are optional and unordered. This
            function will be interpreted as a command
            `func a_value b_value c_value --d d_value --e e_value --f f_value`.
        args : dict of function
        kwargs : dict of function
            The parsers function for positional arguments and keyword arguments.
            Parser function is a function of bound arguments that returning
            ArgumentParser: `parser_func(**self.bound)` is argument parser.
        bound : dict, optional
            The bound arguments.
        """
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
            raise CommandUnfinishError(msg)

        return functools.partial(self.func, **self.bound)

    def parse(self, token):
        # parse positional arguments
        if self.args:
            args = OrderedDict(self.args)
            name, parser_func = args.popitem(False)
            parser = parser_func(**self.bound)

            value = parser.parse(token)
            bound = {**self.bound, name: value}
            return (
                TOKEN_TYPE.ARGUMENT,
                FunctionCommandParser(self.func, args, self.kwargs, bound),
            )

        # parse keyword arguments
        if self.kwargs:
            kwargs = OrderedDict(self.kwargs)
            name = token[2:] if token.startswith("--") else None
            parser_func = kwargs.pop(name, None)

            if parser_func is None:
                msg = f"Unknown argument {token!r}"
                raise CommandParseError(
                    msg, token, ["--" + key for key in self.kwargs.keys()]
                )

            args = OrderedDict([(name, parser_func)])
            return (
                TOKEN_TYPE.KEYWORD,
                FunctionCommandParser(self.func, args, kwargs, self.bound),
            )

        # rest
        raise CommandParseError("Too many arguments")

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
            return None

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
    r"""Command parser for subcommands."""

    def __init__(self, parent):
        r"""Constructor.

        Parameters
        ----------
        parent : object
            The object with fields with command descriptors.
        """
        self.parent = parent
        self.fields = [
            k
            for k, v in type(parent).__dict__.items()
            if isinstance(v, CommandDescriptor)
        ]

    def finish(self):
        desc = self.desc()
        msg = "Unfinished command" + ("\n" + desc if desc is not None else "")
        raise CommandUnfinishError(msg)

    def parse(self, token):
        if token not in self.fields:
            msg = "Unknown command"
            raise CommandParseError(msg, token, self.fields)

        field = getcmd(self.parent, token)
        if not isinstance(field, CommandParser):
            raise CommandParseError("Not a command")

        return TOKEN_TYPE.COMMAND, field

    def suggest(self, token):
        return [val + "\000" for val in fit(token, self.fields)]

    def desc(self):
        return None

    def info(self, token):
        if token not in self.fields:
            return None
        return getcmddesc(self.parent, token)


class RootCommandParser(CommandParser):
    r"""Command parser for root commands."""

    def __init__(self, **parents):
        r"""Constructor.

        Parameters
        ----------
        parents : dict of object
            The objects with fields with command descriptors.
        """
        self.parents = parents
        self.fields = {
            k: group
            for group, parent in self.parents.items()
            for k, v in type(parent).__dict__.items()
            if isinstance(v, CommandDescriptor)
        }

    def get_all_groups(self):
        return list(self.parents.keys())

    def get_group(self, token):
        return self.fields.get(token, None)

    def finish(self):
        desc = self.desc()
        msg = "Unfinished command" + ("\n" + desc if desc is not None else "")
        raise CommandUnfinishError(msg)

    def parse(self, token):
        if token not in self.fields:
            msg = "Unknown command"
            raise CommandParseError(msg, token, self.fields.keys())

        group = self.fields[token]
        parent = self.parents[group]
        field = getcmd(parent, token)
        if not isinstance(field, CommandParser):
            raise CommandParseError("Not a command")

        return TOKEN_TYPE.COMMAND, field

    def suggest(self, token):
        return [val + "\000" for val in fit(token, self.fields.keys())]

    def desc(self):
        return None

    def info(self, token):
        if token not in self.fields:
            return None
        group = self.fields[token]
        parent = self.parents[group]
        return getcmddesc(parent, token)


def getcmd(obj, name):
    clz = type(obj)
    if name not in clz.__dict__:
        return None
    descriptor = clz.__dict__[name]
    if not isinstance(descriptor, CommandDescriptor):
        return None

    return descriptor.__get_command__(obj, clz)


def getcmddesc(obj, name):
    clz = type(obj)
    if name not in clz.__dict__:
        return None
    descriptor = clz.__dict__[name]
    if not isinstance(descriptor, CommandDescriptor):
        return None

    doc = descriptor.__doc__
    if doc is None:
        return None

    return inspect.cleandoc(doc)


class CommandDescriptor:
    r"""Command descriptor."""

    def __init__(self, proxy):
        self.proxy = proxy
        functools.update_wrapper(self, proxy)

    def __get__(self, instance, owner):
        return self.proxy.__get__(instance, owner)

    def __get_command__(self, instance, owner):
        raise NotImplementedError


class subcommand(CommandDescriptor):
    r"""Command descriptor for subcommands."""

    def __init__(self, proxy):
        super(subcommand, self).__init__(property(proxy))

    def __get_command__(self, instance, owner):
        parent = self.proxy.__get__(instance, owner)
        return SubCommandParser(parent)


class function_command(CommandDescriptor):
    r"""Command descriptor for function command."""

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
                raise ValueError(
                    f"No parser for argument {param.name} in {self.proxy.__name__}"
                )

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
        r"""Assign a parser function to an argument of this command.

        Parameters
        ----------
        name : str
            The name of argument to bind.

        Returns
        -------
        arg_parser_dec : function
        """

        def arg_parser_dec(parser):
            self.parsers[name] = parser
            return parser

        return arg_parser_dec

