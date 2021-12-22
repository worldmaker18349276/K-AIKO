"""
A universal Python parser combinator library inspired by Parsec library of Haskell.

This is a fork of https://github.com/sighingnow/parsec.py.
"""

import re
import functools
from typing import Any
import dataclasses


class ParseError(Exception):
    """A class of parse error to explain where and why."""

    def __init__(self, text, index, expected):
        """Create `ParseError` object.

        Parameters
        ----------
        text : str
            The input string to parse.
        index : int
            The position failed to parse.
        expected : str
            The description of expected string.
        """
        self.text = text
        self.index = index
        self.expected = expected

    @staticmethod
    def locate(text, index):
        """Locate the position of text by index.

        Parameters
        ----------
        text : str
            The string to locate.
        index : int
            The index of position.

        Returns
        -------
        tuple of int and int
            Line and column of index in the text.

        Raises
        ------
        ValueError
            If the index is out of bounds.
        """
        if index > len(text):
            raise ValueError("Invalid index.")
        line = text.count("\n", 0, index)
        last_ln = text.rfind("\n", 0, index)
        col = index - (last_ln + 1)
        return (line, col)

    def __str__(self):
        """Represent ParseError as readable text.

        Returns
        -------
        str
            The description of this error.
        """
        if self.index >= len(self.text):
            return f"<out of bounds index {self.index}>"
        line, col = ParseError.locate(self.text, self.index)
        return f"expected: {self.expected} at {line}:{col}"


@dataclasses.dataclass(frozen=True)
class Success:
    """The success state of parsing."""

    index: int
    value: Any


@dataclasses.dataclass(frozen=True)
class Failure:
    """The failure state of parsing."""

    index: int
    expected: str


def parsec(func):
    """Make a parser function in generator form.

    Yield parser to parse text, and the result will be send back.
    Use `return` to return parsing result.

    Parameters
    ----------
    func : function
        The generator function of parser in generator form.

    Returns
    -------
    function
        The parser function.
    """

    @functools.wraps(func)
    def parser_func(*args, **kwargs):
        def parser(text, index):
            it = func(*args, **kwargs)
            value = None
            while True:
                try:
                    parser = it.send(value)
                except StopIteration as e:
                    return Success(index, e.value)

                res = parser(text, index)
                if isinstance(res, Failure):
                    return res
                elif isinstance(res, Success):
                    value, index = res.value, res.index
                else:
                    assert False

        return Parsec(parser)
    return parser_func


class Parsec:
    """The parser combinator.

    Parsec is an object that wrap up all functionality of parser combinator
    in a pythonic way.  It provides monadic methods to manipulate the parsing
    state and result values.  Parser can be built up by simple atomic parsers
    and combinators very easily, so there is no need to manage parser state
    by hand.  But the flaws are hard to backtrack and easy to encounter infinite
    loop.  Nevertheless, it is still very powerful for writing simple parsers.
    """

    def __init__(self, func):
        """Create Parsec by function.

        Parameters
        ----------
        func : function
            The function that do the parsing work.
            Arguments of this function should be a string to be parsed and
            the index on which to begin parsing.
            The function should return either `Success(next_index, value)` if
            parsing successfully, or `Failure(index, expected)` on the failure.
        """
        self.func = func

    def parse(self, text, ret_rest=False):
        """Parse the longest possible prefix of a given string.

        If you want to parse strictly, use `(self << Parsec.eof()).parse(text)`.

        Parameters
        ----------
        text : str
            The string to parse.
        ret_rest : bool
            Return or not the rest of the string.

        Returns
        -------
        result : any
            The result produced by this parser.
        rest : str, optional
            The rest of the string, if `ret_rest` is true.

        Raises
        ------
        ParseError
        """
        res = self.func(text, 0)
        if isinstance(res, Failure):
            raise ParseError(text, res.index, res.expected)
        if ret_rest:
            return (res.value, text[res.index:])
        else:
            return res.value

    # basic

    def bind(self, func):
        """Bind a parser function with the result of this parser.

        If parser is successful, passes the result to `func`, and continues
        with the parser returned from `func`.
        In generator form, it looks like::

            res1 = yield self
            other = func(res1)
            res2 = yield other
            return res2

        Parameters
        ----------
        func : function
            The function eats previous result and produce the next parser.

        Returns
        -------
        Parsec
        """
        def bind_parser(text, index):
            res = self.func(text, index)
            if isinstance(res, Failure):
                return res
            other = func(res.value)
            return other.func(text, res.index)
        return Parsec(bind_parser)

    def map(self, func):
        """Modify the result value by a function.

        If parser is success, transforms the produced value of parser with `func`.
        In generator form, it looks like::

            res = yield self
            return func(res)

        Parameters
        ----------
        func : function
            The function to transform the result.

        Returns
        -------
        Parsec
        """
        def map_parser(text, index):
            res = self.func(text, index)
            if not isinstance(res, Success):
                return res
            return Success(res.index, func(res.value))
        return Parsec(map_parser)

    def starmap(self, func):
        """Transform parser results, which should be wrapped up in a tuple, by a function.

        It is the same as `map` except spreading.
        In generator form, it looks like::

            res = yield self
            return func(*res)

        Parameters
        ----------
        func : function
            The function to transform the result.

        Returns
        -------
        Parsec
        """
        def map_parser(text, index):
            res = self.func(text, index)
            if not isinstance(res, Success):
                return res
            return Success(res.index, func(*res.value))
        return Parsec(map_parser)

    def then(self, other):
        """Sequentially compose two parser, discarding any value produced by the first.

        In generator form, it looks like::

            res1 = yield self
            res2 = yield other
            return res2

        Parameters
        ----------
        other : Parsec
            The parser followed by.

        Returns
        -------
        Parsec
        """
        return self.bind(lambda _: other)

    def skip(self, other):
        """Sequentially compose two parser, discarding any value produced by the second.

        In generator form, it look like::

            res1 = yield self
            res2 = yield other
            return res1

        Parameters
        ----------
        other : Parsec
            The parser followed by.

        Returns
        -------
        Parsec
        """
        def skip_parser(text, index):
            res = self.func(text, index)
            if isinstance(res, Failure):
                return res
            end = other.func(text, res.index)
            if isinstance(end, Failure):
                return end
            return Success(end.index, res.value)
        return Parsec(skip_parser)

    def choice(self, *others):
        """Parse others if this parser failed without consuming any input.

        If you need backtracking, apply `attempt` first.
        Depend on the input text, it looks like::

            res1 = yield self
            return res1

        if parsing `self` successfully.  Or::

            res2 = yield other
            return res2

        Parameters
        ----------
        *others : list of Parsec
            The parsers to combine.

        Returns
        -------
        Parsec
        """
        def choice_parser(text, index):
            res = self.func(text, index)
            if isinstance(res, Success) or res.index != index:
                return res
            results = [res]

            for other in others:
                res = other.func(text, index)
                if isinstance(res, Success) or res.index != index:
                    return res
                results.append(res)

            expected = " or ".join(res.expected for res in results)
            return Failure(res.index, expected)

        return Parsec(choice_parser)

    def concat(self, *others):
        """Concatenate multiple parsers into one.  Return a list of results they produced.

        In generator form, it looks like::

            res1 = yield self
            res2 = yield other
            return [res1, res2]

        Parameters
        ----------
        *others : list of Parsec
            The parsers to concatenate.

        Returns
        -------
        Parsec
        """
        def concat_parser(text, index):
            res = self.func(text, index)
            if isinstance(res, Failure):
                return res
            results = []

            for other in others:
                res = other.func(text, res.index)
                if isinstance(res, Failure):
                    return res
                results.append(res)

            result = tuple(res.value for res in results)
            return Success(res.index, result)

        return Parsec(concat_parser)

    def __or__(self, other):
        """`p | q` means `p.choice(q)`.

        See Also
        --------
        choice
        """
        return self.choice(other)

    def __add__(self, other):
        """`p + q` means `p.concat(q)`.

        See Also
        --------
        concat
        """
        return self.concat(other)

    def __rshift__(self, other):
        """`p >> q` means `p.then(q)`.

        See Also
        --------
        then
        """
        return self.then(other)

    def __lshift__(self, other):
        """`p << q` means `p.skip(q)`.

        See Also
        --------
        skip
        """
        return self.skip(other)

    # behaviors

    def result(self, res):
        """Change the result value when successful.

        In generator form, it looks like::

            res_ = yield self
            return res

        Parameters
        ----------
        res : any
            The result value to return.

        Returns
        -------
        Parsec
        """
        return self >> Parsec.succeed(res)

    def desc(self, expected):
        """Describe expected string of this parser, which will be reported on failure.

        Parameters
        ----------
        expected : str
            The expected string of this parser.

        Returns
        -------
        Parsec
        """
        return self | Parsec.fail(expected)

    def ahead(self):
        """Parse string by looking ahead.

        It will parse without consuming any input if success,
        but still consume input on failure.

        Returns
        -------
        Parsec
        """
        def lookahead_parser(text, index):
            res = self.func(text, index)
            if isinstance(res, Failure):
                return res
            return Success(index, res.value)
        return Parsec(lookahead_parser)

    def attempt(self):
        """Backtrack to the original position.

        It will not consume any input when it failed,
        so that one can parse another case by `choice`.

        Returns
        -------
        Parsec
        """
        def attempt_parser(text, index):
            res = self.func(text, index)
            if isinstance(res, Success):
                return res
            return Failure(index, f"({text[index:res.index]!r} followed by {res.expected})")
        return Parsec(attempt_parser)

    def optional(self):
        """Make a parser as optional.

        It will wrap up the result value in a tuple,
        and return empty tuple on failure.

        Returns
        -------
        Parsec
        """
        def option_parser(text, index):
            res = self.func(text, index)
            if isinstance(res, Success):
                return Success(res.index, (res.value,))
            if isinstance(res, Failure) and res.index == index:
                return Success(index, ())
            return res
        return Parsec(option_parser)

    # atomic

    @staticmethod
    def succeed(res):
        """Create a parser that always succeeds.

        Parameters
        ----------
        res : any
            The result value.

        Returns
        -------
        Parsec
        """
        return Parsec(lambda _, index: Success(index, res))

    @staticmethod
    def fail(expected):
        """Create a parser that always fails.

        Parameters
        ----------
        expected : str
            The description for expected string.

        Returns
        -------
        Parsec
        """
        return Parsec(lambda _, index: Failure(index, expected))

    @staticmethod
    def any():
        """Parse an arbitrary character.

        Returns
        -------
        Parsec
        """
        def any_parser(text, index=0):
            if index < len(text):
                return Success(index + 1, text[index])
            else:
                return Failure(index, "a random char")
        return Parsec(any_parser)

    @staticmethod
    def oneOf(chars):
        """Parse a character from specified string.

        Parameters
        ----------
        chars : str
            The valid characters.

        Returns
        -------
        Parsec
        """
        def one_of_parser(text, index=0):
            if index < len(text) and text[index] in chars:
                return Success(index + 1, text[index])
            else:
                return Failure(index, f"one of {repr(chars)}")
        return Parsec(one_of_parser)

    @staticmethod
    def noneOf(chars):
        """Parse a character not from specified string.

        Parameters
        ----------
        chars : str
            The invalid characters.

        Returns
        -------
        Parsec
        """
        def none_of_parser(text, index=0):
            if index < len(text) and text[index] not in chars:
                return Success(index + 1, text[index])
            else:
                return Failure(index, f"none of {repr(chars)}")
        return Parsec(none_of_parser)

    @staticmethod
    def satisfy(validater, desc="some condition"):
        """Parse a character validated by specified function.

        Parameters
        ----------
        validater : function
            The function to validate input.
        desc : str, optional
            The description of validater, by default `"some condition"`.

        Returns
        -------
        Parsec
        """
        def satisfy_parser(text, index=0):
            if index < len(text) and validater(text[index]):
                return Success(index + 1, text[index])
            else:
                return Failure(index, f"a character that satisfy {desc}")
        return Parsec(satisfy_parser)

    @staticmethod
    def eof():
        """Parse EOF.  The result value is `""`.

        Returns
        -------
        Parsec
        """
        def eof_parser(text, index=0):
            if index >= len(text):
                return Success(index, "")
            else:
                return Failure(index, "EOF")
        return Parsec(eof_parser)

    @staticmethod
    def regex(exp, flags=0):
        """Parse according to a regular expression.

        Parameters
        ----------
        exp : str or regular expression object
            The regular expression to parse.
        flags : int, optional
            The flag of regular expression, by default 0.

        Returns
        -------
        Parsec
        """
        if isinstance(exp, str):
            exp = re.compile(exp, flags)

        def regex_parser(text, index):
            match = exp.match(text, index)
            if match:
                return Success(match.end(), match.group(0))
            else:
                return Failure(index, f"/{exp.pattern}/")
        return Parsec(regex_parser)

    # combinators

    @parsec
    def join(self, parsers):
        """Join parsers just like `str.join`.  Return a tuple of result values of `self`.

        Parameters
        ----------
        parsers : iterable of Parsec
            The parsers to join.

        Returns
        -------
        Parsec
        """
        results = []
        for parser in parsers:
            if results:
                yield self
            res = yield parser
            results.append(res)
        return tuple(results)

    @parsec
    def between(self, opening, closing):
        """Enclose a parser 0 by two parsers.  Return the result value of `self`.

        Parameters
        ----------
        opening : Parsec
            The opening parser.
        closing : Parsec
            The closing parser.

        Returns
        -------
        Parsec
        """
        yield opening
        res = yield self
        yield closing
        return res

    @parsec
    def times(self, n, m=None):
        """Repeat a parser n to m times.  Return a list of result values of `self`.

        Parameters
        ----------
        n : int
            The number of repeating.
        m : int, optional
            The maximum number of repeating, by default n.

        Returns
        -------
        Parsec
        """
        if m is None:
            m = n
        optional_self = self.optional()

        results = []

        for _ in range(n):
            results.append((yield self))

        for _ in range(m-n):
            res = yield optional_self
            if not res:
                break
            results.append(res[0])

        return results

    @staticmethod
    def checkForward():
        """Create a parser to check for infinite pattern on every call.

        When this parser is called twice at the same position, it will raise
        a `ValueError`.
        This parser should be inserted into an infinite loop without state,
        so if the position of each loop does not change, it can be confirmed
        that an infinite loop is encountered.

        Returns
        -------
        Parsec
        """
        prev = None

        def check(text, index):
            nonlocal prev
            if prev == index:
                loc = "{}:{}".format(*ParseError.locate(text, index))
                raise ValueError(f"Infinite pattern happen at {loc}")
            prev = index
            return Success(index, None)

        return Parsec(check)

    @parsec
    def many(self):
        """Repeat a parser as much as possible.  Return a list of result values of `self`.

        Returns
        -------
        Parsec
        """
        check = Parsec.checkForward()
        optional_self = self.optional()
        results = []
        while True:
            maybe_res = yield optional_self
            if not maybe_res:
                return results
            yield check
            results.append(maybe_res[0])

    @parsec
    def manyTill(self, end):
        """Repeat a parser untill `end` succeed.  Return a list of result values of `self`.

        Parameters
        ----------
        end : Parsec
            The parser to stop.

        Returns
        -------
        Parsec
        """
        check = Parsec.checkForward()
        optional_end = end.optional()
        results = []
        while True:
            maybe_res = yield optional_end
            if maybe_res:
                return results
            results.append((yield self))
            yield check

    @parsec
    def sepBy(self, sep):
        """Repeat a parser and separated by `sep`.  Return a list of result values of `self`.

        Parameters
        ----------
        sep : Parsec
            The parser to interpolate.

        Returns
        -------
        Parsec
        """
        res1 = yield self.optional()
        if not res1:
            return []
        res2 = yield (sep >> self).many()
        return [*res1, *res2]

    @parsec
    def sepBy1(self, sep):
        """Repeat a parser at least once, separated by `sep`.  Return a list of result values of `self`.

        Parameters
        ----------
        sep : Parsec
            The parser to interpolate.

        Returns
        -------
        Parsec
        """
        res1 = yield self
        res2 = yield (sep >> self).many()
        return [res1, *res2]

    @parsec
    def sepEndBy(self, sep):
        """Repeat a parser and separated/optionally end by `sep`.  Return a list of result values of `self`.

        Parameters
        ----------
        sep : Parsec
            The parser to interpolate.

        Returns
        -------
        Parsec
        """
        check = Parsec.checkForward()
        results = []
        while True:
            maybe_res = yield self.optional()
            if not maybe_res:
                return results
            yield check
            results.append(maybe_res[0])
            maybe_res = yield sep.optional()
            if not maybe_res:
                return results
