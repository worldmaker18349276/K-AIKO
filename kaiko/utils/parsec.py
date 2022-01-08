"""
A universal Python parser combinator library inspired by Parsec library of Haskell.

This is a fork of https://github.com/sighingnow/parsec.py.
"""

import re
import contextlib
import functools


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
        return (
            f"expecting {self.expected} at {line}:{col}\n"
            + "got (◊ indicate failure position):\n"
            + self.text[:self.index] + "◊" + self.text[self.index:]
        )


class ParseFailure(Exception):
    def __init__(self, expected):
        self.expected = expected

def parsec(func):
    """Make a parser function in generator form.

    Yield parser to parse text, and the result will be send back.
    Catch exception `ParseFailure` to deal with the failure.
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
            with contextlib.closing(it):
                # initializing
                try:
                    sub_parser = it.send(None)
                except StopIteration as stop:
                    return stop.value, index
                except ParseFailure as failure:
                    raise ParseError(text, index, failure.expected) from failure

                while True:
                    try:
                        value, index = sub_parser.func(text, index)

                    except ParseError as error:
                        if error.index != index:
                            raise error

                        # binding failure to the next parser
                        failure = ParseFailure(error.expected)
                        failure.__cause__ = error
                        try:
                            sub_parser = it.throw(ParseFailure, failure)
                        except StopIteration as stop:
                            return stop.value, index
                        except ParseFailure as failure:
                            raise ParseError(text, index, failure.expected) from failure

                    else:
                        # binding result to the next parser
                        try:
                            sub_parser = it.send(value)
                        except StopIteration as stop:
                            return stop.value, index
                        except ParseFailure as failure:
                            raise ParseError(text, index, failure.expected) from failure

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
            The function should return `(value, next_index)` if parsing
            successfully, or raise `ParseError` on the failure.
        """
        self.func = func

    def parse(self, text, ret_rest=False):
        """Parse the longest possible prefix of a given string.

        If you want to parse strictly, use `(self << eof()).parse(text)`.

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
        value, index = self.func(text, 0)
        if ret_rest:
            return value, text[index:]
        else:
            return value

    # basic

    def bind(self, successor=None, catcher=None):
        """Bind a parser function with the result of this parser.

        If the parsing is successful, passes the result to `successor`, and
        continues with the parser returned from `successor`.  If it fails,
        passes the expected message to `catcher`, and continues with the parser
        returned from `catcher`.
        In the generator form, it looks like::

            try:
                res1 = yield self
            except ParseFailure as failure:
                other = catcher(failure)
            else:
                other = successor(res1)
            res2 = yield other
            return res2

        Parameters
        ----------
        successor : function, optional
            The function eats previous result and produce the next parser.
        catcher : function, optional
            The function deal with previous failure and produce the next parser.

        Returns
        -------
        Parsec
        """
        def bind_parser(text, index):
            try:
                value, index = self.func(text, index)

            except ParseError as error:
                if error.index != index or catcher is None:
                    raise error
                other = catcher(ParseFailure(error.expected))
                return other.func(text, index)

            else:
                if successor is None:
                    return value, index
                other = successor(value)
                return other.func(text, index)
        return Parsec(bind_parser)

    def map(self, func):
        """Modify the result value by a function.

        If parser is success, transforms the produced value of parser with `func`.
        In the combinator form, it looks like::

            self.bind(lambda res: nothing(func(res)))

        In the generator form, it looks like::

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
            value, index = self.func(text, index)
            return func(value), index
        return Parsec(map_parser)

    def starmap(self, func):
        """Transform parser results, which should be wrapped up in a tuple, by a function.

        It is the same as `map` except spreading.
        In the combinator form, it looks like::

            self.map(lambda res: func(*res))

        In the generator form, it looks like::

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
        def starmap_parser(text, index):
            value, index = self.func(text, index)
            return func(*value), index
        return Parsec(starmap_parser)

    def then(self, other):
        """Sequentially compose two parser, discarding any value produced by the first.

        In the combinator form, it looks like::

            self.bind(lambda _: other)

        In the generator form, it looks like::

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
        def then_parser(text, index):
            _, index = self.func(text, index)
            value, index = other.func(text, index)
            return value, index
        return Parsec(then_parser)

    def skip(self, other):
        """Sequentially compose two parser, discarding any value produced by the second.

        In the combinator form, it looks like::

            self.bind(lambda res: other.result(res))

        In the generator form, it look like::

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
            value, index = self.func(text, index)
            _, index = other.func(text, index)
            return value, index
        return Parsec(skip_parser)

    def choice(self, *others):
        """Parse others if this parser failed without consuming any input.

        If you need backtracking, apply `attempt` first.
        In the combinator form, it looks like::

            self.bind(catcher=lambda _: other)

        In the generator form, it look like::

            try:
                res = yield self
            except ParseFailure:
                res = yield other
            return res

        Parameters
        ----------
        *others : list of Parsec
            The parsers to combine.

        Returns
        -------
        Parsec
        """
        def choice_parser(text, index):
            expected = []
            for parser in [self, *others]:
                try:
                    return parser.func(text, index)
                except ParseError as error:
                    if error.index != index:
                        raise error
                    expected.append(error.expected)

            raise ParseError(text, index, " or ".join(expected))

        return Parsec(choice_parser)

    def concat(self, *others):
        """Concatenate multiple parsers into one.  Return a tuple of results they produced.

        In the combinator form, it looks like::

            self.bind(lambda res1: other.map(lambda res2: (res1, res2)))

        In the generator form, it looks like::

            res1 = yield self
            res2 = yield other
            return (res1, res2)

        Parameters
        ----------
        *others : list of Parsec
            The parsers to concatenate.

        Returns
        -------
        Parsec
        """
        def concat_parser(text, index):
            results = []
            for parser in [self, *others]:
                value, index = parser.func(text, index)
                results.append(value)
            return tuple(results), index
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

        In the combinator form, it looks like::

            self.bind(lambda _: nothing(res))

        In the generator form, it looks like::

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
        def result_parser(text, index):
            _, index = self.func(text, index)
            return res, index
        return Parsec(result_parser)

    def desc(self, expected):
        """Describe expected string of this parser, which will be reported on failure.

        In the combinator form, it looks like::

            self.bind(catcher=lambda _: fail(expected))

        In the generator form, it looks like::

            try:
                res = yield self
            except ParseFailure:
                raise ParseFailure(expected)
            else:
                return res

        Parameters
        ----------
        expected : str
            The expected string of this parser.

        Returns
        -------
        Parsec
        """
        def desc_parser(text, index):
            try:
                value, index = self.func(text, index)
            except ParseError as error:
                if error.index == index:
                    error.expected = expected
                raise error
            else:
                return value, index
        return Parsec(desc_parser)

    def ahead(self):
        """Parse string by looking ahead.

        It will parse without consuming any input if success,
        but still consume input on failure.

        Returns
        -------
        Parsec
        """
        def ahead_parser(text, index):
            value, _ = self.func(text, index)
            return value, index
        return Parsec(ahead_parser)

    def attempt(self):
        """Backtrack to the original position.

        It will not consume any input when it failed,
        so that one can parse another case by `choice`.

        Returns
        -------
        Parsec
        """
        def attempt_parser(text, index):
            try:
                return self.func(text, index)
            except ParseError as error:
                if error.index != index:
                    expected = f"({text[index:error.index]!r} followed by {error.expected})"
                    raise ParseError(text, index, expected) from error
                raise error
        return Parsec(attempt_parser)

    def reject(self, func):
        """Reject the result value at the original position.

        Parameters
        ----------
        func : function
            The function to determine whether to reject result value by
            returning string as expected message, or None for valid value.

        Returns
        -------
        Parsec
        """
        def reject_parser(text, index):
            res, next_index = self.func(text, index)
            expected = func(res)
            if expected is not None:
                raise ParseError(text, index, expected)
            return res, next_index
        return Parsec(reject_parser)

    def optional(self):
        """Make a parser as optional.

        It will wrap up the result value in a tuple,
        and return empty tuple on failure.
        In the combinator form, it looks like::

            self.bind(
                lambda res: nothing((res,)),
                lambda _: nothing(()),
            )

        In the generator form, it looks like::

            try:
                res = yield self
            except ParseFailure:
                return ()
            else:
                return (res,)

        Returns
        -------
        Parsec
        """
        def optional_parser(text, index):
            try:
                value, index = self.func(text, index)
            except ParseError as error:
                if error.index != index:
                    raise error
                return (), index
            else:
                return (value,), index
        return Parsec(optional_parser)

    # atomic

    @staticmethod
    def fail(expected):
        """Create a parser that always fails.

        In the generator form, it looks like::

            raise ParseFailure(expected)

        Parameters
        ----------
        expected : str
            The description for expected string.

        Returns
        -------
        Parsec
        """
        def fail_parser(text, index):
            raise ParseError(text, index, expected)
        return Parsec(fail_parser)

    @staticmethod
    def nothing(res=None):
        """Create a parser that parse nothing.

        In the generator form, it looks like::

            return res

        Parameters
        ----------
        res : any, optional
            The parsing result, by default None.

        Returns
        -------
        Parsec
        """
        def nothing_parser(text, index):
            return res, index
        return Parsec(nothing_parser)

    # combinators

    def join(self, parsers):
        """Join parsers just like `str.join`.  Return a tuple of result values of `self`.

        In the combinator form, it looks like::

            parsers[0].concat(*[self >> parser for parser in parsers[1:]]).map(tuple)

        In the generator form, it looks like::

            results = []
            is_first = True
            for parser in parsers:
                if is_first:
                    yield self
                is_first = False
                results.append((yield parser))
            return tuple(results)

        Parameters
        ----------
        parsers : sequence of Parsec
            The parsers to join.

        Returns
        -------
        Parsec
        """
        def join_parser(text, index):
            results = []
            is_first = True

            for parser in parsers:
                if not is_first:
                    _, index = self.func(text, index)
                is_first = False

                value, index = parser.func(text, index)
                results.append(value)

            return tuple(results), index
        return Parsec(join_parser)

    def between(self, opening, closing):
        """Enclose a parser by `opening` and `closing`.  Return the result value of `self`.

        In the combinator form, it looks like::

            opening >> self << closing

        In the generator form, it looks like::

            yield opening
            res = yield self
            yield closing
            return res

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
        def between_parser(text, index):
            _, index = opening.func(text, index)
            value, index = self.func(text, index)
            _, index = closing.func(text, index)
            return value, index
        return Parsec(between_parser)

    def times(self, n, m=None):
        """Repeat a parser n to m times.  Return a list of result values of `self`.

        In the combinator form, it looks like::

            concat(*[self]*n).concat(*[self.optional()]*(m-n)).map(lambda res: [e for elem in res for e in elem])

        In the generator form, it looks like::

            results = []
            for i in range(n):
                try:
                    results.append((yield self))
                except ParseFailure:
                    if i < m:
                        raise
                    else:
                        break
            return results

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

        def times_parser(text, index):
            results = []
            for i in range(n):
                try:
                    value, index = self.func(text, index)
                except ParseError as error:
                    if error.index != index or i < m:
                        raise error
                    break
                results.append(value)
            return results, index
        return Parsec(times_parser)

    def many(self):
        """Repeat a parser as much as possible.  Return a list of result values of `self`.

        In the combinator form, it looks like::

            self.bind(
                lambda head: self.many().map(lambda tail: [head, *tail]),
                lambda _: nothing([])
            )

        In the generator form, it looks like::

            results = []
            while True:
                try:
                    results.append((yield self))
                except ParseFailure:
                    return results

        Returns
        -------
        Parsec
        """
        def many_parser(text, index):
            check = check_forward()
            results = []
            while True:
                try:
                    res, index = self.func(text, index)
                except ParseError as error:
                    if error.index != index:
                        raise error
                    return results, index
                else:
                    check.func(text, index)
                    results.append(res)
        return Parsec(many_parser)

    def many_till(self, end):
        """Repeat a parser until `end` succeed.  Return a list of result values of `self`.

        In the combinator form, it looks like::

            end.bind(
                lambda _: (self + self.many_till(end)).starmap(lambda head, tail: [head, *tail]),
                lambda _: nothing([])
            )

        In the generator form, it looks like::

            results = []
            while not (yield end.optional()):
                results.append((yield self))
            return results

        Parameters
        ----------
        end : Parsec
            The parser to stop.

        Returns
        -------
        Parsec
        """
        def many_till_parser(text, index):
            check = check_forward()
            results = []
            while True:
                try:
                    _, index = end.func(text, index)
                except ParseError as error:
                    if error.index != index:
                        raise error
                else:
                    return results, index

                res, index = self.func(text, index)
                check.func(text, index)
                results.append(res)
        return Parsec(many_till_parser)

    def sep_by(self, sep):
        """Repeat a parser and separated by `sep`.  Return a list of result values of `self`.

        In the combinator form, it looks like::

            self.bind(
                lambda head: (sep >> self).many().map(lambda tail: [head, *tail]),
                lambda _: nothing([])
            )

        In the generator form, it looks like::

            results = []

            try:
                results.append((yield self))
            except ParseFailure:
                return []

            while (yield sep.optional()):
                results.append((yield self))
            return results

        Parameters
        ----------
        sep : Parsec
            The parser to interpolate.

        Returns
        -------
        Parsec
        """
        def sep_by_parser(text, index):
            check = check_forward()
            results = []
            is_first = True

            while True:
                try:
                    res, index = self.func(text, index)
                    check.func(text, index)
                    results.append(res)
                except ParseError as error:
                    if error.index != index or not is_first:
                        raise error
                    return results, index

                is_first = False

                try:
                    _, index = sep.func(text, index)
                except ParseError as error:
                    if error.index != index:
                        raise error
                    return results, index

        return Parsec(sep_by_parser)

    def sep_by1(self, sep):
        """Repeat a parser at least once, separated by `sep`.  Return a list of result values of `self`.

        In the combinator form, it looks like::

            (self + (sep >> self).many()).starmap(lambda head, tail: [head, *tail])

        In the generator form, it looks like::

            results = []
            results.append((yield self))
            while (yield sep.optional()):
                results.append((yield self))
            return results

        Parameters
        ----------
        sep : Parsec
            The parser to interpolate.

        Returns
        -------
        Parsec
        """
        def sep_by1_parser(text, index):
            check = check_forward()
            results = []

            while True:
                res, index = self.func(text, index)
                check.func(text, index)
                results.append(res)

                try:
                    _, index = sep.func(text, index)
                except ParseError as error:
                    if error.index != index:
                        raise error
                    return results, index

        return Parsec(sep_by1_parser)

    def sep_end_by(self, sep):
        """Repeat a parser and separated/optionally end by `sep`.  Return a list of result values of `self`.

        In the combinator form, it looks like::

            self.bind(
                lambda head: sep.bind(
                                lambda _: self.sep_end_by(sep),
                                lambda _: nothing([]),
                            ).map(lambda tail: [head, *tail]),
                lambda _: nothing([])
            )

        In the generator form, it looks like::

            results = []
            try:
                while True:
                    results.append((yield self))
                    yield sep
            except ParseFailure:
                return results

        Parameters
        ----------
        sep : Parsec
            The parser to interpolate.

        Returns
        -------
        Parsec
        """
        def sep_end_by_parser(text, index):
            check = check_forward()
            results = []
            while True:
                try:
                    res, index = self.func(text, index)
                    check.func(text, index)
                    results.append(res)
                except ParseError as error:
                    if error.index != index:
                        raise error
                    return results, index

                try:
                    _, index = sep.func(text, index)
                except ParseError as error:
                    if error.index != index:
                        raise error
                    return results, index
        return Parsec(sep_end_by_parser)

    def sep_end_by1(self, sep):
        """Repeat a parser at least once and separated/optionally end by `sep`.  Return a list of result values of `self`.

        In the combinator form, it looks like::

            (
                self
                + sep.bind(
                    lambda _: self.sep_end_by(sep),
                    lambda _: nothing([])
                )
            ).starmap(lambda head, tail: [head, *tail])

        In the generator form, it looks like::

            results = []
            results.append((yield self))

            try:
                while True:
                    yield sep
                    results.append((yield self))
            except ParseFailure:
                return results

        Parameters
        ----------
        sep : Parsec
            The parser to interpolate.

        Returns
        -------
        Parsec
        """
        def sep_end_by1_parser(text, index):
            check = check_forward()
            results = []
            is_first = True

            while True:
                try:
                    res, index = self.func(text, index)
                    check.func(text, index)
                    results.append(res)
                except ParseError as error:
                    if error.index != index or is_first:
                        raise error
                    return results, index

                is_first = False

                try:
                    _, index = sep.func(text, index)
                except ParseError as error:
                    if error.index != index:
                        raise error
                    return results, index

        return Parsec(sep_end_by1_parser)

    def sep_by_till(self, sep, end):
        """Repeat a parser and separated by `sep` until `end` succeed.  Return a list of result values of `self`.

        In the combinator form, it looks like::

            end.bind(
                lambda _: nothing([])
                lambda _: (self + (sep >> self).many_till(end)).starmap(lambda head, tail: [head, *tail])
            )

        In the generator form, it looks like::

            results = []
            is_first = True
            while not (yield end.optional()):
                if is_first:
                    yield sep
                is_first = False
                results.append((yield self))
            return results

        Parameters
        ----------
        sep : Parsec
            The parser to interpolate.
        end : Parsec
            The parser to stop.

        Returns
        -------
        Parsec
        """
        def sep_by_till_parser(text, index):
            check = check_forward()
            results = []
            is_first = True

            while True:
                try:
                    _, index = end.func(text, index)
                except ParseError as error:
                    if error.index != index:
                        raise error
                else:
                    return results, index

                if is_first:
                    _, index = sep.func(text, index)
                is_first = False
                
                res, index = self.func(text, index)
                check.func(text, index)
                results.append(res)

        return Parsec(sep_by_till_parser)


def proxy(parser_getter):
    return Parsec(lambda text, index: parser_getter().func(text, index))

def choice(*parsers):
    if not parsers:
        raise ValueError("no choice")
    return parsers[0].choice(*parsers[1:])

def concat(*parsers):
    if not parsers:
        raise ValueError("nothing to concatenate")
    return parsers[0].concat(*parsers[1:])

def fail(expected):
    return Parsec.fail(expected)

def nothing(res=None):
    return Parsec.nothing(res)

def any():
    """Parse an arbitrary character.  It failed at the EOF.

    Returns
    -------
    Parsec
    """
    def any_parser(text, index):
        if index < len(text):
            return text[index], index + 1
        else:
            raise ParseError(text, index, "a random char")
    return Parsec(any_parser)

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
    def one_of_parser(text, index):
        if index < len(text) and text[index] in chars:
            return text[index], index + 1
        else:
            raise ParseError(text, index, f"one of {repr(chars)}")
    return Parsec(one_of_parser)

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
    def none_of_parser(text, index):
        if index < len(text) and text[index] not in chars:
            return text[index], index + 1
        else:
            raise ParseError(text, index, f"none of {repr(chars)}")
    return Parsec(none_of_parser)

def satisfy(validater, desc=None):
    """Parse a character validated by specified function.

    Parameters
    ----------
    validater : function
        The function to validate input.
    desc : str, optional
        The description of validater.

    Returns
    -------
    Parsec
    """
    if desc is None:
        desc = repr(validater)
    def satisfy_parser(text, index):
        if index < len(text) and validater(text[index]):
            return text[index], index + 1
        else:
            raise ParseError(text, index, f"a character that satisfy {desc}")
    return Parsec(satisfy_parser)

def eof():
    """Parse EOF.  The result value is `""`.

    Returns
    -------
    Parsec
    """
    def eof_parser(text, index):
        if index >= len(text):
            return "", index
        else:
            raise ParseError(text, index, "EOF")
    return Parsec(eof_parser)

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
            return match.group(0), match.end()
        else:
            raise ParseError(text, index, f"/{exp.pattern}/")
    return Parsec(regex_parser)

def tokens(tokens):
    """Try to match a list of strings.  Return the matching string.
    
    This method sorts the strings to prevent conflicts.  For example,
    it will try to match "letter" before "let", otherwise "letter"
    will be parsed as "let" with rest "ter".

    Parameters
    ----------
    tokens : sequence of str
        The strings to match.

    Returns
    -------
    Parsec
    """
    desc = " or ".join(repr(token) for token in tokens) or "nothing"
    tokens = sorted(tokens, reverse=True)

    def tokens_parser(text, index):
        for token in tokens:
            next_index = index + len(token)
            if text[index:next_index] == token:
                return token, next_index
        else:
            raise ParseError(text, index, desc)
    return Parsec(tokens_parser)

def check_forward():
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
        return None, index

    return Parsec(check)

