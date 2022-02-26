from fractions import Fraction
import dataclasses
from typing import List, Tuple, Dict, Union
import ast
from ..utils import parsec as pc


Value = Union[None, bool, int, Fraction, float, str]
Arguments = Tuple[List[Value], Dict[str, Value]]


class Pattern:
    pass


@dataclasses.dataclass
class Note(Pattern):
    #    event note: XXX(arg=...)
    #     text note: "ABC"(arg=...)
    # lengthen note: ~
    #  measure note: |
    #     rest note: _

    symbol: str
    arguments: Arguments


@dataclasses.dataclass
class Division(Pattern):
    # [x o]
    # [x x o]/3

    divisor: int = 2
    patterns: List[Pattern] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Instant(Pattern):
    # {x x o}
    patterns: List[Pattern] = dataclasses.field(default_factory=list)


def IIFE(func):
    return func()


@IIFE
def value_parser():
    none = pc.regex(r"None").map(ast.literal_eval)
    bool = pc.regex(r"True|False").map(ast.literal_eval)
    int = pc.regex(r"[-+]?(0|[1-9][0-9]*)").map(ast.literal_eval)
    frac = pc.regex(r"[-+]?(0|[1-9][0-9]*)\/[1-9][0-9]*").map(Fraction)
    float = pc.regex(r"[-+]?([0-9]+\.[0-9]+(e[-+]?[0-9]+)?|[0-9]+[eE][-+]?[0-9]+)").map(
        ast.literal_eval
    )
    str = pc.regex(
        r'"([^\r\n\\"]|\\[\\"btnrfv]|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})*"'
    ).map(ast.literal_eval)

    desc = "None or bool or str or float or frac or int"
    return pc.choice(none, bool, str, float, frac, int).desc(desc)


@IIFE
@pc.parsec
def arguments_parser():
    key = pc.regex(r"([a-zA-Z_][a-zA-Z0-9_]*)=").desc("'key='").map(lambda k: k[:-1])
    opening = pc.string("(").optional()
    closing = pc.string(")").optional()
    comma = pc.string(", ")

    psargs = []
    kwargs = {}
    keyworded = False

    if not (yield opening):
        return psargs, kwargs
    if (yield closing):
        return psargs, kwargs

    while True:
        try:
            keyword = yield key
        except pc.ParseFailure:
            if keyworded:
                raise
            keyword = None
        value = yield value_parser

        if keyword is not None:
            keyworded = True
            kwargs[keyword] = value
        else:
            psargs.append(value)

        if (yield closing):
            return psargs, kwargs
        yield comma


@IIFE
def note_parser():
    symbol = pc.regex(r"[^ \b\t\n\r\f\v()[\]{}\'\"\\#]+")
    text = pc.regex(
        r'"([^\r\n\\"\x00]|\\[\\"btnrfv]|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})*"'
    ).map(ast.literal_eval)
    return (symbol + arguments_parser).starmap(Note) | (
        text + arguments_parser
    ).starmap(lambda text, arg: Note("Text", ([text, *arg[0]], arg[1])))


@pc.parsec
def enclose_by(elem, sep, opening, closing):
    yield opening
    yield sep.optional()

    closing_optional = closing.optional()
    results = []

    while True:
        if (yield closing_optional):
            break
        results.append((yield elem))
        if (yield closing_optional):
            break
        yield sep

    return results


@IIFE
def patterns_parser():
    end = pc.regex(r"[ \t\n]*$").desc("end of file")
    msp = pc.regex(r"([ \t\n$]|#[^\n]*[\n$])+").desc("whitespace")
    div = pc.regex(r"/(\d+)").map(lambda m: int(m[1:])) | pc.nothing(2)

    instant = enclose_by(
        pc.proxy(lambda: pattern), msp, pc.string("{"), pc.string("}")
    ).map(Instant)
    division = (
        enclose_by(pc.proxy(lambda: pattern), msp, pc.string("["), pc.string("]")) + div
    ).starmap(lambda a, b: Division(b, a))
    pattern = instant | division | note_parser
    return enclose_by(pattern, msp, pc.nothing(), end)


class PatternError(Exception):
    pass


def to_events(patterns, beat=0, length=1, notations={}):
    def build(beat, length, last_event, patterns):
        for pattern in patterns:
            if isinstance(pattern, Division):
                beat, last_event = yield from build(
                    beat, length / pattern.divisor, last_event, pattern.patterns
                )

            elif isinstance(pattern, Instant):
                if last_event is not None:
                    yield last_event
                last_event = None

                beat, last_event = yield from build(
                    beat, Fraction(0, 1), last_event, pattern.patterns
                )

            elif pattern.symbol == "~":
                if pattern.arguments[0] or pattern.arguments[1]:
                    raise PatternError("lengthen note don't accept any argument")

                if last_event is not None:
                    last_event.length += length
                beat += length

            elif pattern.symbol == "|":
                if pattern.arguments[0] or pattern.arguments[1]:
                    raise PatternError("measure note don't accept any argument")

            elif pattern.symbol == "_":
                if pattern.arguments[0] or pattern.arguments[1]:
                    raise PatternError("rest note don't accept any argument")

                if last_event is not None:
                    yield last_event
                last_event = None
                beat += length

            else:
                if pattern.symbol not in notations:
                    raise PatternError("unknown symbol: " + pattern.symbol)

                if last_event is not None:
                    yield last_event
                event_type = notations[pattern.symbol]
                last_event = event_type(
                    beat, length, *pattern.arguments[0], **pattern.arguments[1]
                )
                beat += length

        return beat, last_event

    beat = Fraction(1, 1) * beat
    length = Fraction(1, 1) * length
    last_event = None

    events = []
    it = build(beat, length, last_event, patterns)
    while True:
        try:
            event = next(it)
        except StopIteration as e:
            last_beat, last_event = e.value
            break
        else:
            events.append(event)

    if last_event is not None:
        events.append(last_event)

    return events, last_beat


def format_value(value):
    if value is None:
        return "None"
    elif isinstance(value, (bool, int, float)):
        return repr(value)
    elif isinstance(value, Fraction):
        return str(value)
    elif isinstance(value, str):
        return (
            '"' + repr(value + '"')[1:-2].replace('"', r"\"").replace(r"\'", "'") + '"'
        )
    else:
        assert False


def format_arguments(psargs, kwargs):
    if len(psargs) + len(kwargs) == 0:
        return ""
    items = [format_value(value) for value in psargs]
    items += [key + "=" + format_value(value) for key, value in kwargs.items()]
    return "(%s)" % ", ".join(items)


def format_patterns(patterns):
    items = []
    for pattern in patterns:
        if isinstance(pattern, Instant):
            items.append("{%s}" % format_patterns(pattern.patterns))

        elif isinstance(pattern, Division):
            temp = "[%s]" if pattern.divisor == 2 else f"[%s]/{pattern.divisor}"
            items.append(temp % format_patterns(pattern.patterns))

        elif isinstance(pattern, Note):
            items.append(pattern.symbol + format_arguments(*pattern.arguments))

        else:
            assert False

    return " ".join(items)
