from fractions import Fraction
import dataclasses
from typing import List, Tuple, Dict, Union, Optional
import ast
from ..utils import parsec as pc


Value = Union[None, bool, int, Fraction, float, str]
Arguments = Tuple[List[Value], Dict[str, Value]]


class PatternError(Exception):
    pass


class AST:
    pass


@dataclasses.dataclass
class Comment(AST):
    # # abc

    comment: str


@dataclasses.dataclass
class Metadata(AST):
    # #@TITLE: 123

    title: str
    content: Optional[str]


@dataclasses.dataclass
class Symbol(AST):
    # XYZ(arg=...)

    symbol: str
    arguments: Arguments


@dataclasses.dataclass
class Text(AST):
    # "ABC"(arg=...)

    text: str
    arguments: Arguments


@dataclasses.dataclass
class Lengthen(AST):
    # ~

    pass


@dataclasses.dataclass
class Measure(AST):
    # |

    pass


@dataclasses.dataclass
class Rest(AST):
    # _

    pass


@dataclasses.dataclass
class Division(AST):
    # [x o]
    # [x x o]/3

    divisor: int = 2
    patterns: List[AST] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Instant(AST):
    # {x x o}
    patterns: List[AST] = dataclasses.field(default_factory=list)


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
@pc.parsec
def comment_parser():
    comment = yield pc.regex(r"#[^\n]*[\n$]").desc("comment")
    comment = comment[1:].rstrip("\n")
    if comment.startswith("@"):
        metadata = comment[1:].split(":", 1)
        title = metadata[0]
        content = metadata[1] if len(metadata) >= 2 else None
        return Metadata(title, content)
    else:
        return Comment(comment)


@IIFE
def note_parser():
    symbol = pc.regex(r"[^ \b\t\n\r\f\v()[\]{}\'\"\\#]+")
    text = pc.regex(
        r'"([^\r\n\\"\x00]|\\[\\"btnrfv]|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})*"'
    ).map(ast.literal_eval)

    def make_note(sym, arg):
        if sym == "~":
            if arg[0] or arg[1]:
                raise PatternError("lengthen note don't accept any argument")
            return Lengthen()

        elif sym == "|":
            if arg[0] or arg[1]:
                raise PatternError("measure note don't accept any argument")
            return Measure()

        elif sym == "_":
            if arg[0] or arg[1]:
                raise PatternError("rest note don't accept any argument")
            return Rest()

        else:
            return Symbol(sym, arg)

    symbol_parser = (symbol + arguments_parser).starmap(make_note)
    text_parser = (text + arguments_parser).starmap(Text)
    return symbol_parser | text_parser


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
    msp = pc.regex(r"([ \t\n$])+").desc("whitespace")
    div = pc.regex(r"/(\d+)").map(lambda m: int(m[1:])) | pc.nothing(2)

    instant = enclose_by(
        pc.proxy(lambda: pattern), msp, pc.string("{"), pc.string("}")
    ).map(Instant)
    division = (
        enclose_by(pc.proxy(lambda: pattern), msp, pc.string("["), pc.string("]")) + div
    ).starmap(lambda a, b: Division(b, a))
    pattern = instant | division | note_parser | comment_parser
    return enclose_by(pattern, msp, pc.nothing(), end)


@dataclasses.dataclass
class Note:
    symbol: str
    beat: Fraction
    length: Fraction
    arguments: Arguments


def collect(it):
    res = []
    while True:
        try:
            value = next(it)
        except StopIteration as e:
            return res, e.value
        else:
            res.append(value)


def to_notes(patterns, beat=0, length=1):
    def build(beat, length, last_note, patterns):
        for pattern in patterns:
            if isinstance(pattern, Division):
                beat, last_note = yield from build(
                    beat, length / pattern.divisor, last_note, pattern.patterns
                )

            elif isinstance(pattern, Instant):
                if last_note is not None:
                    yield last_note
                last_note = None

                beat, last_note = yield from build(
                    beat, Fraction(0, 1), last_note, pattern.patterns
                )

            elif isinstance(pattern, Lengthen):
                if last_note is not None:
                    last_note.length += length
                beat += length

            elif isinstance(pattern, Measure):
                pass

            elif isinstance(pattern, Rest):
                if last_note is not None:
                    yield last_note
                last_note = None
                beat += length

            elif isinstance(pattern, (Text, Symbol)):
                if isinstance(pattern, Symbol):
                    symbol = pattern.symbol
                    args = pattern.arguments[0]
                    kw = pattern.arguments[1]
                else:
                    symbol = "Text"
                    args = (pattern.text, *pattern.arguments[0])
                    kw = pattern.arguments[1]

                if last_note is not None:
                    yield last_note
                last_note = Note(symbol, beat, length, (args, kw))
                beat += length

            elif isinstance(pattern, (Comment, Metadata)):
                pass

            else:
                raise TypeError

        return beat, last_note

    beat = Fraction(1, 1) * beat
    length = Fraction(1, 1) * length
    last_note = None

    notes, (last_beat, last_note) = collect(build(beat, length, last_note, patterns))
    if last_note is not None:
        notes.append(last_note)

    return notes, last_beat


def snap_to_frac(value, epsilon=0.001, N=float("inf")):
    d = 2
    count = 0
    while True:
        w = epsilon * 0.5 ** d / 2
        for n in range(1, d):
            if n / d - w < value < n / d + w:
                return Fraction(n, d)
            count += 1
            if count > N:
                raise ValueError(f"cannot snap {value} to any fraction")
        d += 1


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

        elif isinstance(pattern, Symbol):
            items.append(pattern.symbol + format_arguments(*pattern.arguments))

        else:
            assert False

    return " ".join(items)
