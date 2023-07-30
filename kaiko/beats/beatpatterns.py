from fractions import Fraction
import dataclasses
import collections
from typing import List, Tuple, Dict, Union, Optional
import math
import ast
from ..utils import parsec as pc


Value = Union[None, bool, int, Fraction, float, str]


@dataclasses.dataclass
class Arguments:
    ps: List[Value]
    kw: Dict[str, Value]


class PatternError(Exception):
    pass


class Pattern:
    pass


@dataclasses.dataclass
class Newline(Pattern):
    pass


@dataclasses.dataclass
class Comment(Pattern):
    # # abc

    comment: str


@dataclasses.dataclass
class Symbol(Pattern):
    # XYZ(arg=...)

    symbol: str
    arguments: Arguments


@dataclasses.dataclass
class Text(Pattern):
    # "ABC"(arg=...)

    text: str
    arguments: Arguments


@dataclasses.dataclass
class Lengthen(Pattern):
    # ~

    pass


@dataclasses.dataclass
class Measure(Pattern):
    # |

    pass


@dataclasses.dataclass
class Rest(Pattern):
    # _

    pass


@dataclasses.dataclass
class Subdivision(Pattern):
    # [x o]
    # [x x o]/3

    divisor: int = 2
    patterns: List[Pattern] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Chord(Pattern):
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
        return Arguments(psargs, kwargs)
    if (yield closing):
        return Arguments(psargs, kwargs)

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
            return Arguments(psargs, kwargs)
        yield comma


@IIFE
@pc.parsec
def comment_parser():
    comment = yield pc.regex(r"#[^\n]*(?=\n|$)").desc("comment")
    return Comment(comment[1:])


@IIFE
def note_parser():
    symbol = pc.regex(r"[^ \b\t\n\r\f\v()[\]{}\'\"\\#]+")
    text = pc.regex(
        r'"([^\r\n\\"\x00]|\\[\\"btnrfv]|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})*"'
    ).map(ast.literal_eval)

    def make_note(sym, arg):
        if sym == "~":
            if arg.ps or arg.kw:
                raise PatternError("lengthen note don't accept any argument")
            return Lengthen()

        elif sym == "|":
            if arg.ps or arg.kw:
                raise PatternError("measure note don't accept any argument")
            return Measure()

        elif sym == "_":
            if arg.ps or arg.kw:
                raise PatternError("rest note don't accept any argument")
            return Rest()

        else:
            return Symbol(sym, arg)

    symbol_parser = (symbol + arguments_parser).starmap(make_note)
    text_parser = (text + arguments_parser).starmap(Text)
    return symbol_parser | text_parser


@IIFE
def patterns_parser():
    msp = (
        pc.regex(r"( |\t|\n|$)+")
        .map(lambda s: [Newline()] * s.count("\n"))
        .desc("whitespace")
    )
    div = pc.regex(r"/(\d+)").map(lambda m: int(m[1:])) | pc.nothing(2)

    @pc.parsec
    def enclose_by(elem, opening, closing):
        closing_optional = closing.optional()
        results = []

        yield opening
        ln = yield msp.optional()
        if ln:
            results.extend(ln[0])

        while True:
            if (yield closing_optional):
                break
            results.append((yield elem))
            if (yield closing_optional):
                break
            ln = yield msp
            results.extend(ln)

        return results

    instant = enclose_by(pc.proxy(lambda: pattern), pc.string("{"), pc.string("}")).map(
        Chord
    )
    division = (
        enclose_by(pc.proxy(lambda: pattern), pc.string("["), pc.string("]")) + div
    ).starmap(lambda a, b: Subdivision(b, a))
    pattern = instant | division | note_parser | comment_parser
    return enclose_by(pattern, pc.nothing(), pc.eof())


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
            if isinstance(pattern, Subdivision):
                beat, last_note = yield from build(
                    beat, length / pattern.divisor, last_note, pattern.patterns
                )

            elif isinstance(pattern, Chord):
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
                    ps = pattern.arguments.ps
                    kw = pattern.arguments.kw
                else:
                    symbol = "Text"
                    ps = (pattern.text, *pattern.arguments.ps)
                    kw = pattern.arguments.kw

                if last_note is not None:
                    yield last_note
                last_note = Note(symbol, beat, length, Arguments(ps, kw))
                beat += length

            elif isinstance(pattern, Comment):
                if last_note is not None:
                    yield last_note
                last_note = Note(
                    "#", beat, Fraction(0, 1), Arguments([pattern.comment], {})
                )

            elif isinstance(pattern, Newline):
                pass

            else:
                raise TypeError(pattern)

        return beat, last_note

    beat = Fraction(1, 1) * beat
    length = Fraction(1, 1) * length
    last_note = None

    notes, (last_beat, last_note) = collect(build(beat, length, last_note, patterns))
    if last_note is not None:
        notes.append(last_note)

    return notes, last_beat


def parse_notes(patterns_str, beat=0, length=1):
    return to_notes(patterns_parser.parse(patterns_str), beat, length)


def snap_to_frac(value, epsilon=0.001, D=1024):
    for d in range(2, D):
        w = epsilon * 0.5**d / 2
        n = round(value * d)
        if n / d - w < value < n / d + w:
            return Fraction(n, d)
    else:
        raise ValueError(f"cannot snap {value} to any fraction")


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


def format_arguments(arguments):
    if len(arguments.ps) + len(arguments.kw) == 0:
        return ""
    items = [format_value(value) for value in arguments.ps]
    items += [key + "=" + format_value(value) for key, value in arguments.kw.items()]
    return "(%s)" % ", ".join(items)


def patterns_to_str(patterns):
    items = []
    for pattern in patterns:
        if isinstance(pattern, Chord):
            items.append("{%s}" % patterns_to_str(pattern.patterns))

        elif isinstance(pattern, Subdivision):
            temp = "[%s]" if pattern.divisor == 2 else f"[%s]/{pattern.divisor}"
            items.append(temp % patterns_to_str(pattern.patterns))

        elif isinstance(pattern, Symbol):
            items.append(pattern.symbol + format_arguments(pattern.arguments))

        elif isinstance(pattern, Text):
            items.append(
                format_value(pattern.text) + format_arguments(pattern.arguments)
            )

        elif isinstance(pattern, Lengthen):
            items.append("~")

        elif isinstance(pattern, Measure):
            items.append("|")

        elif isinstance(pattern, Rest):
            items.append("_")

        elif isinstance(pattern, Comment):
            items.append(f"\n#{pattern.comment}\n")

        elif isinstance(pattern, Newline):
            items.append("\n")

        else:
            raise TypeError(pattern)

    def join_sp(items):
        res = []
        nosp = True
        for item in items:
            if not nosp and not item.startswith("\n"):
                res.append(" ")
            res.append(item)
            nosp = item.endswith("\n")
        return "".join(res)

    return join_sp(items)


@dataclasses.dataclass
class Grid:
    offset: Fraction = Fraction(0, 1)
    start: Fraction = Fraction(0, 1)
    end: Fraction = Fraction(0, 1)
    denominator: int = 1
    subgrids: "List[Grid]" = dataclasses.field(default_factory=list)

    @property
    def unit(self):
        return Fraction(1, self.denominator)

    def as_str(self, denominator=1):
        if self.denominator % denominator != 0:
            raise ValueError

        divisor = self.denominator // denominator
        subgrids = " ".join(
            subgrid.as_str(self.denominator) for subgrid in self.subgrids
        )
        if divisor == 1:
            return f"({self.start} {subgrids} {self.end})"
        else:
            return f"[{self.start} {subgrids} {self.end}]/{divisor}"

    def validate(self):
        return (
            self.denominator % (self.start - self.offset).denominator == 0
            and self.denominator % (self.end - self.offset).denominator == 0
            and self.start <= self.end
            and all(
                self.denominator % (subgrid.start - self.offset).denominator == 0
                for subgrid in self.subgrids
            )
            and all(
                self.denominator % (subgrid.end - self.offset).denominator == 0
                for subgrid in self.subgrids
            )
            and all(
                subgrid.denominator % self.denominator == 0 for subgrid in self.subgrids
            )
            and all(self.start <= subgrid.start < self.end for subgrid in self.subgrids)
            and all(self.start < subgrid.end <= self.end for subgrid in self.subgrids)
            and all(
                subgrid1.end <= subgrid2.start
                for subgrid1, subgrid2 in zip(self.subgrids[:-1], self.subgrids[1:])
            )
            and all(subgrid.validate() for subgrid in self.subgrids)
        )

    @staticmethod
    def align(start, end, offset, denominator):
        start_ = (
            Fraction(math.floor((start - offset) * denominator), denominator) + offset
        )
        end_ = Fraction(math.ceil((end - offset) * denominator), denominator) + offset
        return (start_, end_)

    def get(self, pos, post=False):
        if post:
            for subgrid in self.subgrids:
                if subgrid.start < pos <= subgrid.end:
                    return subgrid
        else:
            for subgrid in self.subgrids:
                if subgrid.start <= pos < subgrid.end:
                    return subgrid
        return None

    def coarsen(self, denominator):
        assert self.denominator % denominator == 0
        assert denominator % (self.start - self.offset).denominator == 0
        assert denominator % (self.end - self.offset).denominator == 0
        if not self.subgrids:
            self.denominator = denominator
            return

        start, end = Grid.align(
            self.subgrids[0].start, self.subgrids[-1].end, self.offset, denominator
        )
        subgrid = Grid(self.offset, start, end, self.denominator, self.subgrids[:])
        # assert subgird.validate()

        self.denominator = denominator
        self.subgrids.clear()
        self.subgrids.append(subgrid)
        # assert self.validate()

    def last_leaf_end(self, default):
        if not self.subgrids:
            return default
        return self.subgrids[-1].last_leaf_end(self.subgrids[-1].end)

    def insert_last(self, leaf):
        assert self.last_leaf_end(self.start) <= leaf.start < self.end
        assert leaf.end <= self.end

        if leaf.denominator % self.denominator != 0:
            common_denominator = math.gcd(leaf.denominator, self.denominator)

            # try to divide subgrid to fit the leaf
            last_end = self.subgrids[-1].end if self.subgrids else self.start
            start_, end_ = Grid.align(
                leaf.start, last_end, self.offset, common_denominator
            )
            if end_ <= start_:
                # grid |xxx:xxx:   :   :   :   |                       |
                # note              ooooo ooooo ooooo ooooo ooooo ooooo
                # res  |xxx,xxx,   :ooooo,ooooo|ooooo,ooooo:ooooo,ooooo|
                self.coarsen(common_denominator)
                # assert self.validate()
            else:
                # grid |xxx:   :   |           |
                # note        ooooo ooooo ooooo
                # res  |xxx: ,o:o,o|o,o:o,o:o,o|
                leaf.denominator = math.lcm(leaf.denominator, self.denominator)
                # assert leaf.validate()

        start, end = Grid.align(leaf.start, leaf.end, self.offset, self.denominator)
        subgrid = self.get(leaf.start)
        if subgrid is None:
            if start == leaf.start and leaf.end == end:
                grid = leaf
            else:
                grid = Grid(self.offset, start, end, leaf.denominator, [leaf])
                # assert grid.validate()

            # insert grid to subgrids
            for i, subgrid in enumerate(self.subgrids[::-1]):
                if subgrid.end <= grid.start:
                    self.subgrids.insert(len(self.subgrids) - i, grid)
                    break
            else:
                self.subgrids.insert(0, grid)
            # assert self.validate()
            return

        # expand subgrid to contain the leaf
        subgrid.end = max(subgrid.end, end)
        subgrid.insert_last(leaf)
        # assert self.validate()

    @staticmethod
    def make(notes, start, end):
        assert start <= notes[0].beat <= end
        assert start <= notes[-1].beat + notes[-1].length <= end
        offset = start
        grid = Grid(start, start, end, 1, [])
        if (grid.start - grid.offset) % 1 == 0:
            grid.start -= 1
        if (grid.end - grid.offset) % 1 == 0:
            grid.end += 1
        for note in notes:
            leaf_start = note.beat
            leaf_end = note.beat + note.length
            leaf_denominator = math.lcm(
                (leaf_start - offset).denominator, (leaf_end - offset).denominator
            )
            leaf = Grid(offset, leaf_start, leaf_end, leaf_denominator, [])
            grid.insert_last(leaf)
        if (grid.start - grid.offset) % 1 == 0:
            grid.start += 1
        if (grid.end - grid.offset) % 1 == 0:
            grid.end -= 1
        return grid

    @staticmethod
    def bar_adder(offset, shape=(4, 4, 2)):
        @IIFE
        def bars():
            beat = offset
            while True:
                beat += 1
                if (beat - offset) % shape[0] == 0:
                    yield Measure(), beat
                if (beat - offset) % (shape[0] * shape[1]) == 0:
                    yield Newline(), beat
                if (beat - offset) % (shape[0] * shape[1] * shape[2]) == 0:
                    yield Newline(), beat

        bar, bar_beat = next(bars, (None, None))

        def add_bar(beat, patterns):
            nonlocal bar, bar_beat
            while bar_beat is not None and bar_beat == beat:
                patterns.append(bar)
                bar, bar_beat = next(bars, (None, None))

        return add_bar

    def build_patterns(self, notes, add_bar):
        notes = iter(notes)

        patterns = []
        beat = self.start

        def add_bar_and_pattern(pattern, length):
            nonlocal beat, patterns
            if beat > self.start:
                add_bar(beat, patterns)
            patterns.append(pattern)
            beat += length

        # leaf grid
        if not self.subgrids:
            note = next(notes, None)
            start_beat = note.beat
            end_beat = note.beat + note.length
            assert (
                note is not None and start_beat == self.start and end_beat == self.end
            )

            if note.symbol == "#":
                pattern = Comment(*note.arguments.ps, **note.arguments.kw)
                add_bar_and_pattern(pattern, 0)
            elif note.symbol == "Text":

                def makeText(text, *ps, **kw):
                    return Text(text, Arguments(ps, kw))

                pattern = makeText(*note.arguments.ps, **note.arguments.kw)
                add_bar_and_pattern(pattern, self.unit)
            else:
                pattern = Symbol(note.symbol, note.arguments)
                add_bar_and_pattern(pattern, self.unit)

            while beat < self.end:
                add_bar_and_pattern(Lengthen(), self.unit)
            return patterns

        # composite grid
        subgrids = iter(self.subgrids)

        subgrid = next(subgrids, None)
        while subgrid is not None:
            while beat < subgrid.start:
                add_bar_and_pattern(Rest(), self.unit)
            assert beat == subgrid.start

            add_bar(beat, patterns)
            children = subgrid.build_patterns(notes, add_bar)
            if subgrid.start == subgrid.end:
                assert len(children) == 1
                if isinstance(children[0], Comment):
                    add_bar_and_pattern(children[0], 0)
                else:
                    add_bar_and_pattern(Chord(children), 0)
            elif subgrid.denominator == self.denominator:
                patterns.extend(children)
                beat = subgrid.end
            else:
                divisor = subgrid.denominator // self.denominator
                add_bar_and_pattern(
                    Subdivision(divisor, children), subgrid.end - subgrid.start
                )
            subgrid = next(subgrids, None)

        while beat < self.end:
            add_bar_and_pattern(Rest(), self.unit)
        assert beat == self.end

        return patterns


def extend_notes(patterns, symbols):
    res = []
    last = None

    for pattern in patterns:
        assert last is None or isinstance(last, Chord)
        if isinstance(pattern, Lengthen):
            if last is not None:
                if last.patterns[:-1]:
                    res.append(Chord(last.patterns[:-1]))
                res.append(last.patterns[-1])
                last = None
            else:
                res.append(pattern)

        elif isinstance(pattern, Rest):
            if last is not None:
                if (
                    isinstance(last.patterns[-1], Symbol)
                    and last.patterns[-1].symbol in symbols
                ):
                    if last.patterns[:-1]:
                        res.append(Chord(last.patterns[:-1]))
                    res.append(last.patterns[-1])
                else:
                    res.append(last)
                    res.append(pattern)
                last = None
            else:
                res.append(pattern)

        elif isinstance(pattern, (Comment, Text, Symbol)):
            if last is not None:
                res.append(last)
                last = None
            res.append(pattern)

        elif isinstance(pattern, (Measure, Newline)):
            res.append(pattern)

        elif isinstance(pattern, Subdivision):
            patterns_ = pattern.patterns[:]
            if last is not None:
                patterns_.insert(0, last)
                last = None
            patterns_ = extend_notes(patterns_, symbols)
            if patterns_ and isinstance(patterns_[0], Chord):
                res.append(patterns_.pop(0))
            if patterns_ and isinstance(patterns_[-1], Chord):
                last = patterns_.pop()
            res.append(dataclasses.replace(pattern, patterns=patterns_))

        elif isinstance(pattern, Chord):
            if last is not None:
                last = Chord(last.patterns + pattern.patterns)
            else:
                last = pattern

        else:
            raise TypeError(pattern)

    if last is not None:
        res.append(last)

    return res


@dataclasses.dataclass
class Part:
    shape: Tuple[int, int, int]
    metadata: Note
    notes: List[Note]
    start: Optional[Fraction] = None
    end: Optional[Fraction] = None


def parse_shape_note(note):
    if note.symbol != "#":
        return None

    if note.arguments.ps:
        comment = note.arguments.ps[0]
    elif "comment" in note.arguments.kw:
        comment = note.arguments.kw["comment"]
    else:
        return None

    if not isinstance(comment, str):
        return None

    if not comment.startswith("SHAPE:"):
        return None

    shape_str = comment[len("SHAPE:") :]
    shape = shape_str.split(",")

    if not all(s.strip().isdigit() for s in shape):
        return None

    return tuple(int(s) for s in shape)


def format_notes(notes, beat=Fraction(0, 1), width=None, lengthless_symbols=[]):
    # partition notes by shapes
    default_shape = 4, 4, 2
    parts = [Part(default_shape, None, [])]

    for note in notes:
        shape = parse_shape_note(note)
        if shape is not None:
            shape = tuple(
                shape[i] if i < len(shape) else default
                for i, default in enumerate(default_shape)
            )
            parts[-1].end = note.beat
            parts.append(Part(shape, note, [], note.beat))
            continue

        parts[-1].notes.append(note)

    if not parts[0].notes:
        parts.pop(0)

    def enum_last(list):
        l = len(list)
        for i, elem in enumerate(list):
            yield i == l - 1, elem

    res = []
    for is_last, part in enum_last(parts):
        start = part.start if part.start is not None else part.notes[0].beat
        end = (
            part.end
            if part.end is not None
            else part.notes[-1].beat + part.notes[-1].length
        )
        grid = Grid.make(part.notes, start, end)
        add_bar = Grid.bar_adder(start, part.shape)
        patterns = grid.build_patterns(part.notes, add_bar)

        if is_last:
            # for extending the last lengthless symbol
            patterns.append(Rest())
            grid.end += 1
        patterns = extend_notes(patterns, lengthless_symbols)
        if is_last and isinstance(patterns[-1], Rest):
            patterns.pop()
            grid.end -= 1

        add_bar(grid.end, patterns)

        if part.metadata is not None:
            metadata_node = Comment(
                *part.metadata.arguments.ps, **part.metadata.arguments.kw
            )
            res.append(metadata_node)
            res.append(Newline())
        res.extend(patterns)
        if not is_last and not isinstance(res[-1], Newline):
            res.append(Newline())

    return patterns_to_str(res)
