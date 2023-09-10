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
        raise TypeError(value)


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
    r"""Grid of notes.

    This class aligns notes in a measure of time, so that notes can be represented
    in series of patterns. It can be visualized as a ruler on time.

    ..code::

        x  x  o     o     o  x     x  o     o        ===> notes
        |--|--|           |--|--|--|--|              ===> subgrid
        |-----|-----|-----|-----|-----|-----|-----|  ===> grid

    With the help of grid, those notes can be written as patterns

    ..code::

        [x x] o o [o x _ x] o o

    But not

    ..code::

            no need to wrap in a subdivision
              /
        [x x o ~] o [o] x [x] o o
                     \    /
                weird, hard-to-read patterns

    This class is designed for finding a grid aligning all notes in a concise way.

    Grid System
    -----------

    A grid describe a linear sequence of ticks. One this grid, one can specify a
    time span, such as (1, 4). To specify a fractional time span, the grid should
    be subdivided into another grid, called a subgrid. Grids and subgrids form a
    tree structure, where the start point and the end point of a subgrid should
    aligned with the beats of its parent, and subgrids should disjoint with each
    others. With a proper grid system, one can specify a series of exclusive time
    span, which can be used to align note sequence. Our goal is to find a proper
    grid system to fit a given note sequence.

    Build Grids
    -----------

    The start time and the end time of a note should align on ticks.  For example,
    a note with time span (1/4, 5/4) can not be aligned by grid with unit 1, even
    though it has length 1. A note can be treated as a smallest subgrid of this
    grid system. In this example, it is a subgrid with range (1/4, 5/4) and unit
    1/4, denoted as (1/4 5/4)/4. To fit the subgrid of a note into this grid
    system. One should further include a subgrid (0 2)/4 in between the root grid
    and the subgrid of this note.

    ..code::

            |---|---|---|---|                              ===> note (unit=1/4)
        |---|---|---|---|---|---|---|---|                  ===> subgrid
        |---------------|---------------|---------------|  ===> root grid

    A zero-length note can be treated as a zero-length subgrid. For example, a
    zero-length note (3/4 3/4)/4 can fit in the grid system as

    ..code::

                    |                                      ===> note (unit=1/4)
        |---|---|---|---|                                  ===> subgrid
        |---------------|---------------|---------------|  ===> root grid

    To align all notes by this grid system, one should fit all subgrids of notes
    into this grid system. Even though notes cannot overlap each others, the
    subgrid in between still can.

    If added note have overlapped subgrid to the previous one, extend this subgrid,
    and try to fit this note on this extended subgrid.

    ..code::

                                    |-|-|-|-|-|            ===> a new added subgrid of note (unit=1/8)
            |---|---|---|---|   |-|-|-|-|-|-|-|-|          ===> add a subgrid to fit the note
        |---|---|---|---|---|---|---|---|---|---|---|---|  ===> extend this subgrid
        |---------------|---------------|---------------|

    If the unit of the note cannot fit into the extended subgrid, try to coarsen
    this subgrid, so that their subgrids are disjoint.

    ..code::

                          |-----|  ===> a note (unit=1/4) to insert
        |---|---|                  ===> previous inserted note (unit=1/6)
        |---|---|---|---|---|---|  ===> subgrid
        |-----------------------|  ===> root grid

        become

                          |-----|  ===> a note (unit=1/4) to insert
                    |-----|-----|  ===> this subgrid now is disjoint to the previous one
        |---|---|
        |---|---|---|              ===> this subgrid is shrinked
        |-----------|-----------|  ===> a coarsen subgrid (unit=1/2)
        |-----------------------|

    If failed, subdivide the unit of note, so that it fit into the subgrid.

    ..code::

              |-----|  ===> a note (unit=1/2) to insert
        |---|          ===> previous inserted note (unit=1/3)
        |---|---|---|  ===> subgrid
        |-----------|  ===> root grid

        become

              |-|-|-|  ===> a subdivided note (unit=1/6), which is now can be inserted
            |-|-|-|-|  ===> subgrid
        |---|
        |---|---|---|
        |-----------|

    Format Notes
    ------------

    After building grid system for a given note sequence, each leaf note
    corresponds to each note. By filling empty parts, the patterns of given notes
    can be built. For example,

    ..code::

              |-|-|   |        ===> the leaf note of "x"
            |-|-|-|-|-|-|      ===> subgrid
        |---|---|---|---|---|  ===> grid

        being formetted as:

          _ [_ x ~ _ _ {x} _] _

    Where leaf notes are filled by lengthen patterns, and non-leaf notes are
    filled by rest patterns. Subgrids being formatted as a subdivision pattern,
    and zero-length notes being formatted as chords.

    There may be multiple zero-length notes at the same beat. In this case,
    chords should be merged, like "{x} {x} {o}" to "{x x o}". Moreover,
    zero-length note may be actually a lengthless note (whose length doesn't
    change the meaning of the note), that means it can be extended freely. For
    example, "{x} _" can be replaced by "x", or "{x} [_ o]" by "[x o]".

    In the definition of subgrids, a zero-length subgird cannot be contained in a
    grid on the boundary, for example, grid (1 1)/1 cannot be contained in (1 2)/2.
    So to build a grid system for given notes, a margin between boundary of the
    root grid and first and last notes is needed. The margin can be removed after
    building. To remove the margin, two special notes are added as anchors, and
    being removed after formatting patterns. By this approach, the start and end
    time of the patterns can be fractional.

    Fields
    ------
    offset : Fraction
        The time offset (in beat) of the origin of ticks.
        All grids in a grid system should have the same offset.
    start, end : Fraction
        The start time and end time (in beat) of this grid. The start time and end
        time should be on the scale. The start time should be less than or equal to
        the end time.
    denominator : int
        The denominator of the unit of ticks. See `unit`.
    unit : Fraction
        The time differences (in beat) between ticks.
    subgrids : list of Grid
        The subgrids of this grid. The denominators of subgrids should be divisible
        by the denominator of this grid. The start time and end time of subgrids
        should be on the scale of this grid. All subgrids are ordered and exculsively
        contained in the grid.
    """
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
            return f"({self.start} {subgrids} {self.end})/{divisor}"

    def validate(self):
        return (
            self.denominator % (self.start - self.offset).denominator == 0
            and self.denominator % (self.end - self.offset).denominator == 0
            and self.start <= self.end
            and all(self.offset == subgrid.offset for subgrid in self.subgrids)
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
        """Find the region containing (start, end) and align to ticks specified
        by offset and denominator.
        """
        start_ = (
            Fraction(math.floor((start - offset) * denominator), denominator) + offset
        )
        end_ = Fraction(math.ceil((end - offset) * denominator), denominator) + offset
        return (start_, end_)

    def get(self, pos, post=False):
        """Get the child subgird that contains the position `pos`, or None if
        not found.
        """
        if post:
            for subgrid in self.subgrids:
                if subgrid.start < pos <= subgrid.end:
                    return subgrid
            return None
        else:
            for subgrid in self.subgrids:
                if subgrid.start <= pos < subgrid.end:
                    return subgrid
            return None

    def coarsen(self, denominator):
        """Coarsen this grid to specified denominator.

        If it has subgrids, they will be contained in the subgrid with the
        same unit, and this subgrid will become the only subgrid of this
        grid. Caller should check it is valid to do this.
        """
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
        """Find the end time of the last leaf subgrid, or `default` if not found."""
        if not self.subgrids:
            return default
        return self.subgrids[-1].last_leaf_end(self.subgrids[-1].end)

    def insert_last(self, leaf):
        """Insert a leaf subgrid into this grid system.
        The inserted leaf subgrid should be the last leaves in time.
        """
        assert self.last_leaf_end(self.start) <= leaf.start <= self.end
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
            # no overlap
            if start == leaf.start and leaf.end == end:
                grid = leaf
            else:
                grid = Grid(self.offset, start, end, leaf.denominator, [leaf])
                # assert grid.validate()

            # insert grid to subgrids
            # assert not self.subgrids or self.subgrids[-1].end <= grid.start
            self.subgrids.append(grid)
            # assert self.validate()
            return

        # expand subgrid to contain the leaf
        # assert subgrid is self.subgrids[-1]
        subgrid.end = max(subgrid.end, end)
        subgrid.insert_last(leaf)
        # assert self.validate()

    def build_patterns(self, notes):
        """Build patterns for given note sequence.
        Where grid is built by insert_last all notes to an empty grid, so that each
        leaf grid correspond to each note.
        """
        notes = iter(notes)

        patterns = []
        beat = self.start

        if not self.subgrids:
            # leaf grid
            note = next(notes, None)
            assert note is not None

            start_beat = note.beat
            end_beat = note.beat + note.length
            assert (
                note is not None and start_beat == self.start and end_beat == self.end
            )

            if note.symbol == "#":
                pattern = Comment(*note.arguments.ps, **note.arguments.kw)
                patterns.append(pattern)
            elif note.symbol == "Text":

                def makeText(text, *ps, **kw):
                    return Text(text, Arguments(ps, kw))

                pattern = makeText(*note.arguments.ps, **note.arguments.kw)
                patterns.append(pattern)
                beat += self.unit
            else:
                pattern = Symbol(note.symbol, note.arguments)
                patterns.append(pattern)
                beat += self.unit

            while beat < self.end:
                patterns.append(Lengthen())
                beat += self.unit
            return patterns

        else:
            # composite grid
            for subgrid in self.subgrids:
                while beat < subgrid.start:
                    patterns.append(Rest())
                    beat += self.unit
                assert beat == subgrid.start

                children = subgrid.build_patterns(notes)

                if subgrid.start == subgrid.end:
                    assert len(children) == 1
                    if isinstance(children[0], Comment):
                        patterns.append(children[0])
                    else:
                        patterns.append(Chord(children))
                elif subgrid.denominator == self.denominator:
                    patterns.extend(children)
                    beat = subgrid.end
                else:
                    divisor = subgrid.denominator // self.denominator
                    patterns.append(Subdivision(divisor, children))
                    beat = subgrid.end

            while beat <= self.end - self.unit:
                patterns.append(Rest())
                beat += self.unit
            assert beat == self.end

            return patterns

    @staticmethod
    def to_patterns(notes, start, end):
        """Format given notes to patterns in the range (start, end).
        Where notes should be inlcuded by this range.
        """
        # assert start <= notes[0].beat <= end
        # assert start <= notes[-1].beat + notes[-1].length <= end

        # extend range
        offset = start
        start_ = start - 1
        end_ = start + math.ceil(end - start) + 1
        grid = Grid(offset, start_, end_, 1, [])

        # add start/end anchor point
        START = Note("#START#", start, Fraction(0, 1), Arguments([], {}))
        END = Note("#END#", end, Fraction(0, 1), Arguments([], {}))
        notes_ = [START, *notes, END]

        # build grid
        for note in notes_:
            leaf_start = note.beat
            leaf_end = note.beat + note.length
            leaf_denominator = math.lcm(
                (leaf_start - offset).denominator, (leaf_end - offset).denominator
            )
            leaf = Grid(offset, leaf_start, leaf_end, leaf_denominator, [])
            grid.insert_last(leaf)

        patterns = grid.build_patterns(notes_)

        # trim
        def trim_start(patterns):
            while patterns:
                pattern = patterns.pop(0)
                if isinstance(pattern, Symbol) and pattern.symbol == START.symbol:
                    return True
                if isinstance(pattern, (Subdivision, Chord)):
                    children = pattern.patterns[:]
                    is_finished = trim_start(children)
                    if is_finished:
                        if children:
                            pattern = dataclasses.replace(pattern, patterns=children)
                            patterns.insert(0, pattern)
                        return True
            return False

        def trim_end(patterns):
            while patterns:
                pattern = patterns.pop()
                if isinstance(pattern, Symbol) and pattern.symbol == END.symbol:
                    return True
                if isinstance(pattern, (Subdivision, Chord)):
                    children = pattern.patterns[:]
                    is_finished = trim_end(children)
                    if is_finished:
                        if children:
                            pattern = dataclasses.replace(pattern, patterns=children)
                            patterns.append(pattern)
                        return True
            return False

        trim_start(patterns)
        trim_end(patterns)

        return patterns


def simplify_chords(patterns, lengthless_symbols):
    """Simplify chords by merging "{x x} {x o}" => "{x x x o}", lengthening
    "{x o x} ~" => "{x o} x" or "{x o x} _" => "{x o} x" for lengthless notes,
    escaping "[{x o} x x]" => "{x o} [x x]" or "[x x {x o}]" => "[x x] {x o}".
    """
    res = []
    for pattern in patterns:
        if isinstance(pattern, (Comment, Text, Symbol, Measure, Newline)):
            res.append(pattern)

        elif isinstance(pattern, Chord):
            if res and isinstance(res[-1], Chord):
                # {x o x} {x o x}   =>   {x o x x o x}
                last_chord = res.pop()
                last_chord = Chord(last_chord.patterns + pattern.patterns)
                res.append(last_chord)
            else:
                res.append(pattern)

        elif isinstance(pattern, Lengthen):
            if res and isinstance(res[-1], Chord):
                # {x o x} ~   =>   {x o} x
                last_chord = res.pop()
                if last_chord.patterns[:-1]:
                    res.append(Chord(last_chord.patterns[:-1]))
                res.append(last_chord.patterns[-1])
            else:
                res.append(pattern)

        elif isinstance(pattern, Rest):
            if res and isinstance(res[-1], Chord):
                # {x o x} _   =>   {x o} x   if x is lengthless
                last_chord = res.pop()
                if (
                    isinstance(last_chord.patterns[-1], Symbol)
                    and last_chord.patterns[-1].symbol in lengthless_symbols
                ):
                    if last_chord.patterns[:-1]:
                        res.append(Chord(last_chord.patterns[:-1]))
                    res.append(last_chord.patterns[-1])
                else:
                    res.append(last_chord)
                    res.append(pattern)
            else:
                res.append(pattern)

        elif isinstance(pattern, Subdivision):
            # {x o x} [x x x x]   =>   [{x o x} x x x x]
            children = pattern.patterns[:]
            if res and isinstance(res[-1], Chord):
                last_chord = res.pop()
                children.insert(0, last_chord)

            children = simplify_chords(children, lengthless_symbols)

            # [{x o x} x x x x]   =>   {x o x} [x x x x]
            # [x x x x {x o x}]   =>   [x x x x] {x o x}
            subres = [None, None, None]
            if children and isinstance(children[0], Chord):
                subres[0] = children.pop(0)
            if children and isinstance(children[-1], Chord):
                subres[2] = children.pop()
            if children:
                subres[1] = dataclasses.replace(pattern, patterns=children)
            res.extend(pattern for pattern in subres if pattern is not None)

        else:
            raise TypeError(pattern)

    return res


def insert_measures(
    patterns, shape=(4, 4, 2), offset=Fraction(0, 1), denominator=1, last_measure=True
):
    """Insert measures and newlines to patterns. The shape is a triple of
    integer. For example, (3, 4, 2) means 3 beats per section, 4 sections per
    line, and 2 lines per paragraph, which represents the format:

    ..code::

        x x o | x x o | x x o | x x o |
        x x o | x x o | x x o | x x o |

        x x o | x x o | x x o | x x o |
        x x o | x x o | x x o | x x o |

        ...

    Some edge cases:

    ..code::

                     zero-length patterns are placed after measure
                    _____/
          o x x x | {x x} x x x o | x x o [x x | o x] x o o  -- the last measure is dropped
        ^-- the first measure is dropped  ^^^^^^^^^^^-- measure in subdivision
    """

    def length_of_patterns(patterns, denominator=1):
        unit = Fraction(1, denominator)
        length = unit * 0
        for pattern in patterns:
            if isinstance(pattern, (Comment, Measure, Newline, Chord)):
                pass
            elif isinstance(pattern, (Text, Symbol, Lengthen, Rest)):
                length += unit
            elif isinstance(pattern, Subdivision):
                length += length_of_patterns(
                    pattern.patterns, denominator * pattern.divisor
                )
            else:
                raise TypeError(pattern)
        return length

    res = []
    beat = last_beat = length_of_patterns(patterns, denominator)
    for pattern in patterns[::-1]:
        length = length_of_patterns([pattern], denominator)
        if isinstance(pattern, Subdivision):
            children = insert_measures(
                pattern.patterns,
                shape,
                offset - beat + length,
                denominator * pattern.divisor,
                False,
            )
            pattern = dataclasses.replace(pattern, patterns=children)
        if length > 0 and beat != 0 and (last_measure or beat != last_beat):
            if (beat - offset) % (shape[0] * shape[1] * shape[2]) == 0:
                res.append(Newline())
            if (beat - offset) % (shape[0] * shape[1]) == 0:
                res.append(Newline())
            if (beat - offset) % shape[0] == 0:
                res.append(Measure())
        res.append(pattern)
        beat -= length

    return res[::-1]


def format_notes(notes, start=None, end=None, lengthless_symbols=[]):
    """Format a note sequence in the range (start, end).

    A sepcial comment note can be used to control formatting. For example,
    the comment "#SHAPE: 3, 4, 2" will formatting following notes in the
    shape (3, 4, 2). The whole note sequence is partitioned by those
    formatting comment notes.
    """

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

    # assert all(start is None or start <= note.beat for note in notes)
    # assert all(end is None or note.beat + note.length <= end for note in notes)

    # partition notes by shapes
    default_shape = 4, 4, 2
    parts = [Part(default_shape, None, [], start, None)]

    for note in notes:
        shape = parse_shape_note(note)
        if shape is not None:
            shape = tuple(
                shape[i] if i < len(shape) else default
                for i, default in enumerate(default_shape)
            )
            parts[-1].end = note.beat
            parts.append(Part(shape, note, [], note.beat, None))
            continue

        parts[-1].notes.append(note)
    else:
        parts[-1].end = end

    if not parts[0].notes and (
        parts[0].start is None or parts[0].end is None or parts[0].start == parts[0].end
    ):
        parts.pop(0)

    # format each part
    def enum_last(list):
        l = len(list)
        for i, elem in enumerate(list):
            yield i == l - 1, elem

    res = []
    for is_last, part in enum_last(parts):
        part_start = (
            part.start
            if part.start is not None
            else Fraction(math.floor(part.notes[0].beat), 1)
        )
        part_end = (
            part.end
            if part.end is not None
            else Fraction(math.ceil(part.notes[-1].beat + part.notes[-1].length), 1)
        )

        patterns = Grid.to_patterns(part.notes, part_start, part_end)
        patterns = simplify_chords(patterns, lengthless_symbols)
        patterns = insert_measures(patterns, part.shape)

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
