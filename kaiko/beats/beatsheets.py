import os
import math
from fractions import Fraction
import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Dict, Union
import ast
from ..utils import parsec as pc
from ..utils import formattec as fc
from . import beatmaps

version = "0.2.0"

def Context(beat, length, **update):
    return beatmaps.UpdateContext(update)

class BeatmapParseError(Exception):
    pass

class BeatSheet(beatmaps.Beatmap):
    _notations = {
        'x': beatmaps.Soft,
        'o': beatmaps.Loud,
        '<': beatmaps.Incr,
        '%': beatmaps.Roll,
        '@': beatmaps.Spin,
        'Context': Context,
        'Text': beatmaps.Text,
        'Flip': beatmaps.Flip,
        'Shift': beatmaps.Shift,
    }

    # audio
    audio: str
    volume: float
    # info
    info: str
    preview: float
    # timings
    offset: float
    tempo: float
    # playfield
    bar_shift: float
    bar_flip: bool

    @staticmethod
    def to_events(track):
        if track.hide:
            return

        notations = BeatSheet._notations

        def build(beat, length, last_event, patterns):
            for pattern in patterns:
                if isinstance(pattern, Division):
                    beat, last_event = yield from build(beat, length / pattern.divisor, last_event, pattern.patterns)

                elif isinstance(pattern, Instant):
                    if last_event is not None:
                        yield last_event
                    last_event = None

                    beat, last_event = yield from build(beat, Fraction(0, 1), last_event, pattern.patterns)

                elif pattern.symbol == "~":
                    if pattern.arguments[0] or pattern.arguments[1]:
                        raise BeatmapParseError("lengthen note don't accept any argument")

                    if last_event is not None:
                        last_event.length += length
                    beat += length

                elif pattern.symbol == "|":
                    if pattern.arguments[0] or pattern.arguments[1]:
                        raise BeatmapParseError("measure note don't accept any argument")

                    if (beat - track.beat) % track.meter != 0:
                        raise BeatmapParseError("wrong measure")

                elif pattern.symbol == "_":
                    if pattern.arguments[0] or pattern.arguments[1]:
                        raise BeatmapParseError("rest note don't accept any argument")

                    if last_event is not None:
                        yield last_event
                    last_event = None
                    beat += length

                else:
                    if pattern.symbol not in notations:
                        raise BeatmapParseError("unknown symbol: " + pattern.symbol)

                    if last_event is not None:
                        yield last_event
                    event_type = notations[pattern.symbol]
                    last_event = event_type(beat, length, *pattern.arguments[0], **pattern.arguments[1])
                    beat += length

            return beat, last_event

        beat = Fraction(1, 1) * track.beat
        length = Fraction(1, 1) * track.length
        last_event = None
        beat, last_event = yield from build(beat, length, last_event, track.patterns)

        if last_event is not None:
            yield last_event

        return beat

    @staticmethod
    def to_track(sequence):
        raise NotImplementedError

    @property
    def chart(self):
        tracks = [self.to_track(seq) for seq in self.event_sequences]
        return chart_formatter.format(tracks)

    @chart.setter
    def chart(self, value):
        tracks = chart_parser.parse(value)
        self.event_sequences = [list(self.to_events(track)) for track in tracks]

    @staticmethod
    def parse_patterns(patterns_str):
        try:
            patterns = patterns_parser().parse(patterns_str)
            track = Track(beat=0, length=1, meter=4, patterns=patterns)

            events = []
            it = BeatSheet.to_events(track)
            while True:
                try:
                    event = next(it)
                except StopIteration as e:
                    width = e.value
                    break
                else:
                    events.append(event)

            return events, width

        except Exception as e:
            raise BeatmapParseError(f"failed to parse patterns") from e

    @staticmethod
    def read(filename, hack=False, metadata_only=False):
        filename = os.path.abspath(filename)
        if filename.endswith((".kaiko", ".ka")):
            sheet = open(filename).read()

            try:
                if hack:
                    beatmap = BeatSheet()
                    exec(sheet, dict(), dict(beatmap=beatmap))
                else:
                    beatmap = beatsheet_parser(metadata_only=metadata_only).parse(sheet)
            except Exception as e:
                raise BeatmapParseError(f"failed to read beatmap {filename}") from e

            beatmap.root = os.path.dirname(filename)
            return beatmap

        elif filename.endswith(".osu"):
            try:
                return OSU_FORMAT.read(filename, metadata_only=metadata_only)
            except Exception as e:
                raise BeatmapParseError(f"failed to read beatmap {filename}") from e

        else:
            raise BeatmapParseError(f"unknown file extension: {filename}")


Value = Union[None, bool, int, Fraction, float, str]
Arguments = Tuple[List[Value], Dict[str, Value]]

class Pattern:
    pass

@dataclass
class Track:
    # TRACK(beat=0, length=1/2):
    #     ...

    beat: Union[int, Fraction] = 0
    length: Union[int, Fraction] = 1
    meter: Union[int, Fraction] = 4
    hide: bool = False
    patterns: List[Pattern] = dataclasses.field(default_factory=list)

    def set_arguments(self, beat=0, length=1, meter=4, hide=False):
        self.beat = beat
        self.length = length
        self.meter = meter
        self.hide = hide

    def get_arguments(self):
        return dict(beat=self.beat, length=self.length, meter=self.meter, hide=self.hide)

@dataclass
class Note(Pattern):
    #    event note: XXX(arg=...)
    #     text note: "ABC"(arg=...)
    # lengthen note: ~
    #  measure note: |
    #     rest note: _

    symbol: str
    arguments: Arguments

@dataclass
class Division(Pattern):
    # [x o]
    # [x x o]/3

    divisor: int = 2
    patterns: List[Pattern] = dataclasses.field(default_factory=list)

@dataclass
class Instant(Pattern):
    # {x x o}
    patterns: List[Pattern] = dataclasses.field(default_factory=list)

def IIFE(func):
    return func()

@IIFE
def value_parser():
    none  = pc.regex(r"None").map(ast.literal_eval)
    bool  = pc.regex(r"True|False").map(ast.literal_eval)
    int   = pc.regex(r"[-+]?(0|[1-9][0-9]*)").map(ast.literal_eval)
    frac  = pc.regex(r"[-+]?(0|[1-9][0-9]*)\/[1-9][0-9]*").map(Fraction)
    float = pc.regex(r"[-+]?([0-9]+\.[0-9]+(e[-+]?[0-9]+)?|[0-9]+[eE][-+]?[0-9]+)").map(ast.literal_eval)
    str   = pc.regex(
        r'"([^\r\n\\"]|\\[\\"btnrfv]|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})*"').map(ast.literal_eval)

    desc = "None or bool or str or float or frac or int"
    return pc.choice(none, bool, str, float, frac, int).desc(desc)

@fc.Formattec
def value_formatter(value, **contexts):
    if value is None:
        yield "None"
    elif isinstance(value, (bool, int, float)):
        yield repr(value)
    elif isinstance(value, Fraction):
        yield str(value)
    elif isinstance(value, str):
        yield '"' + repr(value + '"')[1:-2].replace('"', r'\"').replace(r"\'", "'") + '"'
    else:
        raise fc.FormatError(value, "None or bool or str or float or frac or int")

@IIFE
@pc.parsec
def arguments_parser():
    key = pc.regex(r"([a-zA-Z_][a-zA-Z0-9_]*)=").desc("'key='").map(lambda k: k[:-1])
    opening = pc.tokens(["("]).optional()
    closing = pc.tokens([")"]).optional()
    comma = pc.tokens([", "])

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

@fc.Formattec
def arguments_formatter(value, **contexts):
    psargs, kwargs = value

    if len(psargs) + len(kwargs) == 0:
        yield ""

    yield "("

    is_first = True
    for value in psargs:
        if not is_first:
            yield ", "
        is_first = False
        yield from value_formatter.func(value)

    for key, value in kwargs.items():
        if not is_first:
            yield ", "
        is_first = False
        yield key
        yield "="
        yield from value_formatter.func(value)

    yield ")"

@IIFE
def note_parser():
    symbol = pc.regex(r"[^ \b\t\n\r\f\v()[\]{}\'\"\\#]+")
    text = pc.regex(
        r'"([^\r\n\\"\x00]|\\[\\"btnrfv]|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})*"').map(ast.literal_eval)
    return (
        (symbol + arguments_parser).starmap(Note)
        | (text + arguments_parser).starmap(lambda text, arg: Note('Text', ([text, *arg[0]], arg[1])))
    )

@fc.Formattec
def note_formatter(value, **contexts):
    yield value.symbol
    yield from arguments_formatter.func(value.arguments)

@pc.parsec
def msp_parser(indent=0):
    # return None  ->  no match
    # return ""    ->  end of block
    # return " "   ->  whitespace between token
    # return "\n"  ->  newline between token (including comments)

    sp = pc.regex(r"[ \t]+").optional()
    eol = pc.regex(r"(?=\n|$)").optional()
    nl = pc.regex(r"\n+(?=[ ]{%s,})" % (indent,)).optional()
    ind = pc.regex(r"[ ]{%s}(?![ ])" % (indent,)).optional()
    cmt = pc.regex(r"#[^\n]*").optional()

    # whitespace
    spaced = yield sp

    # end of line
    if not (yield eol):
        if (yield cmt.ahead()):
            raise pc.ParseFailure("comment occupied a whole line")
        if spaced:
            return " "
        return None

    # newline
    while (yield nl):
        if not (yield ind):
            raise pc.ParseFailure(f"indentation with level {indent}")
        if (yield eol):
            continue
        if (yield cmt):
            continue
        return "\n"

    return ""

@pc.parsec
def patterns_parser(indent=0, until=""):
    opening = pc.tokens(["{", "["]).optional()
    closing = pc.tokens([until]).optional() if until else pc.nothing("")
    div = pc.regex(r"/(\d+)").map(lambda m: int(m[1:])) | pc.nothing(2)
    msp = msp_parser(indent=indent)
    if until:
        msp = msp.validate(lambda sp: sp != "", repr(until))

    sp = " "
    patterns = []

    while True:
        # end of block
        if sp == "":
            return patterns

        # closing bracket
        if (yield closing):
            return patterns

        # no space
        if sp is None:
            raise pc.ParseFailure("space")

        # pattern
        opened = yield opening
        if opened == ("{",):
            yield msp_parser(indent=indent)
            subpatterns = yield patterns_parser(indent=indent, until="}")
            pattern = Instant(subpatterns)

        elif opened == ("[",):
            yield msp_parser(indent=indent)
            subpatterns = yield patterns_parser(indent=indent, until="]")
            divisor = yield div
            pattern = Division(divisor, subpatterns)

        else:
            pattern = yield note_parser

        patterns.append(pattern)

        # spacing
        sp = yield msp

@fc.Formattec
def patterns_formatter(value, **contexts):
    is_first = True
    for pattern in value:
        if not is_first:
            yield " "
        is_first = False

        if isinstance(pattern, Instant):
            yield "{"
            yield from patterns_formatter.func(value.patterns)
            yield "}"

        elif isinstance(pattern, Division):
            yield "["
            yield from patterns_formatter.func(value.patterns)
            yield "]"
            yield "" if value.divisor == 2 else f"/{value.divisor}"

        elif isinstance(pattern, Note):
            yield from note_formatter.func(pattern)

        else:
            assert False

@IIFE
@pc.parsec
def chart_parser():
    tracks = []

    while True:
        sp = yield msp_parser(indent=0).validate(lambda sp: sp is not None, "whitespace or newline")

        if sp == "":
            return tracks

        elif sp == "\n":
            yield pc.tokens(["TRACK"])
            arguments = yield arguments_parser

            track = Track()
            try:
                track.set_arguments(*arguments[0], **arguments[1])
            except Exception as e:
                raise pc.ParseFailure("valid arguments") from e

            yield pc.regex(r":(?=\n)")
            yield msp_parser(indent=4).validate(lambda sp: sp == "\n", "newline")

            patterns = yield patterns_parser(indent=4)
            track.patterns = patterns
            tracks.append(track)

        else:
            assert False

@fc.Formattec
def chart_formatter(value, *, indent=0, **contexts):
    for track in value:
        kwargs = track.get_arguments()
        yield "\n" + " "*indent + "TRACK"
        yield from arguments_formatter.func(([], kwargs))
        yield ":\n"
        yield " "*indent + "    "
        yield from patterns_formatter.func(track.patterns, indent=indent+4)
        yield "\n"


@pc.parsec
def beatsheet_parser(metadata_only=False):
    header = yield pc.regex(r"#K-AIKO-std-(\d+\.\d+\.\d+)(?=\n|$)").desc("header")
    vernum = header[len("#K-AIKO-std-"):].split(".")
    vernum0 = version.split(".")
    if vernum[0] != vernum0[0] or vernum[1:] > vernum0[1:]:
        raise BeatmapParseError("incompatible version")

    beatsheet = BeatSheet()
    fields = BeatSheet.__annotations__
    valid_fields = list(fields.keys())
    valid_fields.append("chart")

    while True:
        sp = yield msp_parser(indent=0).validate(lambda sp: sp in ("\n", ""), "newline")
        if sp == "":
            return beatsheet

        yield pc.tokens(["beatmap."])
        name = yield pc.tokens(valid_fields)
        yield pc.tokens([" = "])
        valid_fields.remove(name)

        if name == "info":
            field_parser = pc.mstr_parser
        elif name == "chart":
            field_parser = pc.rmstr_parser
        else:
            field_parser = pc.from_type_hint(fields[name])

        value = yield field_parser

        if not metadata_only or name != "chart":
            setattr(beatsheet, name, value)

@fc.Formattec
def beatsheet_formatter(value, **contexts):
    fields = BeatSheet.__annotations__

    yield f"#K-AIKO-std-{version}\n"

    for name, type_hint in fields.items():
        field_formatter = fc.mstr_formatter if name == "info" else fc.from_type_hint(type_hint)
        yield f"beatmap.{name} = "
        yield from field_formatter.func(getattr(value, name))
        yield "\n"

    name = 'chart'
    field_formatter = fc.rmstr_formatter
    yield f"beatmap.{name} = "
    yield from field_formatter.func(getattr(value, name))
    yield "\n"


class OSU:
    def read(self, filename, metadata_only=False):
        path = os.path.dirname(filename)
        index = 0

        with open(filename, encoding='utf-8-sig') as file:
            format = file.readline()
            index += 1
            # if format != "osu file format v14\n":
            #     raise BeatmapParseError(f"invalid file format: {repr(format)}")

            beatmap = beatmaps.Beatmap()
            beatmap.event_sequences = [[]]
            context = {}

            parse = None

            line = "\n"
            while line != "":
                if line == "\n" or line.startswith(r"\\"):
                    pass
                elif line == "[General]\n":
                    parse = self.parse_general
                elif line == "[Editor]\n":
                    parse = self.parse_editor
                elif line == "[Metadata]\n":
                    parse = self.parse_metadata
                elif line == "[Difficulty]\n":
                    parse = self.parse_difficulty
                elif line == "[Events]\n":
                    parse = self.parse_events
                elif line == "[TimingPoints]\n":
                    parse = self.parse_timingpoints
                elif line == "[Colours]\n":
                    parse = self.parse_colours
                elif line == "[HitObjects]\n":
                    parse = self.parse_hitobjects
                else:
                    try:
                        if not metadata_only or parse != self.parse_timingpoints and parse != self.parse_hitobjects:
                            parse(beatmap, context, line)
                    except Exception as e:
                        raise BeatmapParseError(f"parse error at line {index}") from e

                line = file.readline()
                index += 1

        beatmap.root = path
        return beatmap

    def parse_general(self, beatmap, context, line):
        option, value = line.split(": ", maxsplit=1)
        if option == 'AudioFilename':
            beatmap.audio = value.rstrip("\n")
        elif option == 'PreviewTime':
            beatmap.preview = int(value)/1000

    def parse_editor(self, beatmap, context, line): pass

    def parse_metadata(self, beatmap, context, line):
        beatmap.info += line

    def parse_difficulty(self, beatmap, context, line):
        option, value = line.split(":", maxsplit=1)
        if option == 'SliderMultiplier':
            context['multiplier0'] = float(value)

    def parse_events(self, beatmap, context, line): pass

    def parse_timingpoints(self, beatmap, context, line):
        time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects = line.rstrip("\n").split(",")
        time = float(time)
        beatLength = float(beatLength)
        meter = int(meter)
        volume = float(volume)
        multiplier = context['multiplier0']

        if 'timings' not in context:
            context['timings'] = []
        if 'beatLength0' not in context:
            context['beatLength0'] = beatLength
            beatmap.offset = time/1000
            beatmap.tempo = 60 / (beatLength/1000)

        if uninherited == "0":
            multiplier = multiplier / (-0.01 * beatLength)
            beatLength = context['timings'][-1][1]
            meter = context['timings'][-1][2]

        speed = multiplier / 1.4
        volume = 20 * math.log10(volume / 100)
        sliderVelocity = (multiplier * 100) / (beatLength/context['beatLength0'])
        density = (8/meter) / (beatLength/context['beatLength0']) # 8 per measure

        context['timings'].append((time, beatLength, meter, speed, volume, sliderVelocity, density))

    def parse_colours(self, beatmap, context, line): pass

    def parse_hitobjects(self, beatmap, context, line):
        x,y,time,type,hitSound,*objectParams,hitSample = line.rstrip("\n").split(",")
        time = float(time)
        type = int(type)
        hitSound = int(hitSound)

        beat = beatmap.beat(time/1000)
        speed, volume, sliderVelocity, density = next(vs for t, b, m, *vs in context['timings'][::-1] if t <= time)

        # type : [_:_:_:_:Spinner:_:Slider:Circle]
        # hitSound : [Kat:Large:Kat:Don]

        if type & 1: # circle
            if hitSound == 0 or hitSound & 1: # don
                event = beatmaps.Soft(beat=beat, length=Fraction(0), speed=speed, volume=volume)
                beatmap.event_sequences[0].append(event)

            elif hitSound & 10: # kat
                event = beatmaps.Loud(beat=beat, length=Fraction(0), speed=speed, volume=volume)
                beatmap.event_sequences[0].append(event)

        elif type & 2: # slider
            # curve,slides,sliderLength,edgeSounds,edgeSets = objectParams
            sliderLength = float(objectParams[2]) if len(objectParams) >= 3 else 0.0
            sliderLength = float(sliderLength)
            length = sliderLength / sliderVelocity

            event = beatmaps.Roll(beat=beat, length=length, density=density, speed=speed, volume=volume)
            beatmap.event_sequences[0].append(event)

        elif type & 8: # spinner
            end_time, = objectParams
            end_time = float(end_time)
            length = (end_time - time)/context['beatLength0']

            event = beatmaps.Spin(beat=beat, length=length, density=density, speed=speed, volume=volume)
            beatmap.event_sequences[0].append(event)

OSU_FORMAT = OSU()
