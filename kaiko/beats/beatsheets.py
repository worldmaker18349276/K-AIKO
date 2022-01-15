import os
import math
from fractions import Fraction
import dataclasses
from pathlib import Path
import re
from typing import List, Tuple, Dict, Union
import ast
from ..utils import parsec as pc
from ..utils import serializers as sz
from . import beatmaps

version = "0.3.0"

def Context(beat, length, **update):
    return beatmaps.UpdateContext(update)

class BeatmapParseError(Exception):
    pass

class PatternError(Exception):
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

    _fields = {
        "info": str,
        "audio.path": Path,
        "audio.volume": float,
        "audio.preview": float,
        "audio.info": str,
        "metronome.offset": float,
        "metronome.tempo": float,
        "beatbar_state.bar_shift": float,
        "beatbar_state.bar_flip": bool,
        "chart": str,
    }

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
                        raise PatternError("lengthen note don't accept any argument")

                    if last_event is not None:
                        last_event.length += length
                    beat += length

                elif pattern.symbol == "|":
                    if pattern.arguments[0] or pattern.arguments[1]:
                        raise PatternError("measure note don't accept any argument")

                    if (beat - track.beat) % track.meter != 0:
                        raise PatternError("wrong measure")

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
        return format_chart(tracks)

    @chart.setter
    def chart(self, value):
        tracks = chart_parser.parse(value)
        self.event_sequences = [list(self.to_events(track)) for track in tracks]

    @staticmethod
    def parse_patterns(patterns_str):
        try:
            patterns = make_patterns_parser().parse(patterns_str)
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
                    local = {}
                    exec(sheet, {'__file__': filename}, local)
                    beatmap = local['beatmap']
                else:
                    root = Path(filename).resolve().parent
                    beatmap = make_beatsheet_parser(root, metadata_only=metadata_only).parse(sheet)
            except Exception as e:
                raise BeatmapParseError(f"failed to read beatmap {filename}") from e

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

@dataclasses.dataclass
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
    none  = pc.regex(r"None").map(ast.literal_eval)
    bool  = pc.regex(r"True|False").map(ast.literal_eval)
    int   = pc.regex(r"[-+]?(0|[1-9][0-9]*)").map(ast.literal_eval)
    frac  = pc.regex(r"[-+]?(0|[1-9][0-9]*)\/[1-9][0-9]*").map(Fraction)
    float = pc.regex(r"[-+]?([0-9]+\.[0-9]+(e[-+]?[0-9]+)?|[0-9]+[eE][-+]?[0-9]+)").map(ast.literal_eval)
    str   = pc.regex(
        r'"([^\r\n\\"]|\\[\\"btnrfv]|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})*"').map(ast.literal_eval)

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
        r'"([^\r\n\\"\x00]|\\[\\"btnrfv]|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})*"').map(ast.literal_eval)
    return (
        (symbol + arguments_parser).starmap(Note)
        | (text + arguments_parser).starmap(lambda text, arg: Note('Text', ([text, *arg[0]], arg[1])))
    )

@pc.parsec
def make_msp_parser(indent=0):
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
        raise pc.ParseFailure("whitespace or newline or end of block")

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

def make_patterns_parser(indent=0):
    end = make_msp_parser(indent=indent).reject(lambda sp: None if sp == "" else "end of block")
    msp = make_msp_parser(indent=indent).reject(lambda sp: None if sp in (" ", "\n") else "whitespace or newline")
    div = pc.regex(r"/(\d+)").map(lambda m: int(m[1:])) | pc.nothing(2)

    instant = enclose_by(pc.proxy(lambda: pattern), msp, pc.string("{"), pc.string("}")).map(Instant)
    division = (
        enclose_by(pc.proxy(lambda: pattern), msp, pc.string("["), pc.string("]")) + div
    ).starmap(lambda a, b: Division(b, a))
    pattern = instant | division | note_parser
    return enclose_by(pattern, msp, pc.nothing(), end)

@IIFE
@pc.parsec
def chart_parser():
    tracks = []

    while True:
        sp = yield make_msp_parser(indent=0).reject(lambda sp: None if sp in ("\n", "") else "newline or end of block")
        if sp == "":
            return tracks

        yield pc.string("TRACK")
        arguments = yield arguments_parser

        track = Track()
        try:
            track.set_arguments(*arguments[0], **arguments[1])
        except Exception as e:
            raise pc.ParseFailure("valid arguments") from e

        yield pc.regex(r":(?=\n)")
        yield make_msp_parser(indent=4).reject(lambda sp: None if sp == "\n" else "newline")

        patterns = yield make_patterns_parser(indent=4)
        track.patterns = patterns
        tracks.append(track)


mstr_parser = pc.regex(
    # always start/end with newline
    r'"""(?=\n)('
    r'(?!""")[^\\\x00]'
    r'|\\[0-7]{1,3}'
    r'|\\x[0-9a-fA-F]{2}'
    r'|\\u[0-9a-fA-F]{4}'
    r'|\\U[0-9a-fA-F]{8}'
    r'|\\(?![xuUN\x00]).'
    r')*(?<=\n)"""',
).map(ast.literal_eval).desc("triple quoted string")

rmstr_parser = pc.regex(
    r'r"""(?=\n)('
    r'(?!""")[^\\\x00]'
    r'|\\[^\x00]'
    r')*(?<=\n)"""',
).map(ast.literal_eval).desc("raw triple quoted string")

@pc.parsec
def make_beatsheet_parser(root, metadata_only=False):
    beatmap_name = "beatmap"
    cls = BeatSheet

    header = yield pc.regex(r"#K-AIKO-std-(\d+\.\d+\.\d+)(?=\n|$)").desc("header")
    vernum = header[len("#K-AIKO-std-"):].split(".")
    vernum0 = version.split(".")
    if vernum[0] != vernum0[0] or vernum[1:] > vernum0[1:]:
        raise ValueError("incompatible version")

    prepare = [
        pc.string("from pathlib import Path"),
        pc.string(f"from {cls.__module__} import {cls.__name__}"),
        pc.string(f"{beatmap_name} = {cls.__name__}(root=Path(__file__).resolve().parent)"),
    ]
    for prepare_parser in prepare:
        yield make_msp_parser(indent=0).reject(lambda sp: None if sp is "\n" else "newline")
        yield prepare_parser

    beatsheet = BeatSheet(root=root)

    valid_fields = {}
    for field, typ in beatsheet._fields.items():
        if field == "chart":
            valid_fields[field] = rmstr_parser
        elif typ is str:
            valid_fields[field] = mstr_parser
        elif typ is Path:
            valid_fields[field] = (
                pc.string(f"{beatmap_name}.root / ")
                >> sz.make_parser_from_type_hint(str)
            ).map(lambda path: root / path)
        else:
            valid_fields[field] = sz.make_parser_from_type_hint(typ)

    while True:
        sp = yield make_msp_parser(indent=0).reject(lambda sp: None if sp in ("\n", "") else "newline or end of block")
        if sp == "":
            return beatsheet

        yield pc.string(f"{beatmap_name}.")
        name = yield pc.tokens([field + " = " for field in valid_fields.keys()])
        name = name[:-3]

        value = yield valid_fields[name]
        del valid_fields[name]

        if not metadata_only or name != "chart":
            subfield = beatsheet
            for field in name.split(".")[:-1]:
                subfield = getattr(subfield, field)
            setattr(subfield, name.split(".")[-1], value)


def format_value(value):
    if value is None:
        return "None"
    elif isinstance(value, (bool, int, float)):
        return repr(value)
    elif isinstance(value, Fraction):
        return str(value)
    elif isinstance(value, str):
        return '"' + repr(value + '"')[1:-2].replace('"', r'\"').replace(r"\'", "'") + '"'
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

def format_chart(chart):
    return "".join(
        f"\nTRACK{format_arguments([], track.get_arguments())}:\n"
        f"    {format_patterns(track.patterns)}\n"
        for track in chart
    )

def format_mstr(value):
    if not value.startswith("\n") or not value.endswith("\n"):
        raise ValueError("string should start and end with newline")
    return '"""' + repr(value + '"')[1:-2].replace('"', r'\"').replace(r"\'", "'").replace(r"\n", "\n") + '"""'

def format_rmstr(value):
    if not value.startswith("\n") or not value.endswith("\n"):
        raise ValueError("string should start and end with newline")
    m = re.search(r'\x00|\r|"""|\\$', value)
    if m:
        raise ValueError("string cannot contain '\\x00', '\\r', '\"\"\"' and single '\\'")
    return 'r"""' + value + '"""'

def format_beatsheet(beatsheet):
    res = []
    beatmap_name = "beatmap"
    cls = BeatSheet

    res.append(f"#K-AIKO-std-{version}\n")
    res.append("from pathlib import Path\n")
    res.append(f"from {cls.__module__} import {cls.__name__}\n")
    res.append("\n")
    res.append(f"{beatmap_name} = {cls.__name__}(root=Path(__file__).resolve().parent)\n")

    for name in BeatSheet.__annotations__.keys():
        format_field = format_mstr if name == "info" else sz.format_value
        res.append(f"{beatmap_name}.{name} = {format_field(getattr(beatsheet, name))}\n")

    name = 'chart'
    format_field = format_rmstr
    res.append(f"{beatmap_name}.{name} = {format_field(getattr(beatsheet, name))}\n")

    return "".join(res)


class OSU:
    def read(self, filename, metadata_only=False):
        path = Path(filename).resolve().parent
        index = 0

        with open(filename, encoding='utf-8-sig') as file:
            format = file.readline()
            index += 1
            # if format != "osu file format v14\n":
            #     raise BeatmapParseError(f"invalid file format: {repr(format)}")

            beatmap = beatmaps.Beatmap()
            beatmap.root = path
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

        return beatmap

    def parse_general(self, beatmap, context, line):
        option, value = line.split(": ", maxsplit=1)
        if option == 'AudioFilename':
            beatmap.audio.path = beatmap.root / value.rstrip("\n")
        elif option == 'PreviewTime':
            beatmap.audio.preview = int(value)/1000

    def parse_editor(self, beatmap, context, line): pass

    def parse_metadata(self, beatmap, context, line):
        beatmap.info += line
        if re.match("(Title|TitleUnicode|Artist|ArtistUnicode|Source):", line):
            beatmap.audio.info += line

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
            beatmap.metronome.offset = time/1000
            beatmap.metronome.tempo = 60 / (beatLength/1000)

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

        beat = beatmap.metronome.beat(time/1000)
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
