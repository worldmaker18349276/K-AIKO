import os
import math
from collections import OrderedDict, namedtuple
from fractions import Fraction
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Union
from ast import literal_eval
from .biparser import (
    startswith, match,
    EncodeError, DecodeError,
    Biparser, LiteralBiparser,
    from_type_hint,
)
from .beatmap import (
    Beatmap,
    UpdateContext, Text, Flip, Shift,
    Soft, Loud, Incr, Roll, Spin,
)

def Context(beat, length, **update):
    return UpdateContext(update)

class BeatmapParseError(Exception):
    pass

class BeatSheet(Beatmap):
    _notations = {
        'x': Soft,
        'o': Loud,
        '<': Incr,
        '%': Roll,
        '@': Spin,
        'Context': Context,
        'Text': Text,
        'Flip': Flip,
        'Shift': Shift,
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

    def _to_events(self, track):
        if track.hide:
            return

        notations = self._notations

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

    def to_events(self, track):
        return list(self._to_events(track))

    def to_track(self, sequence):
        raise NotImplementedError

    @property
    def chart(self):
        tracks = [self.to_track(seq) for seq in self.event_sequences]
        return chart_biparser.encode(tracks)

    @chart.setter
    def chart(self, value):
        tracks, _ = chart_biparser.decode(value)
        self.event_sequences = [self.to_events(track) for track in tracks]

    @staticmethod
    def read(filename, hack=False):
        filename = os.path.abspath(filename)
        if filename.endswith((".kaiko", ".ka")):
            sheet = open(filename).read()

            try:
                if hack:
                    beatmap = BeatSheet()
                    exec(sheet, dict(), dict(beatmap=beatmap))
                else:
                    beatmap, _ = beatsheet_biparser.decode(sheet)
            except Exception as e:
                raise
                # raise BeatmapParseError(f"failed to read beatmap {filename}") from e

            beatmap.root = os.path.dirname(filename)
            return beatmap

        elif filename.endswith(".osu"):
            try:
                return OSU_FORMAT.read(filename)
            except Exception as e:
                raise BeatmapParseError(f"failed to read beatmap {filename}") from e

        else:
            raise ValueError(f"unknown file extension: {filename}")


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
    patterns: List[Pattern] = field(default_factory=list)

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
    #     text note: 'ABC'(arg=...)
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
    patterns: List[Pattern] = field(default_factory=list)

@dataclass
class Instant(Pattern):
    # {x x o}
    patterns: List[Pattern] = field(default_factory=list)


class MStrBiparser(LiteralBiparser):
    # always start/end with newline => easy to parse
    # no named escape sequence '\N{...}'

    regex = (r'"""(?=\n)('
             r'(?!""")[^\\\x00]'
             r'|\\[0-7]{1,3}'
             r'|\\x[0-9a-fA-F]{2}'
             r'|\\u[0-9a-fA-F]{4}'
             r'|\\U[0-9a-fA-F]{8}'
             r'|\\(?![xuUN\x00]).'
             r')*(?<=\n)"""')
    expected = ['"""\n"""']
    type = str

    def encode(self, value):
        if not value.startswith("\n") or not value.endswith("\n"):
            raise EncodeError(value, "", [], info="it should start and end with newline")
        return '"""' + repr(value + '"')[1:-2].replace('"', r'\"').replace(r"\'", "'").replace(r"\n", "\n") + '"""'

class RMStrBiparser(LiteralBiparser):
    regex = (r'r"""(?=\n)('
             r'(?!""")[^\\\x00]'
             r'|\\[^\x00]'
             r')*(?<=\n)"""')
    expected = ['r"""\n"""']
    type = str

    def encode(self, value):
        if not value.startswith("\n") or not value.endswith("\n"):
            raise EncodeError(value, "", [], info="it should start and end with newline")

        m = re.search(r'\x00|\r|"""|\\$', value)
        if m:
            raise EncodeError(value, f"[{m.start()}]", [],
                info="unable to repr '\\x00', '\\r', '\"\"\"' and single '\\' as raw string")

        return 'r"""' + value + '"""'


class ValueBiparser(Biparser):
    none  = r"None"
    bool  = r"True|False"
    int   = r"[-+]?(0|[1-9][0-9]*)"
    frac  = r"[-+]?(0|[1-9][0-9]*)\/[1-9][0-9]*"
    float = r"[-+]?([0-9]+\.[0-9]+(e[-+]?[0-9]+)?|[0-9]+[eE][-+]?[0-9]+)"
    str   = r'"([^\r\n\\"]|\\[\\"btnrfv]|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})*"'

    def encode(self, value):
        if value is None:
            return "None"
        elif isinstance(value, (bool, int, float)):
            return repr(value)
        elif isinstance(value, Fraction):
            return str(value)
        elif isinstance(value, str):
            return '"' + repr(value + '"')[1:-2].replace('"', r'\"').replace(r"\'", "'") + '"'
        else:
            raise EncodeError(value, "", (None, bool, int, float, Fraction, str))

    def decode(self, text, index=0, partial=False):
        for regex in [self.none, self.bool, self.str, self.float, self.frac, self.int]:
            res, index = match(regex, ["None"], text, index, optional=True, partial=partial)
            if res:
                if regex == self.frac:
                    return Fraction(res.group()), index
                else:
                    return literal_eval(res.group()), index

        raise DecodeError(text, index, ["None"])
value_biparser = ValueBiparser()

class ArgumentsBiparser(Biparser):
    key = r"([a-zA-Z_][a-zA-Z0-9_]*)="

    def encode(self, value):
        psargs, kwargs = value

        psargs_str = [value_biparser.encode(value) for value in psargs]
        kwargs_str = [key+"="+value_biparser.encode(value) for key, value in kwargs.items()]

        if len(psargs_str) + len(kwargs_str) == 0:
            return ""

        return "(" + ", ".join([*psargs_str, *kwargs_str]) + ")"

    def decode(self, text, index=0, partial=False):
        psargs = []
        kwargs = {}
        keyworded = False

        m, index = startswith(["("], text, index, optional=True, partial=True)
        if not m: return (psargs, kwargs), index
        m, index = startswith([")"], text, index, optional=True, partial=partial)
        if m: return (psargs, kwargs), index

        while True:
            key, index = match(self.key, ["k="], text, index, optional=not keyworded, partial=True)
            value, index = value_biparser.decode(text, index, partial=True)
            if key:
                keyworded = True
                kwargs[key.group(1)] = value
            else:
                psargs.append(value)

            m, index = startswith([")"], text, index, optional=True, partial=partial)
            if m: return (psargs, kwargs), index

            _, index = startswith([", "], text, index, partial=True)
arguments_biparser = ArgumentsBiparser()

class NoteBiparser(Biparser):
    symbol = (r"[^ \b\t\n\r\f\v()[\]{}\'\"\\#]+"
              r'|"([^\r\n\\"\x00]|\\[\\"btnrfv]|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})*"')

    def encode(self, value):
        return value.symbol + arguments_biparser.encode(value.arguments)

    def decode(self, text, index=0, partial=False):
        m, index = match(self.symbol, [], text, index, partial=True)
        symbol = m.group(0)
        arguments, index = arguments_biparser.decode(text, index, partial=partial)

        if symbol.startswith('"'):
            arguments = ([literal_eval(symbol), *arguments[0]], arguments[1])
            symbol = 'Text'

        return Note(symbol, arguments), index
note_biparser = NoteBiparser()

def parse_msp(text, index, indent=0, optional=False):
    # return None  ->  no match
    # return ""    ->  end of block
    # return " "   ->  whitespace between token
    # return "\n"  ->  newline between token (including comments)

    sp = r"[ \t]+"
    eol = r"(?=\n|$)"
    nl = r"\n+([ ]{" + str(indent) + r",})"
    cmt = r"#[^\n]*"

    # whitespace
    m_sp, index = match(sp, [], text, index, optional=True, partial=True)
    if m_sp:
        m_cmt, _ = match(cmt, [], text, index, optional=True, partial=True)
        if m_cmt:
            raise DecodeError(text, index, [], info="comment should occupy a whole line")

    # end of line
    m_eol, index = match(eol, [], text, index, optional=True, partial=True)
    if not m_eol:
        if m_sp:
            return " ", index
        elif optional:
            return None, index
        else:
            raise DecodeError(text, index, [])

    # newline
    m_cmt = None
    m_nl, index = match(nl, [], text, index, optional=True, partial=True)
    while m_nl:
        if m_nl.group(1) != " "*indent:
            raise DecodeError(text, index, [], info="wrong indentation level")

        m_eol, index = match(eol, [], text, index, optional=True, partial=True)
        m_cmt, index = match(cmt, [], text, index, optional=True, partial=True)
        if not m_eol and not m_cmt:
            return "\n", index

        m_nl, index = match(nl, [], text, index, optional=True, partial=True)

    else:
        return "", index

class PatternsBiparser(Biparser):
    def encode(self, value):
        patterns_str = []
        for pattern in value:
            if isinstance(pattern, Instant):
                pattern_str = "{" + self.encode(value.patterns) + "}"

            elif isinstance(pattern, Division):
                div = "" if value.divisor == 2 else f"/{value.divisor}"
                pattern_str = "[" + self.encode(value.patterns) + "]" + div

            elif isinstance(pattern, Note):
                pattern_str = note_biparser.encode(pattern)

            patterns_str.append(pattern_str)

        return " ".join(patterns_str)

    def decode(self, text, index=0, partial=False, closed_by=None, indent=0):
        sp = " "
        patterns = []

        while True:
            # end of block
            if sp == "":
                if closed_by:
                    raise DecodeError(text, index, [])
                else:
                    return patterns, index

            # closing bracket
            if closed_by:
                closed, index = match(closed_by, [], text, index, optional=True, partial=partial)
                if closed:
                    return patterns, index

            # no space
            if sp is None:
                raise DecodeError(text, index, [])

            # pattern
            opened, index = startswith(["{", "["], text, index, optional=True, partial=True)
            if opened == "{":
                _, index = parse_msp(text, index, indent=indent, optional=True)
                subpatterns, index = self.decode(text, index, partial=True, closed_by=r"\}", indent=indent)
                pattern = Instant(subpatterns)

            elif opened == "[":
                _, index = parse_msp(text, index, indent=indent, optional=True)
                subpatterns, index = self.decode(text, index, partial=True, closed_by=r"\]", indent=indent)
                m, index = match(r"/(\d+)", [""], text, index, optional=True, partial=True)
                divisor = int(m.group(1)) if m else 2
                pattern = Division(divisor, subpatterns)

            else:
                pattern, index = note_biparser.decode(text, index, partial=True)

            patterns.append(pattern)

            # spacing
            sp, index = parse_msp(text, index, indent=indent, optional=True)
patterns_biparser = PatternsBiparser()

class ChartBiparser(Biparser):
    def encode(self, value):
        res = ""
        for track in value:
            kwargs = track.get_arguments()
            res += "\nTRACK" + arguments_biparser.encode(([], kwargs)) + ":\n"
            res += "    " + patterns_biparser.encode(track.patterns) + "\n"

    def decode(self, text, index=0, partial=False):
        tracks = []

        while True:
            sp, index = parse_msp(text, index, indent=0)

            if sp == "":
                return tracks, index

            elif sp == "\n":
                _, index = startswith(["TRACK"], text, index, partial=True)
                arguments, index = arguments_biparser.decode(text, index, partial=True)
                _, index = match(r":(?=\n)", [":"], text, index, partial=True)

                track = Track()
                try:
                    track.set_arguments(*arguments[0], **arguments[1])
                except Exception:
                    raise DecodeError(text, index, [])

                sp, index = parse_msp(text, index, indent=4, optional=True)
                if sp != "\n":
                    raise DecodeError(text, index, ["\n    "])

                patterns, index = patterns_biparser.decode(text, index, partial=True, indent=4)
                track.patterns = patterns

                tracks.append(track)

            else:
                raise ValueError("impossible condition")
chart_biparser = ChartBiparser()

class BeatSheetBiparser(Biparser):
    version = "0.2.0"

    def encode(self, value):
        fields = BeatSheet.__annotations__

        sheet = ""
        sheet += "#K-AIKO-std-" + self.version + "\n"

        for name, type_hint in fields.items():
            if name == "info":
                field_biparser = MStrBiparser()
            else:
                field_biparser = from_type_hint(fields[name])

            sheet += "beatmap." + name + " = " + field_biparser.encode(getattr(value, name)) + "\n"

        field_biparser = RMStrBiparser()
        sheet += "beatmap.chart = " + field_biparser.encode(getattr(value, 'chart')) + "\n"

    def decode(self, text, index=0, partial=False):
        m, index = match(r"#K-AIKO-std-(\d+\.\d+\.\d+)(?=\n|$)",
                                  ["#K-AIKO-std-" + self.version],
                                  text, index, partial=True)
        version = m.group(1)

        vernum = version.split(".")
        vernum0 = self.version.split(".")
        if vernum[0] != vernum0[0] or vernum[1:] > vernum0[1:]:
            raise BeatmapParseError("incompatible version")

        beatsheet = BeatSheet()
        fields = BeatSheet.__annotations__
        is_set = []
        after_chart = False

        while True:
            prev_index = index
            sp, index = parse_msp(text, index, indent=0)
            if sp == "":
                return beatsheet, index
            if sp == " ":
                raise DecodeError(text, prev_index, ["\n"])

            m, index = match(r"beatmap\.([_a-zA-Z][_a-zA-Z0-9]*) = ", [], text, index, partial=True)
            name = m.group(1)
            if name not in fields and name != "chart":
                raise DecodeError(text, index, [], info=f"unknown field {name}")
            if name in is_set:
                raise DecodeError(text, index, [], info=f"field {name} has been set")
            if name != "chart" and after_chart:
                raise DecodeError(text, index, [], info=f"field {name} should be set before chart")
            is_set.append(name)

            if name == "info":
                field_biparser = MStrBiparser()
            elif name == "chart":
                field_biparser = RMStrBiparser()
                after_chart = True
            else:
                field_biparser = from_type_hint(fields[name])

            value, index = field_biparser.decode(text, index, partial=True)

            setattr(beatsheet, name, value)
beatsheet_biparser = BeatSheetBiparser()


class OSU:
    def read(self, filename):
        path = os.path.dirname(filename)
        index = 0

        with open(filename, encoding='utf-8-sig') as file:
            format = file.readline()
            index += 1
            # if format != "osu file format v14\n":
            #     raise BeatmapParseError(f"invalid file format: {repr(format)}")

            beatmap = Beatmap()
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

        # type: [_:_:_:_:Spinner:_:Slider:Circle]
        # hitSound: [Kat:Large:Kat:Don]

        if type & 1: # circle
            if hitSound == 0 or hitSound & 1: # don
                event = Soft(beat=beat, length=0, speed=speed, volume=volume)
                beatmap.event_sequences[0].append(event)

            elif hitSound & 10: # kat
                event = Loud(beat=beat, length=0, speed=speed, volume=volume)
                beatmap.event_sequences[0].append(event)

        elif type & 2: # slider
            # curve,slides,sliderLength,edgeSounds,edgeSets = objectParams
            sliderLength = float(objectParams[2]) if len(objectParams) >= 3 else 0.0
            sliderLength = float(sliderLength)
            length = sliderLength / sliderVelocity

            event = Roll(beat=beat, length=length, density=density, speed=speed, volume=volume)
            beatmap.event_sequences[0].append(event)

        elif type & 8: # spinner
            end_time, = objectParams
            end_time = float(end_time)
            length = (end_time - time)/context['beatLength0']

            event = Spin(beat=beat, length=length, density=density, speed=speed, volume=volume)
            beatmap.event_sequences[0].append(event)

OSU_FORMAT = OSU()
