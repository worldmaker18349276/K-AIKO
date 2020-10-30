import math
from collections import OrderedDict, namedtuple
from fractions import Fraction
import inspect
from ast import literal_eval
from lark import Lark, Transformer


# #K-AIKO-std-1.1.0
# beatmap.info = '''
# ...
# '''
# beatmap.audio = '...'
# beatmap.offset = 2.44
# beatmap.tempo = 140.0
# beatmap += r'''
# (beat=0, length=1, meter=4)
# x x o x | x x [x x] o | x x [x x] x | [x x x x] [_ x] x |
# %(2) ~ ~ ~ | < < < < | < < < < | @(2) ~ ~ ~ |
# '''

k_aiko_grammar = r"""
_NEWLINE: /\r\n?|\n/
_WHITESPACE: /[ \t]+/
_COMMENT: /\#[^\r\n]*/
_MCOMMENT: /\#((?![\r\n]|''').)*/
_n: (_WHITESPACE? _COMMENT? _NEWLINE)+
_s: (_WHITESPACE? _MCOMMENT? _NEWLINE)* _WHITESPACE?
_e: (_WHITESPACE? _COMMENT? _NEWLINE)* _WHITESPACE? _COMMENT?

none:  /None/
bool:  /True|False/
int:   /[-+]?(0|[1-9][0-9]*)/
frac:  /[-+]?(0|[1-9][0-9]*)\/[1-9][0-9]*/
float: /[-+]?[0-9]+\.[0-9]+/
str:   /'((?![\\\r\n]|').|\\.|\\\r\n)*'/s
mstr:  /'''((?!\\|''').|\\.)*'''/s

value: none | bool | float | frac | int | str
key: /[a-zA-Z_][a-zA-Z0-9_]*/
arg:         value  ->  pos
   | key "=" value  ->  kw
arguments: ["(" [ arg (", " arg)* ] ")"]

symbol: /[^ \b\t\n\r\f\v()[\]{}\'"\\#|~]+/
note: symbol arguments
lengthen: "~"
measure: "|"
division: "[" arguments pattern "]"
instant: "{" pattern "}"
pattern: _s ((lengthen | measure | note | division | instant) _s)*
chart: _s arguments pattern

version: /[0-9]+\.[0-9]+\.[0-9]+(?=\r\n?|\n|$)/
header: "#K-AIKO-std-" version
info:   [_n "beatmap.info"   " = " mstr]
audio:  [_n "beatmap.audio"  " = " str]
offset: [_n "beatmap.offset" " = " float]
tempo:  [_n "beatmap.tempo"  " = " float]
charts: (_n "beatmap += r'''\n" chart "\n'''")*

contents: info audio offset tempo charts _e
k_aiko_std: header contents
"""

class K_AIKO_STD_Transformer(Transformer):
    # defaults: k_aiko_std, header, contents,
    #           chart, note, lengthen, measure, division, instant
    charts = pattern = lambda self, args: args
    version = symbol = key = value = lambda self, args: args[0]
    info = audio = offset = tempo = lambda self, args: None if len(args) == 0 else args[0]
    none = bool = int = float = str = mstr = lambda self, args: literal_eval(args[0])
    frac = lambda self, args: Fraction(args[0])
    pos = lambda self, args: (None, args[0])
    kw = lambda self, args: (args[0], args[1])

    def arguments(self, args):
        psargs = []
        kwargs = dict()

        for key, value in args:
            if key is None:
                if len(kwargs) > 0:
                    raise ValueError("positional argument follows keyword argument")
                psargs.append(value)
            else:
                if key in kwargs:
                    raise ValueError("keyword argument repeated")
                kwargs[key] = value

        return psargs, kwargs

class K_AIKO_STD:
    version = "1.1.1"
    k_aiko_std_parser = Lark(k_aiko_grammar, start="k_aiko_std")
    chart_parser = Lark(k_aiko_grammar, start="chart")
    transformer = K_AIKO_STD_Transformer()

    def __init__(self, Beatmap, NoteChart):
        self.Beatmap = Beatmap
        self.NoteChart = NoteChart

    def read(self, file):
        # beatmap = self.Beatmap()
        # exec(file.read(), dict(), dict(beatmap=beatmap))
        # return beatmap
        return self.load_beatmap(self.transformer.transform(self.k_aiko_std_parser.parse(file.read())))

    def read_chart(self, str, definitions):
        return self.load_chart(self.transformer.transform(self.chart_parser.parse(str)), definitions)

    def _call(self, func, arguments, defaults=dict()):
        psargs, kwargs = arguments
        parameters = inspect.signature(func).parameters
        for key, value in defaults.items():
            if key not in kwargs:
                if key in parameters and parameters[key].kind == inspect.Parameter.KEYWORD_ONLY:
                    kwargs[key] = value
        return func(*psargs, **kwargs)

    def load_beatmap(self, node):
        header, contents = node.children
        version, = header.children
        info, audio, offset, tempo, charts = contents.children

        if version.split(".")[:2] != self.version.split(".")[:2]:
            raise ValueError("incompatible version")

        beatmap = self.Beatmap()

        if info is not None:
            beatmap.info = info
        if audio is not None:
            beatmap.audio = audio
        if offset is not None:
            beatmap.offset = offset
        if tempo is not None:
            beatmap.tempo = tempo

        for chart_node in charts:
            chart = self.load_chart(chart_node, beatmap.definitions)
            beatmap.charts.append(chart)

        return beatmap

    def load_chart(self, node, definitions):
        arguments, pattern = node.children

        chart = self._call(self.NoteChart, arguments)
        self.load_pattern(pattern, chart, chart.beat, chart.length, None, definitions)

        return chart

    def load_pattern(self, pattern, chart, beat, length, note, definitions):
        for node in pattern:
            if node.data == "note":
                symbol, arguments = node.children

                if symbol == "_":
                    note = self._call((lambda:None), arguments)

                elif symbol in definitions:
                    note = self._call(definitions[symbol], arguments, dict(beat=beat, length=length))
                    chart.notes.append(note)

                else:
                    raise ValueError(f"undefined symbol {node.symbol}")

                beat = beat + length

            elif node.data == "lengthen":
                if note is not None and "length" in note.bound.arguments:
                    note.length = note.length + length

                beat = beat + length

            elif node.data == "measure":
                if (beat - chart.beat) % chart.meter != 0:
                    raise ValueError("wrong measure")

            elif node.data == "division":
                arguments, pattern = node.children
                divisor = self._call((lambda divisor=2:divisor), arguments)
                divided_length = Fraction(1, 1) * length / divisor
                chart, beat, _, note = self.load_pattern(pattern, chart, beat, divided_length, note, definitions)

            elif node.data == "instant":
                pattern, = node.children
                chart, beat, _, note = self.load_pattern(pattern, chart, beat, 0, note, definitions)

            else:
                raise ValueError(f"unknown node {str(node)}")

        return chart, beat, length, note


class OSU:
    def __init__(self, Beatmap, NoteChart):
        self.Beatmap = Beatmap
        self.NoteChart = NoteChart

    def read(self, file):
        format = file.readline()
        if format != "osu file format v14\n":
            raise ValueError(f"invalid file format: {repr(format)}")

        beatmap = self.Beatmap()
        beatmap.charts.append(self.NoteChart())
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
                parse(beatmap, context, line)

            line = file.readline()

        return beatmap

    def parse_general(self, beatmap, context, line):
        option, value = line.split(": ", maxsplit=1)
        if option == "AudioFilename":
            beatmap.audio = value.rstrip("\n")

    def parse_editor(self, beatmap, context, line): pass

    def parse_metadata(self, beatmap, context, line):
        beatmap.info += line

    def parse_difficulty(self, beatmap, context, line):
        option, value = line.split(":", maxsplit=1)
        if option == "SliderMultiplier":
            context["multiplier0"] = float(value)

    def parse_events(self, beatmap, context, line): pass

    def parse_timingpoints(self, beatmap, context, line):
        time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects = line.rstrip("\n").split(",")
        time = int(time)
        beatLength = float(beatLength)
        meter = int(meter)
        volume = int(volume)
        multiplier = context["multiplier0"]

        if "timings" not in context:
            context["timings"] = []
        if "beatLength0" not in context:
            context["beatLength0"] = beatLength
            beatmap.offset = time/1000
            beatmap.tempo = 60 / (beatLength/1000)
            beatmap.charts[0].meter = meter

        if uninherited == "0":
            multiplier = multiplier / (-0.01 * beatLength)
            beatLength = context["timings"][-1][1]
            meter = context["timings"][-1][2]

        speed = multiplier / 1.4
        volume = 20 * math.log10(volume / 100)
        sliderVelocity = (multiplier * 100) / (beatLength/context["beatLength0"])
        density = (8/meter) / (beatLength/context["beatLength0"]) # 8 per measure

        context["timings"].append((time, beatLength, meter, speed, volume, sliderVelocity, density))

    def parse_colours(self, beatmap, context, line): pass

    def parse_hitobjects(self, beatmap, context, line):
        x,y,time,type,hitSound,*objectParams,hitSample = line.rstrip("\n").split(",")
        time = int(time)
        type = int(type)
        hitSound = int(hitSound)

        beat = beatmap.beat(time/1000)
        speed, volume, sliderVelocity, density = next(vs for t, b, m, *vs in context["timings"][::-1] if t <= time)

        # type: [_:_:_:_:Spinner:_:Slider:Circle]
        # hitSound: [Kat:Large:Kat:Don]

        if type & 1: # circle
            if hitSound == 0 or hitSound & 1: # don
                note = beatmap.definitions["x"](beat=beat, speed=speed, volume=volume)
                beatmap.charts[0].notes.append(note)

            elif hitSound & 10: # kat
                note = beatmap.definitions["o"](beat=beat, speed=speed, volume=volume)
                beatmap.charts[0].notes.append(note)

        elif type & 2: # slider
            curve,slides,sliderLength,edgeSounds,edgeSets = objectParams
            sliderLength = float(sliderLength)
            length = sliderLength / sliderVelocity

            note = beatmap.definitions["%"](density=density, beat=beat, length=length, speed=speed, volume=volume)
            beatmap.charts[0].notes.append(note)

        elif type & 8: # spinner
            end_time, = objectParams
            end_time = int(end_time)
            length = (end_time - time)/context["beatLength0"]
            # 10

            note = beatmap.definitions["@"](density=density, beat=beat, length=length, speed=speed, volume=volume)
            beatmap.charts[0].notes.append(note)

