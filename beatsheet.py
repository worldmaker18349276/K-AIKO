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
_n: (_WHITESPACE? _COMMENT? _NEWLINE)+
_s: (_WHITESPACE? _COMMENT? _NEWLINE)* _WHITESPACE?
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
arg:               value  ->  pos
   | key _s "=" _s value  ->  kw
arguments: ["(" _s [ arg (_s "," _s arg)* _s ] ")"]

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
    version = "1.1.0"
    k_aiko_std_parser = Lark(k_aiko_grammar, start="k_aiko_std")
    chart_parser = Lark(k_aiko_grammar, start="chart")
    transformer = K_AIKO_STD_Transformer()

    def __init__(self, Beatmap, NoteChart):
        self.Beatmap = Beatmap
        self.NoteChart = NoteChart

    def read(self, str):
        # beatmap = self.Beatmap()
        # exec(str, dict(), dict(beatmap=beatmap))
        # return beatmap
        return self.load_beatmap(self.transformer.transform(self.k_aiko_std_parser.parse(str)))

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

        if version != self.version:
            raise ValueError("wrong version")

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

