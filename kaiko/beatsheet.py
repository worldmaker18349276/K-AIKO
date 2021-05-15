import os
import math
from collections import OrderedDict, namedtuple
from fractions import Fraction
import operator
import inspect
from ast import literal_eval
from lark import Lark, Transformer
from .beatmap import (
    Beatmap,
    Text, Flip, Shift, set_context,
    Soft, Loud, Incr, Roll, Spin,
    )

Rest = lambda: None
Divisor = lambda divisor=2: divisor

class BeatmapParseError(Exception):
    pass

class BeatmapDraft(Beatmap):
    def __init__(self, root=".", info="", audio=None, volume=0.0, preview=0.0, offset=0.0, tempo=60.0):
        super().__init__(None, volume, offset, tempo)
        self.root = root
        self.info = info
        self.audio = audio
        self.preview = preview

        self.notations = self.NotationDict()
        self.chart = self.NoteChart(self.notations)

        self.notations['x'] = Soft
        self.notations['soft'] = Soft
        self.notations['o'] = Loud
        self.notations['loud'] = Loud
        self.notations['<'] = Incr
        self.notations['incr'] = Incr
        self.notations['%'] = Roll
        self.notations['roll'] = Roll
        self.notations['@'] = Spin
        self.notations['spin'] = Spin
        self.notations['TEXT'] = Text
        self.notations['CONTEXT'] = set_context
        self.notations['FLIP'] = Flip
        self.notations['SHIFT'] = Shift

    def build_events(self):
        events = []

        for track in self.chart.tracks:
            for event in track.build_events(self):
                events.append(event)

        return events

    @staticmethod
    def read(filename, hack=False):
        filename = os.path.abspath(filename)
        if filename.endswith((".kaiko", ".ka")):
            if hack:
                beatmap = BeatmapDraft()
                beatmap.root = os.path.dirname(filename)
                exec(open(filename).read(), dict(), dict(beatmap=beatmap))
                return beatmap

            else:
                try:
                    return K_AIKO_STD_FORMAT.read(filename)
                except Exception as e:
                    raise BeatmapParseError(f"failed to read beatmap {filename}") from e

        elif filename.endswith(".osu"):
            try:
                return OSU_FORMAT.read(filename)
            except Exception as e:
                raise BeatmapParseError(f"failed to read beatmap {filename}") from e

        else:
            raise ValueError(f"unknown file extension: {filename}")

    class NotationDict:
        def __init__(self):
            self.definitions = {}

        def __setitem__(self, symbol, builder):
            if any(c in symbol for c in " \b\t\n\r\f\v()[]{}\'\"\\#|~") or symbol == '_':
                raise ValueError(f"invalid symbol `{symbol}`")
            if symbol in self.definitions:
                raise ValueError(f"symbol `{symbol}` is already defined")
            self.definitions[symbol] = NoteType(symbol, builder)

        def __delitem__(self, symbol):
            del self.definitions[symbol]

        def __getitem__(self, symbol):
            return self.definitions[symbol].builder

    class NoteChart:
        def __init__(self, notations):
            self.notations = notations
            self.tracks = []

        def __iadd__(self, track_str):
            try:
                track = K_AIKO_STD_FORMAT.parse_track(track_str, self.notations.definitions)
            except Exception as e:
                raise BeatmapParseError("failed to read track") from e

            self.tracks.append(track)
            return self

class NoteTrack:
    def __init__(self, *, beat=0, length=1, meter=4, hide=False):
        self.beat = beat
        self.length = length
        self.meter = meter
        self.hide = hide
        self.notes = []

    def build_events(self, beatmap):
        if self.hide:
            return
        context = dict()
        for note in self.notes:
            event = note.create(beatmap, context)
            if event is not None:
                yield event

def NoteType(symbol, builder):
    # builder(beatmap, *args, context, **kwargs) -> Event | None
    # => Note(*args, **kwargs)
    signature = inspect.signature(builder)
    parameters = list(signature.parameters.values())[1:]
    contextual = [param.name for param in parameters if param.kind == inspect.Parameter.KEYWORD_ONLY]
    if 'context' in contextual:
        parameters.remove(signature.parameters['context'])
    signature = signature.replace(parameters=parameters)
    attrs = dict(symbol=symbol, builder=staticmethod(builder), signature=signature, contextual=contextual)
    return type(builder.__name__+"Note", (Note,), attrs)

class Note:
    modifiers = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
        '&': operator.and_,
        '|': operator.or_,
        '^': operator.xor,
        }

    def __init__(self, *psargs, **kwargs):
        self.mods = dict()

        # find modified assignments: pretend `f(key+=value)` by `f(**{'key+': value})`
        for key, value in list(kwargs.items()):
            for mod in self.modifiers.keys():
                if key.endswith(mod):
                    # get original key
                    key_ = key[:-len(mod)]
                    if key_ in kwargs:
                        raise ValueError("keyword argument repeated")
                    if key_ not in self.contextual:
                        raise ValueError("unable to modify non-contextual argument")

                    # extract out modifier
                    self.mods[key_] = mod
                    del kwargs[key]
                    kwargs[key_] = value

                    break

        # bind arguments, it allows missing contextual arguments until obtaining context
        self.bound = self.signature.bind_partial(*psargs, **kwargs)

        # check non-contextual arguments
        for key, param in self.signature.parameters.items():
            if key not in self.contextual:
                if key not in self.bound.arguments and param.default == inspect.Parameter.empty:
                    raise ValueError("missing required arguments")

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        psargs_str = [repr(value) for value in self.bound.args]
        kwargs_str = [key+self.mods.get(key, "")+"="+repr(value) for key, value in self.bound.kwargs.items()]
        args_str = ", ".join([*psargs_str, *kwargs_str])
        return f"{self.symbol}({args_str})"

    def create(self, beatmap, context):
        args = self.bound.args
        kwargs = dict(self.bound.kwargs)
        ikwargs = dict()

        # separate modified contextual arguments
        for key in self.contextual:
            if key in kwargs and key in self.mods:
                ikwargs[key] = kwargs[key]
                del kwargs[key]

        # fill in missing contextual arguments by context
        for key in self.contextual:
            if key != 'context' and key not in kwargs and key in context:
                kwargs[key] = context[key]

        # bind unmodified arguments. it will raise error for missing contextual arguments
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        args = bound.args
        kwargs = dict(bound.kwargs)

        # modify contextual arguments if needed
        for key in self.contextual:
            if key in self.mods:
                mod = self.mods[key]
                kwargs[key] = self.modifiers[mod](kwargs[key], ikwargs[key])

        # build
        if 'context' in self.contextual:
            kwargs['context'] = context
        return self.builder(beatmap, *args, **kwargs)


# #K-AIKO-std-1.1.0
# beatmap.info = '''
# ...
# '''
# beatmap.audio = '...'
# beatmap.volume = -20.0
# beatmap.preview = 34.5
# beatmap.offset = 2.44
# beatmap.tempo = 140.0
# beatmap.chart += r'''
# (beat=0, length=1, meter=4)
# x x o x | x x [x x] o | x x [x x] x | [x x x x] [_ x] x |
# %(2) ~ ~ ~ | < < < < | < < < < | @(2) ~ ~ ~ |
# '''

k_aiko_grammar = r"""
_NEWLINE: /\r\n?|\n/
_WHITESPACE: /[ \t]+/
_COMMENT: /#[^\r\n]*/
_MCOMMENT: /#((?!''')[^\r\n])*/
_n: (_NEWLINE _COMMENT?)* _NEWLINE
_e: (_NEWLINE _COMMENT?)*
_s: (_WHITESPACE | (_NEWLINE _MCOMMENT?)* _NEWLINE)*

none:  /None/
bool:  /True|False/
int:   /[-+]?(0|[1-9][0-9]*)/
frac:  /[-+]?(0|[1-9][0-9]*)\/[1-9][0-9]*/
float: /[-+]?[0-9]+\.[0-9]+/
str:   /'([^\r\n\\']|\\[\\'btnrfv]|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})*'/
mstr:  /'''(\r\n?|\n)((?!''')[\s\S])*((?<=\r\n)|(?<=\r)|(?<=\n))'''/

value: none | bool | float | frac | int | str
key: /[a-zA-Z_][a-zA-Z0-9_]*/
mod: /[-+*\/&|^]/
arg:             value  ->  pos
   | key     "=" value  ->  kw
   | key mod "=" value  ->  ikw
arguments: ["(" [ arg (", " arg)* ] ")"]

symbol: /[^ \b\t\n\r\f\v()[\]{}'"\\#|~]+/
note: symbol arguments
text: str arguments
lengthen: "~"
measure: "|"
division: "[" arguments pattern "]"
instant: "{" arguments pattern "}"
pattern: _s ((lengthen | measure | note | text | division | instant) _s)*
track: _s arguments pattern

version: /[0-9]+\.[0-9]+\.[0-9]+(?=\r\n?|\n|$)/
header: "#K-AIKO-std-" version
info:   [_n "beatmap.info"   " = " mstr]
audio:  [_n "beatmap.audio"  " = " str]
volume: [_n "beatmap.volume" " = " float]
preview: [_n "beatmap.preview" " = " float]
offset: [_n "beatmap.offset" " = " float]
tempo:  [_n "beatmap.tempo"  " = " float]
chart: (_n "beatmap.chart"  " += " _MSTR_PREFIX track _MSTR_POSTFIX)*
_MSTR_PREFIX: /r'''(?=\r\n?|\n)/
_MSTR_POSTFIX: /((?<=\r\n)|(?<=\r)|(?<=\n))'''/

contents: info audio volume preview offset tempo chart _e
std: header contents

contents_str: (/[\s\S]+/)?
std_header: header contents_str

track_str: /(\r\n?|\n)((?!''')[\s\S])*/
track_strs: (_n "beatmap.chart"  " += " _MSTR_PREFIX track_str _MSTR_POSTFIX)*
std_metadata: info audio volume preview offset tempo track_strs _e
"""

class K_AIKO_STD_Transformer(Transformer):
    # defaults: std, header, contents, std_header, std_metadata,
    #           track, note, text, lengthen, measure, division, instant
    chart = pattern = track_strs = lambda self, args: args
    version = symbol = key = value = mod = track_str = lambda self, args: args[0]
    contents_str = lambda self, args: "" if len(args) == 0 else args[0]
    info = audio = volume = preview = offset = tempo = lambda self, args: None if len(args) == 0 else args[0]
    none = bool = int = float = str = mstr = lambda self, args: literal_eval(args[0])
    frac = lambda self, args: Fraction(args[0])
    pos = lambda self, args: (None, args[0])
    kw = lambda self, args: (args[0], args[1])
    ikw = lambda self, args: (args[0]+args[1], args[2])

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
    version = "0.1.0"
    std_parser = Lark(k_aiko_grammar, start='std')
    std_header_parser = Lark(k_aiko_grammar, start='std_header')
    std_metadata_parser = Lark(k_aiko_grammar, start='std_metadata')
    std_track_parser = Lark(k_aiko_grammar, start='track')
    transformer = K_AIKO_STD_Transformer()

    def read(self, filename):
        path = os.path.dirname(filename)
        str = open(filename, newline="").read()
        beatmap = self.parse_beatmap(str, path)
        return beatmap

    def parse_beatmap(self, str, path):
        # parse header
        node = self.transformer.transform(self.std_header_parser.parse(str))
        header, contents_str = node.children
        version, = header.children

        vernum = version.split(".")
        vernum0 = self.version.split(".")
        if vernum[0] != vernum0[0] or vernum[1:] > vernum0[1:]:
            raise BeatmapParseError("incompatible version")

        # parse metadata
        contents = self.transformer.transform(self.std_metadata_parser.parse(contents_str))
        info, audio, volume, preview, offset, tempo, track_strs = contents.children

        beatmap = BeatmapDraft()

        beatmap.root = path
        if info is not None:
            beatmap.info = info
        if audio is not None:
            beatmap.audio = audio
            beatmap.audiopath = os.path.join(beatmap.root, beatmap.audio)
        if volume is not None:
            beatmap.volume = volume
        if preview is not None:
            beatmap.preview = preview
        if offset is not None:
            beatmap.offset = offset
        if tempo is not None:
            beatmap.tempo = tempo

        # parse chart
        for track_str in track_strs:
            track = self.parse_track(track_str, beatmap.notations.definitions)
            beatmap.chart.tracks.append(track)

        return beatmap

    def parse_track(self, str, definitions):
        node = self.transformer.transform(self.std_track_parser.parse(str))
        (args, kwargs), pattern = node.children

        track = NoteTrack(*args, **kwargs)
        self.load_pattern(pattern, track, track.beat, track.length, None, definitions)

        return track

    def _make_note(self, notetype, args, kwargs, beat, length):
        # modify beat and length if needed
        modifiers = notetype.modifiers
        for mod in modifiers.keys():
            if 'beat'+mod in kwargs:
                if 'beat' in kwargs:
                    raise ValueError("keyword argument repeated")
                kwargs['beat'] = modifiers[mod](beat, kwargs['beat'+mod])
                del kwargs['beat'+mod]

            if 'length'+mod in kwargs:
                if 'length' in kwargs:
                    raise ValueError("keyword argument repeated")
                kwargs['length'] = modifiers[mod](length, kwargs['length'+mod])
                del kwargs['length'+mod]

        # make note
        note = notetype(*args, **kwargs)

        # assign missing beat/length
        if 'beat' not in note.bound.arguments:
            note.bound.arguments['beat'] = beat
        if 'length' not in note.bound.arguments:
            note.bound.arguments['length'] = length

        return note

    def load_pattern(self, pattern, track, beat, length, note, definitions):
        for node in pattern:
            if node.data == 'note':
                symbol, (args, kwargs) = node.children

                if symbol == '_':
                    note = Rest(*args, **kwargs)

                elif symbol in definitions:
                    note = self._make_note(definitions[symbol], args, kwargs, beat, length)
                    track.notes.append(note)

                else:
                    raise BeatmapParseError(f"undefined symbol {symbol}")

                beat = beat + length

            elif node.data == 'text':
                text, (args, kwargs) = node.children
                args.insert(0, text)

                note = self._make_note(definitions['TEXT'], args, kwargs, beat, length)
                track.notes.append(note)

                beat = beat + length

            elif node.data == 'lengthen':
                if note is not None and 'length' in note.bound.arguments:
                    note.bound.arguments['length'] = note.bound.arguments['length'] + length

                beat = beat + length

            elif node.data == 'measure':
                if (beat - track.beat) % track.meter != 0:
                    raise BeatmapParseError("wrong measure")

            elif node.data == 'division':
                (args, kwargs), pattern = node.children

                divisor = Divisor(*args, **kwargs)
                divided_length = Fraction(1, 1) * length / divisor

                track, beat, _, note = self.load_pattern(pattern, track, beat, divided_length, note, definitions)

            elif node.data == 'instant':
                (args, kwargs), pattern = node.children

                if len(args) + len(kwargs) > 0:
                    note = definitions['CONTEXT'](*args, **kwargs)
                    track.notes.append(note)

                track, beat, _, note = self.load_pattern(pattern, track, beat, 0, note, definitions)

            else:
                raise BeatmapParseError(f"unknown node {str(node)}")

        return track, beat, length, note

K_AIKO_STD_FORMAT = K_AIKO_STD()

class OSU:
    def read(self, filename):
        path = os.path.dirname(filename)
        index = 0

        with open(filename, encoding='utf-8-sig') as file:
            format = file.readline()
            index += 1
            # if format != "osu file format v14\n":
            #     raise BeatmapParseError(f"invalid file format: {repr(format)}")

            beatmap = BeatmapDraft()
            beatmap.root = path
            beatmap.chart.tracks.append(NoteTrack())
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

        return beatmap

    def parse_general(self, beatmap, context, line):
        option, value = line.split(": ", maxsplit=1)
        if option == 'AudioFilename':
            beatmap.audio = value.rstrip("\n")
            beatmap.audiopath = os.path.join(beatmap.root, beatmap.audio)
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
            beatmap.chart.tracks[0].meter = meter

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
        definitions = beatmap.notations.definitions

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
                note = definitions['x'](beat=beat, speed=speed, volume=volume)
                beatmap.chart.tracks[0].notes.append(note)

            elif hitSound & 10: # kat
                note = definitions['o'](beat=beat, speed=speed, volume=volume)
                beatmap.chart.tracks[0].notes.append(note)

        elif type & 2: # slider
            # curve,slides,sliderLength,edgeSounds,edgeSets = objectParams
            sliderLength = float(objectParams[2]) if len(objectParams) >= 3 else 0.0
            sliderLength = float(sliderLength)
            length = sliderLength / sliderVelocity

            note = definitions['%'](density=density, beat=beat, length=length, speed=speed, volume=volume)
            beatmap.chart.tracks[0].notes.append(note)

        elif type & 8: # spinner
            end_time, = objectParams
            end_time = float(end_time)
            length = (end_time - time)/context['beatLength0']

            note = definitions['@'](density=density, beat=beat, length=length, speed=speed, volume=volume)
            beatmap.chart.tracks[0].notes.append(note)

OSU_FORMAT = OSU()
