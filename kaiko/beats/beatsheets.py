import os
import math
from fractions import Fraction
from pathlib import Path
import re
import ast
from ..utils import parsec as pc
from ..utils import serializers as sz
from . import beatmaps
from . import beatpatterns

version = "0.3.0"

class BeatmapParseError(Exception):
    pass

class BeatSheet(beatmaps.Beatmap):
    _notations = {
        'x': beatmaps.Soft,
        'o': beatmaps.Loud,
        '<': beatmaps.Incr,
        '%': beatmaps.Roll,
        '@': beatmaps.Spin,
        'Context': beatpatterns.Context,
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

    @property
    def chart(self):
        tracks = [beatpatterns.Track.from_events(seq) for seq in self.event_sequences]
        return beatpatterns.format_chart(tracks)

    @chart.setter
    def chart(self, value):
        tracks = beatpatterns.chart_parser.parse(value)
        self.event_sequences = [list(track.to_events(BeatSheet._notations)) for track in tracks]

    @staticmethod
    def parse_patterns(patterns_str):
        try:
            patterns = beatpatterns.make_patterns_parser().parse(patterns_str)
            track = beatpatterns.Track(beat=0, length=1, meter=4, patterns=patterns)

            events = []
            it = track.to_events(BeatSheet._notations)
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
                    beatmap = make_beatsheet_parser(filename, metadata_only=metadata_only).parse(sheet)
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

def make_mstr_serializer(suggestions=[]):
    parser = pc.regex(
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

    def validator(value):
        if isinstance(value, str) and value.startswith("\n") and value.endswith("\n"):
            return {str}
        else:
            return set()

    def format_mstr(value):
        return '"""%s"""' % "\n".join(
            repr(line + '"')[1:-2].replace('"""', r'\"""').replace(r"\'", "'")
            for line in value.split("\n")
        )

    return sz.Serializer(parser, format_mstr, validator).suggest(suggestions or ["\n"])


@pc.parsec
def make_beatsheet_parser(filepath, metadata_only=False):
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
        pc.string(f"{beatmap_name} = {cls.__name__}()"),
    ]
    for prepare_parser in prepare:
        yield beatpatterns.make_msp_parser(indent=0).reject(lambda sp: None if sp is "\n" else "newline")
        yield prepare_parser

    beatsheet = BeatSheet()
    root = Path(filepath).parent

    valid_fields = {}
    for field, typ in beatsheet._fields.items():
        if field == "chart":
            valid_fields[field] = rmstr_parser
        elif typ is str:
            valid_fields[field] = mstr_parser
        elif typ is Path:
            valid_fields[field] = (
                pc.string("Path(__file__).parent / ")
                >> sz.make_str_serializer().parser
            ).map(lambda path: root / path)
        else:
            valid_fields[field] = sz.make_serializer_from_type_hint(typ).parser

    while True:
        sp = yield beatpatterns.make_msp_parser(indent=0).reject(lambda sp: None if sp in ("\n", "") else "newline or end of block")
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
    res.append(f"from {cls.__module__} import {cls.__name__}\n")
    res.append("\n")
    res.append(f"{beatmap_name} = {cls.__name__}(__file__)\n")

    for name, typ in BeatSheet._fields.items():
        format_field = format_mstr if name == "info" else sz.make_serializer_from_type_hint(typ).formatter
        res.append(f"{beatmap_name}.{name} = {format_field(getattr(beatsheet, name))}\n")

    name = 'chart'
    format_field = format_rmstr
    res.append(f"{beatmap_name}.{name} = {format_field(getattr(beatsheet, name))}\n")

    return "".join(res)


class OSU:
    def read(self, filename, metadata_only=False):
        index = 0

        with open(filename, encoding='utf-8-sig') as file:
            format = file.readline()
            index += 1
            # if format != "osu file format v14\n":
            #     raise BeatmapParseError(f"invalid file format: {repr(format)}")

            beatmap = beatmaps.Beatmap()
            root = Path(filename).parent
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
                            parse(beatmap, root, context, line)
                    except Exception as e:
                        raise BeatmapParseError(f"parse error at line {index}") from e

                line = file.readline()
                index += 1

        return beatmap

    def parse_general(self, beatmap, root, context, line):
        option, value = line.split(": ", maxsplit=1)
        if option == 'AudioFilename':
            beatmap.audio.path = root / value.rstrip("\n")
        elif option == 'PreviewTime':
            beatmap.audio.preview = int(value)/1000

    def parse_editor(self, beatmap, root, context, line): pass

    def parse_metadata(self, beatmap, root, context, line):
        beatmap.info += line
        if re.match("(Title|TitleUnicode|Artist|ArtistUnicode|Source):", line):
            beatmap.audio.info += line

    def parse_difficulty(self, beatmap, root, context, line):
        option, value = line.split(":", maxsplit=1)
        if option == 'SliderMultiplier':
            context['multiplier0'] = float(value)

    def parse_events(self, beatmap, root, context, line): pass

    def parse_timingpoints(self, beatmap, root, context, line):
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

    def parse_colours(self, beatmap, root, context, line): pass

    def parse_hitobjects(self, beatmap, root, context, line):
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
