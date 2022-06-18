import os
import math
from fractions import Fraction
import re
import ast
from ..utils import parsec as pc
from ..utils import serializers as sz
from . import beatmaps

version = "0.3.0"


class BeatmapParseError(Exception):
    pass


def read(filename, hack=False, metadata_only=False):
    filename = os.path.abspath(filename)
    if filename.endswith((".kaiko", ".ka")):
        sheet = open(filename).read()

        try:
            if hack:
                local = {}
                exec(sheet, {"__file__": filename}, local)
                beatmap = local["beatmap"]
            else:
                beatmap = make_beatmap_parser(
                    filename, metadata_only=metadata_only
                ).parse(sheet)
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


mstr_parser = (
    pc.regex(
        # always start/end with newline
        r'"""(?=\n)('
        r'(?!""")[^\\\x00]'
        r"|\\[0-7]{1,3}"
        r"|\\x[0-9a-fA-F]{2}"
        r"|\\u[0-9a-fA-F]{4}"
        r"|\\U[0-9a-fA-F]{8}"
        r"|\\(?![xuUN\x00])."
        r')*(?<=\n)"""',
    )
    .map(ast.literal_eval)
    .desc("triple quoted string")
)

rmstr_parser = (
    pc.regex(r'r"""(?=\n)(' r'(?!""")[^\\\x00]' r"|\\[^\x00]" r')*(?<=\n)"""')
    .map(ast.literal_eval)
    .desc("raw triple quoted string")
)


def make_mstr_serializer(suggestions=[]):
    parser = (
        pc.regex(
            # always start/end with newline
            r'"""(?=\n)('
            r'(?!""")[^\\\x00]'
            r"|\\[0-7]{1,3}"
            r"|\\x[0-9a-fA-F]{2}"
            r"|\\u[0-9a-fA-F]{4}"
            r"|\\U[0-9a-fA-F]{8}"
            r"|\\(?![xuUN\x00])."
            r')*(?<=\n)"""',
        )
        .map(ast.literal_eval)
        .desc("triple quoted string")
    )

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


_beatmap_fields = {
    "info": str,
    "audio.path": str,
    "audio.volume": float,
    "audio.preview": float,
    "metronome.offset": float,
    "metronome.tempo": float,
    "beatbar_state.bar_shift": float,
    "beatbar_state.bar_flip": bool,
}


@pc.parsec
def make_beatmap_parser(filepath, metadata_only=False):
    beatmap_name = "beatmap"
    Beatmap = beatmaps.Beatmap
    BeatTrack = beatmaps.BeatTrack

    # parse header
    header = yield pc.regex(r"#K-AIKO-std-(\d+\.\d+\.\d+)(?=\n|$)").desc("header")
    vernum = header[len("#K-AIKO-std-") :].split(".")
    vernum0 = version.split(".")
    if vernum[0] != vernum0[0] or vernum[1:] > vernum0[1:]:
        raise ValueError("incompatible version")

    # parse imports, initialization
    prepare = [
        pc.string("from kaiko.beats.beatmaps import Beatmap, BeatTrack"),
        pc.string(f"{beatmap_name} = Beatmap(__file__)"),
    ]
    for prepare_parser in prepare:
        yield make_msp_parser(indent=0).reject(
            lambda sp: None if sp is "\n" else "newline"
        )
        yield prepare_parser

    beatmap = Beatmap(filepath)

    # parse fields
    valid_fields = dict(_beatmap_fields)
    valid_fields["tracks"] = dict

    while True:
        sp = yield make_msp_parser(indent=0).reject(
            lambda sp: None if sp in ("\n", "") else "newline or end of block"
        )
        if sp == "":
            return beatmap

        yield pc.string(f"{beatmap_name}.")
        name = yield pc.tokens(list(valid_fields.keys()))

        if name != "tracks":
            yield pc.string(" = ")
            field_type = valid_fields[name]
            value = (
                yield mstr_parser
                if name == "info"
                else sz.make_serializer_from_type_hint(field_type).parser
            )
            del valid_fields[name]

            subfield = beatmap
            for field in name.split(".")[:-1]:
                subfield = getattr(subfield, field)
            setattr(subfield, name.split(".")[-1], value)

        else:
            # parse track:

            # beatmap.tracks["main"] = BeatTrack.parse(r"""
            # ...
            # """)

            track_name = (
                yield pc.string("[")
                >> sz.make_str_serializer().parser
                << pc.string("]")
            )
            if track_name in beatmap.tracks:
                raise ValueError("duplicated name")
            yield pc.string(" = ")
            track_str = (
                yield pc.string("BeatTrack.parse(") >> rmstr_parser << pc.string(")")
            )

            track = BeatTrack.parse(track_str) if not metadata_only else BeatTrack([])
            beatmap.tracks[track_name] = track


def format_mstr(value):
    if not value.startswith("\n") or not value.endswith("\n"):
        raise ValueError("string should start and end with newline")
    return (
        '"""'
        + repr(value + '"')[1:-2]
        .replace('"', r"\"")
        .replace(r"\'", "'")
        .replace(r"\n", "\n")
        + '"""'
    )


def format_rmstr(value):
    if not value.startswith("\n") or not value.endswith("\n"):
        raise ValueError("string should start and end with newline")
    m = re.search(r'\x00|\r|"""|\\$', value)
    if m:
        raise ValueError(
            "string cannot contain '\\x00', '\\r', '\"\"\"' and single '\\'"
        )
    return 'r"""' + value + '"""'


def format_beatmap(beatmap):
    res = []
    beatmap_name = "beatmap"

    res.append(f"#K-AIKO-std-{version}\n")
    res.append("from kaiko.beats.beatmaps import Beatmap, BeatTrack\n")
    res.append("\n")
    res.append(f"{beatmap_name} = Beatmap(__file__)\n")

    for name, typ in _beatmap_fields.items():
        format_field = (
            format_mstr
            if name == "info"
            else sz.make_serializer_from_type_hint(typ).formatter
        )
        res.append(f"{beatmap_name}.{name} = {format_field(getattr(beatmap, name))}\n")

    format_str = sz.make_serializer_from_type_hint(str).formatter
    for track_name, track in beatmap.tracks.items():
        track_str = "\n" + track.to_str() + "\n"
        res.append(
            f"{beatmap_name}.tracks[{format_str(track_name)}] = BeatTrack.parse({format_rmstr(track_str)})\n"
        )

    return "".join(res)


class OSU:
    def read(self, filename, metadata_only=False):
        index = 0

        with open(filename, encoding="utf-8-sig") as file:
            format = file.readline()
            index += 1
            # if format != "osu file format v14\n":
            #     raise BeatmapParseError(f"invalid file format: {repr(format)}")

            beatmap = beatmaps.Beatmap(filename)
            beatmap.tracks = {"main": beatmaps.BeatTrack([])}
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
                        if (
                            not metadata_only
                            or parse != self.parse_timingpoints
                            and parse != self.parse_hitobjects
                        ):
                            parse(beatmap, context, line)
                    except Exception as e:
                        raise BeatmapParseError(f"parse error at line {index}") from e

                line = file.readline()
                index += 1

        return beatmap

    def parse_general(self, beatmap, context, line):
        option, value = line.split(": ", maxsplit=1)
        if option == "AudioFilename":
            beatmap.audio.path = value.rstrip("\n")
        elif option == "PreviewTime":
            beatmap.audio.preview = int(value) / 1000

    def parse_editor(self, beatmap, context, line):
        pass

    def parse_metadata(self, beatmap, context, line):
        beatmap.info += line

    def parse_difficulty(self, beatmap, context, line):
        option, value = line.split(":", maxsplit=1)
        if option == "SliderMultiplier":
            context["multiplier0"] = float(value)

    def parse_events(self, beatmap, context, line):
        pass

    def parse_timingpoints(self, beatmap, context, line):
        (
            time,
            beatLength,
            meter,
            sampleSet,
            sampleIndex,
            volume,
            uninherited,
            effects,
        ) = line.rstrip("\n").split(",")
        time = float(time)
        beatLength = float(beatLength)
        meter = int(meter)
        volume = float(volume)
        multiplier = context["multiplier0"]

        if "timings" not in context:
            context["timings"] = []
        if "beatLength0" not in context:
            context["beatLength0"] = beatLength
            beatmap.metronome.offset = time / 1000
            beatmap.metronome.tempo = 60 / (beatLength / 1000)

        if uninherited == "0":
            multiplier = multiplier / (-0.01 * beatLength)
            beatLength = context["timings"][-1][1]
            meter = context["timings"][-1][2]

        speed = multiplier / 1.4
        volume = 20 * math.log10(volume / 100)
        sliderVelocity = (multiplier * 100) / (beatLength / context["beatLength0"])
        density = (8 / meter) / (beatLength / context["beatLength0"])  # 8 per measure

        context["timings"].append(
            (time, beatLength, meter, speed, volume, sliderVelocity, density)
        )

    def parse_colours(self, beatmap, context, line):
        pass

    def parse_hitobjects(self, beatmap, context, line):
        x, y, time, type, hitSound, *objectParams, hitSample = line.rstrip("\n").split(
            ","
        )
        time = float(time)
        type = int(type)
        hitSound = int(hitSound)

        beat = beatmap.metronome.beat(time / 1000)
        speed, volume, sliderVelocity, density = next(
            (vs for t, b, m, *vs in context["timings"][::-1] if t <= time),
            context["timings"][0][3:],
        )

        # type : [_:_:_:_:Spinner:_:Slider:Circle]
        # hitSound : [Kat:Large:Kat:Don]

        if type & 1:  # circle
            if hitSound == 0 or hitSound & 1:  # don
                event = beatmaps.Soft(
                    beat=beat, length=Fraction(0), speed=speed, volume=volume
                )
                beatmap.tracks["main"].events.append(event)

            elif hitSound & 10:  # kat
                event = beatmaps.Loud(
                    beat=beat, length=Fraction(0), speed=speed, volume=volume
                )
                beatmap.tracks["main"].events.append(event)

        elif type & 2:  # slider
            # curve,slides,sliderLength,edgeSounds,edgeSets = objectParams
            sliderLength = float(objectParams[2]) if len(objectParams) >= 3 else 0.0
            sliderLength = float(sliderLength)
            length = sliderLength / sliderVelocity

            event = beatmaps.Roll(
                beat=beat, length=length, density=density, speed=speed, volume=volume
            )
            beatmap.tracks["main"].events.append(event)

        elif type & 8:  # spinner
            (end_time,) = objectParams
            end_time = float(end_time)
            length = (end_time - time) / context["beatLength0"]

            event = beatmaps.Spin(
                beat=beat, length=length, density=density, speed=speed, volume=volume
            )
            beatmap.tracks["main"].events.append(event)


OSU_FORMAT = OSU()
