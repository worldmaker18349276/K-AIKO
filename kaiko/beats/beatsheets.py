import os
import math
from fractions import Fraction
from typing import Optional
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

    def formatter(value):
        return '"""{}"""'.format(
            repr(value + '"')[1:-2]
            .replace('"', r"\"")
            .replace(r"\'", "'")
            .replace(r"\n", "\n")
        )

    return sz.Serializer(parser, formatter, validator).suggest(suggestions or ["\n"])


def make_rmstr_serializer(suggestions=[]):
    parser = (
        pc.regex(
            # always start/end with newline
            r'r"""(?=\n)('
            r'(?!""")[^\\\x00]'
            r"|\\[^\x00]"
            r')*(?<=\n)"""'
        )
        .map(ast.literal_eval)
        .desc("raw triple quoted string")
    )

    def validator(value):
        if isinstance(value, str) and value.startswith("\n") and value.endswith("\n"):
            return {str}
        else:
            return set()

    def formatter(value):
        if re.search(r'\x00|\r|"""|\\$', value):
            raise ValueError(
                "string cannot contain '\\x00', '\\r', '\"\"\"' and single '\\'"
            )
        return f'r"""{value}"""'

    return sz.Serializer(parser, formatter, validator).suggest(suggestions or ["\n"])


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
def make_beatmap_parser(filepath, metadata_only=False):
    beatmap_name = "beatmap"
    Beatmap = beatmaps.Beatmap
    BeatTrack = beatmaps.BeatTrack
    BeatState = beatmaps.BeatState
    BeatPoints = beatmaps.BeatPoints

    # parse header
    header = yield pc.regex(r"#K-AIKO-std-(\d+\.\d+\.\d+)(?=\n|$)").desc("header")
    vernum = [int(v) for v in header[len("#K-AIKO-std-") :].split(".")]
    vernum0 = [int(v) for v in version.split(".")]
    if vernum[0] != vernum0[0] or vernum[1:] > vernum0[1:]:
        raise ValueError("incompatible version")

    # parse imports, initialization
    prepare = [
        pc.string(
            "from kaiko.beats.beatmaps import Beatmap, BeatTrack, BeatState, BeatPoints"
        ),
        pc.string(f"{beatmap_name} = Beatmap(__file__)"),
    ]
    for prepare_parser in prepare:
        yield make_msp_parser(indent=0).reject(
            lambda sp: None if sp is "\n" else "newline"
        )
        yield prepare_parser

    beatmap = Beatmap(filepath)

    # parsers
    mstr_parser = make_mstr_serializer().parser
    rmstr_parser = make_rmstr_serializer().parser
    str_parser = sz.make_str_serializer().parser
    float_parser = sz.make_float_serializer().parser

    # parse fields
    valid_fields = [
        "info",
        "audio.path",
        "audio.volume",
        "audio.preview",
        "beatstate",
        "beatpoints",
        "tracks",
    ]

    while True:
        sp = yield make_msp_parser(indent=0).reject(
            lambda sp: None if sp in ("\n", "") else "newline or end of block"
        )
        if sp == "":
            return beatmap

        yield pc.string(f"{beatmap_name}.")
        name = yield pc.tokens(valid_fields)

        if name == "info":
            yield pc.string(" = ")
            beatmap.info = yield mstr_parser
            valid_fields.remove(name)

        elif name == "audio.path":
            yield pc.string(" = ")
            beatmap.audio.path = yield str_parser
            valid_fields.remove(name)

        elif name == "audio.volume":
            yield pc.string(" = ")
            beatmap.audio.volume = yield float_parser
            valid_fields.remove(name)

        elif name == "audio.preview":
            yield pc.string(" = ")
            beatmap.audio.preview = yield float_parser
            valid_fields.remove(name)

        elif name == "beatpoints":
            # parse beatpoints:

            # beatmap.beatpoints = BeatPoints.parse(r"""
            # ...
            # """)

            yield pc.string(" = ")

            try:
                offset, tempo = yield pc.template(
                    "BeatPoints.fixed(offset={}, tempo={})", float_parser, float_parser
                )
            except pc.ParseFailure:
                pass
            else:
                beatpoints = BeatPoints.fixed(offset=offset, tempo=tempo)
                beatmap.beatpoints = beatpoints
                continue

            (beatpoints_str,) = yield pc.template("BeatPoints.parse({})", rmstr_parser)

            beatpoints = (
                BeatPoints.parse(beatpoints_str)
                if not metadata_only
                else BeatPoints([])
            )
            beatmap.beatpoints = beatpoints

            valid_fields.remove(name)

        elif name == "beatstate":
            # parse beatstate:

            # beatmap.beatstate = BeatState.parse(r"""
            # ...
            # """)

            yield pc.string(" = ")

            shift_parser = sz.make_serializer_from_type_hint(Optional[float]).parser
            flip_parser = sz.make_serializer_from_type_hint(Optional[bool]).parser
            try:
                shift, flip = yield pc.template(
                    "BeatState.fixed(shift={}, flip={})", shift_parser, flip_parser
                )
            except pc.ParseFailure:
                pass
            else:
                beatstate = BeatState.fixed(shift=shift, flip=flip)
                beatmap.beatstate = beatstate
                continue

            (beatbarstate_str,) = yield pc.template("BeatState.parse({})", rmstr_parser)

            beatstate = (
                BeatState.parse(beatbarstate_str)
                if not metadata_only
                else BeatState([])
            )
            beatmap.beatstate = beatstate

            valid_fields.remove(name)

        elif name == "tracks":
            # parse track:

            # beatmap.tracks["main"] = BeatTrack.parse(r"""
            # ...
            # """)

            (track_name,) = yield pc.template("[{}]", sz.make_str_serializer().parser)
            if track_name in beatmap.tracks:
                raise ValueError("duplicated name")
            yield pc.string(" = ")
            (track_str,) = yield pc.template("BeatTrack.parse({})", rmstr_parser)

            track = BeatTrack.parse(track_str) if not metadata_only else BeatTrack([])
            beatmap.tracks[track_name] = track

        else:
            assert False


def format_beatmap(beatmap):
    res = []
    beatmap_name = "beatmap"

    res.append(f"#K-AIKO-std-{version}\n")
    res.append("from kaiko.beats.beatmaps import Beatmap, BeatTrack\n")
    res.append("\n")
    res.append(f"{beatmap_name} = Beatmap(__file__)\n")

    format_mstr = make_mstr_serializer().formatter
    format_rmstr = make_rmstr_serializer().formatter
    format_str = sz.make_str_serializer().formatter
    format_float = sz.make_float_serializer().formatter

    res.append(f"{beatmap_name}.info = {format_mstr(beatmap.info)}\n")
    res.append(f"{beatmap_name}.audio.path = {format_str(beatmap.audio.path)}\n")
    res.append(f"{beatmap_name}.audio.volume = {float_float(beatmap.audio.volume)}\n")
    res.append(f"{beatmap_name}.audio.preview = {float_float(beatmap.audio.preview)}\n")

    res.append(f"{beatmap_name}.beatpoints = BeatPoints.parse({format_rmstr(...)})\n")

    res.append(f"{beatmap_name}.beatstate = BeatState.parse({format_rmstr(...)})\n")

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

        if uninherited == "0":
            multiplier = multiplier / (-0.01 * beatLength)
            beatLength = context["timings"][-1][1]
            meter = context["timings"][-1][2]
        else:
            beat = (
                beatmap.beatpoints.beat(time / 1000)
                if beatmap.beatpoints.is_valid()
                else 0.0
            )
            beatpoint = beatmaps.BeatPoint(
                beat=beat, time=time / 1000, tempo=60 / (beatLength / 1000)
            )
            beatmap.beatpoints.points.append(beatpoint)

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

        beat = beatmap.beatpoints.beat(time / 1000)
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
