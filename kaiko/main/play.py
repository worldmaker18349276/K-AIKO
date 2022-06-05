import os
import shutil
import traceback
import dataclasses
import zipfile
from pathlib import Path
from ..utils import datanodes as dn
from ..utils import commands as cmd
from ..beats import beatmaps
from ..beats import beatsheets
from .loggers import Logger
from .files import RecognizedFilePath, RecognizedDirPath, RecognizedWildCardPath, as_pattern, as_child, FileManager, PathParser
from .profiles import ProfileManager


def format_info(info, logger):
    data = dict(
        tuple(line.split(":", maxsplit=1)) for line in info.strip().splitlines()
    )
    return logger.format_dict(data)


def _rm_beatmap(provider, path):
    beatmap_manager = provider.get(BeatmapManager)
    logger = provider.get(Logger)
    path = path.relative_to(beatmap_manager.beatmaps_dir)
    beatmap_manager.remove(logger, path)


class BeatmapFilePath(RecognizedFilePath):
    def desc(self, provider):
        beatmap_manager = provider.get(BeatmapManager)
        path = self.abs.relative_to(beatmap_manager.beatmaps_dir)
        if beatmap_manager.is_beatmap(path):
            return self.__doc__
        else:
            return "(Untracked beatmap)"

    def rm(self, provider):
        _rm_beatmap(provider, self.abs)


class BeatmapsDirPath(RecognizedDirPath):
    "(The place to hold your beatmaps)"

    def mk(self, provider):
        self.abs.mkdir()

    @as_pattern("*")
    class beatmapset(RecognizedDirPath):
        def desc(self, provider):
            beatmap_manager = provider.get(BeatmapManager)
            path = self.abs.relative_to(beatmap_manager.beatmaps_dir)
            if beatmap_manager.is_beatmapset(path):
                return "(Beatmapset of a song)"
            else:
                return "(Untracked beatmapset)"

        def rm(self, provider):
            _rm_beatmap(provider, self.abs)

        @as_pattern("*.kaiko")
        class beatmap_KAIKO(BeatmapFilePath):
            "(Beatmap file in kaiko format)"

        @as_pattern("*.ka")
        class beatmap_KA(BeatmapFilePath):
            "(Beatmap file in kaiko format)"

        @as_pattern("*.osu")
        class beatmap_OSU(BeatmapFilePath):
            "(Beatmap file in osu format)"

        @as_pattern("**")
        class inner_file(RecognizedWildCardPath):
            "(Inner file of this beatmapset)"

    @as_pattern("*.osz")
    class beamap_zip(RecognizedFilePath):
        "(Compressed beatmapset file)"


class BeatmapManager:
    def __init__(self, beatmaps_dir):
        self.beatmaps_dir = beatmaps_dir
        self._beatmaps = {}
        self._beatmaps_mtime = None

    def is_uptodate(self):
        return self._beatmaps_mtime == os.stat(str(self.beatmaps_dir)).st_mtime

    def is_beatmapset(self, path):
        return path in self._beatmaps

    def is_beatmap(self, path):
        return path.parent in self._beatmaps and path in self._beatmaps[path.parent]

    def get_beatmap_metadata(self, path):
        if not self.is_beatmap(path):
            raise ValueError(f"Not a beatmap: {str(path)}")

        filepath = self.beatmaps_dir / path
        try:
            beatmap = beatsheets.read(str(filepath), metadata_only=True)
        except beatsheets.BeatmapParseError:
            return None
        else:
            return beatmap

    def get_song(self, path):
        if self.is_beatmapset(path):
            if not self._beatmaps[path]:
                return None
            path = self._beatmaps[path][0]

        beatmap = self.get_beatmap_metadata(path)
        if beatmap is None or beatmap.audio is None or beatmap.audio.path is None:
            return None
        return Song(self.beatmaps_dir / path.parent, beatmap.audio)

    def get_songs(self):
        songs = [self.get_song(path) for path in self._beatmaps.keys()]
        return [song for song in songs if song is not None]

    def reload(self, logger):
        beatmaps_dir = self.beatmaps_dir

        logger.print(
            f"[data/] Load beatmaps from {logger.as_uri(beatmaps_dir)}..."
        )

        for file in beatmaps_dir.iterdir():
            if file.is_file() and file.suffix == ".osz":
                distpath = file.parent / file.stem
                if distpath.exists():
                    continue
                logger.print(f"[data/] Unzip file {logger.as_uri(file)}...")
                distpath.mkdir()
                zf = zipfile.ZipFile(str(file), "r")
                zf.extractall(path=str(distpath))
                file.unlink()

        logger.print("[data/] Load beatmaps...")

        beatmaps_mtime = os.stat(str(beatmaps_dir)).st_mtime
        self._beatmaps = {}

        for song in beatmaps_dir.iterdir():
            if song.is_dir():
                beatmapset = []
                for beatmap in song.iterdir():
                    if beatmap.suffix in (".kaiko", ".ka", ".osu"):
                        beatmapset.append(beatmap.relative_to(beatmaps_dir))
                if beatmapset:
                    self._beatmaps[song.relative_to(beatmaps_dir)] = beatmapset

        if len(self._beatmaps) == 0:
            logger.print("[data/] There is no song in the folder yet!")
        logger.print(flush=True)
        self._beatmaps_mtime = beatmaps_mtime

    def add(self, logger, path):
        beatmaps_dir = self.beatmaps_dir

        if not path.exists():
            logger.print(f"[warn]File not found: {logger.as_uri(path)}[/]")
            return
        if not path.is_file() and not path.is_dir():
            logger.print(
                f"[warn]Not a file or directory: {logger.as_uri(path)}[/]"
            )
            return

        logger.print(f"[data/] Add a new song from {logger.as_uri(path)}...")

        distpath = beatmaps_dir / path.name
        n = 1
        while distpath.exists():
            n += 1
            distpath = beatmaps_dir / f"{path.stem} ({n}){path.suffix}"
        if n != 1:
            logger.print(
                f"[data/] Name conflict! Rename to {logger.emph(distpath.name, type='all')}"
            )

        if path.is_file():
            shutil.copy(str(path), str(beatmaps_dir))
        elif path.is_dir():
            shutil.copytree(str(path), str(distpath))

        self.reload(logger)

    def remove(self, logger, path):
        beatmaps_dir = self.beatmaps_dir

        if self.is_beatmap(path):
            beatmap_path = beatmaps_dir / path
            logger.print(
                f"[data/] Remove the beatmap at {logger.as_uri(beatmap_path)}..."
            )
            beatmap_path.unlink()
            self.reload(logger)

        elif self.is_beatmapset(path):
            beatmapset_path = beatmaps_dir / path
            logger.print(
                f"[data/] Remove the beatmapset at {logger.as_uri(beatmapset_path)}..."
            )
            shutil.rmtree(str(beatmapset_path))
            self.reload(logger)

        else:
            logger.print(
                f"[warn]Not a beatmap or beatmapset: {logger.as_uri(path)}[/]"
            )

    def make_parser(self, logger, root=".", type="file"):
        return BeatmapParser(root, type, self, logger)


class BeatmapParser(PathParser):
    def __init__(self, root, type, beatmap_manager, logger):
        desc = "It should be a path to the beatmap file"
        super().__init__(root, type=type, desc=desc, filter=self.filter)
        self.beatmap_manager = beatmap_manager
        self.logger = logger

    def filter(self, path):
        path = Path(path)

        try:
            path = path.resolve(strict=True)
        except:
            return False

        if not path.is_relative_to(self.beatmap_manager.beatmaps_dir):
            return False

        path = path.relative_to(self.beatmap_manager.beatmaps_dir)

        return (
            self.beatmap_manager.is_beatmapset(path)
            or self.beatmap_manager.is_beatmap(path)
        )

    def parse(self, token):
        path = super().parse(token)
        path = Path(self.root) / path

        try:
            path = path.resolve(strict=True)
        except:
            raise cmd.CommandParseError(f"Failed to resolve path: {path!s}")

        if not path.is_relative_to(self.beatmap_manager.beatmaps_dir):
            desc = self.desc()
            raise cmd.CommandParseError(
                "Out of beatmaps directory" + ("\n" + desc if desc is not None else "")
            )

        path = path.relative_to(self.beatmap_manager.beatmaps_dir)
        is_beatmapset = self.beatmap_manager.is_beatmapset(path)
        is_beatmap = self.beatmap_manager.is_beatmap(path)

        if self.type == "dir" and not is_beatmapset:
            desc = self.desc()
            raise cmd.CommandParseError(
                "Not a beatmapset" + ("\n" + desc if desc is not None else "")
            )

        if self.type == "file" and not is_beatmap:
            desc = self.desc()
            raise cmd.CommandParseError(
                "Not a beatmap file" + ("\n" + desc if desc is not None else "")
            )

        if self.type == "all" and not is_beatmapset and not is_beatmap:
            desc = self.desc()
            raise cmd.CommandParseError(
                "Not a beatmapset or a beatmap file" + ("\n" + desc if desc is not None else "")
            )

        return path

    def info(self, token):
        path = Path(self.root) / (token or ".")

        try:
            path = path.resolve(strict=True)
        except:
            return None

        if not path.is_relative_to(self.beatmap_manager.beatmaps_dir):
            return None

        path = path.relative_to(self.beatmap_manager.beatmaps_dir)

        if not self.beatmap_manager.is_beatmap(path):
            return None

        beatmap = self.beatmap_manager.get_beatmap_metadata(path)

        if beatmap is None or not beatmap.info.strip():
            return None

        data = dict(
            tuple(line.split(":", maxsplit=1))
            for line in beatmap.info.strip().splitlines()
        )
        return "[rich]" + self.logger.format_dict(data, show_border=False)


@dataclasses.dataclass(frozen=True)
class Song:
    root: Path
    audio: beatmaps.BeatmapAudio

    @property
    def path(self):
        if self.audio.path is None:
            return None
        return self.root / self.audio.path

    @property
    def relpath(self):
        path = self.path
        if path is None:
            return None
        return Path("~") / path.relative_to(Path("~").expanduser())


class PlayCommand:
    def __init__(self, provider, resources_dir, cache_dir):
        self.provider = provider
        self.resources_dir = resources_dir
        self.cache_dir = cache_dir

    @property
    def logger(self):
        return self.provider.get(Logger)

    @property
    def profile_manager(self):
        return self.provider.get(ProfileManager)

    @property
    def file_manager(self):
        return self.provider.get(FileManager)

    @property
    def beatmap_manager(self):
        return self.provider.get(BeatmapManager)

    # beatmaps

    @cmd.function_command
    def play(self, beatmap, start=None):
        """[rich]Let's beat with the song!

        usage: [cmd]play[/] [arg]{beatmap}[/] [[[kw]--start[/] [arg]{START}[/]]]
                       ╱                   ╲
           Path, the path to the        The time to start playing
          beatmap you want to play.    in the middle of the beatmap,
          Only the beatmaps in your      if you want.
        beatmaps folder can be accessed.
        """

        if not self.beatmap_manager.is_beatmap(beatmap):
            self.logger.print("[warn]Not a beatmap.[/]")
            return

        return KAIKOPlay(
            self.resources_dir,
            self.cache_dir,
            self.beatmap_manager.beatmaps_dir / beatmap,
            start,
            self.profile_manager,
            self.logger,
        )

    @cmd.function_command
    def loop(self, pattern, tempo: float = 120.0, offset: float = 1.0):
        """[rich]Beat with the pattern repeatly.

        usage: [cmd]loop[/] [arg]{pattern}[/] [[[kw]--tempo[/] [arg]{TEMPO}[/]]] [[[kw]--offset[/] [arg]{OFFSET}[/]]]
                        ╱                  ╲                  ╲
            text, the pattern     float, the tempo of         float, the offset time
                to repeat.     pattern; default is 120.0.    at start; default is 1.0.
        """

        return KAIKOLoop(
            pattern, tempo, offset, self.resources_dir, self.cache_dir, self.profile_manager, self.logger,
        )

    @loop.arg_parser("pattern")
    def _loop_pattern_parser(self):
        return cmd.RawParser(
            desc="It should be a pattern.", default="x x o x | x [x x] o _"
        )

    @play.arg_parser("beatmap")
    def _play_beatmap_parser(self):
        current = self.file_manager.current.abs
        return self.beatmap_manager.make_parser(self.logger, current, type="file")

    @play.arg_parser("start")
    def _play_start_parser(self, beatmap):
        return cmd.TimeParser(0.0)

    @cmd.function_command
    def reload(self):
        """[rich]Reload your beatmaps.

        usage: [cmd]reload[/]
        """
        self.beatmap_manager.reload(self.logger)

    @cmd.function_command
    def add(self, beatmap):
        """[rich]Add beatmap/beatmapset to your beatmaps folder.

        usage: [cmd]add[/] [arg]{beatmap}[/]
                        ╲
              Path, the path to the
             beatmap you want to add.
             You can drop the file to
           the terminal to paste its path.
        """

        self.beatmap_manager.add(self.logger, beatmap)

    @add.arg_parser("beatmap")
    def _add_beatmap_parser(self):
        return PathParser()

    @cmd.function_command
    def remove(self, beatmap):
        """[rich]Remove beatmap/beatmapset in your beatmaps folder.

        usage: [cmd]remove[/] [arg]{beatmap}[/]
                           ╲
                 Path, the path to the
               beatmap you want to remove.
        """

        self.beatmap_manager.remove(self.logger, beatmap)

    @remove.arg_parser("beatmap")
    def _remove_beatmap_parser(self):
        current = self.file_manager.current.abs
        return self.beatmap_manager.make_parser(self.logger, current, type="all")


def print_hints(logger, settings):
    pause_key = settings.controls.pause_key
    skip_key = settings.controls.skip_key
    stop_key = settings.controls.stop_key
    display_keys = settings.controls.display_delay_adjust_keys
    knock_keys = settings.controls.knock_delay_adjust_keys
    energy_keys = settings.controls.knock_energy_adjust_keys
    logger.print(
        f"[hint/] Press {logger.emph(pause_key, type='all')} to pause/resume the game."
    )
    logger.print(f"[hint/] Press {logger.emph(skip_key, type='all')} to skip time.")
    logger.print(f"[hint/] Press {logger.emph(stop_key, type='all')} to end the game.")
    logger.print(
        f"[hint/] Use {logger.emph(display_keys[0], type='all')} and "
        f"{logger.emph(display_keys[1], type='all')} to adjust display delay."
    )
    logger.print(
        f"[hint/] Use {logger.emph(knock_keys[0], type='all')} and "
        f"{logger.emph(knock_keys[1], type='all')} to adjust hit delay."
    )
    logger.print(
        f"[hint/] Use {logger.emph(energy_keys[0], type='all')} and "
        f"{logger.emph(energy_keys[1], type='all')} to adjust hit strength."
    )


class KAIKOPlay:
    def __init__(self, resources_dir, cache_dir, filepath, start, profile_manager, logger):
        self.resources_dir = resources_dir
        self.cache_dir = cache_dir
        self.filepath = filepath
        self.start = start
        self.profile_manager = profile_manager
        self.logger = logger

    @dn.datanode
    def execute(self, manager):
        logger = self.logger
        devices_settings = self.profile_manager.current.devices
        gameplay_settings = self.profile_manager.current.gameplay

        try:
            beatmap = beatsheets.read(str(self.filepath))

        except beatsheets.BeatmapParseError:
            logger.print(
                f"[warn]Failed to read beatmap {logger.as_uri(self.filepath)}[/]"
            )
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)

        else:
            print_hints(logger, gameplay_settings)
            logger.print()

            score, devices_settings = yield from beatmap.play(
                manager,
                self.resources_dir,
                self.cache_dir,
                self.start,
                devices_settings,
                gameplay_settings,
            ).join()

            logger.print()
            logger.print_scores(
                beatmap.settings.difficulty.performance_tolerance, score.perfs
            )

            if devices_settings is not None:
                logger.print()
                yes = yield from self.logger.ask(
                    "Keep changes to device settings?"
                ).join()
                if yes:
                    logger.print("[data/] Update device settings...")
                    title = self.profile_manager.get_title()
                    old = self.profile_manager.format()
                    self.profile_manager.current.devices = devices_settings
                    self.profile_manager.set_as_changed()
                    new = self.profile_manager.format()

                    self.logger.print(f"[data/] Your changes")
                    logger.print(
                        logger.format_code_diff(old, new, title=title, is_changed=True)
                    )


class KAIKOLoop:
    def __init__(self, pattern, tempo, offset, resources_dir, cache_dir, profile_manager, logger):
        self.pattern = pattern
        self.tempo = tempo
        self.offset = offset
        self.resources_dir = resources_dir
        self.cache_dir = cache_dir
        self.profile_manager = profile_manager
        self.logger = logger

    @dn.datanode
    def execute(self, manager):
        logger = self.logger
        devices_settings = self.profile_manager.current.devices
        gameplay_settings = self.profile_manager.current.gameplay

        try:
            track, width = beatmaps.BeatTrack.parse(self.pattern, ret_width=True)

        except beatsheets.BeatmapParseError:
            logger.print("[warn]Failed to parse pattern.[/]")
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)

        else:
            beatmap = beatmaps.Loop(
                tempo=self.tempo, offset=self.offset, width=width, track=track
            )

            print_hints(logger, gameplay_settings)
            logger.print()

            score, devices_settings = yield from beatmap.play(
                manager,
                self.resources_dir,
                self.cache_dir,
                None,
                devices_settings,
                gameplay_settings,
            ).join()

            logger.print()
            logger.print_scores(
                beatmap.settings.difficulty.performance_tolerance, score.perfs
            )

            if devices_settings is not None:
                logger.print()
                yes = yield from self.logger.ask(
                    "Keep changes to device settings?"
                ).join()
                if yes:
                    logger.print("[data/] Update device settings...")
                    title = self.profile_manager.get_title()
                    old = self.profile_manager.format()
                    self.profile_manager.current.devices = devices_settings
                    self.profile_manager.set_as_changed()
                    new = self.profile_manager.format()

                    self.logger.print(f"[data/] Your changes")
                    logger.print(
                        logger.format_code_diff(old, new, title=title, is_changed=True)
                    )
