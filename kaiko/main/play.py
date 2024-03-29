import shutil
import dataclasses
import zipfile
from collections import defaultdict
from typing import Optional, Dict
from pathlib import Path
from ..utils import markups as mu
from ..utils import datanodes as dn
from ..utils import commands as cmd
from ..utils import providers
from ..devices import clocks
from ..devices import audios as aud
from ..beats import beatmaps
from ..beats import beatsheets
from .loggers import Logger
from .devices import DeviceManager
from .files import (
    RecognizedFilePath,
    RecognizedDirPath,
    UnmovablePath,
    as_pattern,
    InvalidFileOperation,
    FileManager,
)
from .profiles import ProfileManager


class BeatmapFilePath(RecognizedFilePath):
    "Beatmap of a song"

    def info_detailed(self):
        beatmap_manager = providers.get(BeatmapManager)
        logger = providers.get(Logger)

        beatmap = beatmap_manager.get_beatmap_metadata(self)

        if beatmap is None or not beatmap.info.strip():
            return None

        data = dict(
            tuple(line.split(":", maxsplit=1))
            for line in beatmap.info.strip().splitlines()
        )
        return "[rich]" + logger.format_dict(data, show_border=False)

    def rm(self):
        beatmap_manager = providers.get(BeatmapManager)
        succ = beatmap_manager.remove_beatmap(self)
        if not succ:
            return
        beatmap_manager.update_beatmapsdir()
        beatmap_manager.update_beatmapset(self.parent)
        beatmap_manager.update_beatmap(self)

    def cp(self, src):
        beatmap_manager = providers.get(BeatmapManager)
        succ = beatmap_manager.add_beatmap(self, src)
        if not succ:
            return
        beatmap_manager.update_beatmapsdir()
        beatmap_manager.update_beatmapset(self.parent)
        beatmap_manager.update_beatmap(self)

    @property
    def parent(self):
        return BeatmapsetDirPath(self.abs.parent, True)


class BeatmapsetDirPath(RecognizedDirPath):
    "Beatmapset of a song"

    def rm(self):
        beatmap_manager = providers.get(BeatmapManager)
        succ = beatmap_manager.remove_beatmapset(self)
        if not succ:
            return
        beatmap_manager.update_beatmapsdir()
        beatmap_manager.update_beatmapset(self)

    def cp(self, src):
        beatmap_manager = providers.get(BeatmapManager)
        succ = beatmap_manager.add_beatmapset(self, src)
        if not succ:
            return
        beatmap_manager.update_beatmapsdir()
        beatmap_manager.update_beatmapset(self)

    @as_pattern("*.kaiko")
    class beatmap_KAIKO(BeatmapFilePath):
        "Beatmap file in kaiko format"

    @as_pattern("*.ka")
    class beatmap_KA(BeatmapFilePath):
        "Beatmap file in kaiko format"

    @as_pattern("*.osu")
    class beatmap_OSU(BeatmapFilePath):
        "Beatmap file in osu format"

    @property
    def beatmap(self):
        cls = type(self)
        for path in self.iterdir():
            if isinstance(path, (cls.beatmap_KAIKO, cls.beatmap_KA, cls.beatmap_OSU)):
                yield path


class BeatmapsDirPath(RecognizedDirPath, UnmovablePath):
    """The place to hold your beatmaps"""

    beatmapset = as_pattern("*")(BeatmapsetDirPath)

    @as_pattern("*.osz")
    class beatmap_zip(RecognizedFilePath):
        "Compressed beatmapset file"


@dataclasses.dataclass
class BeatmapCache:
    mtime: Optional[float] = None
    cache: Optional[beatmaps.Beatmap] = None


@dataclasses.dataclass
class BeatmapsetCache:
    mtime: Optional[float] = None
    cache: Dict[BeatmapFilePath, BeatmapCache] = dataclasses.field(
        default_factory=lambda: defaultdict(BeatmapCache)
    )


@dataclasses.dataclass
class BeatmapsDirCache:
    mtime: Optional[float] = None
    cache: Dict[BeatmapsetDirPath, BeatmapsetCache] = dataclasses.field(
        default_factory=lambda: defaultdict(BeatmapsetCache)
    )


class BeatmapManager:
    def __init__(self, beatmaps_dir):
        self.beatmaps_dir = beatmaps_dir
        self._beatmaps = BeatmapsDirCache()

    def update_beatmapsdir(self):
        if self._beatmaps.mtime == self.beatmaps_dir.abs.stat().st_mtime:
            return

        logger = providers.get(Logger)
        file_manager = providers.get(FileManager)

        logger.print("[data/] Load beatmaps...")

        # unzip beatmapset file
        for zip_file in self.beatmaps_dir.beatmap_zip:
            dst_path = zip_file.abs.parent / zip_file.abs.stem
            if not dst_path.exists():
                path_mu = file_manager.as_relative_path(zip_file)
                path_mu = logger.format_path(path_mu)
                logger.print(f"[data/] Unzip file {path_mu}...")
                dst_path.mkdir()
                zf = zipfile.ZipFile(str(zip_file), "r")
                zf.extractall(path=str(dst_path))
            zip_file.abs.unlink()

        beatmapsdir_mtime = self.beatmaps_dir.abs.stat().st_mtime

        old_beatmapset_paths = set(self._beatmaps.cache.keys())

        for beatmapset_path in self.beatmaps_dir.beatmapset:
            if beatmapset_path not in self._beatmaps.cache:
                self._beatmaps.cache[beatmapset_path]
            old_beatmapset_paths.discard(beatmapset_path)

        for beatmapset_path in old_beatmapset_paths:
            if beatmapset_path in self._beatmaps.cache:
                del self._beatmaps.cache[beatmapset_path]

        self._beatmaps.mtime = beatmapsdir_mtime

    def update_beatmapset(self, beatmapset_path):
        beatmapset_path = beatmapset_path.normalize()

        if not beatmapset_path.abs.exists():
            if beatmapset_path in self._beatmaps.cache:
                del self._beatmaps.cache[beatmapset_path]
            return

        _beatmapset = self._beatmaps.cache[beatmapset_path]

        beatmapset_mtime = beatmapset_path.abs.stat().st_mtime

        if _beatmapset.mtime == beatmapset_mtime:
            return

        logger = providers.get(Logger)
        file_manager = providers.get(FileManager)
        path_mu = file_manager.as_relative_path(beatmapset_path)
        path_mu = logger.format_path(path_mu)
        logger.log(f"[data/] Load beatmapsets {path_mu}...")

        old_beatmap_paths = set(_beatmapset.cache.keys())

        for beatmap_path in beatmapset_path.beatmap:
            if beatmap_path not in _beatmapset.cache:
                _beatmapset.cache[beatmap_path]
            old_beatmap_paths.discard(beatmap_path)

        for beatmap_path in old_beatmap_paths:
            if beatmap_path in _beatmapset.cache:
                del _beatmapset.cache[beatmap_path]

        _beatmapset.mtime = beatmapset_mtime

    def update_beatmap(self, beatmap_path):
        beatmapset_path = beatmap_path.parent

        if not beatmap_path.parent.abs.exists():
            return

        _beatmapset = self._beatmaps.cache[beatmapset_path]

        if not beatmap_path.abs.exists():
            if beatmap_path in _beatmapset.cache:
                del _beatmapset.cache[beatmap_path]
            return

        _beatmap = _beatmapset.cache[beatmap_path]

        beatmap_mtime = beatmap_path.abs.stat().st_mtime

        if _beatmap.mtime == beatmap_mtime:
            return

        logger = providers.get(Logger)
        file_manager = providers.get(FileManager)
        path_mu = file_manager.as_relative_path(beatmap_path)
        path_mu = logger.format_path(path_mu)
        logger.log(f"[data/] Load beatmap {path_mu}...")

        try:
            beatmap_metadata = beatsheets.read(str(beatmap_path), metadata_only=True)
        except beatsheets.BeatmapParseError:
            beatmap_metadata = None

        _beatmap.cache = beatmap_metadata
        _beatmap.mtime = beatmap_mtime

    def get_beatmapset_paths(self):
        self.update_beatmapsdir()
        return self._beatmaps.cache.keys()

    def get_beatmap_paths(self, beatmapset_path):
        self.update_beatmapset(beatmapset_path)
        if beatmapset_path in self._beatmaps.cache:
            return self._beatmaps.cache[beatmapset_path].cache.keys()
        else:
            return None

    def get_beatmap_metadata(self, beatmap_path):
        self.update_beatmap(beatmap_path)
        beatmapset_path = beatmap_path.parent
        if (
            beatmapset_path in self._beatmaps.cache
            and beatmap_path in self._beatmaps.cache[beatmapset_path].cache
        ):
            return self._beatmaps.cache[beatmapset_path].cache[beatmap_path].cache
        else:
            return None

    def get_song(self, beatmapset_path, beatmap_path=None):
        if beatmap_path is None:
            try:
                beatmap_paths = list(self.get_beatmap_paths(beatmapset_path))
            except ValueError:
                return None
            beatmap_path = beatmap_paths[0] if beatmap_paths else None

        beatmap = self.get_beatmap_metadata(beatmap_path)
        if beatmap is None or beatmap.audio is None or beatmap.audio.path is None:
            return None
        return Song(beatmapset_path.abs, beatmap.audio, beatmap.beatpoints)

    def get_songs(self):
        songs = [
            self.get_song(beatmapset_path)
            for beatmapset_path in self.get_beatmapset_paths()
        ]
        return [song for song in songs if song is not None]

    def validate_beatmapset_path(self, path, should_exist=None):
        file_manager = providers.get(FileManager)
        if not isinstance(path, BeatmapsetDirPath):
            relpath = file_manager.as_relative_path(path)
            raise InvalidFileOperation(f"Not a valid beatmapset path: {relpath}")
        file_manager.validate_path(path, should_exist=should_exist, file_type="dir")

    def validate_beatmap_path(self, path, should_exist=None):
        file_manager = providers.get(FileManager)
        if not isinstance(path, BeatmapFilePath):
            relpath = file_manager.as_relative_path(path)
            raise InvalidFileOperation(f"Not a valid beatmap path: {relpath}")
        file_manager.validate_path(path, should_exist=should_exist, file_type="file")

    def remove_beatmapset(self, beatmapset_path):
        logger = providers.get(Logger)
        file_manager = providers.get(FileManager)

        beatmapset_path = beatmapset_path.normalize()

        try:
            self.validate_beatmapset_path(beatmapset_path, should_exist=True)
        except InvalidFileOperation as e:
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return False

        path_mu = file_manager.as_relative_path(beatmapset_path, self.beatmaps_dir)
        path_mu = logger.format_path(path_mu)
        logger.print(f"[data/] Remove beatmapset {path_mu}...")
        shutil.rmtree(str(beatmapset_path))

    def remove_beatmap(self, beatmap_path):
        logger = providers.get(Logger)
        file_manager = providers.get(FileManager)

        beatmap_path = beatmap_path.normalize()

        try:
            self.validate_beatmap_path(beatmap_path, should_exist=True)
        except InvalidFileOperation as e:
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return False

        path_mu = file_manager.as_relative_path(beatmapset_path, self.beatmaps_dir)
        path_mu = logger.format_path(path_mu)
        logger.print(f"[data/] Remove beatmap {path_mu}...")

        beatmap_path.abs.unlink()

    def add_beatmapset(self, beatmapset_path, src_path):
        logger = providers.get(Logger)
        file_manager = providers.get(FileManager)

        beatmapset_path = beatmapset_path.normalize()

        try:
            self.validate_beatmapset_path(beatmapset_path, should_exist=False)
            file_manager.validate_path(
                src_path, should_exist=True, should_in_range=False, file_type="dir"
            )
        except InvalidFileOperation as e:
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return False

        logger.print(
            f"[data/] Add a new beatmapset from {logger.escape(str(src_path))}..."
        )

        shutil.copytree(str(src_path), str(beatmapset_path))

        return True

    def add_beatmap(self, beatmap_path, src_path):
        logger = providers.get(Logger)
        file_manager = providers.get(FileManager)

        try:
            self.validate_beatmap_path(beatmap_path, should_exist=False)
            file_manager.validate_path(
                src_path, should_exist=True, should_in_range=False, file_type="file"
            )
        except InvalidFileOperation as e:
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return False

        logger.print(
            f"[data/] Add a new beatmap from {logger.escape(str(src_path))}..."
        )

        beatmapset_path = beatmap_path.parent
        if not beatmapset_path.abs.exists():
            beatmapset_path.abs.mkdir()
        shutil.copy(str(src_path), str(beatmap_path))


@dataclasses.dataclass(frozen=True)
class Song:
    root: Path
    audio: beatmaps.BeatmapAudio
    beatpoints: beatmaps.BeatPoints

    @property
    def path(self):
        if self.audio.path is None:
            return None
        return self.root / self.audio.path


class PlayCommand:
    def __init__(self, resources_dir, cache_dir):
        self.resources_dir = resources_dir
        self.cache_dir = cache_dir

    # beatmaps

    @cmd.function_command
    @dn.datanode
    def parse(self, beatmap):
        """[rich]Parse a beatmap.

        usage: [cmd]parse[/] [arg]{beatmap}[/]
                        ╱
           Path, the path to the
          beatmap you want to parse.
        """
        beatmap_manager = providers.get(BeatmapManager)
        file_manager = providers.get(FileManager)
        logger = providers.get(Logger)

        try:
            beatmap_manager.validate_beatmap_path(beatmap, should_exist=True)
        except InvalidFileOperation as e:
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return

        beatmap = yield from load_beatmap(
            beatmap, file_manager, beatmap_manager, logger
        ).join()
        if beatmap is None:
            return
        beatmap_mu = logger.format_value(beatmap, multiline=4)
        logger.print(beatmap_mu)

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
        profile_manager = providers.get(ProfileManager)
        beatmap_manager = providers.get(BeatmapManager)
        device_manager = providers.get(DeviceManager)
        file_manager = providers.get(FileManager)
        logger = providers.get(Logger)

        try:
            beatmap_manager.validate_beatmap_path(beatmap, should_exist=True)
        except InvalidFileOperation as e:
            logger.print(f"[warn]{logger.escape(str(e))}[/]")
            return

        return KAIKOPlay(
            load_beatmap(beatmap, file_manager, beatmap_manager, logger),
            start,
            self.resources_dir,
            file_manager,
            profile_manager,
            device_manager,
            logger,
        )

    @cmd.function_command
    def loop(self, pattern, tempo: float = 120.0, offset: float = 1.0):
        """[rich]Beat with the pattern repeatly.

        usage: [cmd]loop[/] [arg]{pattern}[/] [[[kw]--tempo[/] [arg]{TEMPO}[/]]] [[[kw]--offset[/] [arg]{OFFSET}[/]]]
                        ╱                  ╲                  ╲
            text, the pattern     float, the tempo of         float, the offset time
                to repeat.     pattern; default is 120.0.    at start; default is 1.0.
        """
        profile_manager = providers.get(ProfileManager)
        device_manager = providers.get(DeviceManager)
        file_manager = providers.get(FileManager)
        logger = providers.get(Logger)

        return KAIKOPlay(
            load_pattern(pattern, tempo, offset, logger),
            None,
            self.resources_dir,
            file_manager,
            profile_manager,
            device_manager,
            logger,
        )

    @loop.arg_parser("pattern")
    def _loop_pattern_parser(self):
        return cmd.RawParser(
            desc="It should be a pattern.", default="x x o x | x [x x] o _"
        )

    @play.arg_parser("beatmap")
    @parse.arg_parser("beatmap")
    def _play_beatmap_parser(self):
        file_manager = providers.get(FileManager)
        return file_manager.make_parser(
            desc="It should be a path to the beatmap file",
            filter=lambda path: isinstance(path, BeatmapFilePath),
        )

    @play.arg_parser("start")
    def _play_start_parser(self, beatmap):
        return cmd.TimeParser(0.0)


class KAIKOPlay:
    def __init__(
        self,
        beatmap_loeader,
        start,
        resources_dir,
        file_manager,
        profile_manager,
        device_manager,
        logger,
    ):
        self.beatmap_loeader = beatmap_loeader
        self.start = start
        self.resources_dir = resources_dir
        self.file_manager = file_manager
        self.profile_manager = profile_manager
        self.device_manager = device_manager
        self.logger = logger

    @dn.datanode
    def execute(self):
        logger = self.logger
        gameplay_settings = self.profile_manager.current.gameplay

        beatmap = yield from self.beatmap_loeader.join()

        if beatmap is None:
            return

        hints = KAIKOPlay.generate_hints(logger, gameplay_settings)
        logger.print()

        try:
            with self.profile_manager.restoring() as settings_modified:
                score = yield from beatmap.play(
                    self.start, gameplay_settings, hints=hints
                ).join()
        except beatmaps.BeatmapLoadError as exc:
            logger.print(f"[warn]Failed to load beatmap[/]")
            logger.print_traceback(exc)
            return

        logger.print()
        logger.print(
            logger.format_scores(
                beatmap.settings.difficulty.performance_tolerance, score.perfs
            )
        )

        for section in score.freestyles:
            logger.print("freestyle:")
            logger.print(section.as_patterns_str(), markup=False)

        if self.profile_manager.current == settings_modified:
            return

        logger.print()
        yes = yield from logger.ask(
            "You changed the settings during the game.\nDo you want to keep it?"
        ).join()
        if yes:
            logger.print("[data/] Update device settings...")
            title = self.file_manager.as_relative_path(
                self.profile_manager.current_path
            )
            old = self.profile_manager.format()
            self.profile_manager.current = settings_modified
            self.profile_manager.set_as_changed()
            new = self.profile_manager.format()

            logger.print(f"[data/] Your changes")
            logger.print(
                logger.format_code_diff(old, new, title=title, is_changed=True)
            )

    @staticmethod
    def generate_hints(logger, settings):
        pause_key = settings.controls.pause_key
        skip_key = settings.controls.skip_key
        stop_key = settings.controls.stop_key
        display_keys = settings.controls.display_delay_adjust_keys
        knock_keys = settings.controls.knock_delay_adjust_keys
        energy_keys = settings.controls.knock_energy_adjust_keys

        pause_key = logger.escape(pause_key, type="all")
        skip_key = logger.escape(skip_key, type="all")
        stop_key = logger.escape(stop_key, type="all")
        display_key_1 = logger.escape(display_keys[0], type="all")
        display_key_2 = logger.escape(display_keys[1], type="all")
        knock_key_1 = logger.escape(knock_keys[0], type="all")
        knock_key_2 = logger.escape(knock_keys[1], type="all")
        energy_key_1 = logger.escape(energy_keys[0], type="all")
        energy_key_2 = logger.escape(energy_keys[1], type="all")

        res = []
        res.append(f"[hint/] Press [emph]{pause_key}[/] to pause/resume the game.")
        res.append(f"[hint/] Press [emph]{skip_key}[/] to skip time.")
        res.append(f"[hint/] Press [emph]{stop_key}[/] to end the game.")
        res.append(
            f"[hint/] Use [emph]{display_key_1}[/] and "
            f"[emph]{display_key_2}[/] to adjust display delay."
        )
        res.append(
            f"[hint/] Use [emph]{knock_key_1}[/] and "
            f"[emph]{knock_key_2}[/] to adjust hit delay."
        )
        res.append(
            f"[hint/] Use [emph]{energy_key_1}[/] and "
            f"[emph]{energy_key_2}[/] to adjust hit strength."
        )
        return "\n".join(res)


@dn.datanode
def load_beatmap(beatmap_path, file_manager, beatmap_manager, logger):
    yield
    try:
        return beatsheets.read(str(beatmap_path))

    except beatsheets.BeatmapParseError as exc:
        path_mu = file_manager.as_relative_path(
            beatmap_path, beatmap_manager.beatmaps_dir
        )
        path_mu = logger.format_path(path_mu)
        logger.print(f"[warn]Failed to read beatmap: {path_mu}[/]")
        logger.print_traceback(exc)
        return


@dn.datanode
def load_pattern(pattern, tempo, offset, logger):
    yield
    try:
        track, width = beatmaps.BeatTrack.parse(pattern, ret_width=True)

    except beatsheets.BeatmapParseError as exc:
        logger.print("[warn]Failed to parse pattern[/]")
        logger.print_traceback(exc)
        return

    return beatmaps.Loop(tempo=tempo, offset=offset, width=width, track=track)
