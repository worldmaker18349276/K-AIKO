import os
import shutil
import zipfile
import dataclasses
from pathlib import Path
from ..utils import commands as cmd
from ..beats import beatmaps
from ..beats import beatsheets
from .files import FileDescriptor, DirDescriptor, WildCardDescriptor, as_child


def format_info(info, logger):
    data = dict(
        tuple(line.split(":", maxsplit=1)) for line in info.strip().splitlines()
    )
    return logger.format_dict(data)


class BeatmapsDirDescriptor(DirDescriptor):
    "(The place to hold your beatmaps)"

    @as_child("*")
    class BeatmapSet(DirDescriptor):
        "(Beatmapset of a song)"

        @as_child("*.kaiko")
        class BeatmapKAIKO(FileDescriptor):
            "(Beatmap file in kaiko format)"

        @as_child("*.ka")
        class BeatmapKA(FileDescriptor):
            "(Beatmap file in kaiko format)"

        @as_child("*.osu")
        class BeatmapOSU(FileDescriptor):
            "(Beatmap file in osu format)"

        @as_child("**")
        class InnerFile(WildCardDescriptor):
            "(Inner file of this beatmapset)"

    @as_child("*.osz")
    class BeamapZip(FileDescriptor):
        "(Compressed beatmapset file)"


class BeatmapManager:
    def __init__(self, beatmaps_dir, logger):
        self.beatmaps_dir = beatmaps_dir
        self.logger = logger
        self._beatmaps = {}
        self._beatmaps_mtime = None

    def is_uptodate(self):
        return self._beatmaps_mtime == os.stat(str(self.beatmaps_dir)).st_mtime

    def reload(self):
        logger = self.logger
        beatmaps_dir = self.beatmaps_dir

        logger.print(
            f"[data/] Load beatmaps from {logger.emph(beatmaps_dir.as_uri())}..."
        )

        for file in beatmaps_dir.iterdir():
            if file.is_file() and file.suffix == ".osz":
                distpath = file.parent / file.stem
                if distpath.exists():
                    continue
                logger.print(f"[data/] Unzip file {logger.emph(file.as_uri())}...")
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

    def add(self, path):
        logger = self.logger
        beatmaps_dir = self.beatmaps_dir

        if not path.exists():
            logger.print(f"[warn]File not found: {logger.escape(str(path), type='all')}[/]")
            return
        if not path.is_file() and not path.is_dir():
            logger.print(
                f"[warn]Not a file or directory: {logger.escape(str(path), type='all')}[/]"
            )
            return

        logger.print(f"[data/] Add a new song from {logger.emph(path.as_uri())}...")

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

        self.reload()

    def remove(self, path):
        logger = self.logger
        beatmaps_dir = self.beatmaps_dir

        if self.is_beatmap(path):
            beatmap_path = beatmaps_dir / path
            logger.print(
                f"[data/] Remove the beatmap at {logger.emph(beatmap_path.as_uri())}..."
            )
            beatmap_path.unlink()
            self.reload()

        elif self.is_beatmapset(path):
            beatmapset_path = beatmaps_dir / path
            logger.print(
                f"[data/] Remove the beatmapset at {logger.emph(beatmapset_path.as_uri())}..."
            )
            shutil.rmtree(str(beatmapset_path))
            self.reload()

        else:
            logger.print(
                f"[warn]Not a beatmap or beatmapset: {logger.escape(str(path), type='all')}[/]"
            )

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

    def make_parser(self, root=".", type="file"):
        return BeatmapParser(root, type, self, self.logger)


class BeatmapParser(cmd.PathParser):
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


