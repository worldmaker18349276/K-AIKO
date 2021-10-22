import os
import random
import shutil
import queue
import traceback
import dataclasses
from typing import Optional
from pathlib import Path
from kaiko.utils import commands as cmd
from kaiko.utils import datanodes as dn
from kaiko.utils import engines
from kaiko.beats import beatsheets

@dataclasses.dataclass
class SongMetadata:
    root: str
    audio: str
    volume: float
    info: str
    preview: float

    @classmethod
    def from_beatmap(clz, beatmap):
        if beatmap.audio is None:
            return None
        return clz(root=beatmap.root, audio=beatmap.audio,
                   volume=beatmap.volume, info=beatmap.info, preview=beatmap.preview)

    @property
    def path(self):
        return os.path.join(self.root, self.audio)

    def get_info(self, logger):
        res = {}
        res["path:"] = Path(self.path).as_uri()

        for line in self.info.strip().splitlines():
            index = line.find(":")
            key, value = (line[:index+1], line[index+1:]) if index != -1 else (line, "")
            res[key] = value

        return "\n".join(f"{k} {logger.emph(v)}" for k, v in res.items())

class BeatmapManager:
    def __init__(self, path, logger):
        self.path = path
        self.logger = logger
        self._beatmaps = {}
        self._beatmaps_mtime = None

    def is_uptodate(self):
        return self._beatmaps_mtime == os.stat(str(self.path)).st_mtime

    def reload(self):
        logger = self.logger
        songs_dir = self.path

        logger.print(f"Load songs from {logger.emph(songs_dir.as_uri())}...", prefix="data")

        for file in songs_dir.iterdir():
            if file.is_file() and file.suffix == ".osz":
                distpath = file.parent / file.stem
                if distpath.exists():
                    continue
                logger.print(f"Unzip file {logger.emph(file.as_uri())}...", prefix="data")
                distpath.mkdir()
                zf = zipfile.ZipFile(str(file), 'r')
                zf.extractall(path=str(distpath))
                file.unlink()

        logger.print("Load beatmaps...", prefix="data")

        self._beatmaps_mtime = os.stat(str(songs_dir)).st_mtime
        self._beatmaps = {}

        for song in songs_dir.iterdir():
            if song.is_dir():
                beatmapset = []
                for beatmap in song.iterdir():
                    if beatmap.suffix in (".kaiko", ".ka", ".osu"):
                        beatmapset.append(beatmap.relative_to(songs_dir))
                if beatmapset:
                    self._beatmaps[song.relative_to(songs_dir)] = beatmapset

        if len(self._beatmaps) == 0:
            logger.print("There is no song in the folder yet!", prefix="data")
        logger.print(flush=True)

    def add(self, beatmap):
        logger = self.logger
        songs_dir = self.path

        if not beatmap.exists():
            with logger.warn():
                logger.print(f"File not found: {str(beatmap)}")
            return
        if not beatmap.is_file() and not beatmap.is_dir():
            with logger.warn():
                logger.print(f"Not a file or directory: {str(beatmap)}")
            return

        logger.print(f"Add a new song from {logger.emph(beatmap.as_uri())}...", prefix="data")

        distpath = songs_dir / beatmap.name
        n = 1
        while distpath.exists():
            n += 1
            distpath = songs_dir / f"{beatmap.stem} ({n}){beatmap.suffix}"
        if n != 1:
            logger.print(f"Name conflict! Rename to {logger.emph(distpath.name)}", prefix="data")

        if beatmap.is_file():
            shutil.copy(str(beatmap), str(songs_dir))
        elif beatmap.is_dir():
            shutil.copytree(str(beatmap), str(distpath))

        self.reload()

    def remove(self, beatmap):
        logger = self.logger
        songs_dir = self.path

        beatmap_path = songs_dir / beatmap
        if beatmap_path.is_file():
            logger.print(f"Remove the beatmap at {logger.emph(beatmap_path.as_uri())}...", prefix="data")
            beatmap_path.unlink()
            self.reload()

        elif beatmap_path.is_dir():
            logger.print(f"Remove the beatmapset at {logger.emph(beatmap_path.as_uri())}...", prefix="data")
            shutil.rmtree(str(beatmap_path))
            self.reload()

        else:
            with logger.warn():
                logger.print(f"Not a file: {str(beatmap)}")

    def is_beatmapset(self, path):
        return path in self._beatmaps

    def is_beatmap(self, path):
        return path.parent in self._beatmaps and path in self._beatmaps[path.parent]

    def get_beatmap_metadata(self, path):
        if not self.is_beatmap(path):
            raise ValueError(f"Not a beatmap: {str(path)}")

        filepath = self.path / path
        try:
            beatmap = beatsheets.BeatSheet.read(str(filepath), metadata_only=True)
        except beatsheets.BeatmapParseError:
            return None
        else:
            return beatmap

    def get_song(self, path):
        if self.is_beatmapset(path):
            path = self._beatmaps[path][0]
        beatmap = self.get_beatmap_metadata(path)
        return beatmap and SongMetadata.from_beatmap(beatmap)

    def get_songs(self):
        songs = [self.get_song(path) for path in self._beatmaps.keys()]
        return [song for song in songs if song is not None]

    def make_parser(self, bgm_controller=None):
        return BeatmapParser(self, bgm_controller)

class BeatmapParser(cmd.TreeParser):
    def __init__(self, beatmap_manager, bgm_controller):
        super().__init__(BeatmapParser.make_tree(beatmap_manager._beatmaps))
        self.beatmap_manager = beatmap_manager
        self.bgm_controller = bgm_controller

    @staticmethod
    def make_tree(beatmapsets):
        tree = {}
        for beatmapset_path, beatmapset in beatmapsets.items():
            subtree = {}
            subtree[""] = Path
            for beatmap_path in beatmapset:
                subtree[str(beatmap_path.relative_to(beatmapset_path))] = Path
            tree[os.path.join(str(beatmapset_path), "")] = subtree
        return tree

    def info(self, token):
        path = Path(token)

        song = self.beatmap_manager.get_song(path)
        if self.bgm_controller is not None and song is not None:
            self.bgm_controller.play(song, song.preview)

        if self.beatmap_manager.is_beatmap(path):
            beatmap = self.beatmap_manager.get_beatmap_metadata(path)
            return beatmap.info.strip() if beatmap is not None else None

class KAIKOBGMController:
    def __init__(self, config, logger, beatmap_manager):
        self.config = config
        self.logger = logger
        self._current_bgm = None
        self._action_queue = queue.Queue()
        self.beatmap_manager = beatmap_manager

    @dn.datanode
    def load_mixer(self, manager):
        try:
            mixer_task, mixer = engines.Mixer.create(self.config.current.devices.mixer, manager)

        except Exception:
            with self.logger.warn():
                self.logger.print("Failed to load mixer")
                self.logger.print(traceback.format_exc(), end="")

        self.mixer = mixer
        try:
            with mixer_task:
                yield from mixer_task.join((yield))
        finally:
            self.mixer = None

    @dn.datanode
    def play_song(self, song, start):
        with dn.create_task(lambda event: self.mixer.load_sound(song.path, event)) as task:
            yield from task.join((yield))
            node = dn.DataNode.wrap(task.result)

        self._current_bgm = song
        try:
            with self.mixer.play(node, start=start, volume=song.volume) as song_handler:
                yield
                while not song_handler.is_finalized():
                    yield
        finally:
            self._current_bgm = None

    @dn.datanode
    def load_bgm(self, manager):
        self.mixer = None
        self._current_bgm = None

        while True:
            yield

            if self._action_queue.empty():
                continue

            while not self._action_queue.empty():
                song, start = self._action_queue.get()
            if song is None:
                continue

            with self.load_mixer(manager) as mixer_task:
                while song is not None:
                    with self.play_song(song, start) as song_task:
                        while True:
                            try:
                                mixer_task.send(None)
                            except StopIteration:
                                song, start = None, None
                                break

                            try:
                                song_task.send(None)
                            except StopIteration:
                                song, start = self.random_song(), None
                                break

                            if not self._action_queue.empty():
                                while not self._action_queue.empty():
                                    song, start = self._action_queue.get()
                                break

                            yield

    def random_song(self):
        songs = self.beatmap_manager.get_songs()
        if self._current_bgm is not None:
            songs.remove(self._current_bgm)
        return random.choice(songs) if songs else None

    def stop(self):
        self._action_queue.put((None, None))

    def play(self, song, start=None):
        self._action_queue.put((song, start))

class BGMCommand:
    def __init__(self, bgm_controller, beatmap_manager, logger):
        self.bgm_controller = bgm_controller
        self.beatmap_manager = beatmap_manager
        self.logger = logger

    @cmd.function_command
    def on(self):
        logger = self.logger

        if self.bgm_controller._current_bgm is not None:
            logger.print("now playing:")
            logger.print(self.bgm_controller._current_bgm.get_info(logger))
            return

        song = self.bgm_controller.random_song()
        if song is None:
            logger.print("There is no song in the folder yet!", prefix="data")
            return

        logger.print("will play:")
        logger.print(song.get_info(logger))
        self.bgm_controller.play(song)

    @cmd.function_command
    def off(self):
        self.bgm_controller.stop()

    @cmd.function_command
    def skip(self):
        if self.bgm_controller._current_bgm is not None:
            song = self.bgm_controller.random_song()
            self.logger.print("will play:")
            self.logger.print(song.get_info(self.logger))
            self.bgm_controller.play(song)

    @cmd.function_command
    def play(self, beatmap, start:Optional[float]=None):
        logger = self.logger

        try:
            song = self.beatmap_manager.get_song(beatmap)
        except beatsheets.BeatmapParseError:
            with logger.warn():
                logger.print("Fail to read beatmap")
            return

        if song is None:
            with logger.warn():
                logger.print("This beatmap has no song")
            return

        logger.print("will play:")
        logger.print(song.get_info(logger))
        self.bgm_controller.play(song, start)

    @play.arg_parser("beatmap")
    def _play_beatmap_parser(self):
        return self.beatmap_manager.make_parser()

    @cmd.function_command
    def now_playing(self):
        current = self.bgm_controller._current_bgm
        if current is None:
            self.logger.print("no song")
        else:
            self.logger.print("now playing:")
            self.logger.print(current.get_info(self.logger))
