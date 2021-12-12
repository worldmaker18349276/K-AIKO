import os
import random
import shutil
import queue
import traceback
import dataclasses
from typing import Optional
from pathlib import Path
from ..utils import commands as cmd
from ..utils import datanodes as dn
from ..utils import markups as mu
from ..devices import audios as aud
from ..devices import engines
from ..beats import beatsheets

def make_table(info):
    total_width = 80
    res = {}
    for line in info.strip().splitlines():
        index = line.find(":")
        key, value = (line[:index], line[index+1:]) if index != -1 else (line, "")
        res[key] = value

    width = max(len(k) for k in res.keys())
    return "\n".join(f"{' '*(width-len(k)) + mu.escape(k)} │ [emph]{mu.escape(v)}[/]" for k, v in res.items())

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
        return make_table(self.info)

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

        logger.print(f"[data/] Load songs from {logger.emph(songs_dir.as_uri())}...")

        for file in songs_dir.iterdir():
            if file.is_file() and file.suffix == ".osz":
                distpath = file.parent / file.stem
                if distpath.exists():
                    continue
                logger.print(f"[data/] Unzip file {logger.emph(file.as_uri())}...")
                distpath.mkdir()
                zf = zipfile.ZipFile(str(file), 'r')
                zf.extractall(path=str(distpath))
                file.unlink()

        logger.print("[data/] Load beatmaps...")

        beatmaps_mtime = os.stat(str(songs_dir)).st_mtime
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
            logger.print("[data/] There is no song in the folder yet!")
        logger.print(flush=True)
        self._beatmaps_mtime = beatmaps_mtime

    def add(self, path):
        logger = self.logger
        songs_dir = self.path

        if not path.exists():
            logger.print(f"[warn]File not found: {logger.escape(str(path))}[/]")
            return
        if not path.is_file() and not path.is_dir():
            logger.print(f"[warn]Not a file or directory: {logger.escape(str(path))}[/]")
            return

        logger.print(f"[data/] Add a new song from {logger.emph(path.as_uri())}...")

        distpath = songs_dir / path.name
        n = 1
        while distpath.exists():
            n += 1
            distpath = songs_dir / f"{path.stem} ({n}){path.suffix}"
        if n != 1:
            logger.print(f"[data/] Name conflict! Rename to {logger.emph(distpath.name)}")

        if path.is_file():
            shutil.copy(str(path), str(songs_dir))
        elif path.is_dir():
            shutil.copytree(str(path), str(distpath))

        self.reload()

    def remove(self, path):
        logger = self.logger
        songs_dir = self.path

        if self.is_beatmap(path):
            beatmap_path = songs_dir / path
            logger.print(f"[data/] Remove the beatmap at {logger.emph(beatmap_path.as_uri())}...")
            beatmap_path.unlink()
            self.reload()

        elif self.is_beatmapset(path):
            beatmapset_path = songs_dir / path
            logger.print(f"[data/] Remove the beatmapset at {logger.emph(beatmapset_path.as_uri())}...")
            shutil.rmtree(str(beatmapset_path))
            self.reload()

        else:
            logger.print(f"[warn]Not a beatmap or beatmapset: {logger.escape(str(path))}[/]")

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
            if not self._beatmaps[path]:
                return None
            path = self._beatmaps[path][0]
        beatmap = self.get_beatmap_metadata(path)
        return beatmap and SongMetadata.from_beatmap(beatmap)

    def get_songs(self):
        songs = [self.get_song(path) for path in self._beatmaps.keys()]
        return [song for song in songs if song is not None]

    def make_parser(self, bgm_controller=None):
        return BeatmapParser(self, bgm_controller)

    def print_tree(self, logger):
        beatmapsets = self._beatmaps.items()
        for i, (path, beatmapset) in enumerate(beatmapsets):
            prefix = "└── " if i == len(beatmapsets)-1 else "├── "
            logger.print(prefix + logger.emph(str(path)))

            preprefix = "    " if i == len(beatmapsets)-1 else "│   "
            for j, beatmap in enumerate(beatmapset):
                prefix = "└── " if j == len(beatmapset)-1 else "├── "
                logger.print(preprefix + prefix + logger.escape(str(beatmap.relative_to(path))))

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
            self.bgm_controller.preview(song)

        if self.beatmap_manager.is_beatmap(path):
            beatmap = self.beatmap_manager.get_beatmap_metadata(path)
            return make_table(beatmap.info) if beatmap is not None else None


class BGMAction:
    pass

@dataclasses.dataclass(frozen=True)
class StopBGM(BGMAction):
    pass

@dataclasses.dataclass(frozen=True)
class PlayBGM(BGMAction):
    song: SongMetadata
    start: Optional[float]

@dataclasses.dataclass(frozen=True)
class PreviewBeatmap(BGMAction):
    song: SongMetadata

class KAIKOBGMController:
    def __init__(self, mixer_settings, logger, beatmap_manager):
        self.mixer_settings = mixer_settings
        self.logger = logger
        self._current_bgm = None
        self._action_queue = queue.Queue()
        self.beatmap_manager = beatmap_manager

    def update_mixer_settings(self, mixer_settings):
        self.mixer_settings = mixer_settings

    @dn.datanode
    def _load_mixer(self, manager):
        try:
            mixer_task, mixer = engines.Mixer.create(self.mixer_settings, manager)

        except Exception:
            with self.logger.warn():
                self.logger.print("Failed to load mixer")
                self.logger.print(traceback.format_exc(), end="", markup=False)
            yield
            return

        self.mixer = mixer
        try:
            with mixer_task:
                yield from mixer_task.join((yield))
        finally:
            self.mixer = None

    @dn.datanode
    def _play_song(self, song, start, delay):
        if delay is not None:
            with dn.sleep(delay) as timer:
                yield from timer.join((yield))

        try:
            with dn.create_task(lambda event: self.mixer.load_sound(song.path, event)) as task:
                yield from task.join((yield))
                node = dn.DataNode.wrap(task.result)
        except aud.IOCancelled:
            return

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
        hint_preview_delay = 0.5
        self.mixer = None
        self._current_bgm = None

        while True:
            yield

            if self._action_queue.empty():
                continue

            while not self._action_queue.empty():
                action = self._action_queue.get()
            if isinstance(action, StopBGM):
                continue

            with self._load_mixer(manager) as mixer_task:
                while not isinstance(action, StopBGM):
                    if isinstance(action, PlayBGM):
                        song = action.song
                        start = action.start
                        delay = None
                    elif isinstance(action, PreviewBeatmap):
                        song = action.song
                        start = action.song.preview
                        delay = hint_preview_delay

                    with self._play_song(song, start, delay) as song_task:
                        while True:
                            try:
                                mixer_task.send(None)
                            except StopIteration:
                                action = StopBGM()
                                break

                            try:
                                song_task.send(None)
                            except StopIteration:
                                action = PlayBGM(self.random_song(), None)
                                break

                            if not self._action_queue.empty():
                                while not self._action_queue.empty():
                                    next_action = self._action_queue.get()
                                if action != next_action:
                                    action = next_action
                                    break

                            yield

    def random_song(self):
        songs = self.beatmap_manager.get_songs()
        if self._current_bgm is not None:
            songs.remove(self._current_bgm)
        return random.choice(songs) if songs else None

    def stop(self):
        self._action_queue.put(StopBGM())

    def play(self, song, start=None):
        self._action_queue.put(PlayBGM(song, start))

    def preview(self, song):
        self._action_queue.put(PreviewBeatmap(song))

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
            logger.print("[data/] There is no song in the folder yet!")
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
            logger.print("[warn]Fail to read beatmap[/]")
            return

        if song is None:
            logger.print("[warn]This beatmap has no song[/]")
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
