import os
import contextlib
import random
import shutil
import queue
import zipfile
import dataclasses
from typing import Optional
from pathlib import Path
from ..utils import commands as cmd
from ..utils import datanodes as dn
from ..utils import markups as mu
from ..devices import engines
from ..beats import beatmaps
from ..beats import beatsheets

def format_info(info, logger):
    data = dict(tuple(line.split(":", maxsplit=1)) for line in info.strip().splitlines())
    return logger.format_dict(data)

class BeatmapManager:
    def __init__(self, songs_dir, logger):
        self.songs_dir = songs_dir
        self.logger = logger
        self._beatmaps = {}
        self._beatmaps_mtime = None

    def is_uptodate(self):
        return self._beatmaps_mtime == os.stat(str(self.songs_dir)).st_mtime

    def reload(self):
        logger = self.logger
        songs_dir = self.songs_dir

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
        songs_dir = self.songs_dir

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
        songs_dir = self.songs_dir

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

        filepath = self.songs_dir / path
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
        return beatmap and BGMSong(self.songs_dir / path.parent, beatmap.audio, beatmap.info)

    def get_songs(self):
        songs = [self.get_song(path) for path in self._beatmaps.keys()]
        return [song for song in songs if song is not None]

    def make_parser(self):
        return BeatmapParser(self, self.logger)

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
    def __init__(self, beatmap_manager, logger):
        super().__init__(BeatmapParser.make_tree(beatmap_manager._beatmaps))
        self.beatmap_manager = beatmap_manager
        self.logger = logger

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

        if self.beatmap_manager.is_beatmap(path):
            beatmap = self.beatmap_manager.get_beatmap_metadata(path)
            if beatmap is None or not beatmap.info.strip():
                return None
            data = dict(tuple(line.split(":", maxsplit=1)) for line in beatmap.info.strip().splitlines())
            return "[rich]" + self.logger.format_dict(data, show_border=False)


@dataclasses.dataclass(frozen=True)
class BGMSong:
    root: Path
    audio: beatmaps.BeatmapAudio
    info: str

class BGMAction:
    pass

@dataclasses.dataclass(frozen=True)
class StopBGM(BGMAction):
    pass

@dataclasses.dataclass(frozen=True)
class PlayBGM(BGMAction):
    song: BGMSong
    start: Optional[float]

@dataclasses.dataclass(frozen=True)
class PreviewSong(BGMAction):
    song: BGMSong

@dataclasses.dataclass(frozen=True)
class StopPreview(BGMAction):
    pass

class KAIKOBGMController:
    def __init__(self, mixer_settings, logger, beatmap_manager):
        self.mixer_settings = mixer_settings
        self.logger = logger
        self.current_action = None
        self._action_queue = queue.Queue()
        self.beatmap_manager = beatmap_manager

    def update_mixer_settings(self, mixer_settings):
        self.mixer_settings = mixer_settings

    class MixerLoader:
        def __init__(self, mixer_settings, manager):
            self.mixer_settings = mixer_settings
            self.manager = manager
            self.required = set()
            self.mixer_task = None
            self.mixer = None

        @dn.datanode
        def task(self):
            while True:
                yield
                if not self.required:
                    continue

                assert isinstance(self.mixer_task, dn.DataNode)

                with self.mixer_task:
                    yield

                    while self.required:
                        try:
                            self.mixer_task.send(None)
                        except StopIteration:
                            return

                        yield

                    self.mixer_task = None
                    self.mixer = None

        @contextlib.contextmanager
        def require(self):
            if self.mixer is None:
                self.mixer_task, self.mixer = engines.Mixer.create(self.mixer_settings, self.manager)

            key = object()
            self.required.add(key)
            try:
                yield self.mixer
            finally:
                self.required.remove(key)

    @dn.datanode
    def _play_song(self, mixer, action):
        preview_delay = 0.5

        if isinstance(action, PreviewSong):
            song = action.song
            start = action.song.audio.preview
            delay = preview_delay
        elif isinstance(action, PlayBGM):
            song = action.song
            start = action.start
            delay = None
        else:
            assert False

        if delay is not None:
            yield from dn.sleep(delay).join()

        with mixer.play_file(song.root / song.audio.path, start=start, volume=song.audio.volume) as song_handler:
            yield
            while not song_handler.is_finalized():
                yield

    @dn.datanode
    def _bgm_event_loop(self, require_mixer):
        action = StopBGM()
        self.is_bgm_on = False
        self.current_action = None

        yield
        while True:
            if isinstance(action, StopPreview) and self.is_bgm_on:
                song = self.random_song()
                if song is not None:
                    action = PlayBGM(song, None)

            if isinstance(action, (StopBGM, StopPreview)):
                self.is_bgm_on = False
                self.current_action = None
                yield

                while self._action_queue.empty():
                    yield
                action = self._action_queue.get()

            elif isinstance(action, PreviewSong):
                preview_action = action
                self.current_action = preview_action
                with require_mixer() as mixer:
                    with self._play_song(mixer, preview_action) as preview_task:
                        while True:
                            yield

                            try:
                                preview_task.send(None)
                            except StopIteration:
                                action = preview_action
                                break

                            if not self._action_queue.empty():
                                action = self._action_queue.get()
                                if action != preview_action:
                                    break

            elif isinstance(action, PlayBGM):
                self.is_bgm_on = True
                self.current_action = action
                with require_mixer() as mixer:
                    with self._play_song(mixer, action) as song_task:
                        while True:
                            yield

                            try:
                                song_task.send(None)
                            except StopIteration:
                                song = self.random_song()
                                action = PlayBGM(song, None) if song is not None else StopBGM()
                                break

                            if not self._action_queue.empty():
                                action = self._action_queue.get()
                                if isinstance(action, (PlayBGM, StopBGM, PreviewSong)):
                                    break

            else:
                assert False

    @dn.datanode
    def load_bgm(self, manager):
        mixer_loader = self.MixerLoader(self.mixer_settings, manager)
        with mixer_loader.task() as mixer_task:
            with self._bgm_event_loop(mixer_loader.require) as event_task:
                while True:
                    yield
                    mixer_task.send(None)
                    event_task.send(None)

    def random_song(self):
        songs = self.beatmap_manager.get_songs()
        if self.current_action is not None:
            songs.remove(self.current_action.song)
        return random.choice(songs) if songs else None

    def stop(self):
        self._action_queue.put(StopBGM())

    def play(self, song=None, start=None):
        if song is None:
            song = self.random_song()
        if song is not None:
            self._action_queue.put(PlayBGM(song, start))

    def preview(self, song):
        self._action_queue.put(PreviewSong(song))

    def stop_preview(self):
        self._action_queue.put(StopPreview())

    def preview_handler(self, token):
        path = token and Path(token)
        song = path and self.beatmap_manager.get_song(path)
        if song is not None:
            self.preview(song)
        else:
            self.stop_preview()

class BGMCommand:
    def __init__(self, bgm_controller, beatmap_manager, logger):
        self.bgm_controller = bgm_controller
        self.beatmap_manager = beatmap_manager
        self.logger = logger

    @cmd.function_command
    def on(self):
        """[rich]Turn on bgm.

        usage: [cmd]bgm[/] [cmd]on[/]
        """
        logger = self.logger

        if self.bgm_controller.is_bgm_on:
            self.now_playing()
            return

        song = self.bgm_controller.random_song()
        if song is None:
            logger.print("[data/] There is no song in the folder yet!")
            return

        logger.print("will play:")
        logger.print(format_info(song.info, logger))
        self.bgm_controller.play(song)

    @cmd.function_command
    def off(self):
        """[rich]Turn off bgm.

        usage: [cmd]bgm[/] [cmd]off[/]
        """
        self.bgm_controller.stop()

    @cmd.function_command
    def skip(self):
        """[rich]Skip the currently playing song.

        usage: [cmd]bgm[/] [cmd]skip[/]
        """
        if self.bgm_controller.current_action is not None:
            song = self.bgm_controller.random_song()
            self.logger.print("will play:")
            self.logger.print(format_info(song.info, self.logger))
            self.bgm_controller.play(song)

    @cmd.function_command
    def play(self, beatmap, start:Optional[float]=None):
        """[rich]Play the song of beatmap.

        usage: [cmd]bgm[/] [cmd]play[/] [arg]{beatmap}[/] [[[kw]--start[/] [arg]{START}[/]]]
                           ╱                   ╲
                 Path, the path to the       float, the time
               beatmap you want to play.     the song started.
        """
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
        logger.print(format_info(song.info, logger))
        self.bgm_controller.play(song, start)

    @play.arg_parser("beatmap")
    def _play_beatmap_parser(self):
        return self.beatmap_manager.make_parser()

    @cmd.function_command
    def now_playing(self):
        """[rich]Show the currently playing song.

        usage: [cmd]bgm[/] [cmd]now_playing[/]
        """
        current = self.bgm_controller.current_action
        if isinstance(current, PlayBGM):
            self.logger.print("now playing:")
            self.logger.print(format_info(current.song.info, self.logger))
        elif isinstance(current, PreviewSong):
            self.logger.print("now previewing:")
            self.logger.print(format_info(current.song.info, self.logger))
        elif current is None:
            self.logger.print("no song")
        else:
            assert False
