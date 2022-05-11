import os
import contextlib
import time
import random
import shutil
import threading
import queue
import zipfile
import dataclasses
from typing import Optional
from pathlib import Path
from ..utils import commands as cmd
from ..utils import datanodes as dn
from ..utils import markups as mu
from ..devices import audios as aud
from ..devices import engines
from ..beats import beatmaps
from ..beats import beatsheets


def format_info(info, logger):
    data = dict(
        tuple(line.split(":", maxsplit=1)) for line in info.strip().splitlines()
    )
    return logger.format_dict(data)


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


class BGMAction:
    pass


@dataclasses.dataclass(frozen=True)
class StopBGM(BGMAction):
    pass


@dataclasses.dataclass(frozen=True)
class PlayBGM(BGMAction):
    song: Song
    start: Optional[float]


@dataclasses.dataclass(frozen=True)
class PreviewSong(BGMAction):
    song: Song


@dataclasses.dataclass(frozen=True)
class StopPreview(BGMAction):
    pass


class MixerLoader:
    def __init__(self, manager, delay=0.0, mixer_settings_getter=engines.MixerSettings):
        self._mixer_settings_getter = mixer_settings_getter
        self.manager = manager
        self.delay = delay
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

                expiration = None
                while expiration is None or time.time() < expiration:
                    if expiration is None and not self.required:
                        expiration = time.time() + self.delay
                    elif expiration is not None and self.required:
                        expiration = None

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
            self.mixer_task, self.mixer = engines.Mixer.create(
                self._mixer_settings_getter(), self.manager
            )

        key = object()
        self.required.add(key)
        try:
            yield self.mixer
        finally:
            self.required.remove(key)


@dn.datanode
def play_fadeinout(
    mixer, path, fadein_time, fadeout_time, volume=0.0, start=None, end=None
):
    meta = aud.AudioMetadata.read(path)
    node = aud.load(path)
    node = dn.tslice(node, meta.samplerate, start, end)
    # initialize before attach; it will seek to the starting frame
    node.__enter__()

    samplerate = mixer.settings.output_samplerate
    out_event = threading.Event()
    start = start if start is not None else 0.0
    before = end - start - fadeout_time if end is not None else None
    node = dn.pipe(
        node,
        mixer.resample(meta.samplerate, meta.channels, volume),
        dn.fadein(samplerate, fadein_time),
        dn.fadeout(samplerate, fadeout_time, out_event, before),
    )

    song_handler = mixer.play(node)
    while not song_handler.is_finalized():
        try:
            yield
        except GeneratorExit:
            out_event.set()
            raise


class KAIKOBGMController:
    mixer_loader_delay = 3.0
    preview_delay = 0.5
    fadein_time = 0.5
    fadeout_time = 1.0
    preview_duration = 30.0

    def __init__(
        self, logger, beatmap_manager, mixer_settings_getter=engines.MixerSettings
    ):
        self._mixer_settings_getter = mixer_settings_getter
        self.logger = logger
        self._action_queue = queue.Queue()
        self.beatmap_manager = beatmap_manager
        self.is_bgm_on = False
        self.current_action = None

    @dn.datanode
    def execute(self, manager):
        mixer_loader = MixerLoader(
            manager, self.mixer_loader_delay, self._mixer_settings_getter
        )
        with mixer_loader.task() as mixer_task:
            with self._bgm_event_loop(mixer_loader.require) as event_task:
                while True:
                    yield
                    mixer_task.send(None)
                    event_task.send(None)

    @dn.datanode
    def _play_song(self, mixer, action):
        if isinstance(action, PreviewSong):
            start = action.song.audio.preview
            if start is None:
                start = 0.0
            end = (
                start + self.preview_duration
                if self.preview_duration is not None
                else None
            )

            yield from dn.sleep(self.preview_delay).join()

            yield from play_fadeinout(
                mixer,
                action.song.path,
                fadein_time=self.fadein_time,
                fadeout_time=self.fadeout_time,
                volume=action.song.audio.volume,
                start=start,
                end=end,
            ).join()

        elif isinstance(action, PlayBGM):
            yield from dn.sleep(self.preview_delay).join()

            yield from play_fadeinout(
                mixer,
                action.song.path,
                fadein_time=self.fadein_time,
                fadeout_time=self.fadeout_time,
                volume=action.song.audio.volume,
                start=action.start,
            ).join()

        else:
            assert False

    @dn.datanode
    def _bgm_event_loop(self, require_mixer):
        action = StopBGM()

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
                                action = (
                                    PlayBGM(song, None)
                                    if song is not None
                                    else StopBGM()
                                )
                                break

                            if not self._action_queue.empty():
                                action = self._action_queue.get()
                                if isinstance(action, (PlayBGM, StopBGM, PreviewSong)):
                                    break

            else:
                assert False

    def random_song(self):
        songs = self.beatmap_manager.get_songs()
        if self.current_action is not None and self.current_action.song in songs:
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
        if token is None:
            self.stop_preview()
            return

        path = Path(token)

        if not self.beatmap_manager.is_beatmap(path):
            self.stop_preview()
            return

        song = self.beatmap_manager.get_song(path)

        if song is None:
            self.stop_preview()
            return

        self.preview(song)


class BGMCommand:
    def __init__(self, bgm_controller, beatmap_manager, logger):
        self.subcommand = BGMSubCommand(bgm_controller, beatmap_manager, logger)

    @cmd.subcommand
    def bgm(self):
        """Subcommand to control background music."""
        return self.subcommand


class BGMSubCommand:
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
        logger.print(logger.emph(str(song.relpath), type="all"))
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
            if song is None:
                logger.print("[data/] There is no song in the folder yet!")
                return

            self.logger.print("will play:")
            self.logger.print(self.logger.emph(str(song.relpath), type="all"))
            self.bgm_controller.play(song)

    @cmd.function_command
    def play(self, beatmap, start=None):
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
        logger.print(logger.emph(str(song.relpath), type="all"))
        self.bgm_controller.play(song, start)

    @play.arg_parser("beatmap")
    def _play_beatmap_parser(self):
        return self.beatmap_manager.make_parser()

    @play.arg_parser("start")
    def _play_start_parser(self, beatmap):
        return cmd.TimeParser(0.0)

    @cmd.function_command
    def now_playing(self):
        """[rich]Show the currently playing song.

        usage: [cmd]bgm[/] [cmd]now_playing[/]
        """
        current = self.bgm_controller.current_action
        if isinstance(current, PlayBGM):
            self.logger.print("now playing:")
            self.logger.print(self.logger.emph(str(current.song.relpath), type="all"))
        elif isinstance(current, PreviewSong):
            self.logger.print("now previewing:")
            self.logger.print(self.logger.emph(str(current.song.relpath), type="all"))
        elif current is None:
            self.logger.print("no song")
        else:
            assert False
