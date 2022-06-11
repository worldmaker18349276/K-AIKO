import contextlib
import time
import random
import threading
import queue
import dataclasses
from typing import Optional
from pathlib import Path
from ..utils import commands as cmd
from ..utils import datanodes as dn
from ..devices import audios as aud
from ..devices import engines
from ..beats import beatsheets
from .loggers import Logger
from .files import FileManager
from .play import BeatmapManager, BeatmapFilePath, Song


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


class BGMController:
    mixer_loader_delay = 3.0
    preview_delay = 0.5
    fadein_time = 0.5
    fadeout_time = 1.0
    preview_duration = 30.0

    def __init__(
        self, provider, mixer_settings_getter=engines.MixerSettings
    ):
        self.provider = provider
        self._mixer_settings_getter = mixer_settings_getter
        self._action_queue = queue.Queue()
        self.is_bgm_on = False
        self.current_action = None

    @property
    def beatmap_manager(self):
        return self.provider.get(BeatmapManager)

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
    def __init__(self, provider):
        self.subcommand = BGMSubCommand(provider)

    @cmd.subcommand
    def bgm(self):
        """Subcommand to control background music."""
        return self.subcommand


class BGMSubCommand:
    def __init__(self, provider):
        self.provider = provider

    @property
    def bgm_controller(self):
        return self.provider.get(BGMController)

    @property
    def beatmap_manager(self):
        return self.provider.get(BeatmapManager)

    @property
    def logger(self):
        return self.provider.get(Logger)

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
        logger.print(logger.emph(str(song.path), type="all"))
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
            self.logger.print(self.logger.emph(str(song.path), type="all"))
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
        logger.print(logger.emph(str(song.path), type="all"))
        self.bgm_controller.play(song, start)

    @play.arg_parser("beatmap")
    def _play_beatmap_parser(self):
        file_manager = self.provider.get(FileManager)
        return file_manager.make_parser(
            desc="It should be a path to the beatmap file",
            filter=lambda path: isinstance(path, BeatmapFilePath),
        )

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
            self.logger.print(self.logger.emph(str(current.song.path), type="all"))
        elif isinstance(current, PreviewSong):
            self.logger.print("now previewing:")
            self.logger.print(self.logger.emph(str(current.song.path), type="all"))
        elif current is None:
            self.logger.print("no song")
        else:
            assert False
