import random
import threading
import queue
import dataclasses
from typing import Optional
from ..utils import commands as cmd
from ..utils import config as cfg
from ..utils import datanodes as dn
from ..devices import audios as aud
from ..devices import engines
from ..beats import beatsheets
from .loggers import Logger
from .files import FileManager, UnrecognizedPath
from .play import BeatmapManager, BeatmapFilePath, Song


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


class BGMControllerSettings(cfg.Configurable):
    r"""
    Fields
    ------
    mixer_loader_delay : float
        The mixer loader delay.
    volume : float
        The master volume of bgm controller.
    play_delay : float
        The delay time before the song starts playing.
    fadein_time : float
        The fade-in time when switching between songs.
    fadeout_time : float
        The fade-out time when switching between songs.
    preview_duration : float
        The duration of previewing song.
    """

    mixer_loader_delay: float = 3.0
    volume: float = -10.0
    play_delay: float = 0.5
    fadein_time: float = 0.5
    fadeout_time: float = 1.0
    preview_duration: float = 30.0


class BGMController:
    def __init__(self, provider, settings, mixer_settings):
        self.provider = provider
        self.settings = settings
        self.mixer_settings = mixer_settings
        self._action_queue = queue.Queue()
        self.is_bgm_on = False
        self.current_action = None

    def set_settings(self, settings):
        self.settings = settings

    def set_mixer_settings(self, mixer_settings):
        self.mixer_settings = mixer_settings

    @property
    def beatmap_manager(self):
        return self.provider.get(BeatmapManager)

    @property
    def file_manager(self):
        return self.provider.get(FileManager)

    @property
    def logger(self):
        return self.provider.get(Logger)

    @dn.datanode
    def execute(self, manager):
        mixer_factory = lambda: engines.Mixer.create(self.mixer_settings, manager)
        mixer_loader = engines.EngineLoader(mixer_factory, self.settings.mixer_loader_delay)
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
                start + self.settings.preview_duration
                if self.settings.preview_duration is not None
                else None
            )

            yield from dn.sleep(self.settings.play_delay).join()

            logger = self.logger
            file_manager = self.file_manager
            path_mu = file_manager.as_relative_path(UnrecognizedPath(action.song.path, False), markup=True)
            logger.print(f"[music/] will preview: {path_mu}")

            yield from play_fadeinout(
                mixer,
                action.song.path,
                fadein_time=self.settings.fadein_time,
                fadeout_time=self.settings.fadeout_time,
                volume=action.song.audio.volume + self.settings.volume,
                start=start,
                end=end,
            ).join()

        elif isinstance(action, PlayBGM):
            yield from dn.sleep(self.settings.play_delay).join()

            logger = self.logger
            file_manager = self.file_manager
            path_mu = file_manager.as_relative_path(UnrecognizedPath(action.song.path, False), markup=True)
            logger.print(f"[music/] will play: {path_mu}")

            yield from play_fadeinout(
                mixer,
                action.song.path,
                fadein_time=self.settings.fadein_time,
                fadeout_time=self.settings.fadeout_time,
                volume=action.song.audio.volume + self.settings.volume,
                start=action.start,
            ).join()

        else:
            assert False

    @dn.datanode
    def _bgm_event_loop(self, require_mixer):
        action = StopBGM()

        yield
        while True:
            if isinstance(action, (StopBGM, StopPreview)):
                self.is_bgm_on = False
                self.current_action = None
                yield

                while self._action_queue.empty():
                    yield
                action = self._action_queue.get()

            elif isinstance(action, PreviewSong):
                self.current_action = action
                with require_mixer() as mixer:
                    with self._play_song(mixer, action) as preview_task:
                        while True:
                            yield

                            try:
                                preview_task.send(None)
                            except StopIteration:
                                action = self.current_action
                                break

                            if not self._action_queue.empty():
                                action = self._action_queue.get()
                                if action == self.current_action:
                                    continue
                                if isinstance(action, StopPreview) and self.is_bgm_on:
                                    song = self.random_song()
                                    if song is not None:
                                        action = PlayBGM(song, None)
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
        current_action = self.current_action
        if current_action is not None and current_action.song in songs:
            songs.remove(current_action.song)
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

        path = self.beatmap_manager.beatmaps_dir.recognize(self.file_manager.current.abs / token)

        if not isinstance(path, BeatmapFilePath):
            self.stop_preview()
            return

        song = self.beatmap_manager.get_song(path.parent, path)

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
    def file_manager(self):
        return self.provider.get(FileManager)

    @property
    def logger(self):
        return self.provider.get(Logger)

    @cmd.function_command
    def on(self):
        """[rich]Turn on bgm.

        usage: [cmd]bgm[/] [cmd]on[/]
        """
        logger = self.logger
        bgm_controller = self.bgm_controller

        if bgm_controller.is_bgm_on:
            self.now_playing()
            return

        song = bgm_controller.random_song()
        if song is None:
            logger.print("[data/] There is no song in the folder yet!")
            return

        self._play(song)

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
        logger = self.logger
        bgm_controller = self.bgm_controller

        if bgm_controller.current_action is not None:
            song = bgm_controller.random_song()
            if song is None:
                logger.print("[data/] There is no song in the folder yet!")
                return

            self._play(song)

    @cmd.function_command
    def play(self, beatmap, start=None):
        """[rich]Play the song of beatmap.

        usage: [cmd]bgm[/] [cmd]play[/] [arg]{beatmap}[/] [[[kw]--start[/] [arg]{START}[/]]]
                           ╱                   ╲
                 Path, the path to the       float, the time
               beatmap you want to play.     the song started.
        """
        logger = self.logger
        beatmap_manager = self.beatmap_manager

        try:
            song = beatmap_manager.get_song(beatmap.parent, beatmap)
        except beatsheets.BeatmapParseError:
            logger.print("[warn]Fail to read beatmap[/]")
            return

        if song is None:
            logger.print("[warn]This beatmap has no song[/]")
            return

        self._play(song, start)

    def _play(self, song, start=None):
        self.bgm_controller.play(song, start)

    @play.arg_parser("beatmap")
    def _play_beatmap_parser(self):
        file_manager = self.file_manager
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
        logger = self.logger
        bgm_controller = self.bgm_controller
        file_manager = self.file_manager

        current = bgm_controller.current_action
        if isinstance(current, PlayBGM):
            path_mu = file_manager.as_relative_path(UnrecognizedPath(current.song.path, False), markup=True)
            logger.print(f"[music/] now playing: {path_mu}")

        elif isinstance(current, PreviewSong):
            path_mu = file_manager.as_relative_path(UnrecognizedPath(current.song.path, False), markup=True)
            logger.print(f"[music/] now previewing: {path_mu}")

        elif current is None:
            logger.print("[music/] no song is playing")

        else:
            assert False

