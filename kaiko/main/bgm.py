import random
import threading
import queue
import dataclasses
from typing import Optional
from ..utils import commands as cmd
from ..utils import config as cfg
from ..utils import datanodes as dn
from ..utils import providers
from ..devices import audios as aud
from ..devices import clocks
from ..beats import beatsheets
from .loggers import Logger
from .devices import DeviceManager
from .files import FileManager, UnrecognizedPath
from .play import BeatmapManager, BeatmapFilePath, Song


@dn.datanode
def play_fadeinout(
    mixer,
    path,
    fadein_time,
    fadeout_time,
    volume=0.0,
    start=None,
    end=None,
    beatpoints=None,
    metronome=None,
):
    meta = aud.AudioMetadata.read(path)
    node = aud.load(path)
    node = dn.tslice(node, meta.samplerate, start, end)
    # initialize before attach; it will seek to the starting frame
    node.__enter__()

    samplerate = mixer.settings.output_samplerate
    out_event = threading.Event()
    node = dn.pipe(
        node,
        mixer.resample(meta.samplerate, meta.channels, volume),
        dn.fadein(samplerate, fadein_time),
        dn.fadeout(samplerate, fadeout_time, out_event),
    )
    node = dn.attach(node)

    start = start if start is not None else 0.0
    before = end - start - fadeout_time if end is not None else None

    @dn.datanode
    def sync_tempo():
        if beatpoints is None:
            while True:
                yield

        with beatpoints.tempo_node() as node:
            time = time0 = yield
            time_ = start
            while True:
                res = node.send(time_)
                if res:
                    beat, tempo = res
                    metronome.tempo(time, beat, tempo)
                time = yield
                time_ = start + time - time0

    @dn.datanode
    def play_song():
        tempo_node = sync_tempo()
        with node, tempo_node:
            data, time = yield
            time0 = time
            while (before is None or time - time0 < before) and not out_event.is_set():
                data = node.send(data)
                tempo_node.send(time)
                data, time = yield data

            out_event.set()
            time1 = time
            while time - time1 < fadeout_time:
                data = node.send(data)
                data, time = yield data

    song_handler = mixer.add_effect(mixer.tmask(play_song(), None))
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
    metronome_tempo : float
        The default tempo of metronome.
    """

    volume: float = -10.0
    play_delay: float = 0.5
    fadein_time: float = 0.5
    fadeout_time: float = 1.0
    preview_duration: float = 30.0
    metronome_tempo: float = 120.0


class BGMController:
    def __init__(self, settings, mixer_settings):
        self.metronome = clocks.Metronome(settings.metronome_tempo)
        self.settings = settings
        self.mixer_settings = mixer_settings
        self._action_queue = queue.Queue()
        self.is_bgm_on = False
        self.current_action = None

    def set_settings(self, settings):
        self.settings = settings

    def set_mixer_settings(self, mixer_settings):
        self.mixer_settings = mixer_settings

    def start(self):
        device_manager = providers.get(DeviceManager)
        loader_task, loader = device_manager.load_engine_loader("mixer")
        event_task = self._bgm_event_loop(loader.require)
        return dn.pipe(loader_task, event_task)

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

            logger = providers.get(Logger)
            file_manager = providers.get(FileManager)
            path_mu = file_manager.as_relative_path(
                UnrecognizedPath(action.song.path, False), markup=True
            )
            logger.print(f"[music/] will preview: {path_mu}")

            yield from play_fadeinout(
                mixer,
                action.song.path,
                fadein_time=self.settings.fadein_time,
                fadeout_time=self.settings.fadeout_time,
                volume=action.song.audio.volume + self.settings.volume,
                start=start,
                end=end,
                beatpoints=action.song.beatpoints,
                metronome=self.metronome,
            ).join()

        elif isinstance(action, PlayBGM):
            yield from dn.sleep(self.settings.play_delay).join()

            logger = providers.get(Logger)
            file_manager = providers.get(FileManager)
            path_mu = file_manager.as_relative_path(
                UnrecognizedPath(action.song.path, False), markup=True
            )
            logger.print(f"[music/] will play: {path_mu}")

            yield from play_fadeinout(
                mixer,
                action.song.path,
                fadein_time=self.settings.fadein_time,
                fadeout_time=self.settings.fadeout_time,
                volume=action.song.audio.volume + self.settings.volume,
                start=action.start,
                beatpoints=action.song.beatpoints,
                metronome=self.metronome,
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
                with require_mixer() as (mixer,):
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
                with require_mixer() as (mixer,):
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
        beatmap_manager = providers.get(BeatmapManager)

        songs = beatmap_manager.get_songs()
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
        file_manager = providers.get(FileManager)
        beatmap_manager = providers.get(BeatmapManager)

        if token is None:
            self.stop_preview()
            return

        path = beatmap_manager.beatmaps_dir.recognize(file_manager.current.abs / token)

        if not isinstance(path, BeatmapFilePath):
            self.stop_preview()
            return

        song = beatmap_manager.get_song(path.parent, path)

        if song is None:
            self.stop_preview()
            return

        self.preview(song)


class BGMCommand:
    def __init__(self):
        self.subcommand = BGMSubCommand()

    @cmd.subcommand
    def bgm(self):
        """Subcommand to control background music."""
        return self.subcommand


class BGMSubCommand:
    @cmd.function_command
    def on(self):
        """[rich]Turn on bgm.

        usage: [cmd]bgm[/] [cmd]on[/]
        """
        logger = providers.get(Logger)
        bgm_controller = providers.get(BGMController)

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
        bgm_controller = providers.get(BGMController)
        bgm_controller.stop()

    @cmd.function_command
    def skip(self):
        """[rich]Skip the currently playing song.

        usage: [cmd]bgm[/] [cmd]skip[/]
        """
        logger = providers.get(Logger)
        bgm_controller = providers.get(BGMController)

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
        logger = providers.get(Logger)
        beatmap_manager = providers.get(BeatmapManager)

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
        bgm_controller = providers.get(BGMController)
        bgm_controller.play(song, start)

    @play.arg_parser("beatmap")
    def _play_beatmap_parser(self):
        file_manager = providers.get(FileManager)
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
        logger = providers.get(Logger)
        bgm_controller = providers.get(BGMController)
        file_manager = providers.get(FileManager)

        current = bgm_controller.current_action
        if isinstance(current, PlayBGM):
            path_mu = file_manager.as_relative_path(
                UnrecognizedPath(current.song.path, False), markup=True
            )
            logger.print(f"[music/] now playing: {path_mu}")

        elif isinstance(current, PreviewSong):
            path_mu = file_manager.as_relative_path(
                UnrecognizedPath(current.song.path, False), markup=True
            )
            logger.print(f"[music/] now previewing: {path_mu}")

        elif current is None:
            logger.print("[music/] no song is playing")

        else:
            assert False
