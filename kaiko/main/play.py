import traceback
from pathlib import Path
from ..devices import loggers as log
from ..utils import datanodes as dn
from ..utils import commands as cmd
from ..beats import beatmaps
from ..beats import beatsheets
from .files import FileManager
from .profiles import ProfileManager
from .songs import BeatmapManager


class PlayCommand:
    def __init__(self, provider, resources_dir, cache_dir, beatmaps_dir):
        self.provider = provider
        self.resources_dir = resources_dir
        self.cache_dir = cache_dir
        self.beatmaps_dir = beatmaps_dir

    @property
    def logger(self):
        return self.provider.get(log.Logger)

    @property
    def profiles_manager(self):
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
            self.beatmaps_dir / beatmap,
            start,
            self.profiles_manager,
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
            pattern, tempo, offset, self.resources_dir, self.cache_dir, self.profiles_manager, self.logger,
        )

    @loop.arg_parser("pattern")
    def _loop_pattern_parser(self):
        return cmd.RawParser(
            desc="It should be a pattern.", default="x x o x | x [x x] o _"
        )

    @play.arg_parser("beatmap")
    def _play_beatmap_parser(self):
        current = self.file_manager.root / self.file_manager.current
        return self.beatmap_manager.make_parser(current, type="file")

    @play.arg_parser("start")
    def _play_start_parser(self, beatmap):
        return cmd.TimeParser(0.0)

    @cmd.function_command
    def reload(self):
        """[rich]Reload your beatmaps.

        usage: [cmd]reload[/]
        """
        self.beatmap_manager.reload()

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

        self.beatmap_manager.add(beatmap)

    @add.arg_parser("beatmap")
    def _add_beatmap_parser(self):
        return cmd.PathParser()

    @cmd.function_command
    def remove(self, beatmap):
        """[rich]Remove beatmap/beatmapset in your beatmaps folder.

        usage: [cmd]remove[/] [arg]{beatmap}[/]
                           ╲
                 Path, the path to the
               beatmap you want to remove.
        """

        self.beatmap_manager.remove(beatmap)

    @remove.arg_parser("beatmap")
    def _remove_beatmap_parser(self):
        current = self.file_manager.root / self.file_manager.current
        return self.beatmap_manager.make_parser(current, type="all")


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
    def __init__(self, resources_dir, cache_dir, filepath, start, profiles_manager, logger):
        self.resources_dir = resources_dir
        self.cache_dir = cache_dir
        self.filepath = filepath
        self.start = start
        self.profiles_manager = profiles_manager
        self.logger = logger

    @dn.datanode
    def execute(self, manager):
        logger = self.logger
        devices_settings = self.profiles_manager.current.devices
        gameplay_settings = self.profiles_manager.current.gameplay

        try:
            beatmap = beatsheets.read(str(self.filepath))

        except beatsheets.BeatmapParseError:
            filepath = logger.escape(str(self.filepath), type="all")
            logger.print(
                f"[warn]Failed to read beatmap {filepath}[/]"
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
                    title = self.profiles_manager.get_title()
                    old = self.profiles_manager.format()
                    self.profiles_manager.current.devices = devices_settings
                    self.profiles_manager.set_as_changed()
                    new = self.profiles_manager.format()

                    self.logger.print(f"[data/] Your changes")
                    logger.print(
                        logger.format_code_diff(old, new, title=title, is_changed=True)
                    )


class KAIKOLoop:
    def __init__(self, pattern, tempo, offset, resources_dir, cache_dir, profiles_manager, logger):
        self.pattern = pattern
        self.tempo = tempo
        self.offset = offset
        self.resources_dir = resources_dir
        self.cache_dir = cache_dir
        self.profiles_manager = profiles_manager
        self.logger = logger

    @dn.datanode
    def execute(self, manager):
        logger = self.logger
        devices_settings = self.profiles_manager.current.devices
        gameplay_settings = self.profiles_manager.current.gameplay

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
                    title = self.profiles_manager.get_title()
                    old = self.profiles_manager.format()
                    self.profiles_manager.current.devices = devices_settings
                    self.profiles_manager.set_as_changed()
                    new = self.profiles_manager.format()

                    self.logger.print(f"[data/] Your changes")
                    logger.print(
                        logger.format_code_diff(old, new, title=title, is_changed=True)
                    )
