import sys
import os
import contextlib
import dataclasses
import traceback
import getpass
import shutil
import pkgutil
from pathlib import Path
import appdirs
from ..utils import markups as mu
from ..utils import datanodes as dn
from ..utils import commands as cmd
from ..devices import loggers as log
from ..beats import beatshell
from ..beats import beatmaps
from ..beats import beatsheets
from .profiles import ProfileManager, ProfilesCommand
from .songs import BeatmapManager, KAIKOBGMController, BGMCommand
from .devices import (
    prepare_pyaudio,
    DevicesCommand,
    determine_unicode_version,
    fit_screen,
)


logo = """

  ██▀ ▄██▀   ▄██████▄ ▀██████▄ ██  ▄██▀ █▀▀▀▀▀▀█
  ▀ ▄██▀  ▄▄▄▀█    ██    ██    ██▄██▀   █ ▓▓▓▓ █
  ▄██▀██▄ ▀▀▀▄███████    ██    ███▀██▄  █ ▓▓▓▓ █
  █▀   ▀██▄  ██    ██ ▀██████▄ ██   ▀██▄█▄▄▄▄▄▄█


  🎧  Use headphones for the best experience 🎤 

"""


@dataclasses.dataclass
class KAIKOUser:
    username: str
    data_dir: Path
    cache_dir: Path

    @classmethod
    def create(cls):
        username = getpass.getuser()
        data_dir = Path(appdirs.user_data_dir("K-AIKO", username))
        cache_dir = Path(appdirs.user_cache_dir("K-AIKO", username))
        return cls(username, data_dir, cache_dir)

    @property
    def config_dir(self):
        return self.data_dir / "config"

    @property
    def songs_dir(self):
        return self.data_dir / "songs"

    def is_prepared(self):
        if not self.data_dir.exists():
            return False

        if not self.cache_dir.exists():
            return False

        if not self.config_dir.exists():
            return False

        if not self.songs_dir.exists():
            return False

        return True

    def prepare(self, logger):
        if self.is_prepared():
            return

        # start up
        logger.print("[data/] Prepare your profile...")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.songs_dir.mkdir(parents=True, exist_ok=True)

        logger.print(
            f"[data/] Your data will be stored in {logger.emph(self.data_dir.as_uri())}"
        )
        logger.print(flush=True)

    def remove(self, logger):
        logger.print(
            f"[data/] Remove config directory {logger.emph(self.config_dir.as_uri())}..."
        )
        shutil.rmtree(str(self.config_dir))
        logger.print(
            f"[data/] Remove songs directory {logger.emph(self.songs_dir.as_uri())}..."
        )
        shutil.rmtree(str(self.songs_dir))
        logger.print(
            f"[data/] Remove data directory {logger.emph(self.data_dir.as_uri())}..."
        )
        shutil.rmtree(str(self.data_dir))


class KAIKOMenu:
    update_interval = 0.01

    def __init__(self, profiles, user, manager, logger):
        r"""Constructor.

        Parameters
        ----------
        profiles : ProfileManager
        user : KAIKOUser
        manager : PyAudio
        logger : loggers.Logger
        """
        self.profiles = profiles
        self.user = user
        self.manager = manager
        self.logger = logger
        self.beatmap_manager = BeatmapManager(user.songs_dir, logger)
        self.bgm_controller = KAIKOBGMController(
            logger, self.beatmap_manager, lambda: self.profiles.current.devices.mixer
        )

    @classmethod
    def main(cls):
        # print logo
        print(logo, flush=True)

        try:
            with cls.init() as menu:
                menu.run().exhaust(dt=cls.update_interval, interruptible=True)

        except KeyboardInterrupt:
            pass

        except:
            # print error
            print("\x1b[31m", end="")
            print(traceback.format_exc(), end="")
            print(f"\x1b[m", end="")

    @classmethod
    @contextlib.contextmanager
    def init(cls):
        r"""Initialize KAIKOMenu within a context manager."""
        logger = log.Logger()

        # load user data
        user = KAIKOUser.create()
        user.prepare(logger)

        # load profiles
        profiles = ProfileManager(user.config_dir, logger)
        profiles.on_change(
            lambda settings: logger.recompile_style(
                terminal_settings=settings.devices.terminal,
                logger_settings=settings.devices.logger,
            )
        )
        profiles.update()

        succ = profiles.use()
        if not succ:
            succ = profiles.new()
            if not succ:
                raise RuntimeError("Fail to load profile")

        # load PyAudio
        logger.print("[info/] Load PyAudio...")
        logger.print()

        with prepare_pyaudio(logger) as manager:
            yield cls(profiles, user, manager, logger)

    @dn.datanode
    def run(self):
        r"""Run KAIKOMenu."""
        logger = self.logger

        yield

        if not sys.stdout.isatty():
            raise RuntimeError("please connect to interactive terminal device.")

        # deterimine unicode version
        if (
            self.settings.devices.terminal.unicode_version == "auto"
            and "UNICODE_VERSION" not in os.environ
        ):
            version = yield from determine_unicode_version(logger).join()
            if version is not None:
                os.environ["UNICODE_VERSION"] = version
                self.settings.devices.terminal.unicode_version = version
                self.profiles.set_as_changed()
            logger.print()

        # fit screen size
        size = shutil.get_terminal_size()
        width = self.settings.devices.terminal.best_screen_size
        if size.columns < width:
            logger.print("[hint/] Your screen size seems too small.")

            yield from fit_screen(logger, self.settings.devices.terminal).join()

        # load songs
        self.reload()

        # execute given command
        if len(sys.argv) > 1:
            command = cmd.RootCommandParser(self).build(sys.argv[1:])
            yield from self.execute(command).join()
            return

        # load bgm
        bgm_task = self.bgm_controller.execute(self.manager)

        # tips
        confirm_key = self.settings.shell.input.confirm_key
        help_key = self.settings.shell.input.help_key
        tab_key, _, _ = self.settings.shell.input.autocomplete_keys
        logger.print(
            f"[hint/] Type command and press {logger.emph(confirm_key)} to execute."
        )
        logger.print(f"[hint/] Use {logger.emph(tab_key)} to autocomplete command.")
        logger.print(f"[hint/] If you need help, press {logger.emph(help_key)}.")
        logger.print()

        # prompt
        repl_task = self.repl()
        yield from dn.pipe(repl_task, bgm_task).join()

    @dn.datanode
    def repl(self):
        r"""Start REPL."""
        preview_handler = self.bgm_controller.preview_handler
        input = beatshell.BeatInput(
            self,
            preview_handler,
            self.logger,
            self.user.cache_dir,
            lambda: self.profiles.current.shell,
            lambda: self.profiles.current.devices,
        )
        while True:
            # parse command
            yield from input.prompt().join()

            # execute result
            result = input.result
            if isinstance(result, beatshell.ErrorResult):
                input.prev_session()
                self.logger.print(f"[warn]{self.logger.escape(str(result.error))}[/]")

            elif isinstance(result, beatshell.CompleteResult):
                input.new_session()
                yield from self.execute(result.command).join()

            else:
                assert False

    @dn.datanode
    def execute(self, command):
        r"""Execute a command.

        If it returns executable object (an object has method `execute`), call
        `result.execute(manager)`; if it returns a DataNode, exhaust it;
        otherwise, print repr of result.

        Parameters
        ----------
        command : function
            The command.
        """
        try:
            result = command()

            if hasattr(result, "execute"):
                is_bgm_on = self.bgm_controller.is_bgm_on
                self.bgm_controller.stop()
                yield from result.execute(self.manager).join()
                if is_bgm_on:
                    self.bgm_controller.play()

            elif isinstance(result, dn.DataNode):
                yield from result.join()

            elif result is not None:
                yield
                self.logger.print(self.logger.format_value(result))

        except Exception:
            with self.logger.warn():
                self.logger.print(traceback.format_exc(), end="", markup=False)

    @property
    def settings(self):
        r"""Current settings."""
        return self.profiles.current

    # beatmaps

    @cmd.function_command
    def play(self, beatmap):
        """[rich]Let's beat with the song!

        usage: [cmd]play[/] [arg]{beatmap}[/]
                         ╲
               Path, the path to the
              beatmap you want to play.
              Only the beatmaps in your
             songs folder can be accessed.
        """

        if not self.beatmap_manager.is_beatmap(beatmap):
            self.logger.print("[warn]Not a beatmap.[/]")
            return

        return KAIKOPlay(
            self.user,
            self.user.songs_dir / beatmap,
            self.profiles,
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
            pattern,
            tempo,
            offset,
            self.user,
            self.profiles,
            self.logger,
        )

    @loop.arg_parser("pattern")
    def _loop_pattern_parser(self):
        return cmd.RawParser(
            desc="It should be a pattern.", default="x x o x | x [x x] o _"
        )

    @play.arg_parser("beatmap")
    def _play_beatmap_parser(self):
        return self.beatmap_manager.make_parser()

    @cmd.function_command
    def reload(self):
        """[rich]Reload your songs.

        usage: [cmd]reload[/]
        """
        self.beatmap_manager.reload()

    @cmd.function_command
    def add(self, beatmap):
        """[rich]Add beatmap/beatmapset to your songs folder.

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
        """[rich]Remove beatmap/beatmapset in your songs folder.

        usage: [cmd]remove[/] [arg]{beatmap}[/]
                           ╲
                 Path, the path to the
               beatmap you want to remove.
        """

        self.beatmap_manager.remove(beatmap)

    @remove.arg_parser("beatmap")
    def _remove_beatmap_parser(self):
        return self.beatmap_manager.make_parser()

    @cmd.function_command
    def beatmaps(self):
        """[rich]Show your beatmaps.

        usage: [cmd]beatmaps[/]
        """
        if not self.beatmap_manager.is_uptodate():
            self.reload()

        self.beatmap_manager.print_tree(self.logger)

    @cmd.subcommand
    def bgm(self):
        """Subcommand to control background music."""
        return BGMCommand(self.bgm_controller, self.beatmap_manager, self.logger)

    # devices

    @cmd.subcommand
    def devices(self):
        """Subcommand to manage devices."""
        return DevicesCommand(self.profiles, self.logger, self.manager)

    # profiles

    @cmd.subcommand
    def profiles(self):
        """Subcommand to manage profiles and configurations."""
        return ProfilesCommand(self.profiles, self.logger)

    # system

    @cmd.function_command
    def me(self):
        """[rich]About user.

        usage: [cmd]me[/]
        """
        logger = self.logger

        logger.print(f"username: {logger.emph(self.user.username)}")
        logger.print(f"data directory: {logger.emph(self.user.data_dir.as_uri())}")
        logger.print(f"config directory: {logger.emph(self.user.config_dir.as_uri())}")
        logger.print(f"songs directory: {logger.emph(self.user.songs_dir.as_uri())}")
        logger.print(f"cache directory: {logger.emph(self.user.cache_dir.as_uri())}")

    @cmd.function_command
    def print(self, message, markup=True):
        """[rich]Print something.

        usage: [cmd]print[/] [arg]{message}[/] [[[kw]--markup[/] [arg]{MARKUP}[/]]]
                        ╱                    ╲
              text, the message               ╲
               to be printed.          bool, use markup or not;
                                          default is True.
        """

        try:
            self.logger.print(message, markup=markup)
        except mu.MarkupParseError as e:
            self.logger.print(f"[warn]{self.logger.escape(str(e))}[/]")

    @print.arg_parser("message")
    def _print_message_parser(self):
        return cmd.RawParser(
            desc="It should be some text," " indicating the message to be printed."
        )

    @print.arg_parser("markup")
    def _print_escape_parser(self, message):
        return cmd.LiteralParser(
            bool,
            default=False,
            desc="It should be bool,"
            " indicating whether to use markup;"
            " the default is False.",
        )

    @cmd.function_command
    def clean(self):
        """[rich]Clean screen.

        usage: [cmd]clean[/]
        """
        self.logger.clear()

    @cmd.function_command
    @dn.datanode
    def bye(self):
        """[rich]Close K-AIKO.

        usage: [cmd]bye[/]
        """
        if self.profiles.is_changed():
            yes = yield from self.logger.ask(
                "Exit without saving current configuration?"
            ).join()
            if not yes:
                return
        self.logger.print("Bye~")
        raise KeyboardInterrupt

    @cmd.function_command
    @dn.datanode
    def bye_forever(self):
        """[rich]Clean up all your data and close K-AIKO.

        usage: [cmd]bye_forever[/]
        """
        logger = self.logger

        logger.print("This command will clean up all your data.")

        yes = yield from logger.ask("Do you really want to do that?", False).join()
        if yes:
            self.user.remove(logger)
            logger.print("Good luck~")
            raise KeyboardInterrupt


class KAIKOPlay:
    def __init__(self, user, filepath, profiles, logger):
        self.user = user
        self.filepath = filepath
        self.profiles = profiles
        self.logger = logger

    @dn.datanode
    def execute(self, manager):
        logger = self.logger
        devices_settings = self.profiles.current.devices
        gameplay_settings = self.profiles.current.gameplay

        try:
            beatmap = beatsheets.read(str(self.filepath))

        except beatsheets.BeatmapParseError:
            logger.print(
                f"[warn]Failed to read beatmap {logger.escape(str(self.filepath))}[/]"
            )
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)

        else:
            pause_key = gameplay_settings.controls.pause_key
            stop_key = gameplay_settings.controls.stop_key
            display_keys = gameplay_settings.controls.display_delay_adjust_keys
            knock_keys = gameplay_settings.controls.knock_delay_adjust_keys
            energy_keys = gameplay_settings.controls.knock_energy_adjust_keys
            logger.print(
                f"[hint/] Press {logger.emph(pause_key)} to pause/resume the game."
            )
            logger.print(f"[hint/] Press {logger.emph(stop_key)} to end the game.")
            logger.print(
                f"[hint/] Use {logger.emph(display_keys[0])} and {logger.emph(display_keys[1])} to adjust display delay."
            )
            logger.print(
                f"[hint/] Use {logger.emph(knock_keys[0])} and {logger.emph(knock_keys[1])} to adjust hit delay."
            )
            logger.print(
                f"[hint/] Use {logger.emph(energy_keys[0])} and {logger.emph(energy_keys[1])} to adjust hit strength."
            )
            logger.print()

            score, devices_settings = yield from beatmap.play(
                manager, self.user, devices_settings, gameplay_settings
            ).join()

            if devices_settings is not None:
                yes = yield from self.logger.ask(
                    "Keep changes to device settings?"
                ).join()
                if yes:
                    logger.print("[data/] Update device settings...")
                    self.profiles.current.devices = devices_settings
                    self.profiles.set_as_changed()

            logger.print()
            logger.print_scores(
                beatmap.settings.difficulty.performance_tolerance, score.perfs
            )


class KAIKOLoop:
    def __init__(self, pattern, tempo, offset, user, profiles, logger):
        self.pattern = pattern
        self.tempo = tempo
        self.offset = offset
        self.user = user
        self.profiles = profiles
        self.logger = logger

    @dn.datanode
    def execute(self, manager):
        logger = self.logger
        devices_settings = self.profiles.current.devices
        gameplay_settings = self.profiles.current.gameplay

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

            pause_key = gameplay_settings.controls.pause_key
            stop_key = gameplay_settings.controls.stop_key
            display_keys = gameplay_settings.controls.display_delay_adjust_keys
            knock_keys = gameplay_settings.controls.knock_delay_adjust_keys
            energy_keys = gameplay_settings.controls.knock_energy_adjust_keys
            logger.print(
                f"[hint/] Press {logger.emph(pause_key)} to pause/resume the game."
            )
            logger.print(f"[hint/] Press {logger.emph(stop_key)} to end the game.")
            logger.print(
                f"[hint/] Use {logger.emph(display_keys[0])} and {logger.emph(display_keys[1])} to adjust display delay."
            )
            logger.print(
                f"[hint/] Use {logger.emph(knock_keys[0])} and {logger.emph(knock_keys[1])} to adjust hit delay."
            )
            logger.print(
                f"[hint/] Use {logger.emph(energy_keys[0])} and {logger.emph(energy_keys[1])} to adjust hit strength."
            )
            logger.print()

            score, devices_settings = yield from beatmap.play(
                manager, self.user, devices_settings, gameplay_settings
            ).join()

            if devices_settings is not None:
                yes = yield from self.logger.ask(
                    "Keep changes to device settings?"
                ).join()
                if yes:
                    logger.print("[data/] Update device settings...")
                    self.profiles.current.devices = devices_settings
                    self.profiles.set_as_changed()

            logger.print()
            logger.print_scores(
                beatmap.settings.difficulty.performance_tolerance, score.perfs
            )
