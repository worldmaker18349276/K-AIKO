import sys
import os
import re
import time
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
from ..utils import config as cfg
from ..utils import biparsers as bp
from ..utils import commands as cmd
from ..devices import loggers as log
from ..beats import beatshell
from ..beats import beatmaps
from ..beats import beatsheets
from ..beats import beatanalyzer
from .profiles import ProfileManager, ConfigCommand
from .songs import BeatmapManager, KAIKOBGMController, BGMCommand
from .devices import prepare_pyaudio, DevicesCommand, determine_unicode_version, fit_screen


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
    def create(clz):
        username = getpass.getuser()
        data_dir = Path(appdirs.user_data_dir("K-AIKO", username))
        cache_dir = Path(appdirs.user_cache_dir("K-AIKO", username))
        return clz(username, data_dir, cache_dir)

    @property
    def config_dir(self):
        return self.data_dir / "config"

    @property
    def songs_dir(self):
        return self.data_dir / "songs"

    @property
    def history_file(self):
        return self.cache_dir / ".beatshell_history"

    def is_prepared(self):
        if not self.data_dir.exists():
            return False

        if not self.cache_dir.exists():
            return False

        if not self.history_file.exists():
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
        self.history_file.touch()

        (self.data_dir / "samples/").mkdir(exist_ok=True)
        resources = ["samples/soft.wav",
                     "samples/loud.wav",
                     "samples/incr.wav",
                     "samples/rock.wav",
                     "samples/disk.wav"]
        for rspath in resources:
            logger.print(f"[data/] Load resource {logger.escape(rspath)}...")
            data = pkgutil.get_data("kaiko", rspath)
            open(self.data_dir / rspath, 'wb').write(data)

        logger.print(f"[data/] Your data will be stored in {logger.emph(self.data_dir.as_uri())}")
        logger.print(flush=True)

    def remove(self, logger):
        logger.print(f"[data/] Remove config directory {logger.emph(self.config_dir.as_uri())}...")
        shutil.rmtree(str(self.config_dir))
        logger.print(f"[data/] Remove songs directory {logger.emph(self.songs_dir.as_uri())}...")
        shutil.rmtree(str(self.songs_dir))
        logger.print(f"[data/] Remove data directory {logger.emph(self.data_dir.as_uri())}...")
        shutil.rmtree(str(self.data_dir))


class KAIKOMenu:
    def __init__(self, config, user, manager, logger):
        r"""Constructor.

        Parameters
        ----------
        config : ProfileManager
        user : KAIKOUser
        manager : PyAudio
        logger : loggers.Logger
        """
        self._config = config
        self.user = user
        self.manager = manager
        self.logger = logger
        self.beatmap_manager = BeatmapManager(user.songs_dir, logger)
        self.bgm_controller = KAIKOBGMController(config.current.devices.mixer, logger, self.beatmap_manager)
        config.on_change(lambda settings: self.bgm_controller.update_mixer_settings(settings.devices.mixer))

    @staticmethod
    def main():
        # print logo
        print(logo, flush=True)

        try:
            dt = 0.01
            with KAIKOMenu.init() as menu:
                menu.run().exhaust(dt=dt, interruptible=True)

        except KeyboardInterrupt:
            pass

        except:
            # print error
            print("\x1b[31m", end="")
            print(traceback.format_exc(), end="")
            print(f"\x1b[m", end="")

    @classmethod
    @contextlib.contextmanager
    def init(clz):
        r"""Initialize KAIKOMenu within a context manager."""
        logger = log.Logger()

        # load user data
        user = KAIKOUser.create()
        user.prepare(logger)

        # load config
        config = ProfileManager.initialize(user.config_dir, logger)

        # load PyAudio
        logger.print("[info/] Load PyAudio...")
        logger.print()

        with prepare_pyaudio(logger) as manager:
            yield clz(config, user, manager, logger)

    @dn.datanode
    def run(self):
        r"""Run KAIKOMenu."""
        logger = self.logger

        yield

        if not sys.stdout.isatty():
            raise ValueError("please connect to interactive terminal device.")

        # deterimine unicode version
        if self.settings.devices.terminal.unicode_version == "auto" and "UNICODE_VERSION" not in os.environ:
            version = determine_unicode_version(logger).join()
            if version is not None:
                os.environ["UNICODE_VERSION"] = version
                self.settings.devices.terminal.unicode_version = version
                self._config.set_change()
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
        bgm_task = self.bgm_controller.load_bgm(self.manager)

        # tips
        confirm_key = self.settings.shell.input.confirm_key
        help_key = self.settings.shell.input.help_key
        tab_key, _, _ = self.settings.shell.input.autocomplete_keys
        logger.print(f"[hint/] Type command and press {logger.emph(confirm_key)} to execute.")
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
        input = beatshell.BeatInput(self, preview_handler, self.logger, self.user.history_file, self.settings.shell)
        while True:
            # parse command
            input.update_settings(self.settings.shell)
            yield from input.prompt(self.settings.devices, self.user).join()

            # execute result
            result = input.result
            if isinstance(result, beatshell.ErrorResult):
                input.prev_session()
                self.logger.print(f"[warn]{str(result.error)}[/]")

            elif isinstance(result, beatshell.HelpResult):
                input.prev_session()
                yield from self.execute(result.command).join()

            elif isinstance(result, beatshell.CompleteResult):
                input.new_session()
                yield from self.execute(result.command).join()

            else:
                assert False

    @dn.datanode
    def execute(self, command):
        r"""Execute a command.
        If it returns executable object (an object has method `execute`), call
        `result.execute(manager)`; if it returns a DataNode, exhaust it; otherwise,
        print repr of result.

        Parameters
        ----------
        command : function
            The command.
        """
        result = command()

        if hasattr(result, 'execute'):
            is_bgm_on = self.bgm_controller.is_bgm_on
            if is_bgm_on:
                self.bgm.off()
            yield from result.execute(self.manager).join()
            if is_bgm_on:
                self.bgm.on()

        elif isinstance(result, dn.DataNode):
            yield from result.join()

        elif result is not None:
            yield
            self.logger.print(repr(result), markup=False)

    @property
    def settings(self):
        r"""Current settings."""
        return self._config.current

    # beatmaps

    @cmd.function_command
    def play(self, beatmap):
        """Let's beat with the song!

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

        return KAIKOPlay(self.user, self.user.songs_dir / beatmap,
                         self.settings.devices, self.settings.gameplay, self.logger)

    @cmd.function_command
    def loop(self, pattern, tempo:float=120.0, offset:float=1.0):
        return KAIKOLoop(pattern, tempo, offset,
                         self.user, self.settings.devices, self.settings.gameplay, self.logger)

    @loop.arg_parser("pattern")
    def _loop_pattern_parser(self):
        return cmd.RawParser(desc="It should be a pattern.", default="x x o x | x [x x] o _")

    @play.arg_parser("beatmap")
    def _play_beatmap_parser(self):
        return self.beatmap_manager.make_parser()

    @cmd.function_command
    def reload(self):
        """Reload your songs."""
        self.beatmap_manager.reload()

    @cmd.function_command
    def add(self, beatmap):
        """Add beatmap/beatmapset to your songs folder.

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
        """Remove beatmap/beatmapset in your songs folder.

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
        """Your beatmaps."""
        if not self.beatmap_manager.is_uptodate():
            self.reload()

        self.beatmap_manager.print_tree(self.logger)

    @cmd.subcommand
    @property
    def bgm(self):
        """Background music."""
        return BGMCommand(self.bgm_controller, self.beatmap_manager, self.logger)

    # devices

    @cmd.subcommand
    @property
    def devices(self):
        """Devices."""
        return DevicesCommand(self._config, self.logger, self.manager)

    # config

    @cmd.subcommand
    @property
    def config(self):
        """Configuration."""
        return ConfigCommand(self._config, self.logger)

    # system

    @cmd.function_command
    def me(self):
        """About user."""
        logger = self.logger

        logger.print(f"username: {logger.emph(self.user.username)}")
        logger.print(f"data directory: {logger.emph(self.user.data_dir.as_uri())}")
        logger.print(f"config directory: {logger.emph(self.user.config_dir.as_uri())}")
        logger.print(f"songs directory: {logger.emph(self.user.songs_dir.as_uri())}")
        logger.print(f"command history: {logger.emph(self.user.history_file.as_uri())}")
        logger.print(f"cache directory: {logger.emph(self.user.cache_dir.as_uri())}")

    @cmd.function_command
    def print(self, message, markup=True):
        """Print something.

        usage: [cmd]say[/] [arg]{message}[/] [[[kw]--markup[/] [arg]{MARKUP}[/]]]
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
        return cmd.RawParser(desc="It should be some text,"
                                  " indicating the message to be printed.")

    @print.arg_parser("markup")
    def _print_escape_parser(self, message):
        return cmd.LiteralParser(bool, default=False,
                                       desc="It should be bool,"
                                            " indicating whether to use markup;"
                                            " the default is False.")

    @cmd.function_command
    def clean(self):
        """Clean screen."""
        self.logger.clear()

    @cmd.function_command
    def bye(self):
        """Close K-AIKO."""
        self.logger.print("Bye~")
        raise KeyboardInterrupt

    @cmd.function_command
    @dn.datanode
    def bye_forever(self):
        """Clean up all your data and close K-AIKO."""
        logger = self.logger

        logger.print("This command will clean up all your data.")

        yes = yield from logger.ask("Do you really want to do that?", False).join()
        if yes:
            self.user.remove(logger)
            logger.print("Good luck~")
            raise KeyboardInterrupt


class KAIKOPlay:
    def __init__(self, user, filepath, devices_settings, gameplay_settings, logger):
        self.user = user
        self.filepath = filepath
        self.devices_settings = devices_settings
        self.gameplay_settings = gameplay_settings
        self.logger = logger

    @dn.datanode
    def execute(self, manager):
        logger = self.logger

        try:
            beatmap = beatsheets.BeatSheet.read(str(self.filepath))

        except beatsheets.BeatmapParseError:
            with logger.warn():
                logger.print(f"Failed to read beatmap {logger.escape(str(self.filepath))}")
                logger.print(traceback.format_exc(), end="", markup=False)

        else:
            stop_key = self.gameplay_settings.controls.stop_key
            sound_keys = self.gameplay_settings.controls.sound_delay_adjust_keys
            display_keys = self.gameplay_settings.controls.display_delay_adjust_keys
            knock_keys = self.gameplay_settings.controls.knock_delay_adjust_keys
            energy_keys = self.gameplay_settings.controls.knock_energy_adjust_keys
            logger.print(f"[hint/] Press {logger.emph(stop_key)} to end the game.")
            logger.print(f"[hint/] Use {logger.emph(sound_keys[0])} and {logger.emph(sound_keys[1])} to adjust click sound delay.")
            logger.print(f"[hint/] Use {logger.emph(display_keys[0])} and {logger.emph(display_keys[1])} to adjust display delay.")
            logger.print(f"[hint/] Use {logger.emph(knock_keys[0])} and {logger.emph(knock_keys[1])} to adjust hit delay.")
            logger.print(f"[hint/] Use {logger.emph(energy_keys[0])} and {logger.emph(energy_keys[1])} to adjust hit strength.")
            logger.print()

            score = yield from beatmap.play(manager, self.user, self.devices_settings, self.gameplay_settings).join()

            logger.print()
            beatanalyzer.show_analyze(beatmap.settings.difficulty.performance_tolerance, score.perfs)

class KAIKOLoop:
    def __init__(self, pattern, tempo, offset, user, devices_settings, gameplay_settings, logger):
        self.pattern = pattern
        self.tempo = tempo
        self.offset = offset
        self.user = user
        self.devices_settings = devices_settings
        self.gameplay_settings = gameplay_settings
        self.logger = logger

    @dn.datanode
    def execute(self, manager):
        logger = self.logger

        try:
            events, width = beatsheets.BeatSheet.parse_patterns(self.pattern)

        except beatsheets.BeatmapParseError:
            with logger.warn():
                logger.print(f"Failed to parse pattern.")
                logger.print(traceback.format_exc(), end="", markup=False)

        else:
            beatmap = beatmaps.Loop(tempo=self.tempo, offset=self.offset, width=width, events=events)

            stop_key = self.gameplay_settings.controls.stop_key
            sound_keys = self.gameplay_settings.controls.sound_delay_adjust_keys
            display_keys = self.gameplay_settings.controls.display_delay_adjust_keys
            knock_keys = self.gameplay_settings.controls.knock_delay_adjust_keys
            energy_keys = self.gameplay_settings.controls.knock_energy_adjust_keys
            logger.print(f"[hint/] Press {logger.emph(stop_key)} to end the game.")
            logger.print(f"[hint/] Use {logger.emph(sound_keys[0])} and {logger.emph(sound_keys[1])} to adjust click sound delay.")
            logger.print(f"[hint/] Use {logger.emph(display_keys[0])} and {logger.emph(display_keys[1])} to adjust display delay.")
            logger.print(f"[hint/] Use {logger.emph(knock_keys[0])} and {logger.emph(knock_keys[1])} to adjust hit delay.")
            logger.print(f"[hint/] Use {logger.emph(energy_keys[0])} and {logger.emph(energy_keys[1])} to adjust hit strength.")
            logger.print()

            score = yield from beatmap.play(manager, self.user, self.devices_settings, self.gameplay_settings).join()

            logger.print()
            beatanalyzer.show_analyze(beatmap.settings.difficulty.performance_tolerance, score.perfs)

