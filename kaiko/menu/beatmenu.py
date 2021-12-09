import sys
import os
import re
import time
import contextlib
import dataclasses
import traceback
import zipfile
import getpass
import shutil
import pkgutil
from pathlib import Path
import appdirs
from ..utils import datanodes as dn
from ..utils import config as cfg
from ..utils import biparsers as bp
from ..utils import commands as cmd
from ..devices import loggers as log
from ..devices import engines
from ..beats import beatshell
from ..beats import beatmaps
from ..beats import beatsheets
from ..beats import beatanalyzer
from .profiles import ProfileManager, ConfigCommand, ProfileNameError, ProfileTypeError
from .songs import BeatmapManager, KAIKOBGMController, BGMCommand
from .devices import prepare_pyaudio, DevicesCommand, determine_unicode_version, fit_screen


logo = """

  ██▀ ▄██▀   ▄██████▄ ▀██████▄ ██  ▄██▀ █▀▀▀▀▀▀█
  ▀ ▄██▀  ▄▄▄▀█    ██    ██    ██▄██▀   █ ▓▓▓▓ █
  ▄██▀██▄ ▀▀▀▄███████    ██    ███▀██▄  █ ▓▓▓▓ █
  █▀   ▀██▄  ██    ██ ▀██████▄ ██   ▀██▄█▄▄▄▄▄▄█


  🎧  Use headphones for the best experience 🎤 

"""

def echo_str(escaped_str):
    r"""Interpret a string like bash's echo.
    It interprets the following backslash-escaped characters into:
        \a     alert (bell)
        \b     backspace
        \c     suppress further output
        \e     escape character
        \f     form feed
        \n     new line
        \r     carriage return
        \t     horizontal tab
        \v     vertical tab
        \\     backslash
        \0NNN  the character whose ASCII code is NNN (octal).  NNN can be 0 to 3 octal digits
        \xHH   the eight-bit character whose value is HH (hexadecimal).  HH can be one or two hex digits

    Parameters
    ----------
    escaped_str : str
        The string to be interpreted.

    Returns
    -------
    interpreted_str : str
        The interpreted string.
    """
    regex = r"\\c.*|\\[\\abefnrtv]|\\0[0-7]{0,3}|\\x[0-9a-fA-F]{1,2}|."

    escaped = {
        r"\\": "\\",
        r"\a": "\a",
        r"\b": "\b",
        r"\e": "\x1b",
        r"\f": "\f",
        r"\n": "\n",
        r"\r": "\r",
        r"\t": "\t",
        r"\v": "\v",
        }

    def repl(match):
        matched = match.group(0)

        if matched.startswith("\\c"):
            return ""
        elif matched in escaped:
            return escaped[matched]
        elif matched.startswith("\\0"):
            return chr(int(matched[2:] or "0", 8))
        elif matched.startswith("\\x"):
            return chr(int(matched[2:], 16))
        else:
            return matched

    return re.sub(regex, repl, escaped_str)


class KAIKOSettings(cfg.Configurable):
    devices = engines.DevicesSettings
    shell = beatshell.BeatShellSettings
    gameplay = beatmaps.GameplaySettings


@dataclasses.dataclass
class KAIKOUser:
    username: str
    config_dir: Path
    data_dir: Path
    songs_dir: Path
    history_file: Path

    @classmethod
    def create(clz):
        username = getpass.getuser()
        data_dir = Path(appdirs.user_data_dir("K-AIKO", username))
        config_dir = data_dir / "config"
        songs_dir = data_dir / "songs"
        history_file = data_dir / ".beatshell_history"
        return clz(username, config_dir, data_dir, songs_dir, history_file)

    def is_prepared(self):
        if not self.history_file.exists():
            return False

        if not self.config_dir.exists():
            return False

        if not self.data_dir.exists():
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
        logger : Logger
        """
        self._config = config
        self.user = user
        self.manager = manager
        self.logger = logger
        self.beatmap_manager = BeatmapManager(user.songs_dir, logger)
        self.bgm_controller = KAIKOBGMController(config, logger, self.beatmap_manager)

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
        config = ProfileManager(KAIKOSettings, user.config_dir, logger)

        # load config
        if config.default_name is None:
            config.new()

        else:
            try:
                config.use()
            except (ProfileNameError, ProfileTypeError, bp.DecodeError):
                with logger.warn():
                    logger.print("Failed to load default configuration")
                    logger.print(traceback.format_exc(), end="", markup=False)

                config.new()

        logger.set_settings(config.current.devices.terminal, config.current.devices.logger)

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
            with determine_unicode_version(logger) as task:
                yield from task.join((yield))
                version = task.result
                if version is not None:
                    os.environ["UNICODE_VERSION"] = version
                    self.settings.devices.terminal.unicode_version = version
                    logger.set_settings(terminal_settings=self.settings.devices.terminal)
                    logger.recompile_style()
            logger.print()

        # fit screen size
        size = shutil.get_terminal_size()
        width = self.settings.devices.terminal.best_screen_size
        if size.columns < width:
            logger.print("[hint/] Your screen size seems too small.")

            with fit_screen(logger, self.settings.devices.terminal) as fit_task:
                yield from fit_task.join((yield))

        # load songs
        self.reload()

        # execute given command
        if len(sys.argv) > 1:
            command = cmd.RootCommandParser(self).build(sys.argv[1:])
            with self.execute(command) as command_task:
                yield from command_task.join((yield))
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
        with dn.pipe(repl_task, bgm_task) as task:
            yield from task.join((yield))

    @dn.datanode
    def repl(self):
        r"""Start REPL."""
        input = beatshell.BeatInput(self, self.user.history_file)
        while True:
            # parse command
            with input.prompt(self.settings.devices, self.settings.shell) as prompt_task:
                yield from prompt_task.join((yield))

            # execute result
            result = input.result
            if isinstance(result, beatshell.InputError):
                input.prev_session()
                with self.logger.warn():
                    self.logger.print(result.error, markup=False)
            else:
                input.new_session()
                with self.execute(result.command) as command_task:
                    yield from command_task.join((yield))

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
            has_bgm = bool(self.bgm_controller._current_bgm)
            if has_bgm:
                self.bgm.off()
            with result.execute(self.manager) as command_task:
                yield from command_task.join((yield))
            if has_bgm:
                self.bgm.on()

        elif isinstance(result, dn.DataNode):
            with result:
                yield from result.join((yield))

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
            with self.logger.warn():
                self.logger.print("Not a beatmap.")
                return

        return KAIKOPlay(self.user.data_dir, self.user.songs_dir / beatmap,
                         self.settings.devices, self.settings.gameplay, self.logger)

    @cmd.function_command
    def loop(self, pattern, tempo:float=120.0, offset:float=1.0):
        return KAIKOLoop(pattern, tempo, offset,
                         self.user.data_dir, self.settings.devices, self.settings.gameplay, self.logger)

    @loop.arg_parser("pattern")
    def _loop_pattern_parser(self):
        return cmd.RawParser(desc="It should be a pattern.", default="x x o x | x [x x] o _")

    @play.arg_parser("beatmap")
    def _play_beatmap_parser(self):
        return self.beatmap_manager.make_parser(self.bgm_controller)

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
        options = []
        for song, beatmapset in self.beatmap_manager._beatmaps.items():
            options.append(os.path.join(str(song), ""))
            for beatmap in beatmapset:
                options.append(str(beatmap))

        return cmd.OptionParser(options)

    @cmd.function_command
    def beatmaps(self):
        """Your beatmaps."""
        if not self.beatmap_manager.is_uptodate():
            self.reload()

        for beatmapset in self.beatmap_manager._beatmaps.values():
            for beatmap in beatmapset:
                self.logger.print("• " + self.logger.escape(str(beatmap)))

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

    @cmd.function_command
    def say(self, message, escape=False):
        """Say something to... yourself.

        usage: [cmd]say[/] [arg]{message}[/] \[[kw]--escape[/] [arg]{ESCAPE}[/]]
                      ╱                    ╲
            text, the message               ╲
             to be printed.          bool, use backslash escapes
                                    or not; the default is False.
        """

        if escape:
            self.logger.print(echo_str(message), markup=False)
        else:
            self.logger.print(message, markup=False)

    @cmd.function_command
    def clean(self):
        """Clean screen."""
        self.logger.clear()

    @say.arg_parser("message")
    def _say_message_parser(self):
        return cmd.RawParser(desc="It should be some text,"
                                  " indicating the message to be printed.")

    @say.arg_parser("escape")
    def _say_escape_parser(self, message):
        return cmd.LiteralParser(bool, default=False,
                                       desc="It should be bool,"
                                            " indicating whether to use backslash escapes;"
                                            " the default is False.")

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

        with logger.ask("Do you really want to do that?", False) as task:
            yield from task.join((yield))
            if task.result:
                self.user.remove(logger)
                logger.print("Good luck~")
                raise KeyboardInterrupt


class KAIKOPlay:
    def __init__(self, data_dir, filepath, devices_settings, gameplay_settings, logger):
        self.data_dir = data_dir
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

            with beatmap.play(manager, self.data_dir, self.devices_settings, self.gameplay_settings) as task:
                yield from task.join((yield))
                score = task.result

            logger.print()
            beatanalyzer.show_analyze(beatmap.settings.difficulty.performance_tolerance, score.perfs)

class KAIKOLoop:
    def __init__(self, pattern, tempo, offset, data_dir, devices_settings, gameplay_settings, logger):
        self.pattern = pattern
        self.tempo = tempo
        self.offset = offset
        self.data_dir = data_dir
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

            with beatmap.play(manager, self.data_dir, self.devices_settings, self.gameplay_settings) as task:
                yield from task.join((yield))
                score = task.result

            logger.print()
            beatanalyzer.show_analyze(beatmap.settings.difficulty.performance_tolerance, score.perfs)

