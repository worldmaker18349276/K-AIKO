import sys
import os
import re
import time
from inspect import cleandoc
import traceback
import shutil
import pkgutil
from pathlib import Path
from .. import __version__
from ..utils.providers import Provider
from ..utils import markups as mu
from ..utils import datanodes as dn
from ..utils import commands as cmd
from ..devices import clocks
from . import beatshell
from .files import (
    FileManager,
    FilesCommand,
    DirectCdCommand,
    RecognizedDirPath,
    RecognizedFilePath,
    as_child,
)
from .settings import KAIKOSettings
from .loggers import Logger
from .profiles import ProfileManager, ProfilesCommand, ProfilesDirPath
from .play import BeatmapManager, BeatmapsDirPath, PlayCommand
from .bgm import BGMController, BGMCommand
from .devices import (
    DeviceManager,
    DevicesCommand,
    DevicesDirPath,
)


logo = """

  ██▀ ▄██▀   ▄██████▄ ▀██████▄ ██  ▄██▀ █▀▀▀▀▀▀█
  ▀ ▄██▀  ▄▄▄▀█    ██    ██    ██▄██▀   █ ▓▓▓▓ █
  ▄██▀██▄ ▀▀▀▄███████    ██    ███▀██▄  █ ▓▓▓▓ █
  █▀   ▀██▄  ██    ██ ▀██████▄ ██   ▀██▄█▄▄▄▄▄▄█  {}

"""

logo_animated_text = "  ▣  A sound-controlled terminal-based rhythm game ▣ "


def animated_print(text, kps=30.0, word_delay=0.05, pre_delay=0.5, post_delay=1.0):
    time.sleep(pre_delay)
    for match in re.finditer(r".*?\s+", text):
        for ch in match.group(0):
            print(ch, end="", flush=True)
            time.sleep(1 / kps)
        time.sleep(word_delay)
    time.sleep(post_delay)


class RootDirPath(RecognizedDirPath):
    """The workspace of KAIKO"""

    def info_detailed(self, provider):
        logger = provider.get(Logger)
        profile_manager = provider.get(ProfileManager)

        input_settings = profile_manager.current.shell.input.control

        confirm_key = logger.escape(input_settings.confirm_key, type="all")
        help_key = logger.escape(input_settings.help_key, type="all")
        tab_key = logger.escape(input_settings.autocomplete_keys[0], type="all")

        info = f"""
        [color=bright_green]⠟⣡⡾⠋⣀⡰⡟⠛⢻⡆⠙⢻⡟⠓⢸⣇⣴⠟⠁⡏⣭⣭⢹[/]
        [color=bright_green]⡾⠋⠻⣦⡉⢱⡟⠛⢻⡇⢤⣼⣧⣄⢸⡟⠙⢷⣄⣇⣛⣛⣸[/]

        [hint/] Use headphones for the best experience.
        [hint/] Type command and press [emph]{confirm_key}[/] to execute.
        [hint/] Use [emph]{tab_key}[/] to autocomplete command.
        [hint/] If you need help, press [emph]{help_key}[/].
        """

        return "[rich]" + cleandoc(info)

    def mk(self, provider):
        self.abs.mkdir()

    beatmaps = as_child("Beatmaps", BeatmapsDirPath)

    profiles = as_child("Profiles", ProfilesDirPath)

    devices = as_child("Devices", DevicesDirPath)

    @as_child("Resources")
    class resources(RecognizedDirPath):
        """The place to store some resources of KAIKO"""

        def mk(self, provider):
            self.abs.mkdir()

    @as_child("Cache")
    class cache(RecognizedDirPath):
        """The place to cache some data for better experience

        [rich][color=bright_white]┌──────┐[/]
        [color=bright_white]│☰☲☰☱  │[/]
        [color=bright_white]│☴☲☱☴☱ │[/] Cached data stored here will be used to improve user experience
        [color=bright_white]│☱☴☲   │[/] and debugging, deleting them will not break the system, so feel
        [color=bright_white]│⚌     │[/] free to manage them.
        [color=bright_white]└──────┘[/]
        """

        def mk(self, provider):
            self.abs.mkdir()

        @as_child("logs")
        class logs(RecognizedFilePath):
            "The printed messages will be recorded here"

        beatshell_history = as_child(".beatshell-history")(beatshell.BeatshellHistory)
        prompt_benchmark = as_child("prompt_benchmark.csv")(beatshell.PromptBenchmark)


class KAIKOLauncher:
    update_interval = 0.01
    version = __version__
    METRONOME_TEMPO = 120.0

    def __init__(self):
        self.provider = Provider()

    @classmethod
    def launch(cls):
        # print logo
        print(logo.format(f"v{cls.version}"), flush=True)
        # animated_print(logo_animated_text)
        print(logo_animated_text)
        print("\n\n\n", end="", flush=True)

        try:
            cls.init_and_run().exhaust(dt=cls.update_interval, interruptible=True)

        except KeyboardInterrupt:
            pass

        except:
            # print error
            print("\x1b[31m", end="")
            print(traceback.format_exc(), end="")
            print(f"\x1b[m", end="")

    @classmethod
    @dn.datanode
    def init_and_run(cls):
        r"""Initialize KAIKO and run."""
        launcher = cls()

        # logger
        logger = Logger()
        launcher.provider.set(logger)

        # load workspace
        file_manager = FileManager(RootDirPath, launcher.provider)
        launcher.provider.set(file_manager)

        log_file = file_manager.root.cache.logs.abs
        log_ok = log_file.parent.exists() and not log_file.is_dir()
        if log_ok:
            logger.set_log_file(log_file)

        file_manager.fix()

        if not log_ok:
            logger.set_log_file(log_file)

        file_manager.init_env()

        # load profiles
        profile_manager = ProfileManager(
            KAIKOSettings, file_manager.root.profiles, launcher.provider
        )
        launcher.provider.set(profile_manager)

        profile_manager.on_change(
            lambda settings: logger.recompile_style(
                terminal_settings=settings.devices.terminal,
                logger_settings=settings.logger,
            )
        )
        profile_manager.on_change(
            lambda settings: file_manager.set_settings(settings.files)
        )
        profile_manager.update()

        succ = profile_manager.use()
        if not succ:
            profile_manager.use_empty()

        logger.print(flush=True)

        # load devices
        device_manager = DeviceManager(
            launcher.provider,
            file_manager.root.cache,
            launcher.settings.devices,
        )
        devices_ctxt = yield from device_manager.initialize().join()

        with devices_ctxt as device_manager:
            launcher.provider.set(device_manager)

            beatmap_manager = BeatmapManager(
                file_manager.root.beatmaps, launcher.provider
            )
            launcher.provider.set(beatmap_manager)

            metronome = clocks.Metronome(launcher.METRONOME_TEMPO)
            launcher.provider.set(metronome)

            bgm_controller = BGMController(
                launcher.provider,
                launcher.settings.bgm,
                launcher.settings.devices.mixer,
            )
            launcher.provider.set(bgm_controller)

            prompt = beatshell.BeatPrompt(
                launcher.provider,
                file_manager.root.cache.beatshell_history,
                file_manager.root.cache.prompt_benchmark,
                launcher.settings.shell,
            )
            launcher.provider.set(prompt)

            yield from launcher.run().join()

    @dn.datanode
    def run(self):
        r"""Run KAIKO."""
        logger = self.logger

        yield

        # execute given command
        if len(sys.argv) > 2 and sys.argv[1] == "-c":
            command = self.get_command_parser().build_command(sys.argv[2:])
            yield from self.execute(command).join()
            return
        elif len(sys.argv) != 1:
            raise ValueError("unknown arguments: " + " ".join(sys.argv[1:]))

        # load bgm
        bgm_task = self.bgm_controller.start()

        # prompt
        repl_task = self.repl()
        yield from dn.pipe(repl_task, bgm_task).join()

    @dn.datanode
    def repl(self):
        r"""Start REPL."""
        clock = self.device_manager.clock
        prompt = self.prompt

        self.logger.print()
        prev_dir = None
        clear = True

        while True:
            self.logger.print()
            prompt.print_banner(prev_dir != self.file_manager.current)
            prev_dir = self.file_manager.current

            try:
                command = yield from prompt.new_session(self.get_command_parser(), clear).join()

            except beatshell.PromptError as e:
                self.logger.print(f"[warn]{self.logger.escape(str(e.cause))}[/]")
                clear = False

            else:
                prompt.record_command()
                yield from self.execute(command).join()
                clear = True

            prompt.set_settings(self.settings.shell)
            self.bgm_controller.set_settings(self.settings.bgm)
            self.bgm_controller.set_mixer_settings(self.settings.devices.mixer)

    @dn.datanode
    def execute(self, command):
        r"""Execute a command.

        If it returns executable object (an object has method `execute`), call
        `result.execute()`; if it returns a DataNode, exhaust it; otherwise,
        print repr of result.

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
                yield from result.execute().join()
                if is_bgm_on:
                    self.bgm_controller.play()

            elif isinstance(result, dn.DataNode):
                yield from result.join()

            elif result is not None:
                yield
                self.logger.print(self.logger.format_value(result))

        except Exception:
            self.logger.print_traceback()

    @property
    def profile_manager(self):
        return self.provider.get(ProfileManager)

    @property
    def file_manager(self):
        return self.provider.get(FileManager)

    @property
    def logger(self):
        return self.provider.get(Logger)

    @property
    def device_manager(self):
        return self.provider.get(DeviceManager)

    @property
    def beatmap_manager(self):
        return self.provider.get(BeatmapManager)

    @property
    def metronome(self):
        return self.provider.get(clocks.Metronome)

    @property
    def bgm_controller(self):
        return self.provider.get(BGMController)

    @property
    def prompt(self):
        return self.provider.get(beatshell.BeatPrompt)

    @property
    def settings(self):
        r"""Current settings."""
        return self.profile_manager.current

    def get_command_parser(self):
        commands = {}
        if isinstance(self.file_manager.current, RootDirPath.beatmaps):
            commands["play"] = PlayCommand(
                self.provider,
                self.file_manager.root.resources,
                self.file_manager.root.cache,
            )
        if isinstance(self.file_manager.current, RootDirPath.devices):
            commands["devices"] = DevicesCommand(self.provider)
        if isinstance(self.file_manager.current, RootDirPath.profiles):
            commands["profiles"] = ProfilesCommand(self.provider)
        commands["bgm"] = BGMCommand(self.provider)
        commands["files"] = FilesCommand(self.provider)
        commands["cd"] = DirectCdCommand(self.provider)
        return cmd.RootCommandParser(**commands)
