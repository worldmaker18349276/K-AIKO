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
from ..utils import providers
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

    def info_detailed(self):
        logger = providers.get(Logger)
        profile_manager = providers.get(ProfileManager)

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

    def mk(self):
        self.abs.mkdir()

    beatmaps = as_child("Beatmaps", BeatmapsDirPath)

    profiles = as_child("Profiles", ProfilesDirPath)

    devices = as_child("Devices", DevicesDirPath)

    @as_child("Resources")
    class resources(RecognizedDirPath):
        """The place to store some resources of KAIKO"""

        def mk(self):
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

        def mk(self):
            self.abs.mkdir()

        @as_child("logs")
        class logs(RecognizedFilePath):
            "The printed messages will be recorded here"

        beatshell_history = as_child(".beatshell-history")(beatshell.BeatshellHistory)
        prompt_perf = as_child("prompt_perf.csv")(beatshell.PromptPerf)


class KAIKOLauncher:
    update_interval = 0.01
    version = __version__
    METRONOME_TEMPO = 120.0

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
        providers.set_static(logger)

        # load workspace
        file_manager = FileManager(RootDirPath)
        providers.set_static(file_manager)

        file_manager.fix()
        logger.set_log_file(file_manager.root.cache.logs.abs)

        file_manager.init_env()

        # load profiles
        profile_manager = ProfileManager(KAIKOSettings, file_manager.root.profiles)
        providers.set_static(profile_manager)

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
            file_manager.root.cache,
            file_manager.root.resources,
            launcher.settings.devices,
        )
        devices_ctxt = yield from device_manager.initialize().join()

        with devices_ctxt as device_manager:
            providers.set_static(device_manager)

            beatmap_manager = BeatmapManager(file_manager.root.beatmaps)
            providers.set_static(beatmap_manager)

            metronome = clocks.Metronome(launcher.METRONOME_TEMPO)
            providers.set_static(metronome)

            bgm_controller = BGMController(
                launcher.settings.bgm,
                launcher.settings.devices.mixer,
            )
            providers.set_static(bgm_controller)

            prompt = beatshell.BeatPrompt(
                file_manager.root.cache.beatshell_history,
                file_manager.root.cache.prompt_perf,
                launcher.settings.shell,
            )

            yield from launcher.run(prompt).join()

    @dn.datanode
    def run(self, prompt):
        r"""Run KAIKO."""
        logger = providers.get(Logger)
        bgm_controller = providers.get(BGMController)

        yield

        # execute given command
        if len(sys.argv) > 2 and sys.argv[1] == "-c":
            command = self.get_command_parser().build_command(sys.argv[2:])
            yield from self.execute(command).join()
            return
        elif len(sys.argv) != 1:
            raise ValueError("unknown arguments: " + " ".join(sys.argv[1:]))

        # load bgm
        bgm_task = bgm_controller.start()

        # prompt
        repl_task = self.repl(prompt)
        yield from dn.pipe(repl_task, bgm_task).join()

    @dn.datanode
    def repl(self, prompt):
        r"""Start REPL."""
        logger = providers.get(Logger)
        file_manager = providers.get(FileManager)
        device_manager = providers.get(DeviceManager)
        bgm_controller = providers.get(BGMController)

        clock = device_manager.clock

        logger.print()
        prev_dir = None
        clear = True

        while True:
            logger.print()
            prompt.print_banner(prev_dir != file_manager.current)
            prev_dir = file_manager.current

            try:
                command = yield from prompt.new_session(
                    self.get_command_parser(), clear
                ).join()

            except beatshell.PromptError as e:
                logger.print(f"[warn]{logger.escape(str(e.cause))}[/]")
                clear = False

            else:
                prompt.record_command()
                yield from self.execute(command).join()
                clear = True

            prompt.set_settings(self.settings.shell)
            bgm_controller.set_settings(self.settings.bgm)
            bgm_controller.set_mixer_settings(self.settings.devices.mixer)
            device_manager.set_settings(self.settings.devices)

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
        logger = providers.get(Logger)
        bgm_controller = providers.get(BGMController)

        try:
            result = command()

            if hasattr(result, "execute"):
                is_bgm_on = bgm_controller.is_bgm_on
                bgm_controller.stop()
                yield from result.execute().join()
                if is_bgm_on:
                    bgm_controller.play()

            elif isinstance(result, dn.DataNode):
                yield from result.join()

            elif result is not None:
                yield
                logger.print(logger.format_value(result))

        except Exception:
            logger.print_traceback()

    @property
    def settings(self):
        r"""Current settings."""
        return providers.get(ProfileManager).current

    def get_command_parser(self):
        file_manager = providers.get(FileManager)

        commands = {}
        if isinstance(file_manager.current, RootDirPath.beatmaps):
            commands["play"] = PlayCommand(
                file_manager.root.resources,
                file_manager.root.cache,
            )
        if isinstance(file_manager.current, RootDirPath.devices):
            commands["devices"] = DevicesCommand()
        if isinstance(file_manager.current, RootDirPath.profiles):
            commands["profiles"] = ProfilesCommand()
        commands["bgm"] = BGMCommand()
        commands["files"] = FilesCommand()
        commands["cd"] = DirectCdCommand()
        return cmd.RootCommandParser(**commands)
