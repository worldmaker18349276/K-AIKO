import sys
import os
import traceback
import shutil
import pkgutil
from pathlib import Path
from .. import __version__
from ..utils.providers import Provider
from ..utils import markups as mu
from ..utils import datanodes as dn
from ..utils import commands as cmd
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


  🎧  Use headphones for the best experience 🎤 

"""


class RootDirPath(RecognizedDirPath):
    """The workspace of KAIKO

    [rich]⠟⣡⡾⠋⣀⡰⡟⠛⢻⡆⠙⢻⡟⠓⢸⣇⣴⠟⠁⡏⣭⣭⢹
    ⡾⠋⠻⣦⡉⢱⡟⠛⢻⡇⢤⣼⣧⣄⢸⡟⠙⢷⣄⣇⣛⣛⣸

    All data for this game will be stored here, like beatmaps, settings and
    cache data.  To manage your data, use the commands [cmd]rm[/], [cmd]mk[/], [cmd]cp[/] and [cmd]mv[/]
    like bash.  Use the command [cmd]cd[/] or type folder name directly to change
    the current directory.
    """

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

    def __init__(self):
        self.provider = Provider()

    @classmethod
    def launch(cls):
        # print logo
        print(logo.format(f"v{cls.version}"), flush=True)

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
        logger.set_log_file(file_manager.root.cache.logs.abs)

        launcher.provider.set(file_manager)
        file_manager.fix()

        os.environ[file_manager.ROOT_ENVVAR] = str(file_manager.root)

        # load profiles
        profile_manager = ProfileManager(KAIKOSettings, file_manager.root.profiles, launcher.provider)
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
        devices_ctxt = yield from DeviceManager(launcher.provider).initialize().join()

        with devices_ctxt as device_manager:
            launcher.provider.set(device_manager)

            beatmap_manager = BeatmapManager(file_manager.root.beatmaps, launcher.provider)
            launcher.provider.set(beatmap_manager)

            bgm_controller = BGMController(
                launcher.provider, launcher.settings.bgm, launcher.settings.devices.mixer
            )
            launcher.provider.set(bgm_controller)

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
        bgm_task = self.bgm_controller.execute(self.device_manager.audio_manager)

        # prompt
        repl_task = self.repl()
        yield from dn.pipe(repl_task, bgm_task).join()

    @dn.datanode
    def repl(self):
        r"""Start REPL."""
        preview_handler = self.bgm_controller.preview_handler
        prompt = beatshell.BeatPrompt(
            self.logger,
            self.file_manager.root.cache.beatshell_history,
            self.file_manager.root.cache.prompt_benchmark,
            self.get_command_parser(),
            self.settings.shell,
            preview_handler,
        )

        prompt.print_tips()
        self.logger.print()

        while True:
            self.logger.print()
            prompt.print_banner(self.provider)

            try:
                command = yield from prompt.prompt(self.settings.devices).join()

            except beatshell.PromptError as e:
                self.logger.print(f"[warn]{self.logger.escape(str(e.cause))}[/]")
                prompt.new_session(self.get_command_parser(), clear=False)

            else:
                prompt.record_command()
                yield from self.execute(command).join()
                prompt.new_session(self.get_command_parser())

            prompt.set_settings(self.settings.shell)
            self.bgm_controller.set_settings(self.settings.bgm)
            self.bgm_controller.set_mixer_settings(self.settings.devices.mixer)

    @dn.datanode
    def execute(self, command):
        r"""Execute a command.

        If it returns executable object (an object has method `execute`), call
        `result.execute(audio_manager)`; if it returns a DataNode, exhaust it;
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
                yield from result.execute(self.device_manager.audio_manager).join()
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
    def bgm_controller(self):
        return self.provider.get(BGMController)

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

