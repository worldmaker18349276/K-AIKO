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
from ..beats import beatshell
from .files import (
    FileManager,
    FilesCommand,
    CdCommand,
    RecognizedDirPath,
    RecognizedFilePath,
    RecognizedWildCardPath,
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
    "(The workspace of KAIKO)"

    def mk(self, provider):
        self.abs.mkdir()

    beatmaps = as_child("Beatmaps", BeatmapsDirPath)

    profiles = as_child("Profiles", ProfilesDirPath)

    devices = as_child("Devices", DevicesDirPath)

    @as_child("Resources")
    class resources(RecognizedDirPath):
        "(The place to store some resources of KAIKO)"

        def mk(self, provider):
            self.abs.mkdir()

    @as_child("Cache")
    class cache(RecognizedDirPath):
        "(The place to cache some data for better exprience)"

        def mk(self, provider):
            self.abs.mkdir()

        @as_child(".beatshell-history")
        class beatshell_history(RecognizedFilePath):
            "(The command history)"

            def mk(self, provider):
                self.abs.touch()

            def rm(self, provider):
                self.abs.unlink()
                self.abs.touch()


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
        file_manager.fix()
        launcher.provider.set(file_manager)

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

        profile_manager.use()

        logger.print(flush=True)

        # load devices
        devices_ctxt = yield from DeviceManager.initialize(logger, profile_manager).join()

        with devices_ctxt as device_manager:
            launcher.provider.set(device_manager)

            beatmap_manager = BeatmapManager(file_manager.root.beatmaps.abs)
            launcher.provider.set(beatmap_manager)

            bgm_controller = BGMController(
                beatmap_manager, lambda: launcher.settings.devices.mixer
            )
            launcher.provider.set(bgm_controller)

            yield from launcher.run().join()

    @dn.datanode
    def run(self):
        r"""Run KAIKO."""
        logger = self.logger

        yield

        # load beatmaps
        self.beatmap_manager.reload(logger)

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
            self.logger.rich,
            self.file_manager.root.cache.abs,
            self.get_command_parser,
            lambda: self.settings.shell,
            preview_handler,
        )

        self.print_tips()
        while True:
            self.logger.print()
            self.logger.print(prompt.make_banner(self.file_manager, self.profile_manager))

            try:
                command = yield from prompt.prompt(self.settings.devices).join()

            except beatshell.PromptError as e:
                self.logger.print(f"[warn]{self.logger.escape(str(e.cause))}[/]")
                prompt.new_session(clear=False)

            else:
                prompt.record_command()
                yield from self.execute(command).join()
                prompt.new_session()

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
            with self.logger.warn():
                self.logger.print(traceback.format_exc(), end="", markup=False)

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

    def print_tips(self):
        logger = self.logger

        input_settings = self.settings.shell.input

        confirm_key = logger.emph(input_settings.confirm_key, type="all")
        help_key = logger.emph(input_settings.help_key, type="all")
        tab_key = logger.emph(input_settings.autocomplete_keys[0], type="all")

        logger.print(f"[hint/] Type command and press {confirm_key} to execute.")
        logger.print(f"[hint/] Use {tab_key} to autocomplete command.")
        logger.print(f"[hint/] If you need help, press {help_key}.")
        logger.print()

    def get_command_parser(self):
        commands = {}
        if isinstance(self.file_manager.current, RootDirPath.beatmaps):
            commands["play"] = PlayCommand(
                self.provider,
                self.file_manager.root.resources.abs,
                self.file_manager.root.cache.abs,
            )
        if isinstance(self.file_manager.current, RootDirPath.devices):
            commands["devices"] = DevicesCommand(self.provider)
        if isinstance(self.file_manager.current, RootDirPath.profiles):
            commands["profiles"] = ProfilesCommand(self.provider)
        commands["bgm"] = BGMCommand(self.provider)
        commands["files"] = FilesCommand(self.provider)
        commands["cd"] = CdCommand(self.provider)
        return cmd.RootCommandParser(**commands)

