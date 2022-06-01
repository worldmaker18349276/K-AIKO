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
from .files import FileManager, FilesCommand, CdCommand, FileDescriptor, DirDescriptor, WildCardDescriptor, as_child
from .settings import KAIKOSettings
from .loggers import Logger
from .profiles import ProfileManager, ProfilesCommand, ProfilesDirDescriptor
from .play import BeatmapManager, BeatmapsDirDescriptor, PlayCommand
from .bgm import BGMController, BGMCommand
from .devices import (
    DeviceManager,
    DevicesCommand,
    DevicesDirDescriptor,
)


logo = """

  ██▀ ▄██▀   ▄██████▄ ▀██████▄ ██  ▄██▀ █▀▀▀▀▀▀█
  ▀ ▄██▀  ▄▄▄▀█    ██    ██    ██▄██▀   █ ▓▓▓▓ █
  ▄██▀██▄ ▀▀▀▄███████    ██    ███▀██▄  █ ▓▓▓▓ █
  █▀   ▀██▄  ██    ██ ▀██████▄ ██   ▀██▄█▄▄▄▄▄▄█  {}


  🎧  Use headphones for the best experience 🎤 

"""


class RootDirDescriptor(DirDescriptor):
    "(The workspace of KAIKO)"

    beatmaps_name = "Beatmaps"
    profiles_name = "Profiles"
    devices_name = "Devices"
    resources_name = "Resources"
    cache_name = "Cache"

    Beatmaps = as_child(beatmaps_name, is_required=True)(BeatmapsDirDescriptor)

    Profiles = as_child(profiles_name, is_required=True)(ProfilesDirDescriptor)

    Devices = as_child(devices_name, is_required=True)(DevicesDirDescriptor)

    @as_child(resources_name, is_required=True)
    class Resources(DirDescriptor):
        "(The place to store some resources of KAIKO)"

        @as_child("**")
        class Resource(WildCardDescriptor):
            "(Resource file)"

    @as_child(cache_name, is_required=True)
    class Cache(DirDescriptor):
        "(The place to cache some data for better exprience)"

        @as_child(".beatshell-history", is_required=True)
        class BeatShellHistory(FileDescriptor):
            "(The command history)"

        @as_child("**")
        class CacheData(WildCardDescriptor):
            "(Cache data)"


class KAIKOMenu:
    update_interval = 0.01
    version = __version__

    def __init__(self):
        self.provider = Provider()

    @classmethod
    def main(cls):
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
        r"""Initialize KAIKOMenu and run."""
        menu = cls()

        # logger
        logger = Logger()
        menu.provider.set(logger)

        # load workspace
        file_manager = FileManager.create(RootDirDescriptor(menu.provider))
        file_manager.prepare(logger)
        menu.provider.set(file_manager)

        os.environ["KAIKO"] = str(file_manager.root)

        # load profiles
        profile_manager = ProfileManager(KAIKOSettings, menu.profiles_dir)
        menu.provider.set(profile_manager)

        profile_manager.on_change(
            lambda settings: logger.recompile_style(
                terminal_settings=settings.devices.terminal,
                logger_settings=settings.logger,
            )
        )
        profile_manager.on_change(
            lambda settings: file_manager.set_settings(settings.files)
        )
        profile_manager.update(logger)

        succ = profile_manager.use(logger)
        if not succ:
            succ = profile_manager.new(logger)
            if not succ:
                raise RuntimeError("Fail to load profile")

        logger.print(flush=True)

        # load devices
        devices_ctxt = yield from DeviceManager.initialize(logger, profile_manager).join()

        with devices_ctxt as device_manager:
            menu.provider.set(device_manager)

            beatmap_manager = BeatmapManager(menu.beatmaps_dir)
            menu.provider.set(beatmap_manager)

            bgm_controller = BGMController(
                beatmap_manager, lambda: menu.settings.devices.mixer
            )
            menu.provider.set(bgm_controller)

            yield from menu.run().join()

    @dn.datanode
    def run(self):
        r"""Run KAIKOMenu."""
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
            self.cache_dir,
            self.get_command_parser,
            lambda: self.settings.shell,
            preview_handler,
        )

        self.print_tips()
        while True:
            self.print_banner()

            try:
                command = yield from prompt.prompt(self.settings.devices).join()

            except beatshell.PromptError as e:
                self.logger.print(f"[warn]{self.logger.escape(str(e.cause))}[/]")
                prompt.new_session()

            else:
                prompt._record_command()
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

    @property
    def cache_dir(self):
        return self.file_manager.root / self.file_manager.structure.cache_name

    @property
    def profiles_dir(self):
        return self.file_manager.root / self.file_manager.structure.profiles_name

    @property
    def beatmaps_dir(self):
        return self.file_manager.root / self.file_manager.structure.beatmaps_name

    @property
    def resources_dir(self):
        return self.file_manager.root / self.file_manager.structure.resources_name

    @property
    def devices_dir(self):
        return self.file_manager.root / self.file_manager.structure.devices_name

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

    def print_banner(self):
        logger = self.logger

        username = self.file_manager.username
        current_name = self.profile_manager.current_name
        path = str(self.file_manager.current)
        if path == ".":
            path = ""
        path = os.path.join("$KAIKO", path)
        profile_is_changed = self.profile_manager.is_changed()
        path_is_known = self.file_manager.get_desc(self.file_manager.root / self.file_manager.current) is not None

        banner_settings = self.settings.shell.banner

        username = logger.escape(username, type="all")
        profile = logger.escape(current_name, type="all")
        path = logger.escape(path, type="all")

        user_markup = banner_settings.user
        user_markup = logger.rich.parse(user_markup, slotted=True)
        user_markup = mu.replace_slot(
            user_markup,
            user_name=logger.rich.parse(username),
        )

        profile_markup = banner_settings.profile
        profile_markup = profile_markup[0] if not profile_is_changed else profile_markup[1]
        profile_markup = logger.rich.parse(profile_markup, slotted=True)
        profile_markup = mu.replace_slot(
            profile_markup,
            profile_name=logger.rich.parse(profile),
        )

        path_markup = banner_settings.path
        path_markup = path_markup[0] if path_is_known else path_markup[1]
        path_markup = logger.rich.parse(path_markup, slotted=True)
        path_markup = mu.replace_slot(
            path_markup,
            current_path=logger.rich.parse(path),
        )

        banner_markup = banner_settings.banner
        banner_markup = logger.rich.parse(banner_markup, slotted=True)
        banner_markup = mu.replace_slot(
            banner_markup,
            user=user_markup,
            profile=profile_markup,
            path=path_markup,
        )

        logger.print()
        logger.print(banner_markup)

    def get_command_parser(self):
        commands = {}
        if self.file_manager.current == Path("Beatmaps/"):
            commands["play"] = PlayCommand(self.provider, self.resources_dir, self.cache_dir)
        if self.file_manager.current == Path("Devices/"):
            commands["devices"] = DevicesCommand(self.provider)
        if self.file_manager.current == Path("Profiles/"):
            commands["profiles"] = ProfilesCommand(self.provider)
        commands["bgm"] = BGMCommand(self.provider)
        commands["files"] = FilesCommand(self.provider, self.profile_manager.is_changed())
        commands["cd"] = CdCommand(self.provider)
        return cmd.RootCommandParser(**commands)

