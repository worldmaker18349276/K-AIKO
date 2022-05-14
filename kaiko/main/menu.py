import sys
import os
import traceback
import shutil
import pkgutil
from pathlib import Path
from .. import __version__
from ..utils import markups as mu
from ..utils import datanodes as dn
from ..utils import commands as cmd
from ..devices import loggers as log
from ..beats import beatshell
from .files import FileManager, FilesCommand, CdCommand
from .settings import KAIKOSettings
from .profiles import ProfileManager, ProfilesCommand
from .songs import BeatmapManager, KAIKOBGMController, BGMCommand
from .play import PlayCommand
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
  █▀   ▀██▄  ██    ██ ▀██████▄ ██   ▀██▄█▄▄▄▄▄▄█  {}


  🎧  Use headphones for the best experience 🎤 

"""


class KAIKOMenu:
    update_interval = 0.01
    version = __version__

    structure = {
        ".": "(The workspace of KAIKO)",
        "Beatmaps": {
            ".": "(The place to hold your beatmaps)",
            "*": {
                ".": "(Beatmapset of a song)",
                "*.kaiko": "(Beatmap file in kaiko format)",
                "*.ka": "(Beatmap file in kaiko format)",
                "*.osu": "(Beatmap file in osu format)",
                "**": "(Inner file of this beatmapset)",
            },
            "*.osz": "(Compressed beatmapset file)",
        },
        "Profiles": {
            ".": "(The place to manage your profiles)",
            "*.kaiko-profile": "(Your custom profile)",
            ".default-profile": "(The file of default profile name)",
        },
        "Resources": {
            ".": "(The place to store some resources of KAIKO)",
            "**": "(Resource file)",
        },
        "Devices": {
            ".": "(The place to manage your devices)",
        },
        "Cache": {
            ".": "(The place to cache some data for better exprience)",
            ".beatshell-history": "(The command history)",
            "**": "(Cache data)",
        },
    }

    initializer = {
        ".": None,
        "Beatmaps": {
            ".": None,
        },
        "Profiles": {
            ".": None,
        },
        "Resources": {
            ".": None,
        },
        "Devices": {
            ".": None,
        },
        "Cache": {
            ".": None,
            ".beatshell-history": None,
        },
    }

    def __init__(self, profiles_manager, file_manager, manager, logger):
        r"""Constructor.

        Parameters
        ----------
        profiles_manager : ProfileManager
        file_manager : FileManager
        manager : PyAudio
        logger : loggers.Logger
        """
        self.profiles_manager = profiles_manager
        self.file_manager = file_manager
        self.manager = manager
        self.logger = logger
        self.beatmap_manager = BeatmapManager(self.beatmaps_dir, logger)
        self.bgm_controller = KAIKOBGMController(
            logger, self.beatmap_manager, lambda: self.profiles_manager.current.devices.mixer
        )

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
        logger = log.Logger()

        # load workspace
        file_manager = FileManager.create(cls.structure, cls.initializer)
        file_manager.prepare(logger)

        os.environ["KAIKO"] = str(file_manager.root)

        # load profiles
        profiles_dir = file_manager.root / "Profiles"
        profiles_manager = ProfileManager(KAIKOSettings, profiles_dir, logger)
        profiles_manager.on_change(
            lambda settings: logger.recompile_style(
                terminal_settings=settings.devices.terminal,
                logger_settings=settings.devices.logger,
            )
        )
        profiles_manager.update()

        succ = profiles_manager.use()
        if not succ:
            succ = profiles_manager.new()
            if not succ:
                raise RuntimeError("Fail to load profile")

        if not sys.stdout.isatty():
            raise RuntimeError("please connect to interactive terminal device.")

        # deterimine unicode version
        if (
            profiles_manager.current.devices.terminal.unicode_version == "auto"
            and "UNICODE_VERSION" not in os.environ
        ):
            version = yield from determine_unicode_version(logger).join()
            if version is not None:
                os.environ["UNICODE_VERSION"] = version
                profiles_manager.current.devices.terminal.unicode_version = version
                profiles_manager.set_as_changed()
            logger.print()

        # fit screen size
        size = shutil.get_terminal_size()
        width = profiles_manager.current.devices.terminal.best_screen_size
        if size.columns < width:
            logger.print("[hint/] Your screen size seems too small.")

            yield from fit_screen(logger, profiles_manager.current.devices.terminal).join()

        # load PyAudio
        logger.print("[info/] Load PyAudio...")
        logger.print()

        with prepare_pyaudio(logger) as manager:
            menu = cls(profiles_manager, file_manager, manager, logger)
            yield from menu.run().join()

    @dn.datanode
    def run(self):
        r"""Run KAIKOMenu."""
        logger = self.logger

        yield

        # load beatmaps
        self.beatmap_manager.reload()

        # execute given command
        if len(sys.argv) > 2 and sys.argv[1] == "-c":
            command = self.get_command_parser().build_command(sys.argv[2:])
            yield from self.execute(command).join()
            return
        elif len(sys.argv) != 1:
            raise ValueError("unknown arguments: " + " ".join(sys.argv[1:]))

        # load bgm
        bgm_task = self.bgm_controller.execute(self.manager)

        # tips
        confirm_key = logger.emph(self.settings.shell.input.confirm_key, type="all")
        help_key = logger.emph(self.settings.shell.input.help_key, type="all")
        tab_key = logger.emph(self.settings.shell.input.autocomplete_keys[0], type="all")
        logger.print(
            f"[hint/] Type command and press {confirm_key} to execute."
        )
        logger.print(f"[hint/] Use {tab_key} to autocomplete command.")
        logger.print(f"[hint/] If you need help, press {help_key}.")
        logger.print()

        # prompt
        repl_task = self.repl()
        yield from dn.pipe(repl_task, bgm_task).join()

    @dn.datanode
    def repl(self):
        r"""Start REPL."""
        preview_handler = self.bgm_controller.preview_handler
        input = beatshell.BeatInput(
            self.get_command_parser,
            preview_handler,
            self.logger.rich,
            self.cache_dir,
            lambda: self.profiles_manager.current.shell,
            lambda: self.profiles_manager.current.devices,
        )
        while True:
            self.print_banner()

            # parse command
            yield from input.prompt().join()

            # execute result
            result = input.result
            if isinstance(result, beatshell.ErrorResult):
                self.logger.print(f"[warn]{self.logger.escape(str(result.error))}[/]")
                input.prev_session()

            elif isinstance(result, beatshell.CompleteResult):
                input.record_command()
                yield from self.execute(result.command).join()
                input.new_session()

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
        return self.profiles_manager.current

    @property
    def cache_dir(self):
        return self.file_manager.root / "Cache"

    @property
    def profiles_dir(self):
        return self.file_manager.root / "Profiles"

    @property
    def beatmaps_dir(self):
        return self.file_manager.root / "Beatmaps"

    @property
    def resources_dir(self):
        return self.file_manager.root / "Resources"

    @property
    def devices_dir(self):
        return self.file_manager.root / "Devices"

    def print_banner(self):
        username = self.logger.escape(self.file_manager.username, type="all")
        profile = self.logger.escape(self.profiles_manager.current_name, type="all")
        path = str(self.file_manager.current)
        if path == ".":
            path = ""
        path = os.path.join("$KAIKO", path)
        path = self.logger.escape(path, type="all")

        profile_is_changed = self.profiles_manager.is_changed()
        path_is_known = self.file_manager.get_desc(self.file_manager.root / self.file_manager.current) is not None

        user_markup = self.settings.shell.banner.user
        user_markup = self.logger.rich.parse(user_markup, slotted=True)
        user_markup = mu.replace_slot(
            user_markup,
            user_name=self.logger.rich.parse(username),
        )

        profile_markup = self.settings.shell.banner.profile
        profile_markup = profile_markup[0] if not profile_is_changed else profile_markup[1]
        profile_markup = self.logger.rich.parse(profile_markup, slotted=True)
        profile_markup = mu.replace_slot(
            profile_markup,
            profile_name=self.logger.rich.parse(profile),
        )

        path_markup = self.settings.shell.banner.path
        path_markup = path_markup[0] if path_is_known else path_markup[1]
        path_markup = self.logger.rich.parse(path_markup, slotted=True)
        path_markup = mu.replace_slot(
            path_markup,
            current_path=self.logger.rich.parse(path),
        )

        banner_markup = self.settings.shell.banner.banner
        banner_markup = self.logger.rich.parse(banner_markup, slotted=True)
        banner_markup = mu.replace_slot(
            banner_markup,
            user=user_markup,
            profile=profile_markup,
            path=path_markup,
        )

        self.logger.print()
        self.logger.print(banner_markup)

    def get_command_parser(self):
        commands = {}
        if self.file_manager.current == Path("Beatmaps/"):
            commands["play"] = PlayCommand(self)
        if self.file_manager.current == Path("Devices/"):
            commands["devices"] = DevicesCommand(self.profiles_manager, self.logger, self.manager)
        if self.file_manager.current == Path("Profiles/"):
            commands["profiles"] = ProfilesCommand(self.profiles_manager, self.logger)
        commands["bgm"] = BGMCommand(self.bgm_controller, self.beatmap_manager, self.logger)
        commands["files"] = FilesCommand(self)
        commands["cd"] = CdCommand(self)
        return cmd.RootCommandParser(**commands)

