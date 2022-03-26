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
from ..devices import terminals as term
from ..devices import audios as aud
from ..devices import engines
from ..beats import beatshell
from ..beats import beatmaps
from ..beats import beatsheets
from .files import FileManager
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
  █▀   ▀██▄  ██    ██ ▀██████▄ ██   ▀██▄█▄▄▄▄▄▄█  {}


  🎧  Use headphones for the best experience 🎤 

"""


class KAIKOMenu:
    update_interval = 0.01
    version = __version__

    def __init__(self, profiles, workspace, manager, logger):
        r"""Constructor.

        Parameters
        ----------
        profiles : ProfileManager
        workspace : FileManager
        manager : PyAudio
        logger : loggers.Logger
        """
        self.profiles = profiles
        self.workspace = workspace
        self.manager = manager
        self.logger = logger
        self.beatmap_manager = BeatmapManager(workspace.beatmaps_dir, logger)
        self.bgm_controller = KAIKOBGMController(
            logger, self.beatmap_manager, lambda: self.profiles.current.devices.mixer
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
        workspace = FileManager.create()
        workspace.prepare(logger)

        # load profiles
        profiles = ProfileManager(workspace.profiles_dir, logger)
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

        if not sys.stdout.isatty():
            raise RuntimeError("please connect to interactive terminal device.")

        # deterimine unicode version
        if (
            profiles.current.devices.terminal.unicode_version == "auto"
            and "UNICODE_VERSION" not in os.environ
        ):
            version = yield from determine_unicode_version(logger).join()
            if version is not None:
                os.environ["UNICODE_VERSION"] = version
                profiles.current.devices.terminal.unicode_version = version
                profiles.set_as_changed()
            logger.print()

        # fit screen size
        size = shutil.get_terminal_size()
        width = profiles.current.devices.terminal.best_screen_size
        if size.columns < width:
            logger.print("[hint/] Your screen size seems too small.")

            yield from fit_screen(logger, profiles.current.devices.terminal).join()

        # load PyAudio
        logger.print("[info/] Load PyAudio...")
        logger.print()

        with prepare_pyaudio(logger) as manager:
            menu = cls(profiles, workspace, manager, logger)
            yield from menu.run().join()

    @dn.datanode
    def run(self):
        r"""Run KAIKOMenu."""
        logger = self.logger

        yield

        # load beatmaps
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
            self.logger.rich,
            self.workspace.cache_dir,
            lambda: self.profiles.current.shell,
            lambda: self.profiles.current.devices,
        )
        while True:
            self.print_banner()

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

    def print_banner(self):
        username = self.logger.escape(self.workspace.username)
        profile = self.logger.escape(self.profiles.current_name)
        path = str(self.workspace.current)
        if path == ".":
            path = ""
        path = self.logger.escape(path)

        banner = self.settings.shell.prompt.banner
        self.logger.print(banner.format(username=username, profile=profile, path=path))

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
            self.workspace,
            self.workspace.beatmaps_dir / beatmap,
            start,
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
            pattern, tempo, offset, self.workspace, self.profiles, self.logger,
        )

    @loop.arg_parser("pattern")
    def _loop_pattern_parser(self):
        return cmd.RawParser(
            desc="It should be a pattern.", default="x x o x | x [x x] o _"
        )

    @play.arg_parser("beatmap")
    def _play_beatmap_parser(self):
        return self.beatmap_manager.make_parser()

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
        return self.beatmap_manager.make_parser()

    @cmd.function_command
    def list(self):
        """[rich]List your beatmaps.

        usage: [cmd]list[/]
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
    def cd(self, path):
        self.workspace.cd(path, self.logger)

    @cmd.function_command
    def ls(self):
        self.workspace.ls(self.logger)

    @cmd.function_command
    def cat(self, path):
        abspath = self.workspace.get(path, self.logger)

        try:
            content = abspath.read_text()
        except UnicodeDecodeError:
            self.logger.print("[warn]Cannot read binary file.[/]")
            return

        code = self.logger.format_code(
            content, title=str(self.workspace.current / path)
        )
        self.logger.print(code)

    @cd.arg_parser("path")
    def _cd_path_parser(self):
        return cmd.PathParser(self.workspace.root / self.workspace.current, type="dir")

    @cat.arg_parser("path")
    def _cat_path_parser(self):
        return cmd.PathParser(self.workspace.root / self.workspace.current, type="file")

    @cmd.function_command
    def me(self):
        """[rich]About user.

        usage: [cmd]me[/]
        """
        logger = self.logger

        logger.print(f"username: {logger.emph(self.workspace.username)}")
        logger.print(f"root directory: {logger.emph(self.workspace.root.as_uri())}")
        logger.print(
            f"profiles directory: {logger.emph(self.workspace.profiles_dir.as_uri())}"
        )
        logger.print(
            f"beatmaps directory: {logger.emph(self.workspace.beatmaps_dir.as_uri())}"
        )
        logger.print(
            f"resources directory: {logger.emph(self.workspace.resources_dir.as_uri())}"
        )
        logger.print(
            f"cache directory: {logger.emph(self.workspace.cache_dir.as_uri())}"
        )

    @cmd.function_command
    def gen(self, waveform):
        """[rich]Generate sound.

        usage: [cmd]gen[/] [arg]{waveform}[/]
                      ╱
               The function of
               output waveform.
        """
        settings = self.profiles.current.devices.mixer
        return WaveformTest(waveform, self.logger, settings)

    @gen.arg_parser("waveform")
    def _gen_waveform_parser(self):
        return cmd.RawParser(desc="It should be an expression of waveform.")

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
    def clean(self, bottom: bool = False):
        """[rich]Clean screen.

        usage: [cmd]clean[/] [[[kw]--bottom[/] [arg]{BOTTOM}[/]]]
                                   ╲
                          bool, move to bottom or
                           not; default is False.
        """
        self.logger.clear(bottom)

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
            self.workspace.remove(logger)
            logger.print("Good luck~")
            raise KeyboardInterrupt


class WaveformTest:
    def __init__(self, waveform, logger, mixer_settings):
        self.waveform = waveform
        self.logger = logger
        self.mixer_settings = mixer_settings

    def execute(self, manager):
        self.logger.print("[info/] Compile waveform...")

        try:
            node = dn.Waveform(self.waveform).generate(
                self.mixer_settings.output_samplerate,
                self.mixer_settings.output_channels,
                self.mixer_settings.output_buffer_length,
            )

        except:
            self.logger.print("[warn]Fail to compile waveform.[/]")
            with self.logger.warn():
                self.logger.print(traceback.format_exc(), end="", markup=False)
            return dn.DataNode.wrap([])

        self.logger.print("[hint/] Press any key to end test.")
        mixer_task, mixer = engines.Mixer.create(self.mixer_settings, manager)
        mixer.play(node)

        @dn.datanode
        def exit_any():
            keycode = None
            while keycode is None:
                _, keycode = yield

        exit_task = term.inkey(exit_any())

        return dn.pipe(mixer_task, exit_task)


class KAIKOPlay:
    def __init__(self, workspace, filepath, start, profiles, logger):
        self.workspace = workspace
        self.filepath = filepath
        self.start = start
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
            beatmap.print_hints(logger, gameplay_settings)
            logger.print()

            score, devices_settings = yield from beatmap.play(
                manager,
                self.workspace.resources_dir,
                self.workspace.cache_dir,
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
                    title = self.profiles.get_title()
                    old = self.profiles.format()
                    self.profiles.current.devices = devices_settings
                    self.profiles.set_as_changed()
                    new = self.profiles.format()

                    self.logger.print(f"[data/] Your changes")
                    logger.print(
                        logger.format_code_diff(old, new, title=title, is_changed=True)
                    )


class KAIKOLoop:
    def __init__(self, pattern, tempo, offset, workspace, profiles, logger):
        self.pattern = pattern
        self.tempo = tempo
        self.offset = offset
        self.workspace = workspace
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

            beatmap.print_hints(logger, gameplay_settings)
            logger.print()

            score, devices_settings = yield from beatmap.play(
                manager,
                self.workspace.resources_dir,
                self.workspace.cache_dir,
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
                    title = self.profiles.get_title()
                    old = self.profiles.format()
                    self.profiles.current.devices = devices_settings
                    self.profiles.set_as_changed()
                    new = self.profiles.format()

                    self.logger.print(f"[data/] Your changes")
                    logger.print(
                        logger.format_code_diff(old, new, title=title, is_changed=True)
                    )
