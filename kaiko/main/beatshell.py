import threading
from typing import List, Tuple, Union
import dataclasses
from ..utils.providers import Provider
from ..utils import datanodes as dn
from ..utils import config as cfg
from ..utils import markups as mu
from ..devices import clocks
from ..devices import engines
from ..tui import widgets
from ..tui import inputs
from .files import FileManager, UnrecognizedPath, RecognizedFilePath
from .profiles import ProfileManager
from .loggers import Logger
from .bgm import BGMController


# widgets
BLOCKER = [
    "⠶⠦⣚⠀⠶",
    "⢎⣀⡛⠀⠶",
    "⢖⣄⠻⠀⠶",
    "⠖⠐⡩⠂⠶",
    "⠶⠀⡭⠲⠶",
    "⠶⠀⣬⠉⡱",
    "⠶⠀⣦⠙⠵",
    "⠶⠠⣊⠄⠴",
]

TUMBLER = [
    "⠀⢠⢦⠕",
    "⠀⢰⢂⡔",
    "⠀⠴⣂⡤",
    "⠀⢖⣂⣤",
    "⠀⢞⣀⢠",
    "⠀⢟⡀⡄",
    "⠐⢍⢤⠀",
    "⠐⢥⢦⠀",
    "⠐⣄⢲⠀",
    "⠠⣄⡲⠄",
    "⢠⣄⣒⠆",
    "⢠⢀⣘⠆",
    "⠀⡄⣘⠇",
    "⠀⢠⢌⠕",
]

TIMCOEP8 = [
    "⠀⠀⡔⠀⡱",
    "⠀⢀⡤⠀⠶",
    "⠀⢠⠤⠀⠶",
    "⠀⡠⢆⠀⠶",
    "⢀⡰⡦⠀⠶",
    "⠀⡐⢰⠀⠶",
    "⠀⢀⢢⠄⠶",
    "⠀⠀⢰⡴⠶",
]


@dataclasses.dataclass
class PatternsWidgetSettings:
    r"""
    Fields
    ------
    patterns : list of str
        The patterns to loop.
    """
    patterns: List[str] = dataclasses.field(
        default_factory=lambda: [f"[color=cyan]{pattern}[/]" for pattern in TIMCOEP8]
    )


class PatternsWidget:
    def __init__(self, settings):
        self.settings = settings

    def load(self, provider):
        rich = provider.get(mu.RichParser)
        metronome = provider.get(clocks.Metronome)

        patterns = self.settings.patterns

        markuped_patterns = [rich.parse(pattern) for pattern in patterns]
        tick_node = metronome.tick("patterns")

        @dn.datanode
        def patterns_node():
            with tick_node:
                time, ran = yield
                while True:
                    beat, _ = tick_node.send(time)
                    ind = int(beat * len(markuped_patterns) // 1) % len(markuped_patterns)
                    res = markuped_patterns[ind]
                    time, ran = yield [(0, res)]

        return patterns_node()


@dataclasses.dataclass
class MarkerWidgetSettings:
    r"""
    Fields
    ------
    normal_appearance : str
        The appearance of normal-style markers.
    blinking_appearance : str
        The appearance of blinking-style markers.
    blink_ratio : float
        The ratio to blink.
    """
    normal_appearance: str = "❯ "
    blinking_appearance: str = "[weight=bold]❯ [/]"
    blink_ratio: float = 0.3


class MarkerWidget:
    def __init__(self, settings):
        self.settings = settings

    def load(self, provider):
        rich = provider.get(mu.RichParser)
        metronome = provider.get(clocks.Metronome)

        blink_ratio = self.settings.blink_ratio
        normal = [(0, rich.parse(self.settings.normal_appearance))]
        blinking = [(0, rich.parse(self.settings.blinking_appearance))]
        tick_node = metronome.tick("marker")

        @dn.datanode
        def marker_node():
            with tick_node:
                time, ran = yield
                while True:
                    beat, _ = tick_node.send(time)
                    res = blinking if beat % 4 < min(1.0, blink_ratio) else normal
                    time, ran = yield res

        return marker_node()


BeatshellIconWidgetSettings = Union[
    PatternsWidgetSettings,
    widgets.MonitorWidgetSettings,
]


# prompt
class BeatShellSettings(cfg.Configurable):
    r"""
    Fields
    ------
    debug_monitor : bool
        Whether to monitor renderer.
    """

    debug_monitor: bool = False

    input = cfg.subconfig(inputs.InputSettings)

    @cfg.subconfig
    class prompt(cfg.Configurable):
        r"""
        Fields
        ------
        t0 : float
        tempo : float

        icon_width : int
            The text width of icon.
        marker_width : int
            The text width of marker.

        icons : BeatShellIconWidgetSettings
            The appearances of icon.
        marker : MarkerWidgetSettings
            The appearance of marker.
        """
        t0: float = 0.0
        tempo: float = 130.0

        icon_width: int = 5
        marker_width: int = 2

        icons: BeatshellIconWidgetSettings = PatternsWidgetSettings()
        marker: MarkerWidgetSettings = MarkerWidgetSettings()

    @cfg.subconfig
    class banner(cfg.Configurable):
        r"""
        Fields
        ------
        banner : str
            The template of banner with slots `user`, `profile` and `path`.
        info_block : str
            The template for info of current directory.

        user : str
            The template of user with slot `user_name`.
        profile : tuple of str and str
            The templates of profile with slot `profile_name`, the second is for
            changed profile.
        path : tuple of str and str
            The templates of path with slot `current_path`, the second is for
            unknown path.

        unprintable_character : str
            The placeholder for unprintable character.
        """

        banner: str = (
            "[color=bright_black][[[/]"
            "[slot=user/]"
            "[color=bright_black]/[/]"
            "[slot=profile/]"
            "[color=bright_black]]][/]"
            " [slot=path/]"
        )
        info_block: str = f"{'━'*80}\n[slot/]\n{'━'*80}\n"

        user: str = "[color=magenta]♜ [weight=bold][slot=user_name/][/][/]"
        profile: Tuple[str, str] = (
            "[color=blue]⚙ [weight=bold][slot=profile_name/][/][/]",
            "[color=blue]⚙ [weight=bold][slot=profile_name/][/][/]*",
        )
        path: Tuple[str, str] = (
            "[color=cyan]⛩ [weight=bold][slot=current_path/][/][/]",
            "[color=cyan]⛩ [weight=dim][slot=current_path/][/][/]",
        )


class PromptError(Exception):
    def __init__(self, cause):
        self.cause = cause


class BeatshellHistory(RecognizedFilePath):
    "The command history"


class PromptBenchmark(RecognizedFilePath):
    "The prompt benchmark data"


class BeatPrompt:
    r"""Prompt renderer for beatshell."""

    def __init__(
        self,
        provider,
        history_file_path,
        monitor_file_path,
        settings,
    ):
        self.provider = provider
        self.logger = provider.get(Logger)
        self.metronome = provider.get(clocks.Metronome)
        self.settings = settings
        self.history_file_path = history_file_path
        self.monitor_file_path = monitor_file_path

        bgm_controller = provider.get(BGMController)

        self.input = inputs.Input(
            bgm_controller.preview_handler,
            self.history_file_path.abs,
            self.settings.input,
        )

    def set_settings(self, settings):
        self.settings = settings
        self.input._set_settings(self.settings.input)

    def create_widget(self, provider, widget_settings):
        if isinstance(widget_settings, widgets.MonitorWidgetSettings):
            return widgets.MonitorWidget(widget_settings).load(provider)
        elif isinstance(widget_settings, PatternsWidgetSettings):
            return PatternsWidget(widget_settings).load(provider)
        elif isinstance(widget_settings, MarkerWidgetSettings):
            return MarkerWidget(widget_settings).load(provider)
        else:
            raise TypeError

    def register(self, renderer, controller, fin_event):
        # widgets
        settings = self.settings

        provider = Provider()
        provider.set(self.logger.rich)
        provider.set(self.metronome)
        provider.set(renderer)
        provider.set(controller)

        icon = self.create_widget(provider, settings.prompt.icons)
        marker = self.create_widget(provider, settings.prompt.marker)
        textbox = self.input._register(fin_event, provider)

        # layout
        icon_width = settings.prompt.icon_width
        marker_width = settings.prompt.marker_width

        [
            icon_mask,
            marker_mask,
            textbox_mask,
        ] = widgets.layout([icon_width, marker_width, -1])

        # register
        renderer.add_texts(icon, icon_mask, zindex=(2,))
        renderer.add_texts(marker, marker_mask, zindex=(3,))
        renderer.add_texts(textbox, textbox_mask, zindex=(0,))

    @dn.datanode
    def new_session(self, command_parser, clock, devices_settings, clear=True):
        self.input._new_session(command_parser, clear=clear)

        fin_event = threading.Event()

        # engines
        settings = self.settings
        debug_monitor = settings.debug_monitor
        renderer_monitor = (
            engines.Monitor(self.monitor_file_path.abs) if debug_monitor else None
        )
        input_task, controller = engines.Controller.create(
            devices_settings.controller, devices_settings.terminal, clock=clock,
        )
        display_task, renderer = engines.Renderer.create(
            devices_settings.renderer,
            devices_settings.terminal,
            clock=clock,
            monitor=renderer_monitor,
        )

        # handlers
        self.register(renderer, controller, fin_event)

        @dn.datanode
        def stop_when(event):
            yield
            yield
            while not event.is_set():
                yield

        with self.logger.popup(renderer):
            yield from dn.pipe(stop_when(fin_event), display_task, input_task).join()

        result = self.input.result
        if isinstance(result, inputs.ErrorResult):
            self.logger.log(f"parse command: {result.command_str}")
            raise PromptError(result.error)
        elif isinstance(result, inputs.CompleteResult):
            self.logger.log(
                f"parse command: [{result.command_group}] {result.command_str}"
            )
            return result.command
        elif isinstance(result, inputs.EmptyResult):
            return lambda: None
        else:
            raise TypeError

    def record_command(self):
        return self.input._record_command()

    def print_banner(self, print_info):
        file_manager = self.provider.get(FileManager)
        profile_manager = self.provider.get(ProfileManager)

        banner_settings = self.settings.banner

        username = file_manager.username
        current_name = profile_manager.current_path.abs.stem
        path = file_manager.as_relative_path(file_manager.current)
        profile_is_changed = profile_manager.is_changed()
        path_is_known = not isinstance(file_manager.current, UnrecognizedPath)

        username = self.logger.rich.parse(self.logger.escape(username))
        profile = self.logger.rich.parse(self.logger.escape(current_name))
        path = self.logger.rich.parse(self.logger.escape(path))

        user_markup = banner_settings.user
        user_markup = self.logger.rich.parse(user_markup, expand=False, slotted=True)
        user_markup = user_markup(user_name=username)

        profile_markup = banner_settings.profile
        profile_markup = (
            profile_markup[0] if not profile_is_changed else profile_markup[1]
        )
        profile_markup = self.logger.rich.parse(
            profile_markup, expand=False, slotted=True
        )
        profile_markup = profile_markup(profile_name=profile)

        path_markup = banner_settings.path
        path_markup = path_markup[0] if path_is_known else path_markup[1]
        path_markup = self.logger.rich.parse(path_markup, expand=False, slotted=True)
        path_markup = path_markup(current_path=path)

        banner_markup = banner_settings.banner
        banner_markup = self.logger.rich.parse(
            banner_markup, expand=False, slotted=True
        )
        banner_markup = banner_markup(
            user=user_markup,
            profile=profile_markup,
            path=path_markup,
        )

        self.logger.print(banner_markup, log=False)

        if not print_info:
            return

        info = file_manager.current.info_detailed(self.provider)
        if info is None:
            return

        info_markup = self.logger.rich.parse(info, root_tag=True)
        info_block = self.logger.rich.parse(
            banner_settings.info_block, expand=False, slotted=True
        )
        info_markup = info_block(info_markup)
        self.logger.print(info_markup, log=False)
