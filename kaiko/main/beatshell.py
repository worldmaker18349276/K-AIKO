import threading
from typing import List, Tuple, Union
import dataclasses
from ..utils.providers import Provider
from ..utils import datanodes as dn
from ..utils import config as cfg
from ..utils import markups as mu
from ..devices import engines
from ..tui import widgets
from ..tui import inputs
from .files import UnrecognizedPath


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
    patterns: List[str] = dataclasses.field(default_factory=lambda: [f"[color=cyan]{pattern}[/]" for pattern in TIMCOEP8])


class PatternsWidget:
    def __init__(self, settings):
        self.settings = settings

    def load(self, provider):
        rich = provider.get(mu.RichParser)
        metronome = provider.get(engines.Metronome)

        patterns = self.settings.patterns

        markuped_patterns = [rich.parse(pattern) for pattern in patterns]

        def patterns_func(arg):
            time, ran = arg
            beat = metronome.beat(time)
            ind = int(beat * len(markuped_patterns) // 1) % len(markuped_patterns)
            res = markuped_patterns[ind]
            return [(0, res)]

        return patterns_func


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
        metronome = provider.get(engines.Metronome)

        blink_ratio = self.settings.blink_ratio
        normal = [(0, rich.parse(self.settings.normal_appearance))]
        blinking = [(0, rich.parse(self.settings.blinking_appearance))]

        def marker_func(arg):
            time, ran = arg
            beat = metronome.beat(time)
            if beat % 4 < min(1.0, blink_ratio):
                return blinking
            else:
                return normal

        return marker_func


BeatshellIconWidgetSettings = Union[
    PatternsWidgetSettings, widgets.MonitorWidgetSettings,
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
        user: str = "[color=magenta]♜ [weight=bold][slot=user_name/][/][/]"
        profile: Tuple[str, str] = (
            "[color=blue]⚙ [weight=bold][slot=profile_name/][/][/]",
            "[color=blue]⚙ [weight=bold][slot=profile_name/][/][/]*",
        )
        path: Tuple[str, str] = (
            "[color=cyan]⛩ [weight=bold][slot=current_path/][/][/]",
            "[color=cyan]⛩ [weight=dim][slot=current_path/][/][/]",
        )

        unprintable_character: str = "⍰"


class PromptError(Exception):
    def __init__(self, cause):
        self.cause = cause


class BeatPrompt:
    r"""Prompt renderer for beatshell."""

    monitor_file_path = "monitor/prompt.csv"
    history_file_path = ".beatshell-history"

    def __init__(
        self,
        rich,
        cache_dir,
        command_parser,
        settings,
        preview_handler,
    ):
        self.rich = rich
        self.settings = settings
        self.cache_dir = cache_dir

        self.input = inputs.Input(
            preview_handler,
            cache_dir / self.history_file_path,
            self.settings.input,
        )

        self.new_session(command_parser)

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
        t0 = settings.prompt.t0
        tempo = settings.prompt.tempo
        metronome = engines.Metronome(t0, tempo)

        provider = Provider()
        provider.set(self.rich)
        provider.set(metronome)
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
    def prompt(self, devices_settings):
        fin_event = threading.Event()

        # engines
        settings = self.settings
        debug_monitor = settings.debug_monitor
        renderer_monitor = (
            engines.Monitor(self.cache_dir / self.monitor_file_path)
            if debug_monitor
            else None
        )
        input_task, controller = engines.Controller.create(
            devices_settings.controller, devices_settings.terminal
        )
        display_task, renderer = engines.Renderer.create(
            devices_settings.renderer,
            devices_settings.terminal,
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

        yield from dn.pipe(stop_when(fin_event), display_task, input_task).join()

        result = self.input.result
        if isinstance(result, inputs.ErrorResult):
            raise PromptError(result.error)
        elif isinstance(result, inputs.CompleteResult):
            return result.command
        else:
            raise TypeError

    def new_session(self, command_parser, clear=True):
        return self.input._new_session(command_parser, clear=clear)

    def record_command(self):
        return self.input._record_command()

    def make_banner(self, file_manager, profile_manager):
        banner_settings = self.settings.banner

        username = file_manager.username
        current_name = profile_manager.current_path.abs.stem
        path = file_manager.as_relative_path(file_manager.current)
        profile_is_changed = profile_manager.is_changed()
        path_is_known = not isinstance(file_manager.current, UnrecognizedPath)

        unpr = banner_settings.unprintable_character
        username = mu.Text("".join(ch if ch.isprintable() else unpr for ch in username))
        profile = mu.Text("".join(ch if ch.isprintable() else unpr for ch in current_name))
        path = mu.Text("".join(ch if ch.isprintable() else unpr for ch in path))

        user_markup = banner_settings.user
        user_markup = self.rich.parse(user_markup, slotted=True)
        user_markup = user_markup(user_name=username)

        profile_markup = banner_settings.profile
        profile_markup = profile_markup[0] if not profile_is_changed else profile_markup[1]
        profile_markup = self.rich.parse(profile_markup, slotted=True)
        profile_markup = profile_markup(profile_name=profile)

        path_markup = banner_settings.path
        path_markup = path_markup[0] if path_is_known else path_markup[1]
        path_markup = self.rich.parse(path_markup, slotted=True)
        path_markup = path_markup(current_path=path)

        banner_markup = banner_settings.banner
        banner_markup = self.rich.parse(banner_markup, slotted=True)
        banner_markup = banner_markup(
            user=user_markup,
            profile=profile_markup,
            path=path_markup,
        )

        return banner_markup

