import threading
import queue
from typing import List, Tuple, Union
import dataclasses
from ..utils.providers import Provider
from ..utils import datanodes as dn
from ..utils import config as cfg
from ..utils import markups as mu
from ..utils import commands as cmd
from ..devices import engines
from . import beatwidgets
from . import beatinputs


# widgets
@dataclasses.dataclass
class PatternsWidgetSettings:
    r"""
    Fields
    ------
    patterns : list of str
        The patterns to loop.
    """
    patterns: List[str] = dataclasses.field(
        default_factory=lambda: [
            "[color=cyan]⠶⠦⣚⠀⠶[/]",
            "[color=cyan]⢎⣀⡛⠀⠶[/]",
            "[color=cyan]⢖⣄⠻⠀⠶[/]",
            "[color=cyan]⠖⠐⡩⠂⠶[/]",
            "[color=cyan]⠶⠀⡭⠲⠶[/]",
            "[color=cyan]⠶⠀⣬⠉⡱[/]",
            "[color=cyan]⠶⠀⣦⠙⠵[/]",
            "[color=cyan]⠶⠠⣊⠄⠴[/]",
        ]
    )


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


class BeatshellWidgetFactory:
    monitor = beatwidgets.MonitorWidgetSettings
    patterns = PatternsWidgetSettings
    marker = MarkerWidgetSettings

    def __init__(self, rich, renderer, metronome):
        self.provider = Provider()
        self.provider.set(rich)
        self.provider.set(renderer)
        self.provider.set(metronome)

    def create(self, widget_settings):
        if isinstance(widget_settings, BeatshellWidgetFactory.monitor):
            return beatwidgets.MonitorWidget(widget_settings).load(self.provider)
        elif isinstance(widget_settings, BeatshellWidgetFactory.patterns):
            return PatternsWidget(widget_settings).load(self.provider)
        elif isinstance(widget_settings, BeatshellWidgetFactory.marker):
            return MarkerWidget(widget_settings).load(self.provider)
        else:
            raise TypeError


BeatshellIconWidgetSettings = Union[
    PatternsWidgetSettings, beatwidgets.MonitorWidgetSettings,
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

    input = cfg.subconfig(beatinputs.BeatInputSettings)

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
        class textbox(cfg.Configurable, beatwidgets.TextBoxWidgetSettings):
            __doc__ = beatwidgets.TextBoxWidgetSettings.__doc__

    @cfg.subconfig
    class banner(cfg.Configurable):
        r"""
        Fields
        ------
        banner : str
            The template of banner with slots: `user`, `profile`, `path`.

        user : str
            The template of user with slots: `user_name`.
        profile : tuple of str and str
            The templates of profile with slots: `profile_name`, the second is
            for changed profile.
        path : tuple of str and str
            The templates of path with slots: `current_path`, the second is for
            unknown path.
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

    @cfg.subconfig
    class text(cfg.Configurable):
        r"""
        Fields
        ------
        typeahead : str
            The markup template for the type-ahead.
        highlight : str
            The markup template for the highlighted token.

        desc_message : str
            The markup template for the desc message.
        info_message : str
            The markup template for the info message.
        message_max_lines : int
            The maximum number of lines of the message.
        message_overflow_ellipsis : str
            Texts to display when overflowing.

        suggestions_lines : int
            The maximum number of lines of the suggestions.
        suggestion_items : tuple of str and str
            The markup templates for the unselected/selected suggestion.
        suggestion_overflow_ellipses : tuple of str and str
            Texts to display when overflowing top/bottom.
        """
        typeahead: str = "[weight=dim][slot/][/]"
        highlight: str = "[underline][slot/][/]"

        desc_message: str = "[weight=dim][slot/][/]"
        info_message: str = f"{'─'*80}\n[slot/]\n{'─'*80}"
        message_max_lines: int = 16
        message_overflow_ellipsis: str = "[weight=dim]…[/]"

        suggestions_lines: int = 8
        suggestion_items: Tuple[str, str] = ("• [slot/]", "• [invert][slot/][/]")
        suggestion_overflow_ellipses: Tuple[str, str] = ("[weight=dim]ⵗ[/]", "[weight=dim]ⵗ[/]")


class PromptError(Exception):
    def __init__(self, cause):
        self.cause = cause


class BeatPrompt:
    r"""Prompt renderer for beatshell."""

    monitor_file_path = "monitor/prompt.csv"

    def __init__(
        self,
        rich,
        cache_dir,
        command_parser_getter,
        shell_settings_getter,
        preview_handler,
    ):
        self.rich = rich
        self._shell_settings_getter = shell_settings_getter
        self.cache_dir = cache_dir

        self.fin_event = threading.Event()

        self.input = beatinputs.BeatInput(
            command_parser_getter,
            preview_handler,
            rich,
            cache_dir,
            lambda: self._shell_settings_getter().input,
        )

    @property
    def settings(self):
        return self._shell_settings_getter()

    def register(self, renderer):
        # widgets
        settings = self.settings
        t0 = settings.prompt.t0
        tempo = settings.prompt.tempo
        metronome = engines.Metronome(t0, tempo)

        widget_factory = BeatshellWidgetFactory(self.rich, renderer, metronome)

        icon = widget_factory.create(settings.prompt.icons)
        marker = widget_factory.create(settings.prompt.marker)

        state = InputView(self.input)
        text_renderer = TextRenderer(self.rich, settings.text)
        msg_renderer = MsgRenderer(self.rich, settings.text)

        textbox = beatwidgets.TextBox(
            lambda: text_renderer.render_text(state),
            settings.prompt.textbox,
        ).load(widget_factory.provider)

        # layout
        icon_width = settings.prompt.icon_width
        marker_width = settings.prompt.marker_width

        [
            icon_mask,
            marker_mask,
            input_mask,
        ] = beatwidgets.layout([icon_width, marker_width, -1])

        # register
        renderer.add_drawer(state.load(self.fin_event), zindex=())
        renderer.add_texts(icon, icon_mask, zindex=(2,))
        renderer.add_texts(marker, marker_mask, zindex=(3,))
        renderer.add_texts(textbox, input_mask, zindex=(0,))
        renderer.add_drawer(msg_renderer.render_msg(state), zindex=(1,))

    @dn.datanode
    def prompt(self, devices_settings):
        self.fin_event.clear()

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
        self.register(renderer)
        self.input._register(controller)

        @dn.datanode
        def stop_when(event):
            yield
            yield
            while not event.is_set():
                yield

        yield from dn.pipe(stop_when(self.fin_event), display_task, input_task).join()

        result = self.input.result
        if isinstance(result, beatinputs.ErrorResult):
            raise PromptError(result.error)
        elif isinstance(result, beatinputs.CompleteResult):
            return result.command
        else:
            raise TypeError

    def prev_session(self):
        return self.input.prev_session()

    def new_session(self):
        return self.input.new_session()

    def record_command(self):
        return self.input._record_command()


class InputView:
    def __init__(self, input):
        r"""Constructor.

        Parameters
        ----------
        input : BeatInput
        """
        self.input = input

        self.key_pressed = False
        self.buffer = []
        self.tokens = []
        self.pos = 0
        self.highlighted = None
        self.typeahead = ""
        self.clean = False
        self.hint = None
        self.popup = []
        self.suggestions = None
        self.state = "EDIT"

    @dn.datanode
    def load(self, fin_event):
        buffer_modified_counter = None
        key_pressed_counter = None

        res, time, width = yield

        while True:
            with self.input.edit_ctxt.lock:
                if self.input.buffer_modified_counter != buffer_modified_counter:
                    buffer_modified_counter = self.input.buffer_modified_counter
                    self.buffer = list(self.input.text_buffer.buffer)
                    self.tokens = list(self.input.semantic_analyzer.tokens)
                self.pos = self.input.text_buffer.pos

                self.typeahead = self.input.typeahead
                self.clean = self.input.result is not None
                self.hint = self.input.hint_manager.get_hint()
                self.suggestions = (
                    self.input.autocomplete_manager.get_suggestions_list(),
                    self.input.autocomplete_manager.get_suggestions_index(),
                )

                self.popup = []
                while True:
                    try:
                        hint = self.input.hint_manager.popup_queue.get(False)
                    except queue.Empty:
                        break
                    self.popup.append(hint)

                if isinstance(self.input.result, beatinputs.ErrorResult):
                    self.highlighted = self.input.result.index
                else:
                    self.highlighted = self.input.hint_manager.get_hint_location()

                self.state = self.input.state

                self.key_pressed = self.input.key_pressed_counter != key_pressed_counter
                key_pressed_counter = self.input.key_pressed_counter

            res, time, width = yield res

            # fin
            if self.state == "FIN" and not fin_event.is_set():
                fin_event.set()


@dataclasses.dataclass(frozen=True)
class ByAddress:
    value: object

    def __eq__(self, other):
        if not isinstance(other, ByAddress):
            return False
        return self.value is other.value


class TextRenderer:
    def __init__(self, rich, settings):
        self.rich = rich
        self.settings = settings

    @staticmethod
    def _render_grammar_key(buffer, tokens, typeahead, pos, highlighted, clean):
        return (
            ByAddress(buffer),
            typeahead,
            pos,
            highlighted,
            clean,
        )

    def render_grammar(
        self,
        buffer,
        tokens,
        typeahead,
        pos,
        highlighted,
        clean,
        caret_markup,
        typeahead_template,
        highlight_template,
    ):
        length = len(buffer)
        buffer = list(buffer)

        for token in tokens:
            # markup whitespace
            for index in range(token.mask.start, token.mask.stop):
                if buffer[index] == " ":
                    buffer[index] = self.rich.tags["ws"]()

            # markup escape
            for index in token.quotes:
                if buffer[index] == "'":
                    buffer[index] = self.rich.tags["qt"]()
                elif buffer[index] == "\\":
                    buffer[index] = self.rich.tags["bs"]()
                else:
                    assert False

        # markup caret, typeahead
        if clean:
            typeahead = ""

        if pos == length and not typeahead:
            buffer.append(" ")

        if not clean:
            if pos < len(buffer):
                buffer[pos] = caret_markup(mu.join([buffer[pos]]).children)
            else:
                typeahead = caret_markup(mu.join(typeahead[0]).children), typeahead[1:]

        typeahead_markup = mu.replace_slot(typeahead_template, mu.join(typeahead))

        res = []
        prev_index = 0
        for n, token in enumerate(tokens):
            # markup delimiter
            delimiter_markup = mu.join(buffer[prev_index : token.mask.start])
            res.append(delimiter_markup)
            prev_index = token.mask.stop

            # markup token
            token_markup = mu.join(buffer[token.mask])
            if token.type is None:
                if clean or token.mask.stop != length:
                    token_markup = self.rich.tags["unk"](token_markup.children)
            elif token.type is cmd.TOKEN_TYPE.COMMAND:
                token_markup = self.rich.tags["cmd"](token_markup.children)
            elif token.type is cmd.TOKEN_TYPE.KEYWORD:
                token_markup = self.rich.tags["kw"](token_markup.children)
            elif token.type is cmd.TOKEN_TYPE.ARGUMENT:
                token_markup = self.rich.tags["arg"](token_markup.children)
            else:
                assert False

            # markup highlight
            if n == highlighted:
                token_markup = mu.replace_slot(highlight_template, token_markup)

            res.append(token_markup)

        else:
            delimiter_markup = mu.join(buffer[prev_index:])
            res.append(delimiter_markup)

        markup = mu.Group((*res, typeahead_markup))
        markup = markup.expand()
        return markup

    @dn.datanode
    def render_text(self, state):
        typeahead_template = self.rich.parse(self.settings.typeahead, slotted=True)
        highlight_template = self.rich.parse(self.settings.highlight, slotted=True)

        render_grammar = dn.starcachemap(
            self.render_grammar,
            key=self._render_grammar_key,
            caret_markup=beatwidgets.Caret,
            typeahead_template=typeahead_template,
            highlight_template=highlight_template,
        )

        with render_grammar:
            yield
            while True:
                markup = render_grammar.send(
                    (
                        state.buffer,
                        state.tokens,
                        state.typeahead,
                        state.pos,
                        state.highlighted,
                        state.clean,
                    )
                )
                yield markup, state.key_pressed


class MsgRenderer:
    def __init__(self, rich, settings):
        r"""Constructor.

        Parameters
        ----------
        rich : markups.RichParser
        settings : BeatShellSettings.text
        """
        self.rich = rich
        self.settings = settings

    @dn.datanode
    def render_msg(self, state):
        message_max_lines = self.settings.message_max_lines
        sugg_lines = self.settings.suggestions_lines
        sugg_items = self.settings.suggestion_items

        message_overflow_ellipsis = self.settings.message_overflow_ellipsis
        suggestion_overflow_ellipses = self.settings.suggestion_overflow_ellipses

        msg_ellipsis = self.rich.parse(message_overflow_ellipsis)
        msg_ellipsis_width = self.rich.widthof(msg_ellipsis)

        if msg_ellipsis_width == -1:
            raise ValueError(f"invalid ellipsis: {message_overflow_ellipsis!r}")

        sugg_top_ellipsis = self.rich.parse(suggestion_overflow_ellipses[0])
        sugg_top_ellipsis_width = self.rich.widthof(sugg_top_ellipsis)
        sugg_bottom_ellipsis = self.rich.parse(suggestion_overflow_ellipses[1])
        sugg_bottom_ellipsis_width = self.rich.widthof(sugg_bottom_ellipsis)

        if sugg_top_ellipsis_width == -1 or sugg_bottom_ellipsis_width == -1:
            raise ValueError(f"invalid ellipsis: {suggestion_overflow_ellipses!r}")

        sugg_items = (
            self.rich.parse(sugg_items[0], slotted=True),
            self.rich.parse(sugg_items[1], slotted=True),
        )

        desc = self.rich.parse(self.settings.desc_message, slotted=True)
        info = self.rich.parse(self.settings.info_message, slotted=True)

        render_hint = dn.starcachemap(
            self.render_hint,
            key=lambda msgs, hint, suggs: (hint, suggs),
            message_max_lines=message_max_lines,
            msg_ellipsis=msg_ellipsis,
            sugg_lines=sugg_lines,
            sugg_items=sugg_items,
            sugg_ellipses=(sugg_top_ellipsis, sugg_bottom_ellipsis),
            desc=desc,
            info=info,
        )

        with render_hint:
            (view, msgs, logs), time, width = yield
            while True:
                render_hint.send((msgs, state.hint, state.suggestions))
                logs.extend(self.render_popup(state.popup, desc=desc, info=info))
                (view, msgs, logs), time, width = yield (view, msgs, logs)

    def render_hint(
        self,
        msgs,
        hint,
        suggestions,
        *,
        message_max_lines,
        msg_ellipsis,
        sugg_lines,
        sugg_items,
        sugg_ellipses,
        desc,
        info,
    ):
        msgs.clear()

        # draw hint
        msg = None
        if hint is not None and hint.message:
            msg = self.rich.parse(hint.message, root_tag=True)
            lines = 0

            def trim_lines(text):
                nonlocal lines
                if lines >= message_max_lines:
                    return mu.Text("")

                if isinstance(text, mu.Newline):
                    lines += 1
                    if lines == message_max_lines:
                        return mu.Group((text, msg_ellipsis))

                else:
                    for i, ch in enumerate(text.string):
                        if ch == "\n":
                            lines += 1
                        if lines == message_max_lines:
                            return mu.Group((mu.Text(text.string[:i+1]), msg_ellipsis))

                return text

            msg = msg.traverse((mu.Text, mu.Newline), trim_lines)

            if isinstance(hint, beatinputs.DescHint):
                msg = mu.replace_slot(desc, msg)
            elif isinstance(hint, beatinputs.InfoHint):
                msg = mu.replace_slot(info, msg)
            else:
                assert False
            msg = msg.expand()

        if suggestions[0] is not None:
            suggs_list, sugg_index = suggestions
            sugg_start = sugg_index // sugg_lines * sugg_lines
            sugg_end = sugg_start + sugg_lines
            suggs = suggs_list[sugg_start:sugg_end]

            res = []
            for i, sugg in enumerate(suggs):
                sugg = mu.Text(sugg)
                item = sugg_items[1] if i == sugg_index - sugg_start else sugg_items[0]
                sugg = mu.replace_slot(item, sugg)
                res.append(sugg)
                if i == sugg_index - sugg_start and msg is not None:
                    res.append(msg)

            if sugg_start > 0:
                res.insert(0, sugg_ellipses[0])
            if sugg_end < len(suggs_list):
                res.append(sugg_ellipses[1])

            nl = mu.Text("\n")
            is_fst = True
            for block in res:
                if not is_fst:
                    msgs.append(nl)
                msgs.append(block)
                is_fst = False

        else:
            if msg is not None:
                msgs.append(msg)

        return msgs

    def render_popup(self, popup, *, desc, info):
        logs = []

        # draw popup
        for hint in popup:
            msg = None
            if hint.message:
                msg = self.rich.parse(hint.message, root_tag=True)

                if isinstance(hint, beatinputs.DescHint):
                    msg = mu.replace_slot(desc, msg)
                elif isinstance(hint, beatinputs.InfoHint):
                    msg = mu.replace_slot(info, msg)
                else:
                    assert False

                msg = mu.Group((msg, mu.Text("\n")))
                msg = msg.expand()

            if msg is not None:
                logs.append(msg)

        return logs

