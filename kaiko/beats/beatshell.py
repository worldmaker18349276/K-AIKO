from enum import Enum
import functools
import re
import threading
import queue
from typing import Optional, List, Tuple, Dict, Callable, Union
from pathlib import Path
import dataclasses
from ..utils.providers import Provider
from ..utils import datanodes as dn
from ..utils import config as cfg
from ..utils import markups as mu
from ..utils import commands as cmd
from ..devices import engines
from . import beatwidgets


class SHLEXER_STATE(Enum):
    SPACED = " "
    PLAIN = "*"
    BACKSLASHED = "\\"
    QUOTED = "'"


@dataclasses.dataclass(frozen=True)
class ShToken:
    r"""The token of shell-like grammar.

    Attributes
    ----------
    string : str
        The tokenized string.
    type : cmd.TOKEN_TYPE or None
        The type of token.
    mask : slice
        The position of this token.
    quotes : list of int
        The indices of all backslashes and quotation marks used for escaping.
        The token is equal to
        `''.join(raw[i] for i in range(*slice.indices(len(raw))) if i not in quotes)`.
    """
    string: str
    type: Optional[cmd.TOKEN_TYPE]
    mask: slice
    quotes: List[int]

def shlexer_tokenize(raw):
    r"""Tokenizer for shell-like grammar.

    The delimiter is just whitespace, and the token is defined as::

        <nonspace-character> ::= /[^ \\\']/
        <backslashed-character> ::= "\" /./
        <quoted-string> ::= "'" /[^']*/ "'"
        <token> ::= ( <nonspace-character> | <backslashed-character> | <quoted-string> )*

    The backslashes and quotation marks used for escaping will be deleted after
    being interpreted as a string. The input string should be printable, so it
    doesn't contain tab, newline, backspace, etc. In this grammar, the token of
    an empty string can be expressed as `''`.

    Parameters
    ----------
    raw : str or list of str
        The string to tokenize, which should be printable.

    Yields
    ------
    token: ShToken
        the parsed token.

    Returns
    -------
    state : SHLEXER_STATE
        The final state of parsing.
    """
    SPACE = " "
    BACKSLASH = "\\"
    QUOTE = "'"

    length = len(raw)
    raw = enumerate(raw)

    while True:
        try:
            index, char = next(raw)
        except StopIteration:
            return SHLEXER_STATE.SPACED

        # guard space
        if char == SPACE:
            continue

        # parse token
        start = index
        token = []
        quotes = []
        while True:
            if char == SPACE:
                # end parsing token
                yield ShToken("".join(token), None, slice(start, index), quotes)
                break

            elif char == BACKSLASH:
                # escape the next character
                quotes.append(index)

                try:
                    index, char = next(raw)
                except StopIteration:
                    yield ShToken("".join(token), None, slice(start, length), quotes)
                    return SHLEXER_STATE.BACKSLASHED

                token.append(char)

            elif char == QUOTE:
                # escape the following characters until the next quotation mark
                quotes.append(index)

                while True:
                    try:
                        index, char = next(raw)
                    except StopIteration:
                        yield ShToken("".join(token), None, slice(start, length), quotes)
                        return SHLEXER_STATE.QUOTED

                    if char == QUOTE:
                        quotes.append(index)
                        break
                    else:
                        token.append(char)

            else:
                # otherwise, as it is
                token.append(char)

            try:
                index, char = next(raw)
            except StopIteration:
                yield ShToken("".join(token), None, slice(start, length), quotes)
                return SHLEXER_STATE.PLAIN


def shlexer_quoting(compreply, state=SHLEXER_STATE.SPACED):
    r"""Escape a given string so that it can be inserted into an untokenized string.

    The strategy to escape insert string only depends on the state of insert
    position.

    Parameters
    ----------
    compreply : str
        The string to insert. The suffix `'\000'` indicate closing the token.
        But inserting `'\000'` after backslash results in `''`, since it is
        impossible to close it.
    state : SHLEXER_STATE
        The state of insert position.

    Returns
    -------
    raw : str
        The escaped string which can be inserted into untokenized string
        directly.
    """
    partial = not compreply.endswith("\000")
    if not partial:
        compreply = compreply[:-1]

    if state == SHLEXER_STATE.PLAIN:
        raw = re.sub(r"([ \\'])", r"\\\1", compreply)

    elif state == SHLEXER_STATE.BACKSLASHED:
        if compreply == "":
            # cannot close backslash without deleting it
            return ""
        raw = compreply[0] + re.sub(r"([ \\'])", r"\\\1", compreply[1:])

    elif state == SHLEXER_STATE.QUOTED:
        if partial:
            raw = compreply.replace("'", r"'\''")
        elif compreply == "":
            raw = "'"
        else:
            raw = compreply[:-1].replace("'", r"'\''") + (
                r"'\'" if compreply[-1] == "'" else compreply[-1] + "'"
            )

    elif state == SHLEXER_STATE.SPACED:
        if compreply != "" and " " not in compreply:
            # use backslash if there is no whitespace
            raw = re.sub(r"([ \\'])", r"\\\1", compreply)
        elif compreply == "":
            raw = "''"
        else:
            raw = (
                "'"
                + compreply[:-1].replace("'", r"'\''")
                + (r"'\'" if compreply[-1] == "'" else compreply[-1] + "'")
            )

    else:
        assert False

    return raw if partial else raw + " "


def shlexer_markup(buffer, tokens, typeahead, tags):
    r"""Markup shell-like grammar.

    Parameters
    ----------
    buffer : list of str
    tokens : list of ShToken
    typeahead : str
    tags : dict
        The dictionary of tags:
        "ws" maps to whitespace tag;
        "qt" maps to quotation tag;
        "bs" maps to backspace tag;
        "cmd" maps to command tag;
        "kw" maps to keyword tag;
        "arg" maps to argument tag;
        "unfinished" map to unfinished tag;
        "unknown" map to unknown tag;
        "typeahead" map to typeahead tag.

    Returns
    -------
    markup : markups.Markup
        The rendered input text.
    """
    # result  ::=  Delimiters + Token + Delimiters + ... + Delimiters + Typeahead
    # Delimiters  ::=  Group[QuasiText]
    # Token  ::=  Unknown[QuasiText] | Command[QuasiText] | Keyword[QuasiText] | Argument[QuasiText]
    # QuasiText  ::=  Text | Whitespace | Quotation | Backslash

    def _wrap(buffer):
        res = [""]
        for ch in buffer:
            if isinstance(ch, str) and isinstance(res[-1], str):
                res[-1] = res[-1] + ch
            else:
                res.append(ch)
        if not res[0]:
            res.pop(0)
        return tuple(mu.Text(e) if isinstance(e, str) else e for e in res)

    length = len(buffer)
    buffer = list(buffer)
    if not typeahead:
        buffer.append(" ")

    for token in tokens:
        # markup whitespace
        for index in range(token.mask.start, token.mask.stop):
            if buffer[index] == " ":
                buffer[index] = tags["ws"]()

        # markup escape
        for index in token.quotes:
            if buffer[index] == "'":
                buffer[index] = tags["qt"]()
            elif buffer[index] == "\\":
                buffer[index] = tags["bs"]()
            else:
                assert False

    markup_children = []
    prev_index = 0
    for n, token in enumerate(tokens):
        # markup delimiter
        markup_children.append(mu.Group(_wrap(buffer[prev_index : token.mask.start])))
        prev_index = token.mask.stop

        # markup token
        if token.type is None:
            if token.mask.stop == length:
                markup_children.append(tags["unfinished"](_wrap(buffer[token.mask])))
            else:
                markup_children.append(tags["unknown"](_wrap(buffer[token.mask])))
        elif token.type is cmd.TOKEN_TYPE.COMMAND:
            markup_children.append(tags["cmd"](_wrap(buffer[token.mask])))
        elif token.type is cmd.TOKEN_TYPE.KEYWORD:
            markup_children.append(tags["kw"](_wrap(buffer[token.mask])))
        elif token.type is cmd.TOKEN_TYPE.ARGUMENT:
            markup_children.append(tags["arg"](_wrap(buffer[token.mask])))
        else:
            assert False

    else:
        markup_children.append(mu.Group(_wrap(buffer[prev_index:])))

        # markup typeahead
        markup_children.append(tags["typeahead"]((mu.Text(typeahead),)))

    return mu.Group(tuple(markup_children))


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


@dataclasses.dataclass(frozen=True)
class CaretPlaceholder(mu.Pair):
    name = "caret"


@dataclasses.dataclass
class CaretWidgetSettings:
    r"""
    Fields
    ------
    normal_appearance : str
        The markup template of the normal-style caret.
    blinking_appearance : str
        The markup template of the blinking-style caret.
    blinking_appearance : str
        The markup template of the highlighted-style caret.
    blink_ratio : float
        The ratio to blink.
    """
    normal_appearance: str = "[slot/]"
    blinking_appearance: str = "[weight=dim][invert][slot/][/][/]"
    highlighted_appearance: str = "[weight=bold][invert][slot/][/][/]"
    blink_ratio: float = 0.3


class CaretWidget:
    def __init__(self, settings):
        self.settings = settings

    def load(self, provider):
        rich = provider.get(mu.RichParser)
        metronome = provider.get(engines.Metronome)

        caret_blink_ratio = self.settings.blink_ratio
        normal = rich.parse(self.settings.normal_appearance, slotted=True)
        blinking = rich.parse(self.settings.blinking_appearance, slotted=True)
        highlighted = rich.parse(self.settings.highlighted_appearance, slotted=True)

        def caret_node(arg):
            time, key_pressed_time = arg
            beat = metronome.beat(time)
            key_pressed_beat = metronome.beat(key_pressed_time) // -1 * -1
            # don't blink while key pressing
            if beat < key_pressed_beat or beat % 1 < caret_blink_ratio:
                if beat % 4 < 1:
                    return highlighted
                else:
                    return blinking
            else:
                return normal

        return caret_node


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
    caret = CaretWidgetSettings

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
        elif isinstance(widget_settings, BeatshellWidgetFactory.caret):
            return CaretWidget(widget_settings).load(self.provider)
        else:
            raise TypeError


BeatshellIconWidgetSettings = Union[
    PatternsWidgetSettings, beatwidgets.MonitorWidgetSettings,
]


# shell
class BeatShellSettings(cfg.Configurable):
    @cfg.subconfig
    class input(cfg.Configurable):
        r"""
        Fields
        ------
        confirm_key : str
            The key for confirming input.
        help_key : str
            The key for help.
        autocomplete_keys : tuple of str and str and str
            The keys for finding the next, previous and canceling suggestions.

        preview_song : bool
            Whether to preview the song when selected.

        history_size : int
            The maximum history size.

        keymap : dict from str to str
            The keymap of beatshell. The key of dict is the keystroke, and the
            value of dict is the action to activate. The format of action is
            just like a normal python code: `input.insert_typeahead() or
            input.move_right()`. The syntax is::

                <function> ::= "input." /(?!_)\w+/ "()"
                <operator> ::= " | " | " & " | " and " | " or "
                <action> ::= (<function> <operator>)* <function>

        """
        confirm_key: str = "Enter"
        help_key: str = "Alt_Enter"
        autocomplete_keys: Tuple[str, str, str] = ("Tab", "Shift_Tab", "Esc")

        preview_song: bool = True

        history_size: int = 500

        keymap: Dict[str, str] = {
            "Backspace": "input.backspace()",
            "Alt_Backspace": "input.delete_backward_token()",
            "Alt_Delete": "input.delete_forward_token()",
            "Delete": "input.delete()",
            "Left": "input.move_left()",
            "Right": "input.insert_typeahead() or input.move_right()",
            "Up": "input.prev()",
            "Down": "input.next()",
            "Home": "input.move_to_start()",
            "End": "input.move_to_end()",
            "Ctrl_Left": "input.move_to_word_start()",
            "Ctrl_Right": "input.move_to_word_end()",
            "Ctrl_Backspace": "input.delete_to_word_start()",
            "Ctrl_Delete": "input.delete_to_word_end()",
            "Esc": "input.cancel_typeahead() | input.cancel_hint()",
            "'\\x04'": "input.delete() or input.exit_if_empty()",
        }

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
        input_margin : int
            The width of margin of input field.

        icons : BeatShellIconWidgetSettings
            The appearances of icon.
        marker : MarkerWidgetSettings
            The appearance of marker.
        caret : CaretWidgetSettings
            The appearance of caret.
        """
        t0: float = 0.0
        tempo: float = 130.0

        icon_width: int = 5
        marker_width: int = 2
        input_margin: int = 3

        icons: BeatshellIconWidgetSettings = PatternsWidgetSettings()
        marker: MarkerWidgetSettings = MarkerWidgetSettings()
        caret: CaretWidgetSettings = CaretWidgetSettings()

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
        desc_message : str
            The markup template for the desc message.
        info_message : str
            The markup template for the info message.
        message_max_lines : int
            The maximum number of lines of the message.

        suggestions_lines : int
            The maximum number of lines of the suggestions.
        suggestion_items : tuple of str and str
            The markup templates for the unselected/selected suggestion.
        """
        desc_message: str = "[weight=dim][slot/][/]"
        info_message: str = f"{'─'*80}\n[slot/]\n{'─'*80}"
        message_max_lines: int = 16

        suggestions_lines: int = 8
        suggestion_items: Tuple[str, str] = ("• [slot/]", "• [invert][slot/][/]")

    debug_monitor: bool = False


class Hint:
    pass


@dataclasses.dataclass(frozen=True)
class DescHint(Hint):
    message: str


@dataclasses.dataclass(frozen=True)
class InfoHint(Hint):
    message: str


@dataclasses.dataclass(frozen=True)
class SuggestionsHint(Hint):
    suggestions: List[str]
    selected: int
    message: str


class Result:
    pass


@dataclasses.dataclass(frozen=True)
class ErrorResult(Result):
    error: Exception


@dataclasses.dataclass(frozen=True)
class CompleteResult(Result):
    command: Callable


@dataclasses.dataclass
class HintState:
    index: int
    hint: Hint
    tokens: Optional[List[str]]


@dataclasses.dataclass
class TabState:
    suggestions: List[str]
    sugg_index: int
    token_index: int
    original_token: List[str]
    original_pos: int
    selection: slice


class ShellSyntaxError(Exception):
    pass


def onstate(*states):
    def onstate_dec(func):
        @functools.wraps(func)
        def onstate_func(self, *args, **kwargs):
            if self.state not in states:
                return False
            return func(self, *args, **kwargs)

        return onstate_func

    return onstate_dec


def locked(func):
    @functools.wraps(func)
    def locked_func(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return locked_func


@dataclasses.dataclass
class HistoryManager:
    history_path: Path
    latest_command: Optional[Tuple[str, str]] = None

    def write_history(self, command_group, command):
        self.history_path.touch()
        command = command.strip()
        if command and command_group and (command_group, command) != self.latest_command:
            open(self.history_path, "a").write(f"\n[{command_group}] {command}")
            self.latest_command = command

    def read_history(self, command_groups, read_size):
        trim_len = 10

        pattern = re.compile(r"\[(\w+)\] (.+)")

        buffers = []
        self.history_path.touch()
        self.latest_command = None
        for command in open(self.history_path):
            command = command.strip()
            match = pattern.fullmatch(command)
            if match:
                self.latest_command = (match.group(1), match.group(2))
                if match.group(1) in command_groups and (not buffers or buffers[-1] != match.group(2)):
                    buffers.append(match.group(2))
            if len(buffers) - read_size > trim_len:
                del buffers[:trim_len]

        return [list(command) for command in buffers[-read_size:]]


class ShellSemanticAnalyzer:
    r"""Sematic analyzer for beatshell.

    Attributes
    ----------
    parser : commands.RootCommandParser
        The root command parser for beatshell.
    tokens : list of ShToken
        The parsed tokens.
    lex_state : SHLEXER_STATE
        The shlexer state.
    group : str or None
        The group name of parsed command.
    result : object or cmd.CommandParseError or cmd.CommandUnfinishError
        The command object or the error.
    length : int
        The parsed length of tokens.
    """

    def __init__(self, parser):
        self.parser = parser
        self.tokens = []
        self.lex_state = SHLEXER_STATE.SPACED
        self.group = None
        self.result = None
        self.length = 0

    def update_parser(self, parser):
        self.parser = parser

    def parse(self, buffer):
        tokenizer = shlexer_tokenize(buffer)

        tokens = []
        while True:
            try:
                token = next(tokenizer)
            except StopIteration as e:
                self.lex_state = e.value
                break

            tokens.append(token)

        types, result = self.parser.parse_command(token.string for token in tokens)
        self.result = result
        self.length = len(types)

        types.extend([None] * (len(tokens) - len(types)))
        self.tokens = [
            dataclasses.replace(token, type=type)
            for token, type in zip(tokens, types)
        ]
        self.group = self.parser.get_group(self.tokens[0].string) if self.tokens else None

    def get_all_groups(self):
        return self.parser.get_all_groups()

    def desc(self, length):
        parents = [token.string for token in self.tokens[:length]]
        return self.parser.desc_command(parents)

    def info(self, length):
        parents = [token.string for token in self.tokens[:length-1]]
        target = self.tokens[length-1].string
        return self.parser.info_command(parents, target)

    def suggest(self, length, target):
        parents = [token.string for token in self.tokens[:length]]
        return self.parser.suggest_command(parents, target)


class BeatInput:
    r"""Input editor for beatshell.

    Attributes
    ----------
    command_parser_getter : function
        The function to produce command parser for beatshell.
    semantic_analyzer : ShellSemanticAnalyzer
        The syntax analyzer.
    preview_handler : function
        A function to preview beatmap.
    rich : markups.RichParser
        The rich parser.
    cache_dir : Path
        The directory of cache data.
    history : HistoryManager
        The input history manager.
    shell_settings : BeatShellSettings
        The shell settings.
    devices_settings : DevicesSettings
        The devices settings.
    buffers : list of list of str
        The editable buffers of input history.
    buffer_index : int
        The negative index of current input buffer.
    buffer : list of str
        The buffer of current input.
    pos : int
        The caret position of input.
    typeahead : str
        The type ahead of input.
    highlighted : int or None
        The index of highlighted token.
    tab_state : TabState or None
        The state of autocomplete.
    hint_state : HintState or None
        The hint state of input.
    popup : queue of DescHint or InfoHint
        The message displayed above the prompt.
    result : Result or None
        The result of input.
    state : str
        The input state.
    modified_counter : int
        The event counter for modifying buffer.
    """

    def __init__(
        self,
        command_parser_getter,
        preview_handler,
        rich,
        cache_dir,
        shell_settings_getter=BeatShellSettings,
        devices_settings_getter=engines.DevicesSettings,
    ):
        r"""Constructor.

        Parameters
        ----------
        command_parser_getter : function
            The function to produce command parser.
        preview_handler : function
        rich : markups.RichParser
        cache_dir : Path
            The directory of cache data.
        shell_settings_getter : BeatShellSettings
            The settings getter of beatshell.
        devices_settings_getter : engines.DevicesSettings
            The settings getter of devices.
        """
        self.command_parser_getter = command_parser_getter
        self.semantic_analyzer = ShellSemanticAnalyzer(None)
        self.preview_handler = preview_handler
        self.rich = rich
        self.cache_dir = cache_dir
        self._shell_settings_getter = shell_settings_getter
        self._devices_settings_getter = devices_settings_getter
        self.history = HistoryManager(self.cache_dir / ".beatshell-history")
        self.buffers = [[]]
        self.buffer_index = -1
        self.buffer = self.buffers[0]
        self.pos = 0
        self.typeahead = ""
        self.highlighted = None
        self.hint_state = None
        self.popup = queue.Queue()
        self.result = None
        self.tab_state = None
        self.state = "FIN"
        self.lock = threading.RLock()
        self.modified_counter = 0
        self.new_session()

    @property
    def shell_settings(self):
        return self._shell_settings_getter()

    @property
    def devices_settings(self):
        return self._devices_settings_getter()

    @dn.datanode
    def prompt(self):
        r"""Start prompt.

        Returns
        -------
        prompt_task : datanodes.DataNode
            The datanode to execute the prompt.
        """
        shell_settings = self.shell_settings
        devices_settings = self.devices_settings

        # engines
        debug_monitor = shell_settings.debug_monitor
        renderer_monitor = (
            engines.Monitor(self.cache_dir / "monitor" / "prompt.csv")
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
        stroke = BeatStroke(self, shell_settings.input)

        # prompt
        t0 = shell_settings.prompt.t0
        tempo = shell_settings.prompt.tempo
        metronome = engines.Metronome(t0, tempo)

        widget_factory = BeatshellWidgetFactory(self.rich, renderer, metronome)

        caret = widget_factory.create(shell_settings.prompt.caret)
        icon = widget_factory.create(shell_settings.prompt.icons)
        marker = widget_factory.create(shell_settings.prompt.marker)

        state = ViewState(self, stroke, metronome)
        textbox = TextBox(self.rich, state, caret, shell_settings)
        msgbox = MsgBox(self.rich, state, shell_settings)
        prompt = BeatPrompt(state, textbox, msgbox, icon, marker, shell_settings)

        # execute
        stroke.register(controller)
        prompt.register(renderer)

        @dn.datanode
        def stop_when(event):
            yield
            yield
            while not event.is_set():
                yield

        yield from dn.pipe(stop_when(state.fin_event), display_task, input_task).join()

    @locked
    @onstate("FIN")
    def new_session(self):
        r"""Start a new session of input.
        """
        self.semantic_analyzer.update_parser(self.command_parser_getter())

        groups = self.semantic_analyzer.get_all_groups()
        history_size = self.shell_settings.input.history_size
        self.buffers = self.history.read_history(groups, history_size)
        self.buffers.append([])
        self.buffer_index = -1

        self.buffer = self.buffers[self.buffer_index]
        self.pos = len(self.buffer)
        self.cancel_typeahead()
        self.update_buffer()
        self.cancel_hint()
        self.clear_result()
        self.state = "EDIT"

    def record_command(self):
        command = "".join(self.buffer).strip()
        self.history.write_history(self.semantic_analyzer.group, command)

    @locked
    @onstate("FIN")
    def prev_session(self):
        r"""Back to previous session of input."""
        self.cancel_typeahead()
        self.update_buffer()
        self.cancel_hint()
        self.clear_result()
        self.state = "EDIT"

    @locked
    @onstate("EDIT")
    def finish(self):
        r"""Finish this session of input.

        Returns
        -------
        succ : bool
        """
        self.state = "FIN"
        return True

    @locked
    def update_buffer(self):
        """Parse syntax.

        Returns
        -------
        succ : bool
        """
        self.semantic_analyzer.parse(self.buffer)
        self.modified_counter += 1
        return True

    @locked
    def show_typeahead(self):
        """Make typeahead.

        Show the possible command you want to type. Only work if the caret is
        at the end of buffer.

        Returns
        -------
        succ : bool
            `False` if unable to complete or the caret is not at the end of
            buffer.
        """
        if self.pos != len(self.buffer):
            self.typeahead = ""
            return False

        if self.semantic_analyzer.lex_state == SHLEXER_STATE.SPACED:
            parents = [token.string for token in self.semantic_analyzer.tokens]
            target = ""
        else:
            parents = [token.string for token in self.semantic_analyzer.tokens[:-1]]
            target = self.semantic_analyzer.tokens[-1].string

        # search history
        length = len(self.buffer)
        for buffer in self.buffers[::-1]:
            if len(buffer) > length and buffer[:length] == self.buffer:
                self.typeahead = "".join(buffer[length:])
                return True

        self.typeahead = ""
        return False

    @locked
    def cancel_typeahead(self):
        """Cancel typeahead.

        Returns
        -------
        succ : bool
        """
        self.typeahead = ""
        return True

    @locked
    def set_result(self, res):
        """Set result.
        Set result of this session.

        Parameters
        ----------
        res : Result
            The result.

        Returns
        -------
        succ : bool
        """
        self.result = res
        return True

    @locked
    def clear_result(self):
        """Clear result.

        Returns
        -------
        succ : bool
        """
        self.result = None
        return True

    @locked
    def set_hint(self, hint, index=None):
        """Set hint.

        Show hint below the prompt.

        Parameters
        ----------
        hint : Hint
            The hint.
        index : int or None
            Index of the token to which the hint is directed, or `None` for
            nothing.

        Returns
        -------
        succ : bool
        """
        self.highlighted = index
        if isinstance(hint, DescHint):
            msg_tokens = (
                [token.string for token in self.semantic_analyzer.tokens[:index]]
                if index is not None
                else None
            )
        elif isinstance(hint, (InfoHint, SuggestionsHint)):
            msg_tokens = (
                [token.string for token in self.semantic_analyzer.tokens[: index + 1]]
                if index is not None
                else None
            )
        else:
            assert False
        self.hint_state = HintState(index, hint, msg_tokens)
        self.update_preview()
        return True

    @locked
    def cancel_hint(self):
        """Cancel hint.

        Remove the hint below the prompt.

        Returns
        -------
        succ : bool
        """
        if self.highlighted is not None:
            self.highlighted = None
        if self.hint_state is not None:
            self.hint_state = None
            self.update_preview()
        return True

    @locked
    def update_hint(self):
        """Update hint.

        Remove hint if the target is updated.

        Returns
        -------
        succ : bool
            `False` if there is no hint or the hint isn't removed.
        """
        if self.hint_state is None:
            return False

        if self.hint_state.tokens is None:
            return self.cancel_hint()

        if self.highlighted is not None and self.highlighted >= len(self.semantic_analyzer.tokens):
            return self.cancel_hint()

        if len(self.hint_state.tokens) > len(self.semantic_analyzer.tokens):
            return self.cancel_hint()

        for token_string, token in zip(self.hint_state.tokens, self.semantic_analyzer.tokens):
            if token_srting != token.string:
                return self.cancel_hint()

        if (
            isinstance(self.hint_state.hint, DescHint)
            and self.semantic_analyzer.tokens[len(self.hint_state.tokens) - 1].type is not None
        ):
            return self.cancel_hint()

        return False

    @locked
    def update_preview(self):
        if not self.shell_settings.input.preview_song:
            return
        if self.hint_state is None:
            self.preview_handler(None)
        elif not isinstance(self.hint_state.hint, (InfoHint, SuggestionsHint)):
            self.preview_handler(None)
        elif (
            isinstance(self.hint_state.hint, SuggestionsHint)
            and not self.hint_state.hint.message
        ):
            self.preview_handler(None)
        elif self.hint_state.tokens is None:
            self.preview_handler(None)
        elif len(self.hint_state.tokens) != 2:
            self.preview_handler(None)
        elif self.hint_state.tokens[0] != "play":
            self.preview_handler(None)
        else:
            self.preview_handler(self.hint_state.tokens[1])

    @locked
    @onstate("EDIT")
    def insert_typeahead(self):
        """Insert typeahead.

        Insert the typeahead if the caret is at the end of buffer.

        Returns
        -------
        succ : bool
            `False` if there is no typeahead or the caret is not at the end of
            buffer.
        """

        if self.typeahead == "" or self.pos != len(self.buffer):
            return False

        self.buffer[self.pos : self.pos] = self.typeahead
        self.pos += len(self.typeahead)
        self.typeahead = ""
        self.update_buffer()
        self.ask_for_hint()

        return True

    @locked
    @onstate("EDIT")
    def insert(self, text):
        """Input.

        Insert some text into the buffer.

        Parameters
        ----------
        text : str
            The text to insert. It shouldn't contain any nongraphic character,
            except for prefix `\\b` which indicate deleting.

        Returns
        -------
        succ : bool
            `False` if buffer isn't changed.
        """
        text = list(text)

        if len(text) == 0:
            return False

        while len(text) > 0 and text[0] == "\b":
            del text[0]
            del self.buffer[self.pos - 1]
            self.pos = self.pos - 1

        if not all(ch.isprintable() for ch in self.buffer):
            raise ValueError("invalid text to insert: " + repr("".join(self.buffer)))

        self.buffer[self.pos : self.pos] = text
        self.pos += len(text)
        self.update_buffer()

        self.show_typeahead()
        self.ask_for_hint()

        return True

    @locked
    @onstate("EDIT")
    def backspace(self):
        """Backspace.

        Delete one character before the caret if exists.

        Returns
        -------
        succ : bool
        """
        if self.pos == 0:
            return False

        self.pos -= 1
        del self.buffer[self.pos]
        self.update_buffer()
        self.cancel_typeahead()
        self.ask_for_hint(clear=True)

        return True

    @locked
    @onstate("EDIT")
    def delete(self):
        """Delete.

        Delete one character after the caret if exists.

        Returns
        -------
        succ : bool
        """
        if self.pos >= len(self.buffer):
            return False

        del self.buffer[self.pos]
        self.update_buffer()
        self.cancel_typeahead()
        self.ask_for_hint(clear=True)

        return True

    @locked
    @onstate("EDIT")
    def delete_all(self):
        """Delete All.

        Returns
        -------
        succ : bool
        """
        if not self.buffer:
            return False

        del self.buffer[:]
        self.pos = 0
        self.update_buffer()
        self.cancel_typeahead()
        self.cancel_hint()

        return True

    @locked
    @onstate("EDIT")
    def delete_range(self, start, end):
        """Delete range.

        Parameters
        ----------
        start : int or None
        end : int or None

        Returns
        -------
        succ : bool
        """
        start = min(max(0, start), len(self.buffer)) if start is not None else 0
        end = (
            min(max(0, end), len(self.buffer)) if end is not None else len(self.buffer)
        )

        if start >= end:
            return False

        del self.buffer[start:end]
        self.pos = start
        self.update_buffer()
        self.cancel_typeahead()
        self.ask_for_hint(clear=True)

        return True

    @locked
    @onstate("EDIT")
    def delete_backward_token(self, index=None):
        """Delete current or backward token.

        Parameters
        ----------
        index : int or None

        Returns
        -------
        succ : bool
        """
        if index is not None:
            token = self.semantic_analyzer.tokens[index]
            return self.delete_range(token.mask.start, token.mask.stop)

        for token in reversed(self.semantic_analyzer.tokens):
            if token.mask.start <= self.pos:
                return self.delete_range(token.mask.start, max(self.pos, token.mask.stop))
        else:
            # find nothing
            return self.delete_range(0, self.pos)

    @locked
    @onstate("EDIT")
    def delete_forward_token(self, index=None):
        """Delete current or forward token.

        Parameters
        ----------
        index : int or None

        Returns
        -------
        succ : bool
        """
        if index is not None:
            token = self.semantic_analyzer.tokens[index]
            return self.delete_range(token.mask.start, token.mask.stop)

        for token in self.semantic_analyzer.tokens:
            if self.pos <= token.mask.stop:
                return self.delete_range(min(self.pos, token.mask.start), token.mask.stop)
        else:
            # find nothing
            return self.delete_range(self.pos, None)

    @locked
    @onstate("EDIT")
    def delete_to_word_start(self):
        """Delete to the word start.

        The word is defined as `\\w+|\\W+`.

        Returns
        -------
        succ : bool
        """
        for match in re.finditer(r"\w+|\W+", "".join(self.buffer)):
            if match.end() >= self.pos:
                return self.delete_range(match.start(), self.pos)
        else:
            return self.delete_range(None, self.pos)

    @locked
    @onstate("EDIT")
    def delete_to_word_end(self):
        """Delete to the word end.

        The word is defined as `\\w+|\\W+`.

        Returns
        -------
        succ : bool
        """
        for match in re.finditer(r"\w+|\W+", "".join(self.buffer)):
            if match.end() > self.pos:
                return self.delete_range(self.pos, match.end())
        else:
            return self.delete_range(self.pos, None)

    @locked
    @onstate("EDIT")
    def move_to(self, pos):
        """Move caret to the specific position.

        Regardless of success or failure, typeahead will be cancelled.

        Parameters
        ----------
        pos : int or None
            Index of buffer, which will be clamped to 0 and length of buffer, or
            `None` for the end of buffer.

        Returns
        -------
        succ : bool
        """
        pos = (
            min(max(0, pos), len(self.buffer)) if pos is not None else len(self.buffer)
        )
        self.cancel_typeahead()

        if self.pos == pos:
            return False

        self.pos = pos
        return True

    @locked
    @onstate("EDIT")
    def move(self, offset):
        """Move caret.

        Parameters
        ----------
        offset : int

        Returns
        -------
        succ : bool
        """
        return self.move_to(self.pos + offset)

    @locked
    @onstate("EDIT")
    def move_left(self):
        """Move caret one character to the left.

        Returns
        -------
        succ : bool
        """
        return self.move(-1)

    @locked
    @onstate("EDIT")
    def move_right(self):
        """Move caret one character to the right.

        Returns
        -------
        succ : bool
        """
        return self.move(+1)

    @locked
    @onstate("EDIT")
    def move_to_start(self):
        """Move caret to the start of buffer.

        Returns
        -------
        succ : bool
        """
        return self.move_to(0)

    @locked
    @onstate("EDIT")
    def move_to_end(self):
        """Move caret to the end of buffer.

        Returns
        -------
        succ : bool
        """
        return self.move_to(None)

    @locked
    @onstate("EDIT")
    def move_to_word_start(self):
        """Move caret to the start of the word.

        Returns
        -------
        succ : bool
        """
        for match in re.finditer(r"\w+|\W+", "".join(self.buffer)):
            if match.end() >= self.pos:
                return self.move_to(match.start())
        else:
            return self.move_to(0)

    @locked
    @onstate("EDIT")
    def move_to_word_end(self):
        """Move caret to the end of the word.

        Returns
        -------
        succ : bool
        """
        for match in re.finditer(r"\w+|\W+", "".join(self.buffer)):
            if match.end() > self.pos:
                return self.move_to(match.end())
        else:
            return self.move_to(None)

    @locked
    @onstate("EDIT")
    def prev(self):
        """Previous buffer.

        Returns
        -------
        succ : bool
        """
        if self.buffer_index == -len(self.buffers):
            return False
        self.buffer_index -= 1

        self.buffer = self.buffers[self.buffer_index]
        self.pos = len(self.buffer)
        self.update_buffer()
        self.cancel_typeahead()
        self.cancel_hint()

        return True

    @locked
    @onstate("EDIT")
    def next(self):
        """Next buffer.

        Returns
        -------
        succ : bool
        """
        if self.buffer_index == -1:
            return False
        self.buffer_index += 1

        self.buffer = self.buffers[self.buffer_index]
        self.pos = len(self.buffer)
        self.update_buffer()
        self.cancel_typeahead()
        self.cancel_hint()

        return True

    @locked
    @onstate("EDIT")
    def ask_for_hint(self, index=None, clear=False):
        """Ask some hint for command.

        Provide some hint for the command on the caret.

        Parameters
        ----------
        index : int
        clear : bool
            Cancel the current hint if no hint was found.

        Returns
        -------
        succ : bool
        """
        if index is None:
            # find the token on the caret
            for index, token in enumerate(self.semantic_analyzer.tokens):
                if token.mask.start <= self.pos <= token.mask.stop:
                    break
            else:
                # find nothing
                if clear:
                    self.cancel_hint()
                return False

        target_type = self.semantic_analyzer.tokens[index].type

        if target_type is None:
            msg = self.semantic_analyzer.desc(index)
            if msg is None:
                self.cancel_hint()
                return False
            hint = DescHint(msg)
            self.set_hint(hint, index)
            return True

        else:
            msg = self.semantic_analyzer.info(index+1)
            if msg is None:
                self.cancel_hint()
                return False
            hint = InfoHint(msg)
            self.set_hint(hint, index)
            return True

    @locked
    @onstate("EDIT")
    def help(self):
        """Help for command.

        Print some hint for the command before the caret.

        Returns
        -------
        succ : bool
        """
        # find the token before the caret
        for index, token in reversed(list(enumerate(self.semantic_analyzer.tokens))):
            if token.mask.start <= self.pos:
                break
        else:
            return False

        if self.hint_state is None or self.hint_state.index != index:
            self.ask_for_hint(index)
            return False

        hint = self.hint_state.hint
        if not hint.message:
            return False
        if isinstance(hint, SuggestionsHint):
            hint = InfoHint(hint.message)

        self.popup.put(hint)
        return True

    @locked
    @onstate("EDIT")
    def confirm(self):
        """Finish the command.

        Returns
        -------
        succ : bool
            `False` if the command is wrong.
        """
        self.cancel_hint()

        if not self.semantic_analyzer.tokens:
            self.set_result(CompleteResult(lambda: None))
            self.finish()
            return True

        if self.semantic_analyzer.lex_state == SHLEXER_STATE.BACKSLASHED:
            res, index = ShellSyntaxError("No escaped character"), len(self.semantic_analyzer.tokens) - 1
        elif self.semantic_analyzer.lex_state == SHLEXER_STATE.QUOTED:
            res, index = ShellSyntaxError("No closing quotation"), len(self.semantic_analyzer.tokens) - 1
        else:
            res, index = self.semantic_analyzer.result, self.semantic_analyzer.length

        if isinstance(res, cmd.CommandUnfinishError):
            self.set_result(ErrorResult(res))
            self.finish()
            return False
        elif isinstance(res, (cmd.CommandParseError, ShellSyntaxError)):
            self.set_result(ErrorResult(res))
            self.highlighted = index
            self.finish()
            return False
        else:
            self.set_result(CompleteResult(res))
            self.finish()
            return True

    @locked
    @onstate("EDIT")
    def exit_if_empty(self):
        """Finish the command.

        Returns
        -------
        succ : bool
            `False` if the command is wrong.
        """
        if self.buffer:
            return False

        self.insert("bye")
        return self.confirm()

    @locked
    @onstate("EDIT")
    def autocomplete(self, action=+1):
        """Autocomplete.

        Complete the token on the caret, or fill in suggestions if caret is
        located in between.

        Parameters
        ----------
        action : +1 or -1 or 0
            Indicating direction for exploration of suggestions. `+1` for next
            suggestion; `-1` for previous suggestion; `0` for canceling the
            process.

        Returns
        -------
        succ : bool
            `False` for canceling the process.
        """

        if self.tab_state is None and action == 0:
            return False

        if self.tab_state is None:
            self.cancel_typeahead()

            # find the token to autocomplete
            token_index = 0
            target = ""
            selection = slice(self.pos, self.pos)
            for token in self.semantic_analyzer.tokens:
                start, stop, _ = token.mask.indices(len(self.buffer))
                if stop < self.pos:
                    token_index += 1
                if start <= self.pos <= stop:
                    target = token.string
                    selection = token.mask

            # generate suggestions
            suggestions = [
                shlexer_quoting(sugg)
                for sugg in self.semantic_analyzer.suggest(token_index, target)
            ]
            sugg_index = len(suggestions) if action == -1 else -1

            if len(suggestions) == 0:
                # no suggestion
                return False

            if len(suggestions) == 1:
                # one suggestion -> complete directly
                self.buffer[selection] = suggestions[0]
                self.pos = selection.start + len(suggestions[0])
                self.update_buffer()
                target_type = self.semantic_analyzer.tokens[token_index].type
                if target_type is None:
                    return False
                msg = self.semantic_analyzer.info(token_index + 1)
                if msg is None:
                    return False
                hint = InfoHint(msg)
                self.set_hint(hint, token_index)
                return False

            # tab state
            original_pos = self.pos

            self.tab_state = TabState(
                suggestions=suggestions,
                sugg_index=sugg_index,
                token_index=token_index,
                original_token=self.buffer[selection],
                original_pos=original_pos,
                selection=selection,
            )

        sugg_index = self.tab_state.sugg_index
        selection = self.tab_state.selection
        suggestions = self.tab_state.suggestions

        if action == +1:
            sugg_index += 1
        elif action == -1:
            sugg_index -= 1
        elif action == 0:
            sugg_index = None
        else:
            raise ValueError(f"invalid action: {action}")

        if sugg_index not in range(len(suggestions)):
            # restore state
            self.buffer[selection] = self.tab_state.original_token
            self.pos = self.tab_state.original_pos

            self.tab_state = None
            self.update_buffer()
            self.cancel_hint()
            return False

        assert sugg_index is not None

        # autocomplete selected token
        self.tab_state.sugg_index = sugg_index
        self.buffer[selection] = suggestions[sugg_index]
        self.pos = selection.start + len(suggestions[sugg_index])
        self.tab_state.selection = slice(selection.start, self.pos)

        self.update_buffer()
        target_type = self.semantic_analyzer.tokens[self.tab_state.token_index].type
        if target_type is None:
            msg = ""
        else:
            msg = self.semantic_analyzer.info(self.tab_state.token_index + 1) or ""
        self.set_hint(
            SuggestionsHint(suggestions, sugg_index, msg), self.tab_state.token_index
        )
        return True

    @locked
    @onstate("EDIT")
    def finish_autocomplete(self):
        r"""Finish autocompletion.

        Returns
        -------
        succ : bool
        """
        if self.tab_state is not None:
            # set hint for complete token
            token_index = self.tab_state.token_index
            self.tab_state = None

            target_type = self.semantic_analyzer.tokens[token_index].type
            if target_type is None:
                self.cancel_hint()
                return True
            msg = self.semantic_analyzer.info(token_index+1)
            if msg is None:
                self.cancel_hint()
                return True
            hint = InfoHint(msg)
            self.set_hint(hint, token_index)
        return True

    @locked
    def unknown_key(self, key):
        self.cancel_hint()
        self.set_result(ErrorResult(ValueError(f"Unknown key: " + key)))
        self.finish()


class BeatStroke:
    r"""Keyboard controller for beatshell."""

    def __init__(self, input, settings):
        self.input = input
        self.settings = settings
        self.key_event = 0

    def register(self, controller):
        r"""Register handler to the given controller.

        Parameters
        ----------
        controller : engines.Controller
        """
        controller.add_handler(self.keypress_handler())

        controller.add_handler(
            self.autocomplete_handler(
                self.settings.autocomplete_keys, self.settings.help_key
            )
        )
        controller.add_handler(self.printable_handler())

        for key, func in self.settings.keymap.items():
            controller.add_handler(self.action_handler(func), key)

        controller.add_handler(self.help_handler(), self.settings.help_key)
        controller.add_handler(self.confirm_handler(), self.settings.confirm_key)
        controller.add_handler(self.unknown_handler(self.settings))

    def keypress_handler(self):
        def keypress(_):
            self.key_event += 1

        return keypress

    def confirm_handler(self):
        return lambda _: self.input.confirm()

    def help_handler(self):
        return lambda _: self.input.help()

    def autocomplete_handler(self, keys, help_key):
        next_key, prev_key, cancel_key = keys

        def handler(args):
            _, time, keyname, keycode = args
            if keyname == next_key:
                self.input.autocomplete(+1)
            elif keyname == prev_key:
                self.input.autocomplete(-1)
            elif keyname == cancel_key:
                self.input.autocomplete(0)
            elif keyname != help_key:
                self.input.finish_autocomplete()

        return handler

    def action_handler(self, func):
        fn = r"input\.(?!_)\w+\(\)"
        op = "(%s)" % "|".join(map(re.escape, (" | ", " & ", " and ", " or ")))
        regex = f"({fn}{op})*{fn}"
        if not re.match(regex, func):
            raise ValueError(f"invalid action: {repr(func)}")
        return lambda _: eval(func, {}, {"input": self.input})

    def printable_handler(self):
        def handler(args):
            _, time, keyname, keycode = args
            if keycode.isprintable():
                self.input.insert(keycode)

        return handler

    def unknown_handler(self, settings):
        keys = list(settings.keymap.keys())
        keys.append(settings.confirm_key)
        keys.append(settings.help_key)
        keys.extend(settings.autocomplete_keys)

        def handler(args):
            _, _, key, code = args
            if key not in keys and not code.isprintable():
                self.input.unknown_key(key)

        return handler


@dn.datanode
def starcache(func, key_func=lambda *a: a):
    data = yield
    key = key_func(*data)
    value = func(*data)

    while True:
        data = yield value
        key_ = key_func(*data)
        if key == key_:
            continue
        key = key_
        value = func(*data)


class BeatPrompt:
    r"""Prompt renderer for beatshell."""

    def __init__(self, state, textbox, msgbox, icon, marker, settings):
        self.state = state
        self.textbox = textbox
        self.msgbox = msgbox
        self.icon = icon
        self.marker = marker
        self.settings = settings

    def register(self, renderer):
        icon_width = self.settings.prompt.icon_width
        marker_width = self.settings.prompt.marker_width
        input_margin = self.settings.prompt.input_margin

        [
            icon_mask,
            marker_mask,
            input_mask,
        ] = beatwidgets.layout([icon_width, marker_width, -1])

        renderer.add_drawer(self.state.load(), zindex=())
        renderer.add_texts(self.icon, icon_mask, zindex=(2,))
        renderer.add_texts(self.marker, marker_mask, zindex=(3,))
        renderer.add_texts(self.textbox.load(), input_mask, zindex=(0,))
        renderer.add_drawer(self.msgbox.load(), zindex=(1,))


class ViewState:
    def __init__(self, input, stroke, metronome):
        r"""Constructor.

        Parameters
        ----------
        input : BeatInput
        stroke : BeatStroke
        metronome : engines.Metronome
        """
        self.input = input
        self.stroke = stroke
        self.metronome = metronome

        # input state
        self.fin_event = threading.Event()
        self.key_pressed_time = 0.0
        self.buffer = []
        self.tokens = []
        self.pos = 0
        self.highlighted = None
        self.typeahead = ""
        self.clean = False
        self.hint = None
        self.state = "EDIT"

    @dn.datanode
    def load(self):
        modified_counter = None
        res, time, width = yield
        key_event = self.stroke.key_event
        self.key_pressed_time = time - 100.0

        while True:
            with self.input.lock:
                if self.input.modified_counter != modified_counter:
                    modified_counter = self.input.modified_counter
                    self.buffer = list(self.input.buffer)
                    self.tokens = list(self.input.semantic_analyzer.tokens)
                self.pos = self.input.pos
                self.highlighted = self.input.highlighted

                self.typeahead = self.input.typeahead
                self.clean = self.input.result is not None
                self.hint = (
                    self.input.hint_state.hint
                    if self.input.hint_state is not None
                    else None
                )
                self.popup = []
                while True:
                    try:
                        msg = self.input.popup.get(False)
                    except queue.Empty:
                        break
                    self.popup.append(msg)
                self.state = self.input.state

            if self.stroke.key_event != key_event:
                key_event = self.stroke.key_event
                self.key_pressed_time = time

            res, time, width = yield res

            # fin
            if self.state == "FIN" and not self.fin_event.is_set():
                self.fin_event.set()


class MsgBox:
    def __init__(self, rich, state, settings):
        r"""Constructor.

        Parameters
        ----------
        rich : markups.RichParser
        state : ViewState
        settings : BeatShellSettings
        """
        self.rich = rich
        self.state = state
        self.settings = settings

    @dn.datanode
    def load(self):
        hint_node = starcache(self.markup_hint, lambda msgs, hint: hint)
        popup_node = starcache(self.markup_popup, lambda logs, popup: popup)
        with hint_node, popup_node:
            (view, msgs, logs), time, width = yield
            while True:
                msgs = hint_node.send((msgs, self.state.hint))
                logs = popup_node.send((logs, self.state.popup))
                (view, msgs, logs), time, width = yield (view, msgs, logs)

    def markup_hint(self, msgs, hint):
        r"""Render hint.

        Parameters
        ----------
        msgs : list of Markup
            The rendered hint.
        hint : Hint
        """
        message_max_lines = self.settings.text.message_max_lines

        sugg_lines = self.settings.text.suggestions_lines
        sugg_items = self.settings.text.suggestion_items

        sugg_items = (
            self.rich.parse(sugg_items[0], slotted=True),
            self.rich.parse(sugg_items[1], slotted=True),
        )

        desc = self.rich.parse(self.settings.text.desc_message, slotted=True)
        info = self.rich.parse(self.settings.text.info_message, slotted=True)

        msgs.clear()

        # draw hint
        if hint is None:
            return msgs

        msg = None
        if hint.message:
            msg = self.rich.parse(hint.message, root_tag=True)
            lines = 0

            def trim_lines(text):
                nonlocal lines
                if lines >= message_max_lines:
                    return ""
                res_string = []
                for ch in text.string:
                    if ch == "\n":
                        lines += 1
                    res_string.append(ch)
                    if lines == message_max_lines:
                        res_string.append("…")
                        break
                return mu.Text("".join(res_string))

            msg = msg.traverse(mu.Text, trim_lines)

            if isinstance(hint, DescHint):
                msg = mu.replace_slot(desc, msg)
            elif isinstance(hint, (InfoHint, SuggestionsHint)):
                msg = mu.replace_slot(info, msg)
            else:
                assert False
            msg = msg.expand()

        if isinstance(hint, SuggestionsHint):
            sugg_start = hint.selected // sugg_lines * sugg_lines
            sugg_end = sugg_start + sugg_lines
            suggs = hint.suggestions[sugg_start:sugg_end]
            if sugg_start > 0:
                msgs.append(mu.Text("…\n"))
            for i, sugg in enumerate(suggs):
                sugg = mu.Text(sugg)
                if i == hint.selected - sugg_start:
                    sugg = mu.replace_slot(sugg_items[1], sugg)
                else:
                    sugg = mu.replace_slot(sugg_items[0], sugg)
                msgs.append(sugg)
                if i == hint.selected - sugg_start and msg is not None:
                    msgs.append(mu.Text("\n"))
                    msgs.append(msg)
                if i != len(suggs) - 1:
                    msgs.append(mu.Text("\n"))
            if sugg_end < len(hint.suggestions):
                msgs.append(mu.Text("\n…"))
            msgs.append(self.rich.parse("\n"))

        else:
            if msg is not None:
                msgs.append(msg)

        return msgs

    def markup_popup(self, logs, popup):
        r"""Render popup.

        Parameters
        ----------
        logs : list of Markup
            The rendered popup.
        popup : list of DescHint or InfoHint
        """
        desc = self.rich.parse(self.settings.text.desc_message, slotted=True)
        info = self.rich.parse(self.settings.text.info_message, slotted=True)

        # draw popup
        for hint in popup:
            msg = None
            if hint.message:
                msg = self.rich.parse(hint.message, root_tag=True)

                if isinstance(hint, DescHint):
                    msg = mu.replace_slot(desc, msg)
                elif isinstance(hint, InfoHint):
                    msg = mu.replace_slot(info, msg)
                else:
                    assert False

                msg = mu.Group((msg, mu.Text("\n")))
                msg = msg.expand()

            if msg is not None:
                logs.append(msg)

        return logs


class TextBox:
    def __init__(self, rich, state, caret, settings):
        r"""Constructor.

        Parameters
        ----------
        rich : markups.RichParser
        state : ViewState
        caret : CaretWidget
        settings : BeatShellSettings
        """
        self.rich = rich
        self.state = state
        self.caret = caret
        self.settings = settings

        self.textbox_pipeline = dn.DynamicPipeline()

    def load(self):
        input_margin = self.settings.prompt.input_margin
        ellipses = ("…", "…")

        self.textbox_pipeline.add_node(self.adjust_text_offset(input_margin), zindex=(0,))
        self.textbox_pipeline.add_node(self.text_handler(), zindex=(1,))
        self.textbox_pipeline.add_node(self.overflow_handler(ellipses), zindex=(2,))

        return self.textbox_node(self.textbox_pipeline)

    @staticmethod
    @dn.datanode
    def textbox_node(pipeline):
        time, ran = yield
        with pipeline:
            while True:
                texts = pipeline.send(([], time, ran))
                time, ran = yield texts

    @dn.datanode
    def overflow_handler(self, ellipses):
        left_ellipsis = self.rich.parse(ellipses[0])
        left_ellipsis_width = self.rich.widthof(ellipses[0])
        right_ellipsis = self.rich.parse(ellipses[1])
        right_ellipsis_width = self.rich.widthof(ellipses[1])

        texts, time, ran = yield
        while True:
            if self.left_overflow:
                texts.append((0, left_ellipsis))
            if self.right_overflow:
                texts.append((len(ran) - right_ellipsis_width, right_ellipsis))
            texts, time, ran = yield texts

    def text_geometry(self, buffer, typeahead, pos):
        text_width = self.rich.widthof(buffer)
        typeahead_width = self.rich.widthof(typeahead)
        caret_dis = self.rich.widthof(buffer[:pos])
        return text_width, typeahead_width, caret_dis

    @dn.datanode
    def adjust_text_offset(self, text_margin):
        self.text_offset = 0
        self.left_overflow = False
        self.right_overflow = False

        geo_key = lambda buffer, typeahead, pos: (id(buffer), typeahead, pos)
        geo_node = starcache(self.text_geometry, geo_key)

        with geo_node:
            texts, time, ran = yield
            while True:
                typeahead = self.state.typeahead if not self.state.clean else ""
                text_width, typeahead_width, caret_dis = geo_node.send(
                    (self.state.buffer, typeahead, self.state.pos)
                )
                box_width = len(ran)

                # adjust text offset
                if text_width - self.text_offset < box_width - 1 - text_margin:
                    # from: ......[....I...    ]
                    #   to: ...[.......I... ]
                    self.text_offset = max(
                        0, text_width - box_width + 1 + text_margin
                    )
                if caret_dis - self.text_offset >= box_width - text_margin:
                    # from: ...[............]..I....
                    #   to: ........[..........I.]..
                    self.text_offset = caret_dis - box_width + text_margin + 1
                elif caret_dis - self.text_offset - text_margin < 0:
                    # from: .....I...[............]...
                    #   to: ...[.I..........].........
                    self.text_offset = max(caret_dis - text_margin, 0)

                # determine overflow
                self.left_overflow = self.text_offset > 0
                self.right_overflow = (
                    text_width + typeahead_width - self.text_offset > box_width - 1
                )

                texts, time, ran = yield texts

    def markup_syntax(self, buffer, tokens, typeahead):
        r"""Markup syntax of input text.

        Parameters
        ----------
        buffer : list of str
        tokens : list of ShToken
        typeahead : str

        Returns
        -------
        markup : markups.Markup
            The syntax highlighted input text.
        """

        # markup tokens
        return shlexer_markup(buffer, tokens, typeahead, self.rich.tags)

    def decorate_tokens(self, markup, pos, highlighted, clean):
        # markup caret
        i = 0
        for n, token in enumerate(markup.children):
            for m, subword in enumerate(token.children):
                l = len(subword.string) if isinstance(subword, mu.Text) else 1
                if pos >= i + l:
                    i += l
                    continue

                if isinstance(subword, mu.Text):
                    subwords = (
                        mu.Text(subword.string[: pos - i]),
                        CaretPlaceholder((mu.Text(subword.string[pos - i]),)),
                        mu.Text(subword.string[pos - i + 1 :]),
                    )
                else:
                    subwords = (CaretPlaceholder((subword,)),)

                token = dataclasses.replace(
                    token,
                    children=token.children[:m] + subwords + token.children[m + 1 :],
                )
                markup = dataclasses.replace(
                    markup,
                    children=markup.children[:n] + (token,) + markup.children[n + 1 :],
                )
                break
            else:
                continue
            break

        # unfinished -> unknown
        if clean and len(markup.children) >= 3:
            n = -3
            token = markup.children[n]
            if isinstance(token, self.rich.tags["unfinished"]):
                token = self.rich.tags["unknown"](token.children)
                markup = dataclasses.replace(
                    markup,
                    children=markup.children[:n] + (token,) + markup.children[n + 1 :],
                )

        # highlight
        if highlighted is not None:
            n = highlighted * 2 + 1
            token = markup.children[n]
            token = self.rich.tags["highlight"]((token,))
            markup = dataclasses.replace(
                markup,
                children=markup.children[:n] + (token,) + markup.children[n + 1 :],
            )

        markup = markup.expand()
        return markup

    def render_caret(self, markup, caret):
        if caret is not None:
            return markup.traverse(
                CaretPlaceholder, lambda m: mu.replace_slot(caret, mu.Group(m.children))
            )
        else:
            return markup.traverse(CaretPlaceholder, lambda m: mu.Group(m.children))

    @dn.datanode
    def text_handler(self):
        syntax_key = lambda buffer, tokens, typeahead: (id(buffer), typeahead)
        syntax_node = starcache(self.markup_syntax, syntax_key)

        dec_key = lambda markup, pos, highlighted, clean: (
            id(markup),
            pos,
            highlighted,
            clean,
        )
        dec_node = starcache(self.decorate_tokens, dec_key)

        caret_widget = dn.DataNode.wrap(self.caret)
        caret_node = starcache(self.render_caret)

        with syntax_node, dec_node, caret_node, caret_widget:
            texts, time, ran = yield
            while True:
                typeahead = self.state.typeahead if not self.state.clean else ""

                markup = syntax_node.send((self.state.buffer, self.state.tokens, typeahead))
                markup = dec_node.send((markup, self.state.pos, self.state.highlighted, self.state.clean))

                caret = caret_widget.send((time, self.state.key_pressed_time))
                if self.state.clean:
                    caret = None

                markup = caret_node.send((markup, caret))

                texts.append((-self.text_offset, markup))
                texts, time, ran = yield texts

