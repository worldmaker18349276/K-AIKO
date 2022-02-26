from enum import Enum
import functools
import re
import threading
from typing import Optional, List, Tuple, Dict, Callable
import dataclasses
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

def shlexer_tokenize(raw):
    r"""Tokenizer for shell-like grammar.
    The delimiter is just whitespace, and the token is defined as::

        <nonspace-character> ::= /[^ \\\']/
        <backslashed-character> ::= "\" /./
        <quoted-string> ::= "'" /[^']*/ "'"
        <token> ::= ( <nonspace-character> | <backslashed-character> | <quoted-string> )*

    The backslashes and quotation marks used for escaping will be deleted after being interpreted as a string.
    The input string should be printable, so it doesn't contain tab, newline, backspace, etc.
    In this grammar, the token of an empty string can be expressed as `''`.

    Parameters
    ----------
    raw : str or list of str
        The string to tokenize, which should be printable.

    Yields
    ------
    token : str
        The tokenized string.
    mask : slice
        The position of this token.
    quotes : list of int
        The indices of all backslashes and quotation marks used for escaping.
        The token is equal to `''.join(raw[i] for i in range(*slice.indices(len(raw))) if i not in quotes)`.

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
                yield "".join(token), slice(start, index), quotes
                break

            elif char == BACKSLASH:
                # escape the next character
                quotes.append(index)

                try:
                    index, char = next(raw)
                except StopIteration:
                    yield "".join(token), slice(start, length), quotes
                    return SHLEXER_STATE.BACKSLASHED

                token.append(char)

            elif char == QUOTE:
                # escape the following characters until the next quotation mark
                quotes.append(index)

                while True:
                    try:
                        index, char = next(raw)
                    except StopIteration:
                        yield "".join(token), slice(start, length), quotes
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
                yield "".join(token), slice(start, length), quotes
                return SHLEXER_STATE.PLAIN

def shlexer_quoting(compreply, state=SHLEXER_STATE.SPACED):
    r"""Escape a given string so that it can be inserted into an untokenized string.
    The strategy to escape insert string only depends on the state of insert position.

    Parameters
    ----------
    compreply : str
        The string to insert.  The suffix `'\000'` indicate closing the token.
        But inserting `'\000'` after backslash results in `''`, since it is impossible to close it.
    state : SHLEXER_STATE
        The state of insert position.

    Returns
    -------
    raw : str
        The escaped string which can be inserted into untokenized string directly.
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
            raw = compreply[:-1].replace("'", r"'\''") + (r"'\'" if compreply[-1] == "'" else compreply[-1] + "'")

    elif state == SHLEXER_STATE.SPACED:
        if compreply != "" and " " not in compreply:
            # use backslash if there is no whitespace
            raw = re.sub(r"([ \\'])", r"\\\1", compreply)
        elif partial:
            raw = "'" + compreply.replace("'", r"'\''")
        elif compreply == "":
            raw = "''"
        else:
            raw = "'" + compreply[:-1].replace("'", r"'\''") + (r"'\'" if compreply[-1] == "'" else compreply[-1] + "'")

    else:
        assert False

    return raw if partial else raw + " "

def shlexer_markup(buffer, tokens, typeahead, tags):
    r"""Markup shlex.

    Parameters
    ----------
    buffer : list of str
    tokens : list
    typeahead : str

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

    for _, type, mask, quotes in tokens:
        # markup whitespace
        for index in range(mask.start, mask.stop):
            if buffer[index] == " ":
                buffer[index] = tags['ws']()

        # markup escape
        for index in quotes:
            if buffer[index] == "'":
                buffer[index] = tags['qt']()
            elif buffer[index] == "\\":
                buffer[index] = tags['bs']()
            else:
                assert False

    markup_children = []
    prev_index = 0
    for n, (_, type, mask, _) in enumerate(tokens):
        # markup delimiter
        markup_children.append(mu.Group(_wrap(buffer[prev_index:mask.start])))
        prev_index = mask.stop

        # markup token
        if type is None:
            if mask.stop == length:
                markup_children.append(tags['unfinished'](_wrap(buffer[mask])))
            else:
                markup_children.append(tags['unknown'](_wrap(buffer[mask])))
        elif type is cmd.TOKEN_TYPE.COMMAND:
            markup_children.append(tags['cmd'](_wrap(buffer[mask])))
        elif type is cmd.TOKEN_TYPE.KEYWORD:
            markup_children.append(tags['kw'](_wrap(buffer[mask])))
        elif type is cmd.TOKEN_TYPE.ARGUMENT:
            markup_children.append(tags['arg'](_wrap(buffer[mask])))
        else:
            assert False

    else:
        markup_children.append(mu.Group(_wrap(buffer[prev_index:])))

        # markup typeahead
        markup_children.append(tags['typeahead']((mu.Text(typeahead),)))

    return mu.Group(tuple(markup_children))


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
            The keymap of beatshell.  The key of dict is the keystroke, and the
            value of dict is the action to activate.  The format of action is just
            like a normal python code: `input.insert_typeahead() or input.move_right()`.
            The syntax are::

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
            "Backspace"     : "input.backspace()",
            "Alt_Backspace" : "input.delete_all()",
            "Delete"        : "input.delete()",
            "Left"          : "input.move_left()",
            "Right"         : "input.insert_typeahead() or input.move_right()",
            "Up"            : "input.prev()",
            "Down"          : "input.next()",
            "Home"          : "input.move_to_start()",
            "End"           : "input.move_to_end()",
            "Ctrl_Left"     : "input.move_to_word_start()",
            "Ctrl_Right"    : "input.move_to_word_end()",
            "Ctrl_Backspace": "input.delete_to_word_start()",
            "Ctrl_Delete"   : "input.delete_to_word_end()",
            "Esc"           : "input.cancel_typeahead() | input.cancel_hint()",
            "'\\x04'"       : "input.delete() or input.exit_if_empty()",
        }

    @cfg.subconfig
    class prompt(cfg.Configurable):
        r"""
        Fields
        ------
        to : float
        tempo : float
        icons : list of str
            The appearances of icon.
        icon_width : int
            The text width of icon.

        markers : tuple of str and str
            The appearance of normal and blinking-style markers.
        marker_width : int
            The text width of marker.

        input_margin : int
            The width of margin of input field.

        caret : tuple of str and str and str
            The markup template of the normal/blinking/highlighted-style caret.
        caret_blink_ratio : float
            The ratio to blink.
        """
        t0: float = 0.0
        tempo: float = 130.0

        icons: List[str] = [
            "[color=cyan]⠶⠦⣚⠀⠶[/]",
            "[color=cyan]⢎⣀⡛⠀⠶[/]",
            "[color=cyan]⢖⣄⠻⠀⠶[/]",
            "[color=cyan]⠖⠐⡩⠂⠶[/]",
            "[color=cyan]⠶⠀⡭⠲⠶[/]",
            "[color=cyan]⠶⠀⣬⠉⡱[/]",
            "[color=cyan]⠶⠀⣦⠙⠵[/]",
            "[color=cyan]⠶⠠⣊⠄⠴[/]",
        ]
        icon_width: int = 5

        markers: Tuple[str, str] = ("❯ ", "[weight=bold]❯ [/]")
        marker_width: int = 2

        input_margin: int = 3

        caret: Tuple[str, str, str] = ("[slot/]", "[weight=dim][invert][slot/][/][/]", "[weight=bold][invert][slot/][/][/]")
        caret_blink_ratio: float = 0.3

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
    message : str

@dataclasses.dataclass(frozen=True)
class InfoHint(Hint):
    message : str

@dataclasses.dataclass(frozen=True)
class SuggestionsHint(Hint):
    suggestions : List[str]
    selected : int
    message : str

class Result:
    pass

@dataclasses.dataclass(frozen=True)
class ErrorResult(Result):
    error : Exception

@dataclasses.dataclass(frozen=True)
class HelpResult(Result):
    command : Callable

@dataclasses.dataclass(frozen=True)
class CompleteResult(Result):
    command : Callable

@dataclasses.dataclass
class HintState:
    index : int
    hint : Hint
    tokens : Optional[List[str]]

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

class BeatInput:
    r"""Input editor for beatshell.

    Attributes
    ----------
    command : commands.RootCommandParser
        The root command parser for beatshell.
    preview_handler : function
        A function to preview beatmap.
    logger : loggers.Logger
        The logger.
    history : Path
        The file of input history.
    settings : BeatShellSettings
        The settings.
    prev_command : str or None
        The previous command.
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
    tokens : list
        The tokens info, which is a list of tuple `(token, type, mask, quotes)`,
        where `type` is cmd.TOKEN_TYPE or None, and the rest are the same as the
        values yielded by `shlexer_tokenize`.
    lex_state : SHLEXER_STATE
        The shlexer state.
    highlighted : int or None
        The index of highlighted token.
    tab_state : TabState or None
        The state of autocomplete.
    hint_state : HintState or None
        The hint state of input.
    result : Result or None
        The result of input.
    state : str
        The input state.
    modified_event : int
        The event counter for modifying buffer.
    """
    def __init__(self, promptable, preview_handler, logger, history, settings=None):
        r"""Constructor.

        Parameters
        ----------
        promptable : object
            The root command.
        preview_handler : function
        logger : loggers.Logger
        history : Path
            The file of input history.
        settings : BeatShellSettings
            The settings of beatshell.
        """
        if settings is None:
            settings = BeatShellSettings()

        self.command = cmd.RootCommandParser(promptable)
        self.preview_handler = preview_handler
        self.logger = logger
        self.history = history
        self.settings = settings
        self.prev_command = None
        self.buffers = [[]]
        self.buffer_index = -1
        self.buffer = self.buffers[0]
        self.pos = 0
        self.typeahead = ""
        self.tokens = []
        self.lex_state = SHLEXER_STATE.SPACED
        self.highlighted = None
        self.hint_state = None
        self.result = None
        self.tab_state = None
        self.state = "FIN"
        self.lock = threading.RLock()
        self.modified_event = 0
        self.new_session(False)

    def update_settings(self, settings):
        self.settings = settings

    @dn.datanode
    def prompt(self, devices_settings, user):
        r"""Start prompt.

        Parameters
        ----------
        devices_settings : engines.DevicesSettings
            The settings of devices.
        user : KAIKOUser

        Returns
        -------
        prompt_task : datanodes.DataNode
            The datanode to execute the prompt.
        """
        debug_monitor = self.settings.debug_monitor
        renderer_monitor = engines.Monitor(user.cache_dir / "monitor" / "prompt.csv") if debug_monitor else None
        input_task, controller = engines.Controller.create(devices_settings.controller, devices_settings.terminal)
        display_task, renderer = engines.Renderer.create(devices_settings.renderer, devices_settings.terminal, monitor=renderer_monitor)
        stroke = BeatStroke(self, self.settings.input)

        t0 = self.settings.prompt.t0
        tempo = self.settings.prompt.tempo
        metronome = engines.Metronome(t0, tempo)

        if debug_monitor:
            monitor_settings = beatwidgets.MonitorWidgetSettings()
            monitor_settings.target = beatwidgets.MonitorTarget.renderer
            icon = yield from beatwidgets.MonitorWidget(renderer, monitor_settings).load().join()
        else:
            patterns_settings = beatwidgets.PatternsWidgetSettings()
            patterns_settings.patterns = self.settings.prompt.icons
            icon = yield from beatwidgets.PatternsWidget(metronome, self.logger.rich, patterns_settings).load().join()

        marker_settings = beatwidgets.MarkerWidgetSettings()
        marker_settings.markers = self.settings.prompt.markers
        marker_settings.blink_ratio = self.settings.prompt.caret_blink_ratio
        marker = yield from beatwidgets.MarkerWidget(metronome, self.logger.rich, marker_settings).load().join()

        caret = yield from Caret(metronome, self.logger.rich, self.settings.prompt).load().join()

        prompt = BeatPrompt(stroke, self, self.settings, self.logger.rich, metronome, icon, marker, caret, renderer_monitor)

        stroke.register(controller)
        prompt.register(renderer)

        @dn.datanode
        def stop_when(event):
            yield
            yield
            while not event.is_set():
                yield

        yield from dn.pipe(stop_when(prompt.fin_event), display_task, input_task).join()

    @locked
    @onstate("FIN")
    def new_session(self, record_current=True):
        r"""Start a new session of input.

        Parameters
        ----------
        record_current : bool, optional
            Recording current input state.
        """
        if record_current:
            command = "".join(self.buffer).strip()
            self.write_history(command)

        self.buffers = self.read_history()
        self.buffers.append([])
        self.buffer_index = -1

        self.buffer = self.buffers[self.buffer_index]
        self.pos = len(self.buffer)
        self.cancel_typeahead()
        self.update_buffer()
        self.cancel_hint()
        self.clear_result()
        self.state = "EDIT"

    def write_history(self, command):
        if command and command != self.prev_command:
            open(self.history, "a").write("\n" + command)
            self.prev_command = command

    def read_history(self):
        history_size = self.settings.input.history_size
        trim_len = 10

        buffers = []
        for command in open(self.history):
            command = command.strip()
            if command:
                buffers.append(command)
            if len(buffers) - history_size > trim_len:
                del buffers[:trim_len]

        buffers = [list(command) for command in buffers[-history_size:]]
        self.prev_command = "".join(buffers[-1]) if buffers else None
        return buffers

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
        tokenizer = shlexer_tokenize(self.buffer)

        tokens = []
        while True:
            try:
                token, mask, quotes = next(tokenizer)
            except StopIteration as e:
                self.lex_state = e.value
                break

            tokens.append((token, mask, quotes))

        types, _ = self.command.parse_command(token for token, _, _ in tokens)
        types.extend([None]*(len(tokens) - len(types)))
        self.tokens = [(token, type, mask, quotes) for (token, mask, quotes), type in zip(tokens, types)]
        self.modified_event += 1
        return True

    @locked
    def show_typeahead(self):
        """Make typeahead.
        Show the possible command you want to type.  Only work if the caret is at
        the end of buffer.

        Returns
        -------
        succ : bool
            `False` if unable to complete or the caret is not at the end of buffer.
        """
        if self.pos != len(self.buffer):
            self.typeahead = ""
            return False

        if self.lex_state == SHLEXER_STATE.SPACED:
            parents = [token for token, _, _, _ in self.tokens]
            target = ""
        else:
            parents = [token for token, _, _, _ in self.tokens[:-1]]
            target, _, _, _ = self.tokens[-1]

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
            Index of the token to which the hint is directed, or `None` for nothing.

        Returns
        -------
        succ : bool
        """
        self.highlighted = index
        if isinstance(hint, DescHint):
            msg_tokens = [token for token, _, _, _ in self.tokens[:index]] if index is not None else None
        elif isinstance(hint, (InfoHint, SuggestionsHint)):
            msg_tokens = [token for token, _, _, _ in self.tokens[:index+1]] if index is not None else None
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

        if self.highlighted is not None and self.highlighted >= len(self.tokens):
            return self.cancel_hint()

        if len(self.hint_state.tokens) > len(self.tokens):
            return self.cancel_hint()

        for token, (token_, _, _, _) in zip(self.hint_state.tokens, self.tokens):
            if token != token_:
                return self.cancel_hint()

        if isinstance(self.hint_state.hint, DescHint) and self.tokens[len(self.hint_state.tokens)-1][1] is not None:
            return self.cancel_hint()

        return False

    @locked
    def update_preview(self):
        if not self.settings.input.preview_song:
            return
        if self.hint_state is None:
            self.preview_handler(None)
        elif not isinstance(self.hint_state.hint, (InfoHint, SuggestionsHint)):
            self.preview_handler(None)
        elif isinstance(self.hint_state.hint, SuggestionsHint) and not self.hint_state.hint.message:
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
            `False` if there is no typeahead or the caret is not at the end of buffer.
        """

        if self.typeahead == "" or self.pos != len(self.buffer):
            return False

        self.buffer[self.pos:self.pos] = self.typeahead
        self.pos += len(self.typeahead)
        self.typeahead = ""
        self.update_buffer()
        self.ask_hint()

        return True

    @locked
    @onstate("EDIT")
    def insert(self, text):
        """Input.
        Insert some text into the buffer.

        Parameters
        ----------
        text : str
            The text to insert.  It shouldn't contain any nongraphic character,
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
            del self.buffer[self.pos-1]
            self.pos = self.pos-1

        if not all(ch.isprintable() for ch in self.buffer):
            raise ValueError("invalid text to insert: " + repr("".join(self.buffer)))

        self.buffer[self.pos:self.pos] = text
        self.pos += len(text)
        self.update_buffer()

        self.show_typeahead()
        self.ask_hint()

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
        self.ask_hint(clear=True)

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
        self.ask_hint(clear=True)

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
        end = min(max(0, end), len(self.buffer)) if end is not None else len(self.buffer)

        if start >= end:
            return False

        del self.buffer[start:end]
        self.pos = start
        self.update_buffer()
        self.cancel_typeahead()
        self.ask_hint(clear=True)

        return True

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
        pos = min(max(0, pos), len(self.buffer)) if pos is not None else len(self.buffer)
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
        return self.move_to(self.pos+offset)

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
    def ask_hint(self, index=None, clear=False):
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
            for index, (_, _, slic, _) in enumerate(self.tokens):
                if slic.start <= self.pos <= slic.stop:
                    break
            else:
                # find nothing
                if clear:
                    self.cancel_hint()
                return False

        target, token_type, _, _ = self.tokens[index]
        parents = [token for token, _, _, _ in self.tokens[:index]]

        if token_type is None:
            msg = self.command.desc_command(parents)
            if msg is None:
                self.cancel_hint()
                return False
            hint = DescHint(msg)
            self.set_hint(hint, index)
            return True

        else:
            msg = self.command.info_command(parents, target)
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
        for index, (_, _, slic, _) in reversed(list(enumerate(self.tokens))):
            if slic.start <= self.pos:
                break
        else:
            return False

        if self.hint_state is None or self.hint_state.index != index:
            self.ask_hint(index)
            return False

        hint = self.hint_state.hint
        if not hint.message:
            return False
        if isinstance(hint, DescHint):
            template = self.logger.rich.parse(self.settings.text.desc_message, slotted=True)
        elif isinstance(hint, (InfoHint, SuggestionsHint)):
            template = self.logger.rich.parse(self.settings.text.info_message, slotted=True)
        else:
            assert False
        msg_markup = mu.replace_slot(template, self.logger.rich.parse(hint.message, root_tag=True))

        self.cancel_hint()
        self.finish_autocomplete()
        self.set_result(HelpResult(lambda:self.logger.print(msg_markup)))
        self.finish()
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

        if len(self.tokens) == 0:
            self.set_result(CompleteResult(lambda:None))
            self.finish()
            return True

        if self.lex_state == SHLEXER_STATE.BACKSLASHED:
            res, index = ShellSyntaxError("No escaped character"), len(self.tokens)-1
        elif self.lex_state == SHLEXER_STATE.QUOTED:
            res, index = ShellSyntaxError("No closing quotation"), len(self.tokens)-1
        else:
            types, res = self.command.parse_command(token for token, _, _, _ in self.tokens)
            index = len(types)

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
            Indicating direction for exploration of suggestions.  `+1` for next
            suggestion; `-1` for previous suggestion; `0` for canceling the process.

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
            parents = []
            target = ""
            selection = slice(self.pos, self.pos)
            for token, _, mask, _ in self.tokens:
                start, stop, _ = mask.indices(len(self.buffer))
                if stop < self.pos:
                    parents.append(token)
                if start <= self.pos <= stop:
                    target = token
                    selection = mask
            token_index = len(parents)

            # generate suggestions
            suggestions = [shlexer_quoting(sugg) for sugg in self.command.suggest_command(parents, target)]
            sugg_index = len(suggestions) if action == -1 else -1

            if len(suggestions) == 0:
                # no suggestion
                return False

            if len(suggestions) == 1:
                # one suggestion -> complete directly
                self.buffer[selection] = suggestions[0]
                self.pos = selection.start + len(suggestions[0])
                self.update_buffer()
                target, target_type, _, _ = self.tokens[token_index]
                if target_type is None:
                    return False
                msg = self.command.info_command(parents, target)
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
                selection=selection)

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
        parents = [token for token, _, _, _ in self.tokens[:self.tab_state.token_index]]
        target, target_type, _, _ = self.tokens[self.tab_state.token_index]
        if target_type is None:
            msg = ""
        else:
            msg = self.command.info_command(parents, target) or ""
        self.set_hint(SuggestionsHint(suggestions, sugg_index, msg), self.tab_state.token_index)
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

            parents = [token for token, _, _, _ in self.tokens[:token_index]]
            target, target_type, _, _ = self.tokens[token_index]
            if target_type is None:
                self.cancel_hint()
                return True
            msg = self.command.info_command(parents, target)
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

        controller.add_handler(self.autocomplete_handler(self.settings.autocomplete_keys, self.settings.help_key))
        controller.add_handler(self.printable_handler(), "PRINTABLE")

        for key, func in self.settings.keymap.items():
            controller.add_handler(self.action_handler(func), key)

        controller.add_handler(self.help_handler(), self.settings.help_key)
        controller.add_handler(self.confirm_handler(), self.settings.confirm_key)
        controller.add_handler(self.unknown_handler(self.settings))

    def keypress_handler(self):
        def keypress(args):
            self.key_event += 1
        return keypress

    def confirm_handler(self):
        return lambda args: self.input.confirm()

    def help_handler(self):
        return lambda args: self.input.help()

    def autocomplete_handler(self, keys, help_key):
        def handler(args):
            key = args[2]
            if key == keys[0]:
                self.input.autocomplete(+1)
            elif key == keys[1]:
                self.input.autocomplete(-1)
            elif key == keys[2]:
                self.input.autocomplete(0)
            elif key != help_key:
                self.input.finish_autocomplete()
        return handler

    def action_handler(self, func):
        fn = r"input\.(?!_)\w+\(\)"
        op = "(%s)" % "|".join(map(re.escape, (" | ", " & ", " and ", " or ")))
        regex = f"({fn}{op})*{fn}"
        if not re.match(regex, func):
            raise ValueError(f"invalid action: {repr(func)}")
        return lambda args: eval(func, {}, {"input": self.input})

    def printable_handler(self):
        return lambda args: self.input.insert(args[3])

    def unknown_handler(self, settings):
        keys = list(settings.keymap.keys())
        keys.append(settings.confirm_key)
        keys.append(settings.help_key)
        keys.extend(settings.autocomplete_keys)
        keys.append("PRINTABLE")
        def handler(args):
            _, _, key, code = args
            if key not in keys:
                self.input.unknown_key(key)
        return handler

class BeatPrompt:
    r"""Prompt renderer for beatshell."""

    def __init__(self, stroke, input, settings, rich, metronome, icon, marker, caret, monitor):
        r"""Constructor.

        Parameters
        ----------
        stroke : BeatStroke
        input : BeatInput
        settings : BeatShellSettings
        rich : markups.RichParser
        metronome : engines.Metronome
        icon : function
        marker : function
        caret : function
        monitor : engines.Monitor or None
        """
        self.stroke = stroke

        self.input = input
        self.settings = settings
        self.monitor = monitor
        self.fin_event = threading.Event()
        self.key_pressed_time = 0.0

        # input state
        self.modified_event = None
        self.buffer = []
        self.tokens = []
        self.pos = 0
        self.highlighted = None
        self.typeahead = ""
        self.clean = False
        self.hint = None
        self.state = "EDIT"

        self.rich = rich

        # widgets
        self.metronome = metronome
        self.icon_func = icon
        self.marker_func = marker
        self.caret_func = caret

    def register(self, renderer):
        icon_width = self.settings.prompt.icon_width
        marker_width = self.settings.prompt.marker_width
        input_margin = self.settings.prompt.input_margin

        icon_mask = slice(None, icon_width)
        marker_mask = slice(icon_width, icon_width+marker_width)
        input_mask = slice(icon_width+marker_width, None)

        icon_drawer = lambda arg: (0, self.icon_func(arg[0], arg[1]))
        marker_drawer = lambda arg: (0, self.marker_func(arg[0], arg[1]))

        renderer.add_drawer(self.state_updater(), zindex=())
        renderer.add_drawer(self.update_metronome(), zindex=(0,))
        renderer.add_drawer(self.adjust_input_offset(), zindex=(0,))
        renderer.add_drawer(self.hint_handler(), zindex=(1,))
        renderer.add_text(self.text_handler(), input_mask, zindex=(1,))
        renderer.add_text(self.get_left_ellipsis_func(), input_mask, zindex=(1,10))
        renderer.add_text(self.get_right_ellipsis_func(), input_mask, zindex=(1,10))
        renderer.add_text(icon_drawer, icon_mask, zindex=(2,))
        renderer.add_text(marker_drawer, marker_mask, zindex=(3,))

    @dn.datanode
    def state_updater(self):
        (view, msg), time, width = yield
        while True:
            # extract input state
            with self.input.lock:
                if self.input.modified_event != self.modified_event:
                    self.modified_event = self.input.modified_event
                    self.buffer = list(self.input.buffer)
                    self.tokens = list(self.input.tokens)
                self.pos = self.input.pos
                self.highlighted = self.input.highlighted

                self.typeahead = self.input.typeahead
                self.clean = self.input.result is not None
                self.hint = self.input.hint_state.hint if self.input.hint_state is not None else None
                self.state = self.input.state

            (view, msg), time, width = yield (view, msg)

            # fin
            if self.state == "FIN" and not self.fin_event.is_set():
                self.fin_event.set()

    @dn.datanode
    def update_metronome(self):
        key_event = None

        (view, msg), time, width = yield
        while True:
            if self.stroke.key_event != key_event:
                key_event = self.stroke.key_event
                self.key_pressed_time = time
            (view, msg), time, width = yield (view, msg)

    def input_geometry(self, buffer, typeahead, pos):
        text_width = self.rich.widthof(buffer)
        typeahead_width = self.rich.widthof(typeahead)
        caret_dis = self.rich.widthof(buffer[:pos])
        return text_width, typeahead_width, caret_dis

    @dn.datanode
    def adjust_input_offset(self):
        icon_width = self.settings.prompt.icon_width
        marker_width = self.settings.prompt.marker_width
        input_margin = self.settings.prompt.input_margin

        input_mask = slice(icon_width+marker_width, None)

        self.input_offset = 0
        self.left_overflow = False
        self.right_overflow = False

        geo_key = lambda buffer, typeahead, pos: (id(buffer), typeahead, pos)
        geo_node = dn.starcache(self.input_geometry, geo_key)

        with geo_node:
            (view, msg), time, width = yield
            while True:
                typeahead = self.typeahead if not self.clean else ""
                text_width, typeahead_width, caret_dis = geo_node.send((self.buffer, typeahead, self.pos))
                input_width = len(range(width)[input_mask])

                # adjust input offset
                if text_width - self.input_offset < input_width - 1 - input_margin:
                    # from: ......[....I...    ]
                    #   to: ...[.......I... ]
                    self.input_offset = max(0, text_width-input_width+1+input_margin)
                if caret_dis - self.input_offset >= input_width - input_margin:
                    # from: ...[............]..I....
                    #   to: ........[..........I.]..
                    self.input_offset = caret_dis - input_width + input_margin + 1
                elif caret_dis - self.input_offset - input_margin < 0:
                    # from: .....I...[............]...
                    #   to: ...[.I..........].........
                    self.input_offset = max(caret_dis - input_margin, 0)

                # determine overflow
                self.left_overflow = self.input_offset > 0
                self.right_overflow = text_width + typeahead_width - self.input_offset > input_width - 1

                (view, msg), time, width = yield (view, msg)

    def markup_syntax(self, buffer, tokens, typeahead):
        r"""Markup syntax of input text.

        Parameters
        ----------
        buffer : list of str
        tokens : list
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
                if pos >= i+l:
                    i += l
                    continue

                if isinstance(subword, mu.Text):
                    subwords = (
                        mu.Text(subword.string[:pos-i]),
                        CaretPlaceholder((mu.Text(subword.string[pos-i]),)),
                        mu.Text(subword.string[pos-i+1:]),
                    )
                else:
                    subwords = CaretPlaceholder((subword,)),

                token = dataclasses.replace(token, children=token.children[:m] + subwords + token.children[m+1:])
                markup = dataclasses.replace(markup, children=markup.children[:n] + (token,) + markup.children[n+1:])
                break
            else:
                continue
            break

        # unfinished -> unknown
        if clean and len(markup.children) >= 3:
            n = -3
            token = markup.children[n]
            if isinstance(token, self.rich.tags['unfinished']):
                token = self.rich.tags['unknown'](token.children)
                markup = dataclasses.replace(markup, children=markup.children[:n] + (token,) + markup.children[n+1:])

        # highlight
        if highlighted is not None:
            n = highlighted*2+1
            token = markup.children[n]
            token = self.rich.tags['highlight']((token,))
            markup = dataclasses.replace(markup, children=markup.children[:n] + (token,) + markup.children[n+1:])

        markup = markup.expand()
        return markup

    def render_caret(self, markup, caret):
        if caret is not None:
            return markup.traverse(CaretPlaceholder, lambda m: mu.replace_slot(caret, mu.Group(m.children)))
        else:
            return markup.traverse(CaretPlaceholder, lambda m: mu.Group(m.children))

    @dn.datanode
    def text_handler(self):
        syntax_key = lambda buffer, tokens, typeahead: (id(buffer), typeahead)
        syntax_node = dn.starcache(self.markup_syntax, syntax_key)

        dec_key = lambda markup, pos, highlighted, clean: (id(markup), pos, highlighted, clean)
        dec_node = dn.starcache(self.decorate_tokens, dec_key)

        caret_node = dn.starcache(self.render_caret)

        with syntax_node, dec_node, caret_node:
            time, ran = yield
            while True:
                typeahead = self.typeahead if not self.clean else ""

                markup = syntax_node.send((self.buffer, self.tokens, typeahead))
                markup = dec_node.send((markup, self.pos, self.highlighted, self.clean))

                caret = self.caret_func(time, self.key_pressed_time) if not self.clean else None
                markup = caret_node.send((markup, caret))

                time, ran = yield -self.input_offset, markup

    def get_left_ellipsis_func(self):
        ellipis = mu.Text("…")
        return lambda arg: ((0, ellipis) if self.left_overflow else None)

    def get_right_ellipsis_func(self):
        ellipis = mu.Text("…")
        return lambda arg: ((len(arg[1])-1, ellipis) if self.right_overflow else None)

    def markup_hint(self, messages, hint):
        r"""Render hint.

        Parameters
        ----------
        messages : list of Markup
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

        messages.clear()

        # draw hint
        if hint is None:
            return messages

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
                messages.append(mu.Text("…\n"))
            for i, sugg in enumerate(suggs):
                sugg = mu.Text(sugg)
                if i == hint.selected-sugg_start:
                    sugg = mu.replace_slot(sugg_items[1], sugg)
                else:
                    sugg = mu.replace_slot(sugg_items[0], sugg)
                messages.append(sugg)
                if i == hint.selected-sugg_start and msg is not None:
                    messages.append(mu.Text("\n"))
                    messages.append(msg)
                if i != len(suggs)-1:
                    messages.append(mu.Text("\n"))
            if sugg_end < len(hint.suggestions):
                messages.append(mu.Text("\n…"))
            messages.append(self.rich.parse("\n"))

        else:
            if msg is not None:
                messages.append(msg)

        return messages

    @dn.datanode
    def hint_handler(self):
        hint_node = dn.starcache(self.markup_hint, lambda msg, hint: hint)
        with hint_node:
            (view, msg), time, width = yield
            while True:
                msg = hint_node.send((msg, self.hint))
                (view, msg), time, width = yield (view, msg)


@dataclasses.dataclass(frozen=True)
class CaretPlaceholder(mu.Pair):
    name = "caret"

@dataclasses.dataclass
class Caret:
    metronome: engines.Metronome
    rich: mu.RichParser
    settings: BeatShellSettings.prompt

    @dn.datanode
    def load(self):
        caret_blink_ratio = self.settings.caret_blink_ratio
        caret = self.settings.caret

        markuped_caret = [
            self.rich.parse(caret[0], slotted=True),
            self.rich.parse(caret[1], slotted=True),
            self.rich.parse(caret[2], slotted=True),
        ]

        def caret_func(time, key_pressed_time):
            beat = self.metronome.beat(time)
            key_pressed_beat = self.metronome.beat(key_pressed_time) // -1 * -1
            # don't blink while key pressing
            if beat < key_pressed_beat or beat % 1 < caret_blink_ratio:
                if beat % 4 < 1:
                    return markuped_caret[2]
                else:
                    return markuped_caret[1]
            else:
                return markuped_caret[0]

        yield
        return caret_func

