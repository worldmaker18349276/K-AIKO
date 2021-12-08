import os
from enum import Enum
import functools
import itertools
import re
import threading
from typing import Union, Optional, List, Set, Tuple, Dict, Callable
import dataclasses
from kaiko.utils import datanodes as dn
from kaiko.utils import biparsers as bp
from kaiko.utils import config as cfg
from kaiko.utils import markups as mu
from kaiko.utils import commands as cmd
from kaiko.utils import engines


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
        raise ValueError

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

    class text(cfg.Configurable):
        r"""
        Fields
        ------
        error_message : str
            The markup template for the error message.
        info_message : str
            The markup template for the info message.
        message_max_lines : int
            The maximum number of lines of the message.

        quotation : str
            The replacement text for quotation marks.
        backslash : str
            The replacement text for backslashes.
        whitespace : str
            The replacement text for escaped whitespaces.
        typeahead : str
            The markup template for the type-ahead.

        suggestions_lines : int
            The maximum number of lines of the suggestions.
        suggestion_items : tuple of str and str
            The markup templates for the unselected/selected suggestion.

        token_unknown : str
            The markup template for the unknown token.
        token_unfinished : str
            The markup template for the unfinished token.
        token_command : str
            The markup template for the command token.
        token_keyword : str
            The markup template for the keyword token.
        token_argument : str
            The markup template for the argument token.
        token_highlight : str
            The markup template for the highlighted token.
        """
        error_message: str = "[weight=dim][color=red][slot/][/][/]"
        info_message: str = f"{'─'*80}\n[weight=dim][slot/][/]\n{'─'*80}"
        message_max_lines: int = 16

        quotation: str = "[weight=dim]'[/]"
        backslash: str = r"[weight=dim]\\[/]"
        whitespace: str = "[weight=dim]⌴[/]"
        typeahead: str = "[weight=dim][slot/][/]"

        suggestions_lines: int = 8
        suggestion_items: Tuple[str, str] = ("• [slot/]", "• [invert][slot/][/]")

        token_unknown: str = "[color=red][slot/][/]"
        token_unfinished: str = "[slot/]"
        token_command: str = "[color=bright_blue][slot/][/]"
        token_keyword: str = "[color=bright_magenta][slot/][/]"
        token_argument: str = "[color=bright_green][slot/][/]"
        token_highlight: str = "[underline][slot/][/]"

    debug_monitor: bool = False


@dataclasses.dataclass(frozen=True)
class InputWarn:
    message : str

@dataclasses.dataclass(frozen=True)
class InputMessage:
    message : str

@dataclasses.dataclass(frozen=True)
class InputSuggestions:
    suggestions : List[str]
    selected : int
    message : Optional[str]

@dataclasses.dataclass(frozen=True)
class InputError:
    error : Exception

@dataclasses.dataclass(frozen=True)
class InputComplete:
    command : Callable

@dataclasses.dataclass
class HintState:
    hint : Union[InputWarn, InputMessage, InputSuggestions]
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
    command : cmd.RootCommandParser
        The root command parser for beatshell.
    history : Path
        The file of input history.
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
    result : InputError or InputComplete or None
        The result of input.
    state : str
        The input state.
    modified_event : int
        The event counter for modifying buffer.
    """
    def __init__(self, promptable, history):
        r"""Constructor.

        Parameters
        ----------
        promptable : object
            The root command.
        history : Path
            The file of input history.
        """
        self.command = cmd.RootCommandParser(promptable)
        self.history = history
        self.prev_command = None
        self.tab_state = None
        self.state = "FIN"
        self.lock = threading.RLock()
        self.modified_event = 0

        self.new_session(False)

    def prompt(self, devices_settings, settings):
        r"""Start prompt.

        Parameters
        ----------
        devices_settings : DevicesSettings
            The settings of devices.
        settings : BeatShellSettings
            The settings of beatshell.

        Returns
        -------
        prompt_task : dn.DataNode
            The datanode to execute the prompt.
        """
        if settings is None:
            settings = BeatShellSettings()

        debug_monitor = settings.debug_monitor
        renderer_monitor = engines.Monitor("prompt_monitor.csv") if debug_monitor else None
        input_task, controller = engines.Controller.create(devices_settings.controller, devices_settings.terminal)
        display_task, renderer = engines.Renderer.create(devices_settings.renderer, devices_settings.terminal, monitor=renderer_monitor)
        stroke = BeatStroke(self, settings.input)
        prompt = BeatPrompt(stroke, self, settings, devices_settings.terminal, renderer_monitor)

        stroke.register(controller)
        prompt.register(renderer)

        @dn.datanode
        def stop_when(event):
            yield
            yield
            while not event.is_set():
                yield

        return dn.pipe(stop_when(prompt.fin_event), display_task, input_task)

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
            if command and command != self.prev_command:
                open(self.history, "a").write("\n" + command)

        self.buffers = list(list(command.strip()) for command in open(self.history) if command.strip())
        self.prev_command = "".join(self.buffers[-1]) if self.buffers else None
        self.buffers.append([])
        self.buffer_index = -1

        self.buffer = self.buffers[self.buffer_index]
        self.pos = len(self.buffer)
        self.typeahead = ""
        self.highlighted = None
        self.update_buffer()

        self.hint_state = None
        self.result = None
        self.state = "EDIT"

    @locked
    @onstate("FIN")
    def prev_session(self):
        r"""Back to previous session of input."""
        self.highlighted = None
        self.result = None
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
        res : InputError or InputComplete
            The result.

        Returns
        -------
        succ : bool
        """
        self.result = res
        self.hint_state = None
        return True

    @locked
    def clear_result(self):
        """Clear result.

        Returns
        -------
        succ : bool
        """
        self.highlighted = None
        self.result = None
        return True

    @locked
    def set_hint(self, hint, index=None):
        """Set hint.
        Show hint below the prompt.

        Parameters
        ----------
        hint : InputWarn or InputMessage or InputSuggestions
            The hint.
        index : int or None
            Index of the token to which the hint is directed, or `None` for nothing.

        Returns
        -------
        succ : bool
        """
        self.highlighted = index
        if isinstance(hint, InputWarn):
            msg_tokens = [token for token, _, _, _ in self.tokens[:index]] if index is not None else None
        elif isinstance(hint, InputMessage):
            msg_tokens = [token for token, _, _, _ in self.tokens[:index+1]] if index is not None else None
        elif isinstance(hint, InputSuggestions):
            msg_tokens = None
        else:
            raise ValueError
        self.hint_state = HintState(hint, msg_tokens)
        return True

    @locked
    def cancel_hint(self):
        """Cancel hint.
        Remove the hint below the prompt.

        Returns
        -------
        succ : bool
        """
        self.highlighted = None
        self.hint_state = None
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

        if isinstance(self.hint_state.hint, InputWarn) and self.tokens[len(self.hint_state.tokens)-1][1] is not None:
            return self.cancel_hint()

        return False

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
        self.ask_hint()

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
        self.ask_hint()

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
        self.ask_hint()

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
        for match in re.finditer("\w+|\W+", "".join(self.buffer)):
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
        for match in re.finditer("\w+|\W+", "".join(self.buffer)):
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
        for match in re.finditer("\w+|\W+", "".join(self.buffer)):
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
        for match in re.finditer("\w+|\W+", "".join(self.buffer)):
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
    def ask_hint(self):
        """Help for command.
        Provide some hint for the command before the caret.

        Returns
        -------
        succ : bool
        """
        # find the token on the caret
        for index, (target, token_type, slic, _) in enumerate(self.tokens):
            if slic.start <= self.pos <= slic.stop:
                break
        else:
            # don't cancel hint if find nothing
            return False

        self.cancel_hint()

        parents = [token for token, _, _, _ in self.tokens[:index]]

        if token_type is None:
            msg = self.command.desc_command(parents)
            if msg is None:
                return False
            hint = InputWarn(msg)
            self.set_hint(hint, index)
            return True

        else:
            msg = self.command.info_command(parents, target)
            if msg is None:
                return False
            hint = InputMessage(msg)
            self.set_hint(hint, index)
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
            self.set_result(InputComplete(lambda:None))
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
            self.set_result(InputError(res))
            self.finish()
            return False
        elif isinstance(res, (cmd.CommandParseError, ShellSyntaxError)):
            self.set_result(InputError(res))
            self.highlighted = index
            self.finish()
            return False
        else:
            self.set_result(InputComplete(res))
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
                msg = self.command.info_command(parents, self.tokens[token_index][0])
                if msg is None:
                    return False
                hint = InputMessage(msg)
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

        if action == +1:
            self.tab_state.sugg_index += 1
        elif action == -1:
            self.tab_state.sugg_index -= 1
        elif action == 0:
            self.tab_state.sugg_index = None
        else:
            raise ValueError

        sugg_index = self.tab_state.sugg_index
        selection = self.tab_state.selection
        suggestions = self.tab_state.suggestions
        if sugg_index in range(len(suggestions)):
            # autocomplete selected token
            self.buffer[selection] = suggestions[sugg_index]
            self.pos = selection.start + len(suggestions[sugg_index])
            self.tab_state.selection = slice(selection.start, self.pos)

            self.update_buffer()
            parents = [token for token, _, _, _ in self.tokens[:self.tab_state.token_index]]
            target = self.tokens[self.tab_state.token_index][0]
            msg = self.command.info_command(parents, target)
            self.set_hint(InputSuggestions(suggestions, sugg_index, msg), self.tab_state.token_index)
            return True

        else:
            # restore state
            self.buffer[selection] = self.tab_state.original_token
            self.pos = self.tab_state.original_pos

            self.tab_state = None
            self.update_buffer()
            self.update_hint()
            return False

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
            parents = [token for token, _, _, _ in self.tokens[:self.tab_state.token_index]]
            target = self.tokens[self.tab_state.token_index][0]
            msg = self.command.info_command(parents, target)
            if msg is None:
                return True
            hint = InputMessage(msg)
            self.set_hint(hint, self.tab_state.token_index)

            self.tab_state = None
        return True

    @locked
    def unknown_key(self, key):
        self.set_result(InputError(ValueError(f"Unknown key: " + key)))
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
        return lambda args: self.input.ask_hint()

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

    def __init__(self, stroke, input, settings, term_settings, monitor):
        r"""Constructor.

        Parameters
        ----------
        stroke : BeatStroke
        input : BeatInput
        settings : BeatShellSettings
        term_settings : TerminalSettings
        monitor : engines.Monitor or None
        """
        self.stroke = stroke

        self.input = input
        self.settings = settings
        self.monitor = monitor
        self.fin_event = threading.Event()
        self.t0 = None
        self.tempo = None

        self.rich = mu.RichTextRenderer(term_settings.unicode_version, term_settings.color_support)
        self.rich.add_pair_template("error", settings.text.error_message)
        self.rich.add_pair_template("info", settings.text.info_message)
        self.rich.add_pair_template("unknown", settings.text.token_unknown)
        self.rich.add_pair_template("unfinished", settings.text.token_unfinished)
        self.rich.add_pair_template("cmd", settings.text.token_command)
        self.rich.add_pair_template("kw", settings.text.token_keyword)
        self.rich.add_pair_template("arg", settings.text.token_argument)
        self.rich.add_pair_template("emph", settings.text.token_highlight)
        self.rich.add_single_template("ws", settings.text.whitespace)
        self.rich.add_single_template("qt", settings.text.quotation)
        self.rich.add_single_template("bs", settings.text.backslash)
        self.rich.add_pair_template("typeahead", settings.text.typeahead)

    def register(self, renderer):
        renderer.add_drawer(self.output_handler())

    @dn.datanode
    def output_handler(self):
        header_node = self.header_node()
        text_node = self.text_node()
        hint_node = dn.starcache(self.markup_hint, lambda msg, hint: hint)
        render_node = self.render_node()
        modified_event = None
        buffer = []
        tokens = []
        with header_node, text_node, hint_node, render_node:
            (view, msg), time, width = yield
            while True:
                # extract input state
                with self.input.lock:
                    if self.input.modified_event != modified_event:
                        modified_event = self.input.modified_event
                        buffer = list(self.input.buffer)
                        tokens = list(self.input.tokens)
                    pos = self.input.pos
                    highlighted = self.input.highlighted

                    typeahead = self.input.typeahead
                    clean = self.input.result is not None
                    hint = self.input.hint_state.hint if self.input.hint_state is not None else None
                    state = self.input.state

                # draw header
                header_data = header_node.send((clean, time))

                # draw text
                text_data = text_node.send((buffer, tokens, typeahead, pos, highlighted, clean))

                # render view
                view = render_node.send((view, width, header_data, text_data))

                # render hint
                msg = hint_node.send((msg, hint))

                (view, msg), time, width = yield (view, msg)

                # fin
                if state == "FIN" and not self.fin_event.is_set():
                    self.fin_event.set()

    def get_monitor_func(self):
        ticks = " ▏▎▍▌▋▊▉█"
        ticks_len = len(ticks)
        icon_width = self.settings.prompt.icon_width

        def monitor_func(period):
            level = int((self.monitor.eff or 0.0) * icon_width*(ticks_len-1))
            return mu.Text("".join(ticks[max(0, min(ticks_len-1, level-i*(ticks_len-1)))] for i in range(icon_width)))

        return monitor_func

    def get_icon_func(self):
        icons = self.settings.prompt.icons

        markuped_icons = [self.rich.parse(icon) for icon in icons]

        def icon_func(period):
            ind = int(period * len(markuped_icons) // 1) % len(markuped_icons)
            return markuped_icons[ind]

        return icon_func

    def get_marker_func(self):
        markers = self.settings.prompt.markers
        caret_blink_ratio = self.settings.prompt.caret_blink_ratio

        markuped_markers = (
            self.rich.parse(markers[0]),
            self.rich.parse(markers[1]),
        )

        def marker_func(period):
            if period % 4 < min(1.0, caret_blink_ratio):
                return markuped_markers[1]
            else:
                return markuped_markers[0]

        return marker_func

    def get_caret_index_func(self):
        caret_blink_ratio = self.settings.prompt.caret_blink_ratio

        def caret_index_func(period, force=False):
            if force or period % 1 < caret_blink_ratio:
                if period % 4 < 1:
                    return 2
                else:
                    return 1
            else:
                return 0

        return caret_index_func

    @dn.datanode
    def header_node(self):
        r"""The datanode to render header and caret.

        Receives
        --------
        clean : bool
            Render header and caret in the clean style: hide caret.
        time : float
            The current time.

        Yields
        ------
        icon: str
            The rendered icon.
        marker : str
            The rendered marker.
        caret_index : int or None
            The index of caret style, or None for no caret.
        """
        icon_func = self.get_monitor_func() if self.monitor else self.get_icon_func()
        marker_func = self.get_marker_func()
        caret_index_func = self.get_caret_index_func()

        self.t0 = self.settings.prompt.t0
        self.tempo = self.settings.prompt.tempo

        clean, time = yield

        period = (0 - self.t0)/(60/self.tempo)
        period_start = period // -1 * -1
        key_event = None
        while True:
            # don't blink while key pressing
            if self.stroke.key_event != key_event:
                key_event = self.stroke.key_event
                period_start = period // -1 * -1

            # render icon, marker, caret
            icon = icon_func(period)
            marker = marker_func(period)
            caret_index = None if clean else caret_index_func(period, period < period_start)

            clean, time = yield icon, marker, caret_index
            period = (time - self.t0)/(60/self.tempo)

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
                        Caret((mu.Text(subword.string[pos-i]),)),
                        mu.Text(subword.string[pos-i+1:]),
                    )
                else:
                    subwords = Caret((subword,)),

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
            token = self.rich.tags['emph']((token,))
            markup = dataclasses.replace(markup, children=markup.children[:n] + (token,) + markup.children[n+1:])

        markup = markup.expand()
        return markup

    def input_geometry(self, buffer, typeahead, pos):
        text_width = self.rich.widthof(buffer)
        typeahead_width = self.rich.widthof(typeahead)
        caret_dis = self.rich.widthof(buffer[:pos])
        return text_width, typeahead_width, caret_dis

    @dn.datanode
    def text_node(self):
        syntax_key = lambda buffer, tokens, typeahead: (id(buffer), typeahead)
        syntax_node = dn.starcache(self.markup_syntax, syntax_key)

        dec_key = lambda markup, pos, highlighted, clean: (id(markup), pos, highlighted, clean)
        dec_node = dn.starcache(self.decorate_tokens, dec_key)

        geo_key = lambda buffer, typeahead, pos: (id(buffer), typeahead, pos)
        geo_node = dn.starcache(self.input_geometry, geo_key)

        text_data = None
        with syntax_node, dec_node, geo_node:
            while True:
                buffer, tokens, typeahead, pos, highlighted, clean = yield text_data
                if clean:
                    typeahead = ""

                markup = syntax_node.send((buffer, tokens, typeahead))
                dec_markup = dec_node.send((markup, pos, highlighted, clean))
                text_width, typeahead_width, caret_dis = geo_node.send((buffer, typeahead, pos))
                text_data = dec_markup, text_width, typeahead_width, caret_dis

    def markup_hint(self, messages, hint):
        r"""Render hint.

        Parameters
        ----------
        messages : list of Markup
            The rendered hint.
        hint : InputWarn or InputMessage or InputSuggestions
        """
        message_max_lines = self.settings.text.message_max_lines

        sugg_lines = self.settings.text.suggestions_lines
        sugg_items = self.settings.text.suggestion_items

        sugg_items = (
            self.rich.parse(sugg_items[0], slotted=True),
            self.rich.parse(sugg_items[1], slotted=True),
        )

        messages.clear()

        # draw hint
        if hint is None:
            return messages

        msg = None
        if hint.message:
            msg = self.rich.parse(hint.message)
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

            if isinstance(hint, InputWarn):
                msg = self.rich.tags['error']((msg,))
            elif isinstance(hint, (InputMessage, InputSuggestions)):
                msg = self.rich.tags['info']((msg,))
            else:
                assert False
            msg = msg.expand()

        if isinstance(hint, InputSuggestions):
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
    def render_caret(self):
        caret = self.settings.prompt.caret

        markuped_caret = [
            self.rich.parse(caret[0], slotted=True),
            self.rich.parse(caret[1], slotted=True),
            self.rich.parse(caret[2], slotted=True),
        ]

        markup_id = None
        cached_res = None
        res = None
        while True:
            markup, caret_index = yield res

            if markup_id != id(markup):
                markup_id = id(markup)
                cached_res = [None, None, None, None]

            if caret_index is not None:
                if cached_res[caret_index] is None:
                    cached_res[caret_index] = markup.traverse(Caret, lambda m: mu.replace_slot(markuped_caret[caret_index], mu.Group(m.children)))
                res = cached_res[caret_index]

            else:
                if cached_res[-1] is None:
                    cached_res[-1] = markup.traverse(Caret, lambda m: mu.Group(m.children))
                res = cached_res[-1]

    @dn.datanode
    def render_node(self):
        r"""The datanode to render whole view.

        Receives
        --------
        view : list of str
            The buffer of the view.
        width : int
            The width of the view.
        header_data : tuple
            The values yielded by `header_node`.
        text_data : tuple
            The values yielded by `text_node`.

        Yields
        ------
        view : list of str
            The buffer of the rendered view.
        """
        caret_node = self.render_caret()

        icon_width = self.settings.prompt.icon_width
        marker_width = self.settings.prompt.marker_width
        input_margin = self.settings.prompt.input_margin
        icon_ran = slice(None, icon_width)
        marker_ran = slice(icon_width, icon_width+marker_width)
        input_ran = slice(icon_width+marker_width, None)

        input_offset = 0

        with caret_node:
            view, width, (icon, marker, caret_index), (markup, text_width, typeahead_width, caret_dis) = yield

            while True:
                xran = range(width)
                input_width = len(xran[input_ran])

                # adjust input offset
                if text_width - input_offset < input_width - 1 - input_margin:
                    # from: ......[....I...    ]
                    #   to: ...[.......I... ]
                    input_offset = max(0, text_width-input_width+1+input_margin)
                if caret_dis - input_offset >= input_width - input_margin:
                    # from: ...[............]..I....
                    #   to: ........[..........I.]..
                    input_offset = caret_dis - input_width + input_margin + 1
                elif caret_dis - input_offset - input_margin < 0:
                    # from: .....I...[............]...
                    #   to: ...[.I..........].........
                    input_offset = max(caret_dis - input_margin, 0)

                # draw caret
                markup = caret_node.send((markup, caret_index))

                # draw input
                view.add_markup(markup, input_ran, -input_offset)
                if input_offset > 0:
                    view.add_markup(mu.Text("…"), input_ran, 0)
                if text_width + typeahead_width - input_offset > input_width - 1:
                    view.add_markup(mu.Text("…"), input_ran, input_width-1)

                # draw header
                view.add_markup(icon, icon_ran, 0)
                view.add_markup(marker, marker_ran, 0)

                view, width, (icon, marker, caret_index), (markup, text_width, typeahead_width, caret_dis) = yield view


@dataclasses.dataclass(frozen=True)
class Caret(mu.Pair):
    name = "caret"

