import os
from enum import Enum
import functools
import re
import threading
from typing import List, Set, Tuple
from dataclasses import dataclass
import wcwidth
from kaiko.utils import datanodes as dn
from kaiko.utils import biparsers as bp
from kaiko.utils import wcbuffers as wcb
from kaiko.utils import config as cfg
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
    ignored : list of int
        The indices of all backslashes and quotation marks used for escaping.
        The token is equal to `''.join(raw[i] for i in range(*slice.indices(len(raw))) if i not in ignored)`.

    Returns
    -------
    state : SHLEXER_STATE
        The final state of parsing.
    """
    SPACE = " "
    BACKSLASH = "\\"
    QUOTE = "'"

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
        ignored = []
        while True:
            if char == SPACE:
                # end parsing token
                yield "".join(token), slice(start, index), ignored
                break

            elif char == BACKSLASH:
                # escape the next character
                ignored.append(index)

                try:
                    index, char = next(raw)
                except StopIteration:
                    yield "".join(token), slice(start, None), ignored
                    return SHLEXER_STATE.BACKSLASHED

                token.append(char)

            elif char == QUOTE:
                # escape the following characters until the next quotation mark
                ignored.append(index)

                while True:
                    try:
                        index, char = next(raw)
                    except StopIteration:
                        yield "".join(token), slice(start, None), ignored
                        return SHLEXER_STATE.QUOTED

                    if char == QUOTE:
                        ignored.append(index)
                        break
                    else:
                        token.append(char)

            else:
                # otherwise, as it is
                token.append(char)

            try:
                index, char = next(raw)
            except StopIteration:
                yield "".join(token), slice(start, None), ignored
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


class BeatShellSettings(cfg.Configurable):
    class input(cfg.Configurable):
        r"""
        Fields
        ------
        confirm_key : str
            The key for confirming input.
        help_key : str
            The key for help.
        autocomplete_keys : Tuple[str, str, str]
            The keys for finding the next, previous and canceling suggestions.
        """
        confirm_key: str = "Enter"
        help_key: str = "Alt_Enter"
        autocomplete_keys: Tuple[str, str, str] = ("Tab", "Shift_Tab", "Esc")

        keymap = {
            "Backspace"     : lambda input: input.backspace(),
            "Alt_Backspace" : lambda input: input.delete_all(),
            "Delete"        : lambda input: input.delete(),
            "Left"          : lambda input: input.move_left(),
            "Right"         : lambda input: input.insert_typeahead() or input.move_right(),
            "Up"            : lambda input: input.prev(),
            "Down"          : lambda input: input.next(),
            "Home"          : lambda input: input.move_to_start(),
            "End"           : lambda input: input.move_to_end(),
            "Ctrl_Left"     : lambda input: input.move_to_word_start(),
            "Ctrl_Right"    : lambda input: input.move_to_word_end(),
            "Ctrl_Backspace": lambda input: input.delete_to_word_start(),
            "Ctrl_Delete"   : lambda input: input.delete_to_word_end(),
            "Esc"           : lambda input: (input.cancel_typeahead(), input.cancel_hint()),
        }

    class prompt(cfg.Configurable):
        r"""
        Fields
        ------
        to : float
        tempo : float
        icons : List[str]
            The appearances of icon.
        icon_width : int
            The text width of icon.

        marker : str
            The appearance of marker.
        marker_attr : Tuple[str, str]
            The text attribute of the normal/blinking-style marker.
        marker_width : int
            The text width of marker.

        caret_attr : Tuple[str, str]
            The text attribute of the normal/blinking-style caret.
        caret_blink_ratio : float
            The ratio to blink.
        """
        t0: float = 0.0
        tempo: float = 130.0

        icons: List[str] = [
            "\x1b[36m⠶⠦⣚⠀⠶\x1b[m",
            "\x1b[36m⢎⣀⡛⠀⠶\x1b[m",
            "\x1b[36m⢖⣄⠻⠀⠶\x1b[m",
            "\x1b[36m⠖⠐⡩⠂⠶\x1b[m",
            "\x1b[36m⠶⠀⡭⠲⠶\x1b[m",
            "\x1b[36m⠶⠀⣬⠉⡱\x1b[m",
            "\x1b[36m⠶⠀⣦⠙⠵\x1b[m",
            "\x1b[36m⠶⠠⣊⠄⠴\x1b[m",
        ]
        icon_width: int = 5

        marker: str = "❯ "
        marker_attr: Tuple[str, str] = ("", "1")
        marker_width: int = 2

        caret_attr: Tuple[str, str] = ("7;2", "7;1")
        caret_blink_ratio: float = 0.3

    class text(cfg.Configurable):
        r"""
        Fields
        ------
        error_message_attr : str
            The text attribute of the error message.
        info_message_attr : str
            The text attribute of the info message.
        message_max_lines : str
            The maximum number of lines of the message.

        escape_attr : str
            The text attribute of the escaped string.
        typeahead_attr
            The text attribute of the type-ahead.
        whitespace : str
            The replacement text of the escaped whitespace.

        suggestions_lines : int
            The maximum number of lines of the suggestions.
        suggestions_selected_attr : str
            The text attribute of the selected suggestion.
        suggestions_bullet : str
            The list bullet of the suggestions.

        token_unknown_attr : str
            The text attribute of the unknown token.
        token_command_attr : str
            The text attribute of the command token.
        token_keyword_attr : str
            The text attribute of the keyword token.
        token_argument_attr : str
            The text attribute of the argument token.
        token_highlight_attr : str
            The text attribute of the highlighted token.
        """
        error_message_attr: str = "31"
        info_message_attr: str = "2"
        message_max_lines: int = 16

        escape_attr: str = "2"
        typeahead_attr: str = "2"
        whitespace: str = "\x1b[2m⌴\x1b[m"

        suggestions_lines: int = 8
        suggestions_selected_attr: str = "7"
        suggestions_bullet: str = "• "

        token_unknown_attr: str = "31"
        token_command_attr: str = "94"
        token_keyword_attr: str = "95"
        token_argument_attr: str = "92"
        token_highlight_attr: str = "4"


class InputWarn:
    def __init__(self, tokens, message):
        self.message = message
        self.tokens = tokens

class InputMessage:
    def __init__(self, tokens, message):
        self.message = message
        self.tokens = tokens

class InputSuggestions:
    def __init__(self, tokens, suggestions, selected):
        self.suggestions = suggestions
        self.selected = selected
        self.tokens = tokens

class InputError:
    def __init__(self, value):
        self.value = value

class InputComplete:
    def __init__(self, value):
        self.value = value

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
        The tokens info, which is a list of tuple `(token, type, mask, ignored)`,
        where `type` is cmd.TOKEN_TYPE or None, and the rest are the same as the
        values yielded by `shlexer_tokenize`.
    lex_state : SHLEXER_STATE
        The shlexer state.
    highlighted : int or None
        The index of highlighted token.
    tab_state : TabState or None
        The state of autocomplete.
    hint : InputWarn or InputMessage or InputSuggestions or None
        The hint of input.
    result : InputError or InputComplete or None
        The result of input.
    state : str
        The input state.
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

        input_task, controller = engines.Controller.create(devices_settings.controller)
        display_task, renderer = engines.Renderer.create(devices_settings.renderer)
        stroke = BeatStroke(self, settings.input)
        prompt = BeatPrompt(stroke, self, settings)

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

        self.tokens = []
        self.lex_state = SHLEXER_STATE.SPACED
        self.highlighted = None

        self.hint = None
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
    def parse_syntax(self):
        """Parse syntax.

        Returns
        -------
        succ : bool
        """
        tokenizer = shlexer_tokenize(self.buffer)

        tokens = []
        while True:
            try:
                token, mask, ignored = next(tokenizer)
            except StopIteration as e:
                self.lex_state = e.value
                break

            tokens.append((token, mask, ignored))

        types, _ = self.command.parse_command(token for token, _, _ in tokens)
        types.extend([None]*(len(tokens) - len(types)))
        self.tokens = [(token, type, mask, ignored) for (token, mask, ignored), type in zip(tokens, types)]
        return True

    @locked
    def make_typeahead(self):
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

        compreply = None
        for suggestion in self.command.suggest_command(parents, target):
            if suggestion.startswith(target):
                compreply = suggestion[len(target):]
                break

        if compreply is None:
            self.typeahead = ""
            return False
        else:
            self.typeahead = shlexer_quoting(compreply, self.lex_state)
            return True

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
    def set_result(self, res_type, value, index=None):
        """Set result.
        Set result of this session.

        Parameters
        ----------
        res_type : InputError or InputComplete
            The type of result.
        value : any
            The value of result.
        index : int or None
            The index of token to highlight.

        Returns
        -------
        succ : bool
        """
        self.highlighted = index
        self.result = res_type(value)
        self.hint = None
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
    def set_hint(self, hint_type, index=None, *params):
        """Set hint.
        Show hint below the prompt.

        Parameters
        ----------
        hint_type : InputWarn or InputMessage or InputSuggestions
            The type of hint.
        index : int or None
            Index of the token to which the hint is directed, or `None` for nothing.
        params : list
            Parameters of hint.

        Returns
        -------
        succ : bool
        """
        self.highlighted = index
        if hint_type == InputWarn:
            msg_tokens = self.tokens[:index] if index is not None else None
        elif hint_type == InputMessage:
            msg_tokens = self.tokens[:index+1] if index is not None else None
        elif hint_type == InputSuggestions:
            msg_tokens = None
        else:
            raise ValueError
        self.hint = hint_type(msg_tokens, *params)
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
        self.hint = None
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
        if self.hint is None:
            return False

        if self.hint.tokens is None:
            return self.cancel_hint()

        if len(self.hint.tokens) > len(self.tokens):
            return self.cancel_hint()

        for t1, t2 in zip(self.hint.tokens, self.tokens):
            if t1[0] != t2[0]:
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
        self.parse_syntax()

        return True

    @locked
    @onstate("EDIT")
    def input(self, text):
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

        if wcwidth.wcswidth("".join(self.buffer)) == -1:
            raise ValueError("invalid text to insert: " + repr("".join(self.buffer)))

        self.buffer[self.pos:self.pos] = text
        self.pos += len(text)
        self.parse_syntax()

        self.make_typeahead()
        self.update_hint()

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
        self.parse_syntax()
        self.cancel_typeahead()
        self.update_hint()

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
        self.parse_syntax()
        self.cancel_typeahead()
        self.update_hint()

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
        self.parse_syntax()
        self.cancel_typeahead()
        self.update_hint()

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
        self.parse_syntax()
        self.cancel_typeahead()
        self.update_hint()

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
        self.parse_syntax()
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
        self.parse_syntax()
        self.cancel_typeahead()
        self.cancel_hint()

        return True

    @locked
    @onstate("EDIT")
    def help(self):
        """Help for command.
        Provide some hint for the command before the caret.

        Returns
        -------
        succ : bool
        """
        self.cancel_hint()

        # find the last token before the caret
        for index, (target, token_type, slic, _) in reversed(list(enumerate(self.tokens))):
            if slic.start is None or slic.start <= self.pos:
                break
        else:
            self.set_hint(InputMessage, None, None)
            return True

        parents = [token for token, _, _, _ in self.tokens[:index]]

        if token_type is None:
            msg = self.command.desc_command(parents)
            hint_type = InputWarn
        else:
            msg = self.command.info_command(parents, target)
            hint_type = InputMessage

        self.set_hint(hint_type, index, msg)
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
        if len(self.tokens) == 0:
            self.set_result(InputComplete, lambda:None)
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
            self.set_result(InputError, res, None)
            self.finish()
            return False
        elif isinstance(res, (cmd.CommandParseError, ShellSyntaxError)):
            self.set_result(InputError, res, index)
            self.finish()
            return False
        else:
            self.set_result(InputComplete, res)
            self.finish()
            return True

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

            # generate suggestions
            suggestions = [shlexer_quoting(sugg) for sugg in self.command.suggest_command(parents, target)]
            sugg_index = len(suggestions) if action == -1 else -1

            # tab state
            original_pos = self.pos

            self.tab_state = TabState(
                suggestions=suggestions,
                sugg_index=sugg_index,
                token_index=len(parents),
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

            self.parse_syntax()
            self.set_hint(InputSuggestions, self.tab_state.token_index, suggestions, sugg_index)
            return True

        else:
            # restore state
            self.buffer[selection] = self.tab_state.original_token
            self.pos = self.tab_state.original_pos

            self.tab_state = None
            self.parse_syntax()
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
            self.tab_state = None
            self.update_hint()
        return True

    @locked
    def unknown_key(self, key):
        self.set_result(InputError, ValueError(f"Unknown key: " + key), None)
        self.finish()

@dataclass
class TabState:
    suggestions: List[str]
    sugg_index: int
    token_index: int
    original_token: List[str]
    original_pos: int
    selection: slice

class BeatStroke:
    r"""Keyboard controller for beatshell."""

    def __init__(self, input, settings):
        self.input = input
        self.settings = settings
        self.key_event = threading.Event()

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
        return lambda args: self.key_event.set()

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
        return lambda args: func(self.input)

    def printable_handler(self):
        return lambda args: self.input.input(args[3])

    def unknown_handler(self, settings):
        keys = list(settings.keymap.keys())
        keys.append(settings.confirm_key)
        keys.append(settings.help_key)
        keys.extend(settings.autocomplete_keys)
        keys.append("PRINTABLE")
        def handler(args):
            _, _, key, code = args
            if key not in keys:
                self.input.unknown_key(key or repr(code))
        return handler

class BeatPrompt:
    r"""Prompt renderer for beatshell."""

    def __init__(self, stroke, input, settings):
        r"""Constructor.

        Parameters
        ----------
        stroke : BeatStroke
        input : BeatInput
        settings : BeatShellSettings
        """
        self.stroke = stroke

        self.input = input
        self.settings = settings
        self.fin_event = threading.Event()
        self.t0 = None
        self.tempo = None

    def register(self, renderer):
        renderer.add_drawer(self.output_handler())

    @dn.datanode
    def output_handler(self):
        header_node = self.header_node()
        text_node = self.text_node()
        hint_node = self.hint_node()
        render_node = self.render_node()
        with header_node, text_node, hint_node, render_node:
            (view, msg), time, width = yield
            while True:
                # extract input state
                with self.input.lock:
                    buffer = list(self.input.buffer)
                    tokens = list(self.input.tokens)
                    pos = self.input.pos
                    highlighted = self.input.highlighted

                    typeahead = self.input.typeahead
                    clean = self.input.result is not None
                    hint = self.input.hint
                    state = self.input.state

                # draw header
                header_data = header_node.send((clean, time))

                # draw text
                text_data = text_node.send((buffer, tokens, typeahead, pos, highlighted, clean))

                # draw hint
                msg = hint_node.send(hint)

                # render view
                view = render_node.send((view, width, header_data, text_data))

                (view, msg), time, width = yield (view, msg)

                # fin
                if state == "FIN" and not self.fin_event.is_set():
                    self.fin_event.set()

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
        caret : function or None
            The function that add a caret to the text, or None for no caret.
        """
        icons = self.settings.prompt.icons
        marker = self.settings.prompt.marker
        marker_attr = self.settings.prompt.marker_attr

        caret_attr = self.settings.prompt.caret_attr
        caret_blink_ratio = self.settings.prompt.caret_blink_ratio

        self.t0 = self.settings.prompt.t0
        self.tempo = self.settings.prompt.tempo

        clean, time = yield

        t = (0 - self.t0)/(60/self.tempo)
        tr = t // 1
        while True:
            # don't blink while key pressing
            if self.stroke.key_event.is_set():
                self.stroke.key_event.clear()
                tr = t // -1 * -1

            # render caret
            if clean:
                caret = None
            elif t < tr or t % 1 < caret_blink_ratio:
                if t % 4 < 1:
                    caret = lambda s: wcb.add_attr(s, caret_attr[1])
                else:
                    caret = lambda s: wcb.add_attr(s, caret_attr[0])
            else:
                caret = None

            # render icon, marker
            ind = int(t * len(icons) // 1) % len(icons)
            icon = icons[ind]

            if t % 4 < min(1, caret_blink_ratio):
                marker_ = wcb.add_attr(marker, marker_attr[1])
            else:
                marker_ = wcb.add_attr(marker, marker_attr[0])

            clean, time = yield icon, marker_, caret
            t = (time - self.t0)/(60/self.tempo)

    @dn.datanode
    def text_node(self):
        r"""The datanode to render input text.

        Receives
        --------
        buffer : list of str
        tokens : list
        typeahead : str
        pos : int
        highlighted : int or None
            See `BeatInput`'s attributes.
        clean : bool
            Render text in the clean style: hide type ahead.

        Yields
        ------
        rendered_text : str
            The rendered input text.
        rendered_typeahead : str
            The rendered type ahead.
        caret_pos : int
            The position of caret.
        """
        escape_attr     = self.settings.text.escape_attr
        typeahead_attr  = self.settings.text.typeahead_attr
        whitespace      = self.settings.text.whitespace

        token_unknown_attr   = self.settings.text.token_unknown_attr
        token_command_attr   = self.settings.text.token_command_attr
        token_keyword_attr   = self.settings.text.token_keyword_attr
        token_argument_attr  = self.settings.text.token_argument_attr
        token_highlight_attr = self.settings.text.token_highlight_attr

        # render buffer
        buffer, tokens, typeahead, pos, highlighted, clean = yield None
        while True:
            indices = range(len(buffer))

            for _, type, mask, ignored in tokens:
                # render whitespace
                for index in indices[mask]:
                    if buffer[index] == " ":
                        buffer[index] = whitespace

                # render escape
                for index in ignored:
                    buffer[index] = wcb.add_attr(buffer[index], escape_attr)

                # render unknown token
                if type is None:
                    if mask.stop is not None or clean:
                        for index in indices[mask]:
                            buffer[index] = wcb.add_attr(buffer[index], token_unknown_attr)

                # render command token
                if type is cmd.TOKEN_TYPE.COMMAND:
                    for index in indices[mask]:
                        buffer[index] = wcb.add_attr(buffer[index], token_command_attr)

                # render keyword token
                elif type is cmd.TOKEN_TYPE.KEYWORD:
                    for index in indices[mask]:
                        buffer[index] = wcb.add_attr(buffer[index], token_keyword_attr)

                # render argument token
                elif type is cmd.TOKEN_TYPE.ARGUMENT:
                    for index in indices[mask]:
                        buffer[index] = wcb.add_attr(buffer[index], token_argument_attr)

            if highlighted in range(len(tokens)):
                # render highlighted token
                _, _, mask, _ = tokens[highlighted]
                for index in indices[mask]:
                    buffer[index] = wcb.add_attr(buffer[index], token_highlight_attr)

            rendered_text = "".join(buffer)

            # render typeahead
            if typeahead and not clean:
                rendered_typeahead = wcb.add_attr(typeahead, typeahead_attr)
            else:
                rendered_typeahead = ""

            # compute caret position
            _, caret_pos = wcb.textrange1(0, "".join(buffer[:pos]))

            buffer, tokens, typeahead, pos, highlighted, clean = yield rendered_text, rendered_typeahead, caret_pos

    @dn.datanode
    def hint_node(self):
        r"""The datanode to render hint.

        Receives
        --------
        hint : InputWarn or InputMessage or InputSuggestions

        Yields
        ------
        msg : str
            The rendered hint.
        """
        message_max_lines = self.settings.text.message_max_lines
        error_message_attr = self.settings.text.error_message_attr
        info_message_attr = self.settings.text.info_message_attr

        suggestions_lines = self.settings.text.suggestions_lines
        suggestions_selected_attr = self.settings.text.suggestions_selected_attr
        suggestions_bullet = self.settings.text.suggestions_bullet

        current_hint = None
        msg = ""
        hint = yield
        while True:
            # track changes of the hint
            if hint is current_hint:
                hint = yield msg
                continue

            current_hint = hint

            # draw hint
            if hint is None:
                msg = ""

            elif isinstance(hint, InputSuggestions):
                sugg_start = hint.selected // suggestions_lines * suggestions_lines
                sugg_end = sugg_start + suggestions_lines
                sugg = hint.suggestions[sugg_start:sugg_end]
                sugg[hint.selected-sugg_start] = wcb.add_attr(sugg[hint.selected-sugg_start], suggestions_selected_attr)
                msg = "\n".join(suggestions_bullet + s for s in sugg)
                if sugg_start > 0:
                    msg = "…\n" + msg
                if sugg_end < len(hint.suggestions):
                    msg = msg + "\n…"

            else:
                msg = hint.message or ""
                if msg.count("\n") >= message_max_lines:
                    msg = "\n".join(msg.split("\n")[:message_max_lines]) + "\x1b[m\n…"

                if msg:
                    if isinstance(hint, InputWarn):
                        msg = wcb.add_attr(msg, error_message_attr)
                    msg = wcb.add_attr(msg, info_message_attr)

            hint = yield msg

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
        icon_width = self.settings.prompt.icon_width
        marker_width = self.settings.prompt.marker_width
        icon_ran = slice(None, icon_width)
        marker_ran = slice(icon_width, icon_width+marker_width)
        input_ran = slice(icon_width+marker_width, None)

        input_offset = 0

        view, width, (icon, marker, caret), (text, typeahead, caret_pos) = yield

        while True:
            input_width = len(range(width)[input_ran])
            _, text_width = wcb.textrange1(0, text)
            _, typeahead_width = wcb.textrange1(0, typeahead)

            # adjust input offset
            if text_width - input_offset < input_width - 1:
                input_offset = max(0, text_width-input_width+1)
            if caret_pos - input_offset >= input_width:
                input_offset = caret_pos - input_width + 1
            elif caret_pos - input_offset < 0:
                input_offset = caret_pos

            # draw input
            wcb.addtext1(view, width, input_ran.start-input_offset, text+typeahead, input_ran)
            if input_offset > 0:
                wcb.addtext1(view, width, input_ran.start, "…", input_ran)
            if text_width + typeahead_width - input_offset > input_width - 1:
                wcb.addtext1(view, width, input_ran.start+input_width-1, "…", input_ran)

            # draw header
            wcb.addtext1(view, width, 0, icon, icon_ran)
            wcb.addtext1(view, width, marker_ran.start, marker, marker_ran)

            # draw caret
            if caret:
                caret_x = input_ran.start - input_offset + caret_pos
                caret_ran = wcb.select1(view, width, slice(caret_x, caret_x+1))
                view[caret_ran.start] = caret(view[caret_ran.start])

            view, width, (icon, marker, caret), (text, typeahead, caret_pos) = yield view
