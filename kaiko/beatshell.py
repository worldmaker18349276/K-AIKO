import os
from enum import Enum
import functools
import re
import threading
from typing import List, Set, Tuple
import wcwidth
from . import engines
from . import datanodes as dn
from . import biparsers as bp
from . import wcbuffers as wcb
from . import config as cfg
from .commands import CommandUnfinishError, CommandParseError, TOKEN_TYPE, RootCommandParser


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
        keymap = {
            "Backspace"     : lambda input: input.backspace(),
            "Delete"        : lambda input: input.delete(),
            "Left"          : lambda input: input.move_left(),
            "Right"         : lambda input: input.insert_typeahead() or input.move_right(),
            "Up"            : lambda input: input.prev(),
            "Down"          : lambda input: input.next(),
            "Home"          : lambda input: input.move_to_start(),
            "End"           : lambda input: input.move_to_end(),
            "Enter"         : lambda input: input.enter(),
            "Esc"           : lambda input: (input.cancel_typeahead(), input.cancel_hint(), input.autocomplete(0)),
            "Alt_Enter"     : lambda input: input.help(),
            "Ctrl_Left"     : lambda input: input.move_to_word_start(),
            "Ctrl_Right"    : lambda input: input.move_to_word_end(),
            "Ctrl_Backspace": lambda input: input.delete_to_word_start(),
            "Ctrl_Delete"   : lambda input: input.delete_to_word_end(),
            "Tab"           : lambda input: input.autocomplete(+1),
            "Shift_Tab"     : lambda input: input.autocomplete(-1),
        }

    class prompt(cfg.Configurable):
        t0: float = 0.0
        tempo: float = 130.0

        headers: List[str] = [
            "\x1b[96;1m⠶⠦⣚⠀⠶\x1b[m\x1b[38;5;255m❯ \x1b[m",
            "\x1b[96;1m⢎⣀⡛⠀⠶\x1b[m\x1b[38;5;255m❯ \x1b[m",
            "\x1b[36m⢖⣄⠻⠀⠶\x1b[m\x1b[38;5;254m❯ \x1b[m",
            "\x1b[36m⠖⠐⡩⠂⠶\x1b[m\x1b[38;5;254m❯ \x1b[m",
            "\x1b[96m⠶⠀⡭⠲⠶\x1b[m\x1b[38;5;253m❯ \x1b[m",
            "\x1b[36m⠶⠀⣬⠉⡱\x1b[m\x1b[38;5;253m❯ \x1b[m",
            "\x1b[36m⠶⠀⣦⠙⠵\x1b[m\x1b[38;5;252m❯ \x1b[m",
            "\x1b[36m⠶⠠⣊⠄⠴\x1b[m\x1b[38;5;252m❯ \x1b[m",

            "\x1b[96m⠶⠦⣚⠀⠶\x1b[m\x1b[38;5;251m❯ \x1b[m",
            "\x1b[36m⢎⣀⡛⠀⠶\x1b[m\x1b[38;5;251m❯ \x1b[m",
            "\x1b[36m⢖⣄⠻⠀⠶\x1b[m\x1b[38;5;250m❯ \x1b[m",
            "\x1b[36m⠖⠐⡩⠂⠶\x1b[m\x1b[38;5;250m❯ \x1b[m",
            "\x1b[96m⠶⠀⡭⠲⠶\x1b[m\x1b[38;5;249m❯ \x1b[m",
            "\x1b[36m⠶⠀⣬⠉⡱\x1b[m\x1b[38;5;249m❯ \x1b[m",
            "\x1b[36m⠶⠀⣦⠙⠵\x1b[m\x1b[38;5;248m❯ \x1b[m",
            "\x1b[36m⠶⠠⣊⠄⠴\x1b[m\x1b[38;5;248m❯ \x1b[m",

            "\x1b[96m⠶⠦⣚⠀⠶\x1b[m\x1b[38;5;247m❯ \x1b[m",
            "\x1b[36m⢎⣀⡛⠀⠶\x1b[m\x1b[38;5;247m❯ \x1b[m",
            "\x1b[36m⢖⣄⠻⠀⠶\x1b[m\x1b[38;5;246m❯ \x1b[m",
            "\x1b[36m⠖⠐⡩⠂⠶\x1b[m\x1b[38;5;246m❯ \x1b[m",
            "\x1b[96m⠶⠀⡭⠲⠶\x1b[m\x1b[38;5;245m❯ \x1b[m",
            "\x1b[36m⠶⠀⣬⠉⡱\x1b[m\x1b[38;5;245m❯ \x1b[m",
            "\x1b[36m⠶⠀⣦⠙⠵\x1b[m\x1b[38;5;244m❯ \x1b[m",
            "\x1b[36m⠶⠠⣊⠄⠴\x1b[m\x1b[38;5;244m❯ \x1b[m",

            "\x1b[96m⠶⠦⣚⠀⠶\x1b[m\x1b[38;5;243m❯ \x1b[m",
            "\x1b[36m⢎⣀⡛⠀⠶\x1b[m\x1b[38;5;243m❯ \x1b[m",
            "\x1b[36m⢖⣄⠻⠀⠶\x1b[m\x1b[38;5;242m❯ \x1b[m",
            "\x1b[36m⠖⠐⡩⠂⠶\x1b[m\x1b[38;5;242m❯ \x1b[m",
            "\x1b[96m⠶⠀⡭⠲⠶\x1b[m\x1b[38;5;241m❯ \x1b[m",
            "\x1b[36m⠶⠀⣬⠉⡱\x1b[m\x1b[38;5;241m❯ \x1b[m",
            "\x1b[36m⠶⠀⣦⠙⠵\x1b[m\x1b[38;5;240m❯ \x1b[m",
            "\x1b[36m⠶⠠⣊⠄⠴\x1b[m\x1b[38;5;240m❯ \x1b[m",
        ]
        header_width: int = 7

        caret_attr: Tuple[str, str] = ("7;2", "7;1")
        caret_blink_ratio: float = 0.3

    class text(cfg.Configurable):
        error_message_attr: str = "31"
        info_message_attr: str = "2"
        message_max_lines: int = 16

        escape_attr: str = "2"
        typeahead_attr: str = "2"
        whitespace: str = "\x1b[2m⌴\x1b[m"

        token_unknown_attr: str = "31"
        token_command_attr: str = "94"
        token_keyword_attr: str = "95"
        token_argument_attr: str = "92"
        token_highlight_attr: str = "4"


class InputWarn:
    def __init__(self, message, tokens):
        self.message = message
        self.tokens = tokens

class InputMessage:
    def __init__(self, message, tokens):
        self.message = message
        self.tokens = tokens

class InputError:
    def __init__(self, value):
        self.value = value

class InputComplete:
    def __init__(self, value):
        self.value = value

class INPUT_STATE(Enum):
    EDIT = "edit"
    TAB = "tab"
    FIN = "fin"

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
    command : RootCommandParser
        The root command parser for beatshell.
    history : list of str
        The input history.
    buffers : list of list of str
        The editable buffers of input history.
    buffers_index : int
        The negative index of current input buffer.
    buffer : list of str
        The buffer of current input.
    pos : int
        The caret position of input.
    typeahead : str
        The type ahead of input.
    tokens : list
        The tokens info, which is a list of tuple `(token, type, mask, ignored)`,
        where `type` is TOKEN_TYPE or None, and the rest are the same as the
        values yielded by `shlexer_tokenize`.
    lex_state : SHLEXER_STATE
        The shlexer state.
    highlighted : int or None
        The index of highlighted token.
    hint : InputWarn or InputMessage or None
        The hint of input.
    result : InputError or InputComplete or None
        The result of input.
    state : INPUT_STATE
        The input state.
    """
    def __init__(self, promptable, history=None):
        r"""Constructor.

        Parameters
        ----------
        promptable : object
            The root command.
        history : list of str
            The input history.
        """
        self.command = RootCommandParser(promptable)
        self.history = history if history is not None else []
        self.lock = threading.RLock()
        self.state = INPUT_STATE.FIN

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
        prompt_knot : dn.DataNode
            The datanode to execute the prompt.
        """
        if settings is None:
            settings = BeatShellSettings()

        input_knot, controller = engines.Controller.create(devices_settings.controller)
        display_knot, renderer = engines.Renderer.create(devices_settings.renderer)
        stroke = BeatStroke(self, settings.input.keymap)
        prompt = BeatPrompt(stroke, self, settings)

        stroke.register(controller)
        prompt.register(renderer)

        @dn.datanode
        def stop_when(event):
            yield
            yield
            while not event.is_set():
                yield

        return dn.pipe(stop_when(prompt.stop_event), display_knot, input_knot)

    @locked
    @onstate(INPUT_STATE.FIN)
    def new_session(self, record_current=True):
        r"""Start a new session of input.

        Parameters
        ----------
        record_current : bool, optional
            Recording current input state.
        """
        if record_current:
            self.history.append("".join(self.buffer))

        self.buffers = [list(history_buffer) for history_buffer in self.history]
        self.buffers.append([])
        self.buffers_index = -1

        self.buffer = self.buffers[self.buffers_index]
        self.pos = len(self.buffer)
        self.typeahead = ""

        self.tokens = []
        self.lex_state = SHLEXER_STATE.SPACED
        self.highlighted = None

        self.hint = None
        self.result = None
        self.state = INPUT_STATE.EDIT

    @locked
    @onstate(INPUT_STATE.FIN)
    def prev_session(self):
        r"""Back to previous session of input."""
        self.highlighted = None
        self.result = None
        self.state = INPUT_STATE.EDIT

    @locked
    @onstate(INPUT_STATE.EDIT)
    def finish(self):
        r"""Finish this session of input.

        Returns
        -------
        succ : bool
        """
        self.state = INPUT_STATE.FIN
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
    def set_hint(self, hint_type, message, index=None):
        """Set hint.
        Show hint below the prompt.

        Parameters
        ----------
        hint_type : InputWarn or InputMessage
            The type of hint.
        message : str
            The message to show.
        index : int or None
            Index of the token to which the hint is directed, or `None` for nothing.

        Returns
        -------
        succ : bool
        """
        self.highlighted = index
        if index is None:
            msg_tokens = None
        elif hint_type == InputWarn:
            msg_tokens = self.tokens[:index]
        else:
            msg_tokens = self.tokens[:index+1]
        self.hint = hint_type(message, msg_tokens)
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
    @onstate(INPUT_STATE.EDIT)
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
    @onstate(INPUT_STATE.EDIT)
    def help(self):
        """Help for command.
        Provide some hint for the command before the caret.

        Returns
        -------
        succ : bool
            `False` if there is no hint.
        """
        self.cancel_hint()

        # find the last token before the caret
        for index, (target, token_type, slic, _) in reversed(list(enumerate(self.tokens))):
            if slic.start is None or slic.start <= self.pos:
                break
        else:
            return False

        parents = [token for token, _, _, _ in self.tokens[:index]]

        if token_type is None:
            msg = self.command.desc_command(parents)
            hint_type = InputWarn
        else:
            msg = self.command.info_command(parents, target)
            hint_type = InputMessage

        if msg is None:
            return False
        else:
            self.set_hint(hint_type, msg, index)
            return True

    @locked
    @onstate(INPUT_STATE.EDIT)
    def enter(self):
        """Enter.
        Finish the command.

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

        if isinstance(res, CommandUnfinishError):
            self.set_result(InputError, res, None)
            self.finish()
            return False
        elif isinstance(res, (CommandParseError, ShellSyntaxError)):
            self.set_result(InputError, res, index)
            self.finish()
            return False
        else:
            self.set_result(InputComplete, res)
            self.finish()
            return True

    @locked
    def unknown_key(self, key):
        self.set_result(InputError, ValueError(f"Unknown key: " + key), None)
        self.finish()

    @locked
    @onstate(INPUT_STATE.EDIT)
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
    @onstate(INPUT_STATE.EDIT)
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
    @onstate(INPUT_STATE.EDIT)
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
    @onstate(INPUT_STATE.EDIT)
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
    @onstate(INPUT_STATE.EDIT)
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
    @onstate(INPUT_STATE.EDIT)
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
    @onstate(INPUT_STATE.EDIT)
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
    @onstate(INPUT_STATE.EDIT)
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
    @onstate(INPUT_STATE.EDIT)
    def move_left(self):
        """Move caret one character to the left.

        Returns
        -------
        succ : bool
        """
        return self.move(-1)

    @locked
    @onstate(INPUT_STATE.EDIT)
    def move_right(self):
        """Move caret one character to the right.

        Returns
        -------
        succ : bool
        """
        return self.move(+1)

    @locked
    @onstate(INPUT_STATE.EDIT)
    def move_to_start(self):
        """Move caret to the start of buffer.

        Returns
        -------
        succ : bool
        """
        return self.move_to(0)

    @locked
    @onstate(INPUT_STATE.EDIT)
    def move_to_end(self):
        """Move caret to the end of buffer.

        Returns
        -------
        succ : bool
        """
        return self.move_to(None)

    @locked
    @onstate(INPUT_STATE.EDIT)
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
    @onstate(INPUT_STATE.EDIT)
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
    @onstate(INPUT_STATE.EDIT)
    def prev(self):
        """Previous buffer.

        Returns
        -------
        succ : bool
        """
        if self.buffers_index == -len(self.buffers):
            return False
        self.buffers_index -= 1

        self.buffer = self.buffers[self.buffers_index]
        self.pos = len(self.buffer)
        self.parse_syntax()
        self.cancel_typeahead()
        self.cancel_hint()

        return True

    @locked
    @onstate(INPUT_STATE.EDIT)
    def next(self):
        """Next buffer.

        Returns
        -------
        succ : bool
        """
        if self.buffers_index == -1:
            return False
        self.buffers_index += 1

        self.buffer = self.buffers[self.buffers_index]
        self.pos = len(self.buffer)
        self.parse_syntax()
        self.cancel_typeahead()
        self.cancel_hint()

        return True

    @locked
    @onstate(INPUT_STATE.EDIT, INPUT_STATE.TAB)
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

        if self.state == INPUT_STATE.EDIT:
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
            self._suggestions = [shlexer_quoting(sugg) for sugg in self.command.suggest_command(parents, target)]
            self._sugg_index = len(self._suggestions) if action == -1 else -1

            # tab state
            self._original_buffer = list(self.buffer)
            self._original_pos = self.pos
            self._selection = selection
            self.state = INPUT_STATE.TAB

        if action == +1:
            self._sugg_index += 1
        elif action == -1:
            self._sugg_index -= 1
        elif action == 0:
            self._sugg_index = None
        else:
            raise ValueError

        self.buffer = list(self._original_buffer)
        self.pos = self._original_pos

        if self._sugg_index in range(len(self._suggestions)):
            # autocomplete selected token
            self.buffer[self._selection] = self._suggestions[self._sugg_index]
            self.pos = self._selection.start + len(self._suggestions[self._sugg_index])
            self.parse_syntax()
            return True

        else:
            # restore state
            del self._original_buffer
            del self._original_pos
            del self._selection
            self.state = INPUT_STATE.EDIT
            self.parse_syntax()
            return False

    @locked
    @onstate(INPUT_STATE.TAB)
    def finish_autocomplete(self):
        r"""Finish autocompletion.

        Returns
        -------
        succ : bool
        """
        del self._original_buffer
        del self._original_pos
        del self._selection
        self.state = INPUT_STATE.EDIT
        return True

class BeatStroke:
    r"""Keyboard controller for beatshell."""

    def __init__(self, input, keymap):
        self.input = input
        self.keymap = keymap
        self.key_event = threading.Event()

    def register(self, controller):
        r"""Register handler to the given controller.

        Parameters
        ----------
        controller : engines.Controller
        """
        controller.add_handler(self.keypress_handler())
        controller.add_handler(self.finish_autocomplete_handler())
        for key, func in self.keymap.items():
            controller.add_handler(self.action_handler(func), key)
        controller.add_handler(self.printable_handler(), "PRINTABLE")
        controller.add_handler(self.unknown_handler(self.keymap))

    def keypress_handler(self):
        return lambda args: self.key_event.set()

    def finish_autocomplete_handler(self):
        return lambda args: self.input.finish_autocomplete() if args[2] not in ("Tab", "Shift_Tab", "Esc") else False

    def action_handler(self, func):
        return lambda args: func(self.input)

    def printable_handler(self):
        return lambda args: self.input.input(args[3])

    def unknown_handler(self, keymap):
        keys = list(keymap.keys())
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
        self.stop_event = threading.Event()
        self.t0 = None
        self.tempo = None

    def register(self, renderer):
        renderer.add_drawer(self.output_handler())

    @dn.datanode
    def output_handler(self):
        header_node = self.header_node()
        hint_node = self.hint_node()
        render_node = self.render_node()
        with header_node, hint_node, render_node:
            (view, msg), time, width = yield
            while True:
                # obtain input state
                with self.input.lock:
                    buffer = list(self.input.buffer)
                    tokens = list(self.input.tokens)
                    pos = self.input.pos
                    highlighted = self.input.highlighted

                    typeahead = self.input.typeahead
                    clean = self.input.result is not None
                    hint = self.input.hint
                    state = self.input.state

                # draw hint
                msg = hint_node.send(hint)

                # draw header
                header_data = header_node.send((clean, time))

                # draw text
                text_data = self.render_text(buffer, tokens, typeahead, pos, highlighted, clean)

                # render
                view = render_node.send((view, width, header_data, text_data))

                (view, msg), time, width = yield (view, msg)

                # end
                if state == INPUT_STATE.FIN and not self.stop_event.is_set():
                    self.stop_event.set()

    @dn.datanode
    def header_node(self):
        headers = self.settings.prompt.headers

        caret_attr = self.settings.prompt.caret_attr
        caret_blink_ratio = self.settings.prompt.caret_blink_ratio

        clean, time = yield
        self.t0 = self.settings.prompt.t0
        self.tempo = self.settings.prompt.tempo

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
                if t % 4 < caret_blink_ratio:
                    caret = lambda s: wcb.add_attr(s, caret_attr[1])
                else:
                    caret = lambda s: wcb.add_attr(s, caret_attr[0])
            else:
                caret = None

            # render header
            ind = int(t / 4 * len(headers) // 1) % len(headers)
            header = headers[ind]

            clean, time = yield header, caret
            t = (time - self.t0)/(60/self.tempo)

    @dn.datanode
    def hint_node(self):
        current_hint = None
        hint = yield
        while True:
            # track changes of the hint
            if hint == current_hint:
                hint = yield None
                continue

            current_hint = hint

            # show hint
            hint = yield self.render_hint(hint)

    def render_hint(self, hint):
        if hint is None:
            return ""

        message_max_lines = self.settings.text.message_max_lines
        error_message_attr = self.settings.text.error_message_attr
        info_message_attr = self.settings.text.info_message_attr

        # show hint
        msg = hint.message or ""
        if msg.count("\n") >= message_max_lines:
            msg = "\n".join(msg.split("\n")[:message_max_lines]) + "\x1b[m\n…"
        if msg:
            if isinstance(hint, InputWarn):
                msg = wcb.add_attr(msg, error_message_attr)
            if isinstance(hint, (InputWarn, InputMessage)):
                msg = wcb.add_attr(msg, info_message_attr)

        return msg

    def render_text(self, rendered_buffer, tokens, typeahead, pos, highlighted, clean):
        escape_attr     = self.settings.text.escape_attr
        typeahead_attr  = self.settings.text.typeahead_attr
        whitespace      = self.settings.text.whitespace

        token_unknown_attr   = self.settings.text.token_unknown_attr
        token_command_attr   = self.settings.text.token_command_attr
        token_keyword_attr   = self.settings.text.token_keyword_attr
        token_argument_attr  = self.settings.text.token_argument_attr
        token_highlight_attr = self.settings.text.token_highlight_attr

        # render buffer
        indices = range(len(rendered_buffer))

        for token, type, mask, ignored in tokens:
            # render whitespace
            for index in indices[mask]:
                if rendered_buffer[index] == " ":
                    rendered_buffer[index] = whitespace

            # render escape
            for index in ignored:
                rendered_buffer[index] = wcb.add_attr(rendered_buffer[index], escape_attr)

            # render unknown token
            if type is None:
                if mask.stop is not None or clean:
                    for index in indices[mask]:
                        rendered_buffer[index] = wcb.add_attr(rendered_buffer[index], token_unknown_attr)

            # render command token
            if type is TOKEN_TYPE.COMMAND:
                for index in indices[mask]:
                    rendered_buffer[index] = wcb.add_attr(rendered_buffer[index], token_command_attr)

            # render keyword token
            elif type is TOKEN_TYPE.KEYWORD:
                for index in indices[mask]:
                    rendered_buffer[index] = wcb.add_attr(rendered_buffer[index], token_keyword_attr)

            # render argument token
            elif type is TOKEN_TYPE.ARGUMENT:
                for index in indices[mask]:
                    rendered_buffer[index] = wcb.add_attr(rendered_buffer[index], token_argument_attr)

        if highlighted in range(len(tokens)):
            # render highlighted token
            _, _, mask, _ = tokens[highlighted]
            for index in indices[mask]:
                rendered_buffer[index] = wcb.add_attr(rendered_buffer[index], token_highlight_attr)

        rendered_text = "".join(rendered_buffer)

        # render typeahead
        if typeahead and not clean:
            rendered_typeahead = wcb.add_attr(typeahead, typeahead_attr)
        else:
            rendered_typeahead = ""

        # compute caret position
        _, caret_pos = wcb.textrange1(0, "".join(rendered_buffer[:pos]))

        return rendered_text, rendered_typeahead, caret_pos

    @dn.datanode
    def render_node(self):
        header_width = self.settings.prompt.header_width
        header_ran = slice(None, header_width)
        input_ran = slice(header_width, None)

        input_offset = 0

        view, width, (header, caret), (text, typeahead, caret_pos) = yield

        while True:
            # adjust input offset
            input_width = len(range(width)[input_ran])
            _, solid_text_width = wcb.textrange1(0, text)

            if solid_text_width - input_offset + 1 <= input_width:
                input_offset = max(0, solid_text_width-input_width+1)
            if caret_pos - input_offset >= input_width:
                input_offset = caret_pos - input_width + 1
            elif caret_pos - input_offset < 0:
                input_offset = caret_pos

            # draw input
            _, text_width = wcb.textrange1(0, text+typeahead)
            wcb.addtext1(view, width, input_ran.start-input_offset, text+typeahead, input_ran)
            if input_offset > 0:
                wcb.addtext1(view, width, input_ran.start, "…", input_ran)
            if text_width-input_offset >= input_width:
                wcb.addtext1(view, width, input_ran.start+input_width-1, "…", input_ran)

            # draw header
            wcb.addtext1(view, width, 0, header, header_ran)

            # draw caret
            if caret:
                caret_x = input_ran.start - input_offset + caret_pos
                caret_ran = wcb.select1(view, width, slice(caret_x, caret_x+1))
                view[caret_ran.start] = caret(view[caret_ran.start])

            view, width, (header, caret), (text, typeahead, caret_pos) = yield view
