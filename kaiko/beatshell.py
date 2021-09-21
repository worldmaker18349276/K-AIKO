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

def echo_str(escaped_str):
    r"""Interpret a string like bash's echo.
    It interprets the following backslash-escaped characters into:
        \a     alert (bell)
        \b     backspace
        \c     suppress further output
        \e     escape character
        \f     form feed
        \n     new line
        \r     carriage return
        \t     horizontal tab
        \v     vertical tab
        \\     backslash
        \0NNN  the character whose ASCII code is NNN (octal).  NNN can be 0 to 3 octal digits
        \xHH   the eight-bit character whose value is HH (hexadecimal).  HH can be one or two hex digits

    Parameters
    ----------
    escaped_str : str
        The string to be interpreted.

    Returns
    -------
    interpreted_str : str
        The interpreted string.
    """
    regex = r"\\c.*|\\[\\abefnrtv]|\\0[0-7]{0,3}|\\x[0-9a-fA-F]{1,2}|."

    escaped = {
        r"\\": "\\",
        r"\a": "\a",
        r"\b": "\b",
        r"\e": "\x1b",
        r"\f": "\f",
        r"\n": "\n",
        r"\r": "\r",
        r"\t": "\t",
        r"\v": "\v",
        }

    def repl(match):
        matched = match.group(0)

        if matched.startswith("\\c"):
            return ""
        elif matched in escaped:
            return escaped[matched]
        elif matched.startswith("\\0"):
            return chr(int(matched[2:] or "0", 8))
        elif matched.startswith("\\x"):
            return chr(int(matched[2:], 16))
        else:
            return matched

    return re.sub(regex, repl, escaped_str)

def pmove(width, x, text, tabsize=8):
    r"""Predict the position after print the given text in the terminal (GNOME terminal).

    Parameters
    ----------
    with : int
        The with of terminal.
    x : int
        The initial position before printing.
    text : str
        The string to print.
    tabsize : int, optional
        The tab size of terminal.

    Returns
    -------
    x : int
    y : int
        The final position after printing.
    """
    y = 0

    for ch, w in wcb.parse_attr(text):
        if ch == "\t":
            if tabsize > 0 and x < width:
                x = min((x+1) // -tabsize * -tabsize, width-1)

        elif ch == "\b":
            x = max(min(x, width-1)-1, 0)

        elif ch == "\r":
            x = 0

        elif ch == "\n":
            y += 1
            x = 0

        elif ch == "\v":
            y += 1

        elif ch == "\f":
            y += 1

        elif ch == "\x00":
            pass

        elif ch[0] == "\x1b":
            pass

        else:
            x += w
            if x > width:
                y += 1
                x = w

    return x, y


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

class INPUT_STATE(Enum):
    EDIT = "edit"
    TAB = "tab"
    FIN = "fin"

class ShellSyntaxError(Exception):
    pass

class BeatInput:
    def __init__(self, promptable, history=None):
        self.command = RootCommandParser(promptable)
        self.history = history if history is not None else []
        self.lock = threading.RLock()
        self.state = INPUT_STATE.FIN

        self.new_session(False)

    def prompt(self, settings=None):
        if settings is None:
            settings = BeatShellSettings()

        input_knot, controller = engines.Controller.create(engines.ControllerSettings())
        stroke = BeatStroke(self, settings.input.keymap)
        prompt = BeatPrompt(stroke, self, settings)

        stroke.register(controller)
        display_knot = dn.show(prompt.output_handler(), 1/settings.prompt.framerate, hide_cursor=True)

        # `dn.show`, `dn.input` will fight each other...
        @dn.datanode
        def slow(dt=0.1):
            import time
            try:
                yield
                prompt.ref_time = time.time()
                yield
                while True:
                    yield
            finally:
                time.sleep(dt)

        prompt_knot = dn.pipe(input_knot, slow(), display_knot)
        return prompt_knot, prompt

    @locked
    @onstate(INPUT_STATE.FIN)
    def new_session(self, record_current=True):
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
        self.result = None
        self.state = INPUT_STATE.EDIT

    @locked
    @onstate(INPUT_STATE.EDIT)
    def finish(self):
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
                token, slic, ignored = next(tokenizer)
            except StopIteration as e:
                self.lex_state = e.value
                break

            tokens.append((token, slic, ignored))

        types, _, _ = self.command.parse_command(token for token, _, _ in tokens)
        self.tokens = [(token, type, slic, ignored) for (token, slic, ignored), type in zip(tokens, types)]
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

        if token_type == TOKEN_TYPE.UNKNOWN:
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
            _, res, index = self.command.parse_command(token for token, _, _, _ in self.tokens)

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
            for token, _, slic, _ in self.tokens:
                start, stop, _ = slic.indices(len(self.buffer))
                if stop < self.pos:
                    parents.append(token)
                if start <= self.pos <= stop:
                    target = token
                    selection = slic

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
        del self._original_buffer
        del self._original_pos
        del self._selection
        self.state = INPUT_STATE.EDIT
        return True

class BeatStroke:
    def __init__(self, input, keymap):
        self.input = input
        self.keymap = keymap
        self.key_event = threading.Event()

    def register(self, controller):
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
            _, _, key, _ = args
            if key not in keys:
                with self.input.lock:
                    self.input.set_result(InputError, ValueError(f"Unknown key: " + repr(key or code)), None)
                    self.finish()
        return handler


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
        framerate: float = 60.0
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

class BeatPrompt:
    def __init__(self, stroke, input, settings):
        self.stroke = stroke

        self.input = input
        self.settings = settings
        self.result = None
        self.ref_time = None
        self.t0 = None
        self.tempo = None

    @dn.datanode
    def output_handler(self):
        size_node = dn.terminal_size()
        header_node = self.header_node()
        hint_node = self.hint_node()
        render_node = self.render_node()
        with size_node, header_node, hint_node, render_node:
            yield
            while True:
                # obtain input state
                with self.input.lock:
                    buffer = list(self.input.buffer)
                    tokens = list(self.input.tokens)
                    pos = self.input.pos
                    highlighted = self.input.highlighted

                    typeahead = self.input.typeahead
                    result = self.input.result
                    hint = self.input.hint
                    state = self.input.state

                size = size_node.send()

                # draw hint
                msg_data = hint_node.send(hint)

                # draw header
                clean = result is not None
                header_data = header_node.send(clean)

                # draw text
                text_data = self.render_text(buffer, tokens, typeahead, pos, highlighted, clean)

                # render
                output_text = render_node.send((header_data, text_data, msg_data, size))

                yield output_text

                # end
                if state == INPUT_STATE.FIN:
                    self.result = result.value
                    self.input.clear_result()
                    return

    @dn.datanode
    def header_node(self):
        framerate = self.settings.prompt.framerate

        headers = self.settings.prompt.headers

        caret_attr = self.settings.prompt.caret_attr
        caret_blink_ratio = self.settings.prompt.caret_blink_ratio

        clean = yield
        self.t0 = self.settings.prompt.t0
        self.tempo = self.settings.prompt.tempo

        n = 0
        t = (self.ref_time - self.t0)/(60/self.tempo)
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

            clean = yield header, caret
            n += 1
            t = (self.ref_time - self.t0 + n/framerate)/(60/self.tempo)

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
        msg = "\n" + msg + "\n" if msg else "\n"

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

        for token, type, slic, ignored in tokens:
            # render whitespace
            for index in indices[slic]:
                if rendered_buffer[index] == " ":
                    rendered_buffer[index] = whitespace

            # render escape
            for index in ignored:
                rendered_buffer[index] = wcb.add_attr(rendered_buffer[index], escape_attr)

            # render unknown token
            if type is TOKEN_TYPE.UNKNOWN:
                if slic.stop is not None or clean:
                    for index in indices[slic]:
                        rendered_buffer[index] = wcb.add_attr(rendered_buffer[index], token_unknown_attr)

            # render command token
            if type is TOKEN_TYPE.COMMAND:
                for index in indices[slic]:
                    rendered_buffer[index] = wcb.add_attr(rendered_buffer[index], token_command_attr)

            # render keyword token
            elif type is TOKEN_TYPE.KEYWORD:
                for index in indices[slic]:
                    rendered_buffer[index] = wcb.add_attr(rendered_buffer[index], token_keyword_attr)

            # render argument token
            elif type is TOKEN_TYPE.ARGUMENT:
                for index in indices[slic]:
                    rendered_buffer[index] = wcb.add_attr(rendered_buffer[index], token_argument_attr)

        if highlighted in range(len(tokens)):
            # render highlighted token
            _, _, slic, _ = tokens[highlighted]
            for index in indices[slic]:
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
        output_text = None

        while True:
            (header, caret), (text, typeahead, caret_pos), msg, size = yield output_text
            width = size.columns
            view = wcb.newwin1(width)

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

            # print message
            if msg is None:
                output_text = "\r\x1b[K" + "".join(view).rstrip() + "\r"
            elif msg == "":
                output_text = "\r\x1b[J" + "".join(view).rstrip() + "\r"
            else:
                _, y = pmove(width, 0, msg)
                output_text = "\r\x1b[J" + "".join(view).rstrip() + "\r" + msg + f"\x1b[{y}A"
