import functools
import contextlib
import re
import queue
import threading
from typing import Optional, List, Tuple, Dict, Callable
from pathlib import Path
import dataclasses
from ..utils import commands as cmd
from ..utils import config as cfg
from ..utils import shells as sh


# input
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
    index : Optional[int]
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


class TextBuffer:
    r"""Text buffer for beatshell.

    Attributes
    ----------
    buffers : list of list of str
        The editable buffers of input history.
    buffer_index : int
        The negative index of current input buffer.
    buffer : list of str
        The buffer of current input.
    pos : int
        The caret position of input.
    """

    def __init__(self, history):
        self.buffers = history
        self.buffers.append([])
        self.buffer_index = -1
        self.pos = len(self.buffer)

    @property
    def buffer(self):
        return self.buffers[self.buffer_index]

    def prev(self):
        if self.buffer_index == -len(self.buffers):
            return False
        self.buffer_index -= 1
        self.pos = len(self.buffer)
        return True

    def next(self):
        if self.buffer_index == -1:
            return False
        self.buffer_index += 1
        self.pos = len(self.buffer)
        return True

    def replace(self, selection, text):
        if not all(ch.isprintable() for ch in text):
            raise ValueError("invalid text to insert: " + repr("".join(text)))

        start, stop, _ = selection.indices(len(self.buffer))
        self.buffer[selection] = text
        selection = slice(start, start + len(text))
        if start <= self.pos < stop:
            self.pos = selection.stop
        elif stop <= self.pos:
            self.pos += selection.stop - stop
        return selection

    def insert(self, text):
        text = list(text)

        if len(text) == 0:
            return False

        while len(text) > 0 and text[0] == "\b":
            del text[0]
            del self.buffer[self.pos - 1]
            self.pos = self.pos - 1

        self.replace(slice(self.pos, self.pos), text)
        return True

    def backspace(self):
        if self.pos == 0:
            return False
        self.replace(slice(self.pos-1, self.pos), "")
        return True

    def delete(self):
        if self.pos >= len(self.buffer):
            return False
        self.replace(slice(self.pos, self.pos+1), "")
        return True

    def delete_all(self):
        if not self.buffer:
            return False
        self.replace(slice(None, None), "")
        return True

    def move_to(self, pos):
        pos = (
            min(max(0, pos), len(self.buffer)) if pos is not None else len(self.buffer)
        )

        if self.pos == pos:
            return False

        self.pos = pos
        return True

    def to_word_start(self):
        for match in re.finditer(r"\w+|\W+", "".join(self.buffer)):
            if match.end() >= self.pos:
                return slice(match.start(), self.pos)
        else:
            return slice(0, self.pos)

    def to_word_end(self):
        for match in re.finditer(r"\w+|\W+", "".join(self.buffer)):
            if match.end() > self.pos:
                return slice(self.pos, match.end())
        else:
            return slice(self.pos, len(self.buffer))


class BeatInputSettings(cfg.Configurable):
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
        The keymap of beatshell. The key of dict is the keystroke, and the
        value of dict is the action to activate. The format of action is
        just like a normal python code: `input.insert_typeahead() or
        input.move_right()`. The syntax is::

            <function> ::= "input." /(?!_)\w+/ "()"
            <operator> ::= " | " | " & " | " and " | " or "
            <action> ::= (<function> <operator>)* <function>

    preview_song : bool
        Whether to preview the song when selected.
    history_size : int
        The maximum history size.
    """
    confirm_key: str = "Enter"
    help_key: str = "Alt_Enter"
    autocomplete_keys: Tuple[str, str, str] = ("Tab", "Shift_Tab", "Esc")

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

    preview_song: bool = True
    history_size: int = 500


class ContextDispatcher:
    def __init__(self):
        self.lock = threading.RLock()
        self.isin = False
        self.before_callbacks = []
        self.after_callbacks = []
        self.onerror_callbacks = []

    def before(self, callback):
        with self.lock:
            self.before_callbacks.append(callback)

    def after(self, callback):
        with self.lock:
            self.after_callbacks.append(callback)

    def onerror(self, callback):
        with self.lock:
            self.onerror_callbacks.append(callback)

    @contextlib.contextmanager
    def on(self):
        with self.lock:
            isin = self.isin
            if isin:
                for callback in self.before_callbacks:
                    callback()
            self.isin = False
            try:
                yield
            except:
                self.isin = isin
                if isin:
                    for callback in self.onerror_callbacks:
                        callback()
            finally:
                self.isin = isin
                if isin:
                    for callback in self.after_callbacks:
                        callback()


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
        with self.edit_ctxt.on():
            return func(self, *args, **kwargs)

    return locked_func


class BeatInput:
    r"""Input editor for beatshell.

    Attributes
    ----------
    input_settings : BeatInputSettings
        The input settings.
    command_parser_getter : function
        The function to produce command parser for beatshell.
    semantic_analyzer : shells.SemanticAnalyzer
        The syntax analyzer.
    rich : markups.RichParser
        The rich parser.
    cache_dir : Path
        The directory of cache data.
    history : HistoryManager
        The input history manager.
    text_buffer : TextBuffer
        The text buffer of beatshell.
    typeahead : str
        The type ahead of input.
    tab_state : TabState or None
        The state of autocomplete.
    hint_state : HintState or None
        The hint state of input.
    popup_queue : queue.Queue
        The message displayed above the prompt.
    result : Result or None
        The result of input.
    state : str
        The input state.
    preview_handler : function
        A function to preview beatmap.
    buffer_modified_counter : int
        The event counter for modifying buffer.
    key_pressed_counter : int
        The event counter for key pressing.
    """

    history_file_path = ".beatshell-history"

    action_regex = "({fn}{op})*{fn}".format(
        fn=r"input\.(?!_)\w+\(\)",
        op=r"( \| | \& | and | or )",
    )

    @classmethod
    def _parse_action(cls, func):
        if not re.match(cls.action_regex, func):
            raise ValueError(f"invalid action: {repr(func)}")
        def action(input):
            with input.edit_ctxt.on():
                eval(func, {}, {"input": input})
        return action

    def __init__(
        self,
        command_parser_getter,
        preview_handler,
        rich,
        cache_dir,
        input_settings_getter=BeatInputSettings,
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
        input_settings_getter : BeatInputSettings
            The settings getter of input.
        """
        self.rich = rich

        self.command_parser_getter = command_parser_getter
        self._input_settings_getter = input_settings_getter

        self.semantic_analyzer = sh.SemanticAnalyzer(None)
        self.cache_dir = cache_dir
        self.history = HistoryManager(self.cache_dir / self.history_file_path)

        self.text_buffer = TextBuffer([])
        self.typeahead = ""
        self.tab_state = None
        self.hint_state = None
        self.popup_queue = queue.Queue()

        self.state = "FIN"
        self.result = None
        self.edit_ctxt = ContextDispatcher()

        self.preview_handler = preview_handler
        self.key_pressed_counter = 0
        self.buffer_modified_counter = 0

        self.new_session()

    @property
    def input_settings(self):
        return self._input_settings_getter()

    def _register(self, controller):
        stroke = BeatStroke(self, self.input_settings)
        stroke.register(controller)

    @locked
    @onstate("FIN")
    def new_session(self):
        r"""Start a new session of input.
        """
        self.semantic_analyzer.update_parser(self.command_parser_getter())

        groups = self.semantic_analyzer.get_all_groups()
        history_size = self.input_settings.history_size
        self.text_buffer = TextBuffer(self.history.read_history(groups, history_size))

        self.cancel_typeahead()
        self.update_buffer()
        self.cancel_hint()
        self.start()

    def record_command(self):
        command = "".join(self.text_buffer.buffer).strip()
        self.history.write_history(self.semantic_analyzer.group, command)

    @locked
    @onstate("FIN")
    def prev_session(self):
        r"""Back to previous session of input."""
        self.cancel_typeahead()
        self.update_buffer()
        self.cancel_hint()
        self.start()

    @locked
    @onstate("EDIT")
    def finish(self, res):
        r"""Finish this session of input.

        Parameters
        ----------
        res : Result
            The result.

        Returns
        -------
        succ : bool
        """
        self.result = res
        self.state = "FIN"
        return True

    @locked
    @onstate("FIN")
    def start(self):
        """Start a session of input.

        Returns
        -------
        succ : bool
        """
        self.result = None
        self.state = "EDIT"
        return True

    @locked
    def update_buffer(self):
        """Parse syntax.

        Returns
        -------
        succ : bool
        """
        self.semantic_analyzer.parse(self.text_buffer.buffer)
        self.buffer_modified_counter += 1
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
        if self.text_buffer.pos != len(self.text_buffer.buffer):
            self.typeahead = ""
            return False

        # search history
        pos = self.text_buffer.pos
        for buffer in reversed(self.text_buffer.buffers):
            if len(buffer) > pos and buffer[:pos] == self.text_buffer.buffer:
                self.typeahead = "".join(buffer[pos:])
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
    def add_popup(self, hint):
        """Add popup.

        Show hint above the prompt.

        Parameters
        ----------
        hint : Hint
            The hint.

        Returns
        -------
        succ : bool
        """
        self.popup_queue.put(hint)
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
        if self.hint_state is None:
            return False
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

        if self.hint_state.index is not None and self.hint_state.index >= len(self.semantic_analyzer.tokens):
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
        if not self.input_settings.preview_song:
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

        if self.typeahead == "" or self.text_buffer.pos != len(self.text_buffer.buffer):
            return False

        selection = slice(self.text_buffer.pos, self.text_buffer.pos)
        self.text_buffer.replace(selection, self.typeahead)

        self.cancel_typeahead()
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
        succ = self.text_buffer.insert(text)
        if not succ:
            return False

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
        succ = self.text_buffer.backspace()
        if not succ:
            return False

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
        succ = self.text_buffer.delete()
        if not succ:
            return False

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
        succ = self.text_buffer.delete_all()
        if not succ:
            return False

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
        self.text_buffer.replace(slice(start, end), "")

        self.update_buffer()
        self.cancel_typeahead()
        self.ask_for_hint(clear=True)

        return True

    def _find_token(self, pos=None):
        pos = pos if pos is not None else self.text_buffer.pos
        for index, token in enumerate(self.semantic_analyzer.tokens):
            if token.mask.start <= pos <= token.mask.stop:
                return index, token
        else:
            return None, None

    def _find_token_before(self, pos=None):
        pos = pos if pos is not None else self.text_buffer.pos
        for index, token in enumerate(reversed(self.semantic_analyzer.tokens)):
            if token.mask.start <= pos:
                return len(self.semantic_analyzer.tokens)-1-index, token
        else:
            return None, None

    def _find_token_after(self, pos=None):
        pos = pos if pos is not None else self.text_buffer.pos
        for token in self.semantic_analyzer.tokens:
            if pos <= token.mask.stop:
                return token
        else:
            return None

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

        _, token = self._find_token_before()
        if token is None:
            return self.delete_range(0, self.text_buffer.pos)
        else:
            return self.delete_range(token.mask.start, max(self.text_buffer.pos, token.mask.stop))

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

        _, token = self._find_token_after()
        if token is None:
            return self.delete_range(self.text_buffer.pos, None)
        else:
            return self.delete_range(min(self.text_buffer.pos, token.mask.start), token.mask.stop)

    @locked
    @onstate("EDIT")
    def delete_to_word_start(self):
        """Delete to the word start.

        The word is defined as `\\w+|\\W+`.

        Returns
        -------
        succ : bool
        """
        mask = self.text_buffer.to_word_start()
        return self.delete_range(mask.start, mask.stop)

    @locked
    @onstate("EDIT")
    def delete_to_word_end(self):
        """Delete to the word end.

        The word is defined as `\\w+|\\W+`.

        Returns
        -------
        succ : bool
        """
        mask = self.text_buffer.to_word_end()
        return self.delete_range(mask.start, mask.stop)

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
        succ = self.text_buffer.move_to(pos)
        self.cancel_typeahead()
        return succ

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
        return self.move_to(self.text_buffer.pos + offset)

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
        mask = self.text_buffer.to_word_start()
        return self.move_to(mask.start)

    @locked
    @onstate("EDIT")
    def move_to_word_end(self):
        """Move caret to the end of the word.

        Returns
        -------
        succ : bool
        """
        mask = self.text_buffer.to_word_end()
        return self.move_to(mask.stop)

    @locked
    @onstate("EDIT")
    def prev(self):
        """Previous buffer.

        Returns
        -------
        succ : bool
        """
        succ = self.text_buffer.prev()
        if not succ:
            return False

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
        succ = self.text_buffer.next()
        if not succ:
            return False

        self.update_buffer()
        self.cancel_typeahead()
        self.cancel_hint()

        return True

    @locked
    @onstate("EDIT")
    def ask_for_hint(self, index=None, clear=False, type="all"):
        """Ask some hint for command.

        Provide some hint for the command on the caret.

        Parameters
        ----------
        index : int
        clear : bool
            Cancel the current hint if token was not found.
        type : one of "info", "desc", "all"
            The type of hint to ask.

        Returns
        -------
        succ : bool
        """
        if index is None:
            index, token = self._find_token()
            if token is None:
                if clear:
                    self.cancel_hint()
                return False

        target_type = self.semantic_analyzer.tokens[index].type

        if target_type is None:
            if type not in ("all", "desc"):
                self.cancel_hint()
                return False
            msg = self.semantic_analyzer.desc(index)
            if msg is None:
                self.cancel_hint()
                return False
            hint = DescHint(msg)
            self.set_hint(hint, index)
            return True

        else:
            if type not in ("all", "info"):
                self.cancel_hint()
                return False
            msg = self.semantic_analyzer.info(index + 1)
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
        index, token = self._find_token_before()
        if token is None:
            return False

        if self.hint_state is None or self.hint_state.index != index:
            self.ask_for_hint(index)
            return False

        hint = self.hint_state.hint
        if not hint.message:
            return False
        if isinstance(hint, SuggestionsHint):
            hint = InfoHint(hint.message)

        self.add_popup(hint)
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
            self.finish(CompleteResult(lambda: None))
            return True

        if self.semantic_analyzer.lex_state == sh.SHLEXER_STATE.BACKSLASHED:
            res, index = ShellSyntaxError("No escaped character"), len(self.semantic_analyzer.tokens) - 1
        elif self.semantic_analyzer.lex_state == sh.SHLEXER_STATE.QUOTED:
            res, index = ShellSyntaxError("No closing quotation"), len(self.semantic_analyzer.tokens) - 1
        else:
            res, index = self.semantic_analyzer.result, self.semantic_analyzer.length

        if isinstance(res, cmd.CommandUnfinishError):
            self.finish(ErrorResult(None, res))
            return False
        elif isinstance(res, (cmd.CommandParseError, ShellSyntaxError)):
            self.finish(ErrorResult(index, res))
            return False
        else:
            self.finish(CompleteResult(res))
            return True

    @locked
    @onstate("EDIT")
    def exit_if_empty(self):
        """Finish the command.

        Returns
        -------
        succ : bool
            `False` if unfinished or the command is wrong.
        """
        if self.text_buffer.buffer:
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
            `True` if is in autocompletion cycle.  Note that it will be `False`
            for no suggestion or one suggestion case.
        """

        if self.tab_state is None and action == 0:
            return False

        if self.tab_state is None:
            self.cancel_typeahead()
            tab_state = self._prepare_tab_state(action)

            if len(tab_state.suggestions) == 0:
                # no suggestion
                return False

            if len(tab_state.suggestions) == 1:
                # one suggestion -> complete directly
                self.text_buffer.replace(tab_state.selection, tab_state.suggestions[0])
                self.update_buffer()
                self.ask_for_hint(tab_state.token_index, type="info")
                return False

            self.tab_state = tab_state

        if action == +1:
            self.tab_state.sugg_index += 1
        elif action == -1:
            self.tab_state.sugg_index -= 1
        elif action == 0:
            self.tab_state.sugg_index = None
        else:
            raise ValueError(f"invalid action: {action}")

        if self.tab_state.sugg_index not in range(len(self.tab_state.suggestions)):
            # restore state
            self.text_buffer.replace(self.tab_state.selection, self.tab_state.original_token)
            self.text_buffer.move_to(self.tab_state.original_pos)

            self.tab_state = None
            self.update_buffer()
            self.cancel_hint()
            return False

        # autocomplete selected token
        self.tab_state.selection = self.text_buffer.replace(
            self.tab_state.selection,
            self.tab_state.suggestions[self.tab_state.sugg_index],
        )

        # update hint
        self.update_buffer()
        target_type = self.semantic_analyzer.tokens[self.tab_state.token_index].type
        if target_type is None:
            msg = ""
        else:
            msg = self.semantic_analyzer.info(self.tab_state.token_index + 1) or ""
        self.set_hint(
            SuggestionsHint(self.tab_state.suggestions, self.tab_state.sugg_index, msg),
            self.tab_state.token_index,
        )
        return True

    def _prepare_tab_state(self, action=+1):
        # find the token to autocomplete
        index, token = self._find_token_before()
        if token is None:
            token_index = 0
            target = ""
            selection = slice(self.text_buffer.pos, self.text_buffer.pos)

        elif token.mask.stop < self.text_buffer.pos:
            token_index = index + 1
            target = ""
            selection = slice(self.text_buffer.pos, self.text_buffer.pos)

        else:
            token_index = index
            target = token.string
            selection = token.mask

        # generate suggestions
        suggestions = [
            sh.quoting(sugg)
            for sugg in self.semantic_analyzer.suggest(token_index, target)
        ]
        sugg_index = len(suggestions) if action == -1 else -1

        # tab state
        original_pos = self.text_buffer.pos
        original_token = self.text_buffer.buffer[selection]

        return TabState(
            suggestions=suggestions,
            sugg_index=sugg_index,
            token_index=token_index,
            original_token=original_token,
            original_pos=original_pos,
            selection=selection,
        )

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
            self.ask_for_hint(self.tab_state.token_index, type="info")
            self.tab_state = None

        return True

    @locked
    def unknown_key(self, key):
        self.cancel_hint()
        self.finish(ErrorResult(None, ValueError(f"Unknown key: " + key)))


class BeatStroke:
    r"""Keyboard controller for beatshell."""

    def __init__(self, input, settings):
        self.input = input
        self.settings = settings

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
            action = self.input._parse_action(func)
            action_handler = lambda _, action=action: action(self.input)
            controller.add_handler(action_handler, key)

        controller.add_handler(self.help_handler(), self.settings.help_key)
        controller.add_handler(self.confirm_handler(), self.settings.confirm_key)
        controller.add_handler(self.unknown_handler(self.settings))

    def keypress_handler(self):
        def keypress(_):
            self.input.key_pressed_counter += 1

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

