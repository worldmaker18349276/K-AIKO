import functools
import contextlib
import re
import queue
import threading
from typing import Optional, List, Tuple, Dict, Callable
from pathlib import Path
import dataclasses
from ..utils import datanodes as dn
from ..utils import commands as cmd
from ..utils import config as cfg
from ..utils import markups as mu
from ..devices import engines
from . import sheditors
from .textboxes import Caret, TextBox, TextBoxWidgetSettings


# hint
class Hint:
    pass


@dataclasses.dataclass(frozen=True)
class DescHint(Hint):
    message: str


@dataclasses.dataclass(frozen=True)
class InfoHint(Hint):
    message: str


@dataclasses.dataclass
class HintState:
    hint: Hint
    index: Optional[int]
    tokens: Optional[List[str]]


class HintManager:
    def __init__(self, editor, preview_handler):
        self.editor = editor
        self.preview_handler = preview_handler
        self.popup_queue = queue.Queue()
        self.hint_state = None

    def get_hint(self):
        return None if self.hint_state is None else self.hint_state.hint

    def get_hint_location(self):
        return None if self.hint_state is None else self.hint_state.index

    def add_popup(self, hint):
        self.popup_queue.put(hint)

    def popup_hint(self):
        hint = self.hint_state.hint
        if not hint.message:
            return False

        self.add_popup(hint)
        return True

    def set_hint(self, hint, index=None):
        if isinstance(hint, DescHint):
            msg_tokens = (
                [token.string for token in self.editor.tokens[:index]]
                if index is not None
                else None
            )
        elif isinstance(hint, InfoHint):
            msg_tokens = (
                [token.string for token in self.editor.tokens[: index + 1]]
                if index is not None
                else None
            )
        else:
            assert False

        self.hint_state = HintState(hint=hint, index=index, tokens=msg_tokens)
        self.update_preview()
        return True

    def cancel_hint(self):
        if self.hint_state is None:
            return False
        self.hint_state = None
        self.update_preview()
        return True

    def update_hint(self):
        if self.hint_state is None:
            return False

        if self.hint_state.tokens is None:
            return self.cancel_hint()

        if self.hint_state.index is not None and self.hint_state.index >= len(self.editor.tokens):
            return self.cancel_hint()

        if len(self.hint_state.tokens) > len(self.editor.tokens):
            return self.cancel_hint()

        for token_string, token in zip(self.hint_state.tokens, self.editor.tokens):
            if token_string != token.string:
                return self.cancel_hint()

        if (
            isinstance(self.hint_state.hint, DescHint)
            and self.editor.tokens[len(self.hint_state.tokens) - 1].type is not None
        ):
            return self.cancel_hint()

        return False

    def update_preview(self):
        if self.hint_state is None:
            self.preview_handler(None)
        elif not isinstance(self.hint_state.hint, InfoHint):
            self.preview_handler(None)
        elif self.hint_state.tokens is None:
            self.preview_handler(None)
        elif len(self.hint_state.tokens) != 2:
            self.preview_handler(None)
        elif self.hint_state.tokens[0] != "play":
            self.preview_handler(None)
        else:
            self.preview_handler(self.hint_state.tokens[1])

    def ask_for_hint(self, index, type="all"):
        if index not in range(len(self.editor.tokens)):
            return False

        target_type = self.editor.tokens[index].type

        if target_type is None:
            if type not in ("all", "desc"):
                self.cancel_hint()
                return False
            msg = self.editor.desc(index)
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
            msg = self.editor.info(index + 1)
            if msg is None:
                self.cancel_hint()
                return False
            hint = InfoHint(msg)
            self.set_hint(hint, index)
            return True


# autocomplete
@dataclasses.dataclass
class TabState:
    suggestions: List[str]
    sugg_index: int
    token_index: int
    original_token: List[str]
    original_pos: int
    selection: slice


class AutocompleteManager:
    def __init__(self, editor):
        self.editor = editor

        self.tab_state = None

    def get_suggestions_list(self):
        if self.tab_state is None:
            return None
        else:
            return self.tab_state.suggestions

    def get_suggestions_index(self):
        if self.tab_state is None:
            return None
        else:
            return self.tab_state.sugg_index

    def is_in_cycle(self):
        return self.tab_state is not None

    def prepare_tab_state(self, action=+1):
        # find the token to autocomplete
        index, token = self.editor.find_token_before(self.editor.pos)
        if token is None:
            token_index = 0
            target = ""
            selection = slice(self.editor.pos, self.editor.pos)

        elif token.mask.stop < self.editor.pos:
            token_index = index + 1
            target = ""
            selection = slice(self.editor.pos, self.editor.pos)

        else:
            token_index = index
            target = token.string
            selection = token.mask

        # generate suggestions
        suggestions = [
            sheditors.quoting(sugg)
            for sugg in self.editor.suggest(token_index, target)
        ]
        sugg_index = len(suggestions) if action == -1 else -1

        # tab state
        original_pos = self.editor.pos
        original_token = self.editor.buffer[selection]

        return TabState(
            suggestions=suggestions,
            sugg_index=sugg_index,
            token_index=token_index,
            original_token=original_token,
            original_pos=original_pos,
            selection=selection,
        )

    def autocomplete(self, action=+1):
        if self.tab_state is None:
            tab_state = self.prepare_tab_state(action)

            if len(tab_state.suggestions) == 0:
                # no suggestion
                return None

            if len(tab_state.suggestions) == 1:
                # one suggestion -> complete directly
                self.editor.replace(tab_state.selection, tab_state.suggestions[0])
                return tab_state.token_index

            self.tab_state = tab_state

        if action == +1:
            self.tab_state.sugg_index += 1
        elif action == -1:
            self.tab_state.sugg_index -= 1
        else:
            raise ValueError(f"invalid action: {action}")

        if self.tab_state.sugg_index not in range(len(self.tab_state.suggestions)):
            self.cancel_autocomplete()
            return None

        # fill in selected token
        self.tab_state.selection = self.editor.replace(
            self.tab_state.selection,
            self.tab_state.suggestions[self.tab_state.sugg_index],
        )

        return self.tab_state.token_index

    def cancel_autocomplete(self):
        if self.tab_state is None:
            return
        self.editor.replace(self.tab_state.selection, self.tab_state.original_token)
        self.editor.move_to(self.tab_state.original_pos)
        self.tab_state = None

    def finish_autocomplete(self):
        if self.tab_state is None:
            return None
        index = self.tab_state.token_index
        self.tab_state = None
        return index


# input
class Result:
    pass


@dataclasses.dataclass(frozen=True)
class EmptyResult(Result):
    pass


@dataclasses.dataclass(frozen=True)
class ErrorResult(Result):
    command_str: str
    index : Optional[int]
    error: Exception


@dataclasses.dataclass(frozen=True)
class CompleteResult(Result):
    command_group: str
    command_str: str
    command: Callable


class ShellSyntaxError(Exception):
    pass


class HistoryManager:
    r"""
    Fields
    ------
    history_path : Path
    latest_command : tuple of str and str, optional
        lastest group and command.
    buffers : list of list of str
        The buffers of editor.
    buffer_index : int
        The index of current buffer.
    """

    TRIM_LEN = 10
    PATTERN = re.compile(r"\[(\w+)\] (.+)")

    def __init__(self, history_path, latest_command=None):
        self.history_path = history_path
        self.latest_command = latest_command
        self.buffers = [[]]
        self.buffer_index = -1

    @property
    def buffer(self):
        return self.buffers[self.buffer_index]

    def prev(self):
        if self.buffer_index == -len(self.buffers):
            return False
        self.buffer_index -= 1
        return True

    def next(self):
        if self.buffer_index == -1:
            return False
        self.buffer_index += 1
        return True

    def write_history(self, command_group, command):
        self.history_path.touch()
        command = command.strip()
        if command and command_group and (command_group, command) != self.latest_command:
            open(self.history_path, "a").write(f"\n[{command_group}] {command}")
            self.latest_command = (command_group, command)

    def read_history(self, command_groups, read_size):
        buffers = []
        self.history_path.touch()
        self.latest_command = None
        for command in open(self.history_path):
            command = command.strip()
            match = self.PATTERN.fullmatch(command)
            if match:
                self.latest_command = (match.group(1), match.group(2))
                if match.group(1) in command_groups and (not buffers or buffers[-1] != match.group(2)):
                    buffers.append(match.group(2))
            if len(buffers) - read_size > self.TRIM_LEN:
                del buffers[:self.TRIM_LEN]

        self.buffers = [list(command) for command in buffers[-read_size:]]
        self.buffers.append([])
        self.buffer_index = -1


class InputSettings(cfg.Configurable):
    r"""
    Fields
    ------
    preview_song : bool
        Whether to preview the song when selected.
    history_size : int
        The maximum history size.
    """

    preview_song: bool = True
    history_size: int = 500

    @cfg.subconfig
    class control(cfg.Configurable):
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
            The keymap of input. The key of dict is the keystroke, and the
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

        keymap: Dict[str, str] = {
            "Backspace": "input.backspace()",
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
            "Alt_Left": "input.move_to_token_start()",
            "Alt_Right": "input.move_to_token_end()",
            "Alt_Backspace": "input.delete_backward_token()",
            "Alt_Delete": "input.delete_forward_token()",
            "Esc": "input.cancel_typeahead() | input.cancel_hint()",
            "'\\x04'": "input.delete() or input.exit_if_empty()",
        }

    @cfg.subconfig
    class hint(cfg.Configurable):
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
        suggestion_overflow_ellipses: Tuple[str, str] = ("[weight=dim]ⵗ [slot/][/]", "[weight=dim]ⵗ [slot/][/]")

    @cfg.subconfig
    class textbox(cfg.Configurable, TextBoxWidgetSettings):
        __doc__ = TextBoxWidgetSettings.__doc__

        def __init__(self):
            pass


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
                raise
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


class Input:
    r"""Input editor.

    Attributes
    ----------
    settings : InputSettings
        The input settings.
    history : HistoryManager
        The input history manager.
    editor : sheditors.Editor
        The editor of command.
    typeahead : str
        The type ahead of input.
    hint_manager : HintManager
    autocomplete_manager : AutocompleteManager
    result : Result or None
        The result of input.
    state : str
        The input state.
    buffer_modified_counter : int
        The event counter for modifying buffer.
    key_pressed_counter : int
        The event counter for key pressing.
    """

    def __init__(
        self,
        preview_handler,
        history_path,
        settings,
    ):
        r"""Constructor.

        Parameters
        ----------
        preview_handler : function
        history_path : Path
            The path of command history.
        settings : InputSettings
            The settings of input.
        """
        self.settings = settings

        self.history = HistoryManager(history_path)
        self.editor = sheditors.Editor(None, self.history.buffer)
        self.typeahead = ""
        self.hint_manager = HintManager(
            self.editor,
            lambda song: preview_handler(song) if self.settings.preview_song else None,
        )
        self.autocomplete_manager = AutocompleteManager(self.editor)

        self.state = "FIN"
        self.result = None
        self.edit_ctxt = ContextDispatcher()

        self.key_pressed_counter = 0
        self.buffer_modified_counter = 0

    def _set_settings(self, settings):
        self.settings = settings

    def _register(self, fin_event, provider):
        rich = provider.get(mu.RichParser)
        renderer = provider.get(engines.Renderer)
        controller = provider.get(engines.Controller)

        stroke = InputStroke(self, self.settings.control)
        stroke.register(controller)

        state = InputView(self)
        text_renderer = TextRenderer(rich, self.settings.hint)
        msg_renderer = MsgRenderer(rich, self.settings.hint)

        renderer.add_drawer(state.load(fin_event), zindex=())
        renderer.add_drawer(msg_renderer.render_msg(state), zindex=(1,))

        textbox = TextBox(
            text_renderer.render_text(state),
            self.settings.textbox,
        ).load(provider)

        return textbox

    def _record_command(self):
        command = "".join(self.editor.buffer).strip()
        self.history.write_history(self.editor.group, command)

    @locked
    @onstate("EDIT")
    def _finish_session(self, res):
        r"""Finish this session of input.

        Parameters
        ----------
        res : Result
            The result.
        """
        self.result = res
        self.state = "FIN"

    @locked
    @onstate("FIN")
    def _new_session(self, command_parser, clear=True):
        r"""Start a new session of input.

        Parameters
        ----------
        command_parser : cmd.CommandParser
        clear : bool, optional
        """
        self.editor.update_parser(command_parser)

        if clear:
            groups = self.editor.get_all_groups()
            history_size = self.settings.history_size
            self.history.read_history(groups, history_size)
            self.editor.init(self.history.buffer)

        self.update_buffer(clear=True)
        self.start()

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
    @onstate("EDIT")
    def prev(self):
        """Previous buffer.

        Returns
        -------
        succ : bool
        """
        succ = self.history.prev()
        if not succ:
            return False
        self.editor.init(self.history.buffer)
        self.update_buffer(clear=True)

        return True

    @locked
    @onstate("EDIT")
    def next(self):
        """Next buffer.

        Returns
        -------
        succ : bool
        """
        succ = self.history.next()
        if not succ:
            return False
        self.editor.init(self.history.buffer)
        self.update_buffer(clear=True)

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
        if self.editor.pos != len(self.editor.buffer):
            self.typeahead = ""
            return False

        # search history
        pos = self.editor.pos
        for buffer in reversed(self.history.buffers):
            if len(buffer) > pos and buffer[:pos] == self.editor.buffer:
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

        if self.typeahead == "" or self.editor.pos != len(self.editor.buffer):
            return False

        self.editor.insert(self.typeahead)

        self.update_buffer()
        self.ask_for_hint()

        return True

    @locked
    def add_popup(self, msg):
        """Add popup.

        Show hint above the prompt.

        Parameters
        ----------
        msg : str
            The message of hint.

        Returns
        -------
        succ : bool
        """
        self.hint_manager.add_popup(DescHint(msg))
        return True

    @locked
    def set_hint(self, msg, index=None):
        """Set hint.

        Show hint below the prompt.

        Parameters
        ----------
        msg : str
            The message of hint.
        index : int or None
            Index of the token to which the hint is directed, or `None` for
            nothing.

        Returns
        -------
        succ : bool
        """
        return self.hint_manager.set_hint(DescHint(msg), index=index)

    @locked
    def cancel_hint(self):
        """Cancel hint.

        Remove the hint below the prompt.

        Returns
        -------
        succ : bool
        """
        return self.hint_manager.cancel_hint()

    @locked
    def update_hint(self):
        """Update hint.

        Remove hint if the target is updated.

        Returns
        -------
        succ : bool
            `False` if there is no hint or the hint isn't removed.
        """
        return self.hint_manager.update_hint()

    @locked
    @onstate("EDIT")
    def ask_for_hint(self, index=None, type="all"):
        """Ask some hint for command.

        Provide some hint for the command on the caret.

        Parameters
        ----------
        index : int, optional
        type : one of "info", "desc", "all", optional
            The type of hint to ask.

        Returns
        -------
        succ : bool
        """
        if index is None:
            index, token = self.editor.find_token_before(self.editor.pos)

        return self.hint_manager.ask_for_hint(index, type=type)

    @locked
    def update_buffer(self, clear=False):
        """Update buffer.

        Parameters
        ----------
        clear : bool, optional

        Returns
        -------
        succ : bool
        """
        self.editor.parse()
        self.buffer_modified_counter += 1
        self.cancel_typeahead()
        if clear:
            self.cancel_hint()
        else:
            self.update_hint()
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
        succ = self.editor.insert(text)
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
        succ = self.editor.backspace()
        if not succ:
            return False

        self.update_buffer()
        self.ask_for_hint()

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
        succ = self.editor.delete()
        if not succ:
            return False

        self.update_buffer()
        self.ask_for_hint()

        return True

    @locked
    @onstate("EDIT")
    def delete_all(self):
        """Delete All.

        Returns
        -------
        succ : bool
        """
        succ = self.editor.delete_all()
        if not succ:
            return False

        self.update_buffer()
        self.ask_for_hint()

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
        self.editor.replace(slice(start, end), "")

        self.update_buffer()
        self.ask_for_hint()

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
        mask = self.editor.to_word_start()
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
        mask = self.editor.to_word_end()
        return self.delete_range(mask.start, mask.stop)

    @locked
    @onstate("EDIT")
    def delete_token(self, index):
        """Delete current token.

        Parameters
        ----------
        index : int

        Returns
        -------
        succ : bool
        """
        token = self.editor.tokens[index]
        return self.delete_range(token.mask.start, token.mask.stop)

    @locked
    @onstate("EDIT")
    def delete_backward_token(self):
        """Delete backward token.

        Returns
        -------
        succ : bool
        """
        _, token = self.editor.find_token_before(self.editor.pos)
        if token is None:
            return self.delete_range(0, self.editor.pos)
        else:
            return self.delete_range(token.mask.start, max(self.editor.pos, token.mask.stop))

    @locked
    @onstate("EDIT")
    def delete_forward_token(self):
        """Delete forward token.

        Returns
        -------
        succ : bool
        """
        _, token = self.editor.find_token_after(self.editor.pos)
        if token is None:
            return self.delete_range(self.editor.pos, None)
        else:
            return self.delete_range(min(self.editor.pos, token.mask.start), token.mask.stop)

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
        succ = self.editor.move_to(pos)
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
        return self.move_to(self.editor.pos + offset)

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
        mask = self.editor.to_word_start()
        return self.move_to(mask.start)

    @locked
    @onstate("EDIT")
    def move_to_word_end(self):
        """Move caret to the end of the word.

        Returns
        -------
        succ : bool
        """
        mask = self.editor.to_word_end()
        return self.move_to(mask.stop)

    @locked
    @onstate("EDIT")
    def move_to_token_start(self):
        """Move caret to the start of the token.

        Returns
        -------
        succ : bool
        """
        _, token = self.editor.find_token_before(self.editor.pos)
        if token is None:
            return self.move_to(0)
        else:
            return self.move_to(token.mask.start)

    @locked
    @onstate("EDIT")
    def move_to_token_end(self):
        """Move caret to the end of the word.

        Returns
        -------
        succ : bool
        """
        _, token = self.editor.find_token_after(self.editor.pos)
        if token is None:
            return self.move_to(None)
        else:
            return self.move_to(token.mask.stop)

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
        index, token = self.editor.find_token_before(self.editor.pos)
        if token is None:
            return False

        if self.hint_manager.get_hint_location() != index:
            self.ask_for_hint(index)
            return False

        return self.hint_manager.popup_hint()

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

        if not self.editor.tokens:
            self._finish_session(EmptyResult())
            return True

        command_str = "".join(self.editor.buffer).strip()

        if self.editor.lex_state == sheditors.SHLEXER_STATE.BACKSLASHED:
            res, index = ShellSyntaxError("No escaped character"), len(self.editor.tokens) - 1
        elif self.editor.lex_state == sheditors.SHLEXER_STATE.QUOTED:
            res, index = ShellSyntaxError("No closing quotation"), len(self.editor.tokens) - 1
        else:
            res, index = self.editor.result, self.editor.length

        if isinstance(res, cmd.CommandUnfinishError):
            self._finish_session(ErrorResult(command_str, None, res))
            return False
        elif isinstance(res, (cmd.CommandParseError, ShellSyntaxError)):
            self._finish_session(ErrorResult(command_str, index, res))
            return False
        else:
            self._finish_session(CompleteResult(str(self.editor.group), command_str, res))
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
        if self.editor.buffer:
            return False

        self.insert("bye")
        return self.confirm()

    @locked
    @onstate("EDIT")
    def forward_autocomplete(self):
        """Autocomplete forwardly.

        Complete the token on the caret, or fill in suggestions if caret is
        located in between.

        Returns
        -------
        succ : bool
            `True` if is in autocompletion cycle.  Note that it will be `False`
            for no suggestion or one suggestion case.
        """
        index = self.autocomplete_manager.autocomplete(action=+1)
        is_in_cycle = self.autocomplete_manager.is_in_cycle()
        self.update_buffer(clear=True)
        if index is not None:
            self.ask_for_hint(index, type="info")

        return is_in_cycle

    @locked
    @onstate("EDIT")
    def backward_autocomplete(self):
        """Autocomplete backwardly.

        Complete the token on the caret backwardly, or fill in suggestions if
        caret is located in between.

        Returns
        -------
        succ : bool
            `True` if is in autocompletion cycle.  Note that it will be `False`
            for no suggestion or one suggestion case.
        """
        index = self.autocomplete_manager.autocomplete(action=-1)
        is_in_cycle = self.autocomplete_manager.is_in_cycle()
        self.update_buffer(clear=True)
        if index is not None:
            self.ask_for_hint(index, type="info")

        return is_in_cycle

    @locked
    @onstate("EDIT")
    def finish_autocomplete(self):
        r"""Finish autocompletion.

        Returns
        -------
        succ : bool
        """
        index = self.autocomplete_manager.finish_autocomplete()
        if index is not None:
            self.ask_for_hint(index, type="info")
        return True

    @locked
    @onstate("EDIT")
    def cancel_autocomplete(self):
        r"""Cancel autocompletion.

        Returns
        -------
        succ : bool
        """
        self.autocomplete_manager.cancel_autocomplete()
        self.update_buffer(clear=True)
        return True

    @locked
    def unknown_key(self, key):
        self.cancel_hint()
        command_str = "".join(self.editor.buffer).strip()
        self._finish_session(ErrorResult(command_str, None, ValueError(f"Unknown key: " + key)))


class InputStroke:
    r"""Keyboard controller."""

    def __init__(self, input, settings):
        self.input = input
        self.settings = settings

    @staticmethod
    def _parse_action(func):
        ACTION_REGEX = "({fn}{op})*{fn}".format(
            fn=r"input\.(?!_)\w+\(\)",
            op=r"( \| | \& | and | or )",
        )

        if not re.match(ACTION_REGEX, func):
            raise ValueError(f"invalid action: {repr(func)}")
        def action(input):
            with input.edit_ctxt.on():
                eval(func, {}, {"input": input})
        return action

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
            action = self._parse_action(func)
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
                self.input.forward_autocomplete()
            elif keyname == prev_key:
                self.input.backward_autocomplete()
            elif keyname == cancel_key:
                self.input.cancel_autocomplete()
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


class InputView:
    def __init__(self, input):
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
                    self.buffer = list(self.input.editor.buffer)
                    self.tokens = list(self.input.editor.tokens)
                self.pos = self.input.editor.pos

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

                if isinstance(self.input.result, ErrorResult):
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
                typeahead = caret_markup(mu.join(typeahead[:1]).children), typeahead[1:]

        typeahead_markup = typeahead_template(mu.join(typeahead))

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
                token_markup = highlight_template(token_markup)

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
            caret_markup=Caret,
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

        sugg_top_ellipsis = self.rich.parse(suggestion_overflow_ellipses[0], slotted=True)
        sugg_bottom_ellipsis = self.rich.parse(suggestion_overflow_ellipses[1], slotted=True)

        sugg_items_templates = (
            self.rich.parse(sugg_items[0], slotted=True),
            self.rich.parse(sugg_items[1], slotted=True),
        )

        desc_template = self.rich.parse(self.settings.desc_message, slotted=True)
        info_template = self.rich.parse(self.settings.info_message, slotted=True)

        render_hint = dn.starcachemap(
            self.render_hint,
            message_max_lines=message_max_lines,
            msg_ellipsis=msg_ellipsis,
            sugg_lines=sugg_lines,
            sugg_items_templates=sugg_items_templates,
            sugg_ellipses=(sugg_top_ellipsis, sugg_bottom_ellipsis),
            desc_template=desc_template,
            info_template=info_template,
        )

        with render_hint:
            (view, msgs, logs), time, width = yield
            while True:
                msg = render_hint.send((state.hint, state.suggestions))
                if msg is None:
                    if len(msgs) != 0:
                        msgs.clear()
                else:
                    if len(msgs) != 1 or msgs[0] is not msg:
                        msgs.clear()
                        msgs.append(msg)
                logs.extend(self.render_popup(state.popup, desc_template=desc_template, info_template=info_template))
                (view, msgs, logs), time, width = yield (view, msgs, logs)

    def render_hint(
        self,
        hint,
        suggestions,
        *,
        message_max_lines,
        msg_ellipsis,
        sugg_lines,
        sugg_items_templates,
        sugg_ellipses,
        desc_template,
        info_template,
    ):
        msgs = []

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

            if isinstance(hint, DescHint):
                msg = desc_template(msg)
            elif isinstance(hint, InfoHint):
                msg = info_template(msg)
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
                item_template = sugg_items_templates[1] if i == sugg_index - sugg_start else sugg_items_templates[0]
                sugg = item_template(sugg)
                res.append(sugg)
                if i == sugg_index - sugg_start and msg is not None:
                    res.append(msg)

            if sugg_start > 0:
                res.insert(0, sugg_ellipses[0](mu.Text(f"{sugg_start} more")))
            if sugg_end < len(suggs_list):
                res.append(sugg_ellipses[1](mu.Text(f"{len(suggs_list) - sugg_end} more")))

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

        return mu.Group(tuple(msgs)) if msgs else None

    def render_popup(self, popup, *, desc_template, info_template):
        logs = []

        # draw popup
        for hint in popup:
            msg = None
            if hint.message:
                msg = self.rich.parse(hint.message, root_tag=True)

                if isinstance(hint, DescHint):
                    msg = desc_template(msg)
                elif isinstance(hint, InfoHint):
                    msg = info_template(msg)
                else:
                    assert False

                msg = mu.Group((msg, mu.Text("\n")))
                msg = msg.expand()

            if msg is not None:
                logs.append(msg)

        return logs

