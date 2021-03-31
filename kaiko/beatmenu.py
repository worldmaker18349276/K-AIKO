import os
import time
import contextlib
from typing import Tuple
import queue
import threading
import zipfile
import psutil
import appdirs
from . import cfg
from . import datanodes as dn
from . import tui
from .beatmap import BeatmapPlayer
from .beatsheet import BeatmapDraft, BeatmapParseError
from . import beatanalyzer


default_keymap = {
    "Backspace"        : "\x7f",
    "Delete"           : "\x1b[3~",
    "Left"             : "\x1b[D",
    "Right"            : "\x1b[C",
    "Home"             : "\x1b[H",
    "End"              : "\x1b[F",
    "Up"               : "\x1b[A",
    "Down"             : "\x1b[B",
    "PageUp"           : "\x1b[5~",
    "PageDown"         : "\x1b[6~",
    "Tab"              : "\t",
    "Esc"              : "\x1b",
    "Enter"            : "\n",
    "Shift+Tab"        : "\x1b[Z",
    "Ctrl+Backspace"   : "\x08",
    "Ctrl+Delete"      : "\x1b[3;5~",
    "Shift+Left"       : "\x1b[1;2D",
    "Shift+Right"      : "\x1b[1;2C",
    "Alt+Left"         : "\x1b[1;3D",
    "Alt+Right"        : "\x1b[1;3C",
    "Alt+Shift+Left"   : "\x1b[1;4D",
    "Alt+Shift+Right"  : "\x1b[1;4C",
    "Ctrl+Left"        : "\x1b[1;5D",
    "Ctrl+Right"       : "\x1b[1;5C",
    "Ctrl+Shift+Left"  : "\x1b[1;6D",
    "Ctrl+Shift+Right" : "\x1b[1;6C",
    "Alt+Ctrl+Left"    : "\x1b[1;7D",
    "Alt+Ctrl+Right"   : "\x1b[1;7C",
    "Alt+Ctrl+Shift+Left"  : "\x1b[1;8D",
    "Alt+Ctrl+Shift+Right" : "\x1b[1;8C",
}

class EditSegment:
    def __init__(self, text=None, pos=0, suggestions=None):
        self.text = text or []
        self.pos = pos
        self.suggestions = suggestions or []
        self.hint = ""
        self.event = threading.Event()
        self.keymap = default_keymap

    @dn.datanode
    def input_handler(self):
        while True:
            _, key = yield
            self.event.set()

            self.hint = ""

            # suggestions
            while key == self.keymap["Tab"]:
                original_text = "".join(self.text)
                for suggestion in self.suggestions:
                    if suggestion.startswith(original_text):
                        self.text = list(suggestion)
                        self.pos = len(self.text)

                        _, key = yield
                        self.event.set()
                        if key != self.keymap["Tab"]:
                            break

                else:
                    self.text = list(original_text)
                    self.pos = len(self.text)

                    _, key = yield
                    self.event.set()

            # edit
            if len(key) == 1 and key.isprintable():
                self.text[self.pos:self.pos] = [key]
                self.pos += 1

                # hint
                input_text = "".join(self.text)
                for suggestion in self.suggestions:
                    if suggestion.startswith(input_text):
                        self.hint = suggestion
                        break

            elif key == self.keymap["Backspace"]:
                if self.pos == 0:
                    continue
                self.pos -= 1
                del self.text[self.pos]

            elif key == self.keymap["Delete"]:
                if self.pos >= len(self.text):
                    continue
                del self.text[self.pos]

            elif key == self.keymap["Left"]:
                if self.pos == 0:
                    continue
                self.pos -= 1

            elif key == self.keymap["Right"]:
                if self.pos >= len(self.text):
                    continue
                self.pos += 1

            elif key == self.keymap["Home"]:
                self.pos = 0

            elif key == self.keymap["End"]:
                self.pos = len(self.text)

            elif key == self.keymap["Esc"]:
                self.text.clear()


def edit(framerate=60.0, suggestions=[]):
    seg = EditSegment(suggestions=["play", "say", "settings", "exit"])
    input_knot = dn.input(seg.input_handler())

    @dn.datanode
    def prompt_node():
        size_node = dn.terminal_size()
        headers = [ # game of life - Blocker
            "\x1b[96;1m⠶⠦⣚⠀⠶\x1b[m\x1b[38;5;255m❯\x1b[m ",
            "\x1b[96;1m⢎⣀⡛⠀⠶\x1b[m\x1b[38;5;255m❯\x1b[m ",
            "\x1b[36m⢖⣄⠻⠀⠶\x1b[m\x1b[38;5;254m❯\x1b[m ",
            "\x1b[36m⠖⠐⡩⠂⠶\x1b[m\x1b[38;5;254m❯\x1b[m ",
            "\x1b[96m⠶⠀⡭⠲⠶\x1b[m\x1b[38;5;253m❯\x1b[m ",
            "\x1b[36m⠶⠀⣬⠉⡱\x1b[m\x1b[38;5;253m❯\x1b[m ",
            "\x1b[36m⠶⠀⣦⠙⠵\x1b[m\x1b[38;5;252m❯\x1b[m ",
            "\x1b[36m⠶⠠⣊⠄⠴\x1b[m\x1b[38;5;252m❯\x1b[m ",

            "\x1b[96m⠶⠦⣚⠀⠶\x1b[m\x1b[38;5;251m❯\x1b[m ",
            "\x1b[36m⢎⣀⡛⠀⠶\x1b[m\x1b[38;5;251m❯\x1b[m ",
            "\x1b[36m⢖⣄⠻⠀⠶\x1b[m\x1b[38;5;250m❯\x1b[m ",
            "\x1b[36m⠖⠐⡩⠂⠶\x1b[m\x1b[38;5;250m❯\x1b[m ",
            "\x1b[96m⠶⠀⡭⠲⠶\x1b[m\x1b[38;5;249m❯\x1b[m ",
            "\x1b[36m⠶⠀⣬⠉⡱\x1b[m\x1b[38;5;249m❯\x1b[m ",
            "\x1b[36m⠶⠀⣦⠙⠵\x1b[m\x1b[38;5;248m❯\x1b[m ",
            "\x1b[36m⠶⠠⣊⠄⠴\x1b[m\x1b[38;5;248m❯\x1b[m ",

            "\x1b[96m⠶⠦⣚⠀⠶\x1b[m\x1b[38;5;247m❯\x1b[m ",
            "\x1b[36m⢎⣀⡛⠀⠶\x1b[m\x1b[38;5;247m❯\x1b[m ",
            "\x1b[36m⢖⣄⠻⠀⠶\x1b[m\x1b[38;5;246m❯\x1b[m ",
            "\x1b[36m⠖⠐⡩⠂⠶\x1b[m\x1b[38;5;246m❯\x1b[m ",
            "\x1b[96m⠶⠀⡭⠲⠶\x1b[m\x1b[38;5;245m❯\x1b[m ",
            "\x1b[36m⠶⠀⣬⠉⡱\x1b[m\x1b[38;5;245m❯\x1b[m ",
            "\x1b[36m⠶⠀⣦⠙⠵\x1b[m\x1b[38;5;244m❯\x1b[m ",
            "\x1b[36m⠶⠠⣊⠄⠴\x1b[m\x1b[38;5;244m❯\x1b[m ",

            "\x1b[96m⠶⠦⣚⠀⠶\x1b[m\x1b[38;5;243m❯\x1b[m ",
            "\x1b[36m⢎⣀⡛⠀⠶\x1b[m\x1b[38;5;243m❯\x1b[m ",
            "\x1b[36m⢖⣄⠻⠀⠶\x1b[m\x1b[38;5;242m❯\x1b[m ",
            "\x1b[36m⠖⠐⡩⠂⠶\x1b[m\x1b[38;5;242m❯\x1b[m ",
            "\x1b[96m⠶⠀⡭⠲⠶\x1b[m\x1b[38;5;241m❯\x1b[m ",
            "\x1b[36m⠶⠀⣬⠉⡱\x1b[m\x1b[38;5;241m❯\x1b[m ",
            "\x1b[36m⠶⠀⣦⠙⠵\x1b[m\x1b[38;5;240m❯\x1b[m ",
            "\x1b[36m⠶⠠⣊⠄⠴\x1b[m\x1b[38;5;240m❯\x1b[m ",
            ]
        period = 0.6

        with size_node:
            yield
            t = 0
            tr = 0
            while True:
                if seg.event.is_set():
                    seg.event.clear()
                    tr = t // -1 * -1

                ind = int(t / 4 * len(headers)) % len(headers)
                header = headers[ind]

                # cursor
                if t-tr < 0 or (t-tr) % 1 < 0.3:
                    cursor_pos = seg.pos
                    if ind == 0 or ind == 1:
                        cursor_wrapper = "\x1b[7;1m", "\x1b[m"
                    else:
                        cursor_wrapper = "\x1b[7;2m", "\x1b[m"
                else:
                    cursor_wrapper = None

                # input
                input_text = "".join(seg.text)
                input_hint = "".join(f"\x1b[2m{ch}\x1b[m" for ch in seg.hint) if seg.hint else ""

                # size
                try:
                    size = size_node.send(None)
                except StopIteration:
                    return
                width = size.columns

                # draw
                view = tui.newwin1(width)
                view, x = tui.addtext1(view, width, 0, header)
                tui.addtext1(view, width, x, input_hint)
                tui.addtext1(view, width, x, input_text)
                if cursor_wrapper:
                    _, cursor_x = tui.textrange1(x, input_text[:cursor_pos])
                    cursor_ran = tui.select1(view, width, slice(cursor_x, cursor_x+1))
                    view[cursor_ran.start] = view[cursor_ran.start].join(cursor_wrapper)

                yield "\r" + "".join(view) + "\r"
                t += 1/framerate/period

    display_knot = dn.interval(prompt_node(), dn.show(hide_cursor=True), 1/framerate)

    menu_knot = dn.pipe(input_knot, display_knot)
    dn.exhaust(menu_knot, dt=0.01, interruptible=True)


class KAIKOTheme(metaclass=cfg.Configurable):
    data_icon: str = "\x1b[92m🗀\x1b[0m "
    info_icon: str = "\x1b[94m🛠\x1b[0m "
    hint_icon: str = "\x1b[93m💡\x1b[0m "

    verb: Tuple[str, str] = ("\x1b[2m", "\x1b[0m")
    emph: Tuple[str, str] = ("\x1b[1m", "\x1b[21m")
    warn: Tuple[str, str] = ("\x1b[31m", "\x1b[0m")

class BeatMenuUser:
    def __init__(self, theme):
        self.theme = theme
        self.data_dir = None
        self.songs_dir = None

    def load(self):
        emph = self.theme.emph
        data_icon = self.theme.data_icon

        self.data_dir = appdirs.user_data_dir("K-AIKO", psutil.Process().username())
        self.songs_dir = os.path.join(self.data_dir, "songs")
        if not os.path.isdir(self.data_dir):
            # start up
            print(f"{data_icon} preparing your profile...")
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.songs_dir, exist_ok=True)
            print(f"{data_icon} your data will be stored in {('file://'+self.data_dir).join(emph)}")
            print(flush=True)

class BeatMenuGame:
    def __init__(self, user):
        self.user = user
        self._beatmaps = []
        self.songs_mtime = None

    def reload(self):
        info_icon = self.user.theme.info_icon
        emph = self.user.theme.emph
        songs_dir = self.user.songs_dir

        print(f"{info_icon} Loading songs from {('file://'+songs_dir).join(emph)}...")

        self._beatmaps = []

        for root, dirs, files in os.walk(songs_dir):
            for file in files:
                if file.endswith(".osz"):
                    filepath = os.path.join(root, file)
                    distpath, _ = os.path.splitext(filepath)
                    if os.path.isdir(distpath):
                        continue
                    print(f"{info_icon} Unzip file {('file://'+filepath).join(emph)}...")
                    os.makedirs(distpath)
                    zf = zipfile.ZipFile(filepath, 'r')
                    zf.extractall(path=distpath)

        for root, dirs, files in os.walk(songs_dir):
            for file in files:
                if file.endswith((".kaiko", ".ka", ".osu")):
                    filepath = os.path.join(root, file)
                    self._beatmaps.append(filepath)

        if len(self._beatmaps) == 0:
            print("{info_icon} There is no song in the folder yet!")
        print(flush=True)

        self.songs_mtime = os.stat(songs_dir).st_mtime

    @property
    def beatmaps(self):
        if self.songs_mtime != os.stat(self.user.songs_dir).st_mtime:
            self.reload()

        return list(self._beatmaps)

    def play(self, beatmap:str="<path to the beatmap>"):
        return BeatMenuPlay(beatmap)

class BeatMenuPlay:
    def __init__(self, filepath):
        self.filepath = filepath

    @contextlib.contextmanager
    def execute(self, manager):
        try:
            beatmap = BeatmapDraft.read(self.filepath)

        except BeatmapParseError:
            print(f"failed to read beatmap {self.filepath}")

        else:
            game = BeatmapPlayer(beatmap)
            game.execute(manager)

            print()
            beatanalyzer.show_analyze(beatmap.settings.performance_tolerance, game.perfs)

def explore(menu_tree, sep="\x1b[32m❯\x1b[m ", framerate=60.0):
    keymap = {
        "\x1b[B": 'NEXT',
        "\x1b[A": 'PREV',
        "\n": 'ENTER',
        "\x1b": 'EXIT',
    }

    prompts = []
    result = None

    @dn.datanode
    def input_handler(menu_tree):
        nonlocal prompts, result
        try:
            prompts = menu_tree.send(None)
        except StopIteration:
            return

        while True:
            _, key = yield
            if key not in keymap:
                continue

            try:
                res = menu_tree.send(keymap[key])
            except StopIteration:
                return

            if isinstance(res, list):
                prompts = res
            else:
                result = res
                return

    input_knot = dn.input(input_handler(menu_tree))

    @dn.datanode
    def prompt_node():
        size_node = dn.terminal_size()
        headers = [ # game of life - Blocker
            "\x1b[36m⠶⠦⣚⠀⠶\x1b[m",
            "\x1b[36m⢎⣀⡛⠀⠶\x1b[m",
            "\x1b[36m⢖⣄⠻⠀⠶\x1b[m",
            "\x1b[36m⠖⠐⡩⠂⠶\x1b[m",
            "\x1b[36m⠶⠀⡭⠲⠶\x1b[m",
            "\x1b[36m⠶⠀⣬⠉⡱\x1b[m",
            "\x1b[36m⠶⠀⣦⠙⠵\x1b[m",
            "\x1b[36m⠶⠠⣊⠄⠴\x1b[m",
            ]
        period = 1/8

        with size_node:
            yield
            ind = 0
            while True:
                header = headers[int(ind/framerate/period) % len(headers)] + sep

                try:
                    size = size_node.send(None)
                except StopIteration:
                    return
                width = size.columns

                view = tui.newwin1(width)
                tui.addtext1(view, width, 0, header + sep.join(prompts))
                yield "\r" + "".join(view) + "\r"
                ind += 1

    display_knot = dn.interval(prompt_node(), dn.show(hide_cursor=True), 1/framerate)

    menu_knot = dn.pipe(input_knot, display_knot)
    dn.exhaust(menu_knot, dt=0.01, interruptible=True)
    return result

@dn.datanode
def menu_tree(items):
    index = 0
    length = len(items)
    if length == 0:
        return

    prompt, func = items[index]

    action = yield

    while True:
        if action is None:
            pass

        elif action == 'NEXT':
            index = min(index+1, length-1)

        elif action == 'PREV':
            index = max(index-1, 0)

        elif action == 'ENTER':
            if func is None:
                # None -> no action
                pass

            elif hasattr(func, 'execute'):
                # executable -> suspend to execute
                action = yield func
                continue

            elif hasattr(func, '__call__'):
                # datanode function -> forward action to submenu
                with func() as node:
                    action = None
                    while True:
                        try:
                            res = node.send(action)
                        except StopIteration:
                            break
                        res = res if hasattr(res, 'execute') else [prompt, *res]
                        action = yield res

            else:
                raise ValueError(f"unknown function: {repr(func)}")

        elif action == 'EXIT':
            break

        prompt, func = items[index]
        action = yield [prompt]

