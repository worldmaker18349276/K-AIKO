import os
import time
import contextlib
import queue
import threading
from . import cfg
from . import datanodes as dn
from . import tui
from .beatmap import BeatmapPlayer
from .beatsheet import BeatmapDraft, BeatmapParseError


class BeatMenuPlay:
    def __init__(self, filepath):
        self.filepath = filepath

    @contextlib.contextmanager
    def connect(self, kerminal):
        try:
            beatmap = BeatmapDraft.read(self.filepath)

        except BeatmapParseError:
            print(f"failed to read beatmap {file}")
            yield

        else:
            game = BeatmapPlayer(beatmap)
            with game.connect(kerminal) as main:
                yield main

class BeatMenu:
    keymap = {
        'NEXT': "\x1b[B",
        'PREV': "\x1b[A",
        'ENTER': "\n",
        'EXIT': "\x1b",
    }

    def __init__(self):
        self.prompts = []
        self.sessions = queue.Queue()
        self.stop_event = threading.Event()
        self.sep = "❯ "

    @contextlib.contextmanager
    def connect(self, kerminal):
        self.kerminal = kerminal

        with dn.interval(consumer=self.run(), dt=0.1) as thread:
            yield thread

    def get_menu_node(self):
        songs = []

        for root, dirs, files in os.walk("./songs"):
            for file in files:
                if file.endswith((".kaiko", ".ka", ".osu")):
                    filepath = os.path.join(root, file)
                    songs.append((file, BeatMenuPlay(filepath)))

        return menu_node([
            ("play", lambda: menu_node(songs, self.keymap)),
            ("settings", None),
            ("quit", self.get_quit_node),
        ], self.keymap)

    @dn.datanode
    def get_quit_node(self):
        self.stop_event.set()
        yield

    @dn.datanode
    def input_handler(self, menu_node):
        while True:
            _, _, key = yield
            res = menu_node.send(key)
            if isinstance(res, list):
                self.prompts = res
            else:
                self.sessions.put(res)

    @dn.datanode
    def prompt_drawer(self):
        yield
        while True:
            yield self.sep + self.sep.join(self.prompts)

    @dn.datanode
    def run(self):
        menu_node = self.get_menu_node()

        with menu_node:
            self.prompts = menu_node.send(None)
            while True:
                if self.sessions.empty():
                    with self.kerminal.renderer.add_text(self.prompt_drawer(), 0),\
                         self.kerminal.controller.add_handler(self.input_handler(menu_node)):
                        while self.sessions.empty():
                            if self.stop_event.is_set():
                                return
                            if menu_node.finalized:
                                return
                            yield

                else:
                    session = self.sessions.get()
                    with session.connect(self.kerminal) as main:
                        main.start()
                        while main.is_alive():
                            yield

@dn.datanode
def menu_node(items, keymap):
    index = 0
    length = len(items)
    prompt, func = items[index]

    # ignore the first action and yield initial prompt
    yield
    key = yield [prompt]

    while True:
        if key == keymap['NEXT']:
            index = min(index+1, length-1)

        elif key == keymap['PREV']:
            index = max(index-1, 0)

        elif key == keymap['ENTER']:
            if func is None:
                # None -> no action
                pass

            elif hasattr(func, 'connect'):
                # knockable -> suspend to execute
                key = yield func
                continue

            elif hasattr(func, '__call__'):
                # datanode function -> forward action to submenu
                with func() as node:
                    while True:
                        try:
                            res = node.send(key)
                        except StopIteration:
                            break
                        res = res if hasattr(res, 'connect') else [prompt, *res]
                        key = yield res

            else:
                raise ValueError(f"unknown function: {repr(func)}")

        elif key == keymap['EXIT']:
            break

        prompt, func = items[index]
        key = yield [prompt]
