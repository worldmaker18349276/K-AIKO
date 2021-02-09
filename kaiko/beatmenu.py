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
    def __init__(self):
        songs = dict()

        for root, dirs, files in os.walk("./songs"):
            for file in files:
                if file.endswith((".k-aiko", ".kaiko", ".ka", ".osu")):
                    filepath = os.path.join(root, file)
                    songs[file] = BeatMenuPlay(filepath)

        self.tree = {
            "songs": songs,
            "settings": None,
            "quit": self.quit,
        }
        self.indices = (0,)
        self.sessions = queue.Queue()
        self.stop_event = threading.Event()

    @contextlib.contextmanager
    def connect(self, kerminal):
        self.kerminal = kerminal

        with dn.interval(consumer=self.run(), dt=0.1) as thread:
            yield thread

    @dn.datanode
    def input_handler(self):
        while True:
            _, _, key = yield

            if key == "\x1b[A":
                self.prev()
            elif key == "\x1b[B":
                self.next()
            elif key == "\n":
                self.enter()
            elif key == "\x1b":
                self.esc()

    @dn.datanode
    def prompt_drawer(self):
        yield
        while True:
            keys, _ = self.get_item(self.indices)
            yield ">" + ">".join(keys)

    @dn.datanode
    def run(self):
        while True:
            if self.sessions.empty():
                with self.kerminal.renderer.add_text(self.prompt_drawer(), 0),\
                     self.kerminal.controller.add_handler(self.input_handler()):
                    while self.sessions.empty():
                        if self.stop_event.is_set():
                            return
                        yield

            else:
                session = self.sessions.get()
                with session.connect(self.kerminal) as main:
                    main.start()
                    while main.is_alive():
                        yield

    def get_item(self, indices):
        keys = []
        tree = self.tree
        for i in indices:
            key, tree = list(tree.items())[i]
            keys.append(key)
        return keys, tree

    def next(self):
        _, tree = self.get_item(self.indices[:-1])
        index = min(self.indices[-1] + 1, len(tree)-1)
        self.indices = self.indices[:-1] + (index,)

    def prev(self):
        index = max(self.indices[-1] - 1, 0)
        self.indices = self.indices[:-1] + (index,)

    def enter(self):
        _, tree = self.get_item(self.indices)
        if tree is None:
            pass
        elif hasattr(tree, '__call__'):
            tree()
        elif hasattr(tree, 'connect'):
            self.sessions.put(tree)
        else:
            self.indices = self.indices + (0,)

    def esc(self):
        if len(self.indices) > 1:
            self.indices = self.indices[:-1]

    def quit(self):
        self.stop_event.set()
