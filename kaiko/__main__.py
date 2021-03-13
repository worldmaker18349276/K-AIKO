import os
import contextlib
from .beatmap import BeatmapPlayer
from .beatsheet import BeatmapDraft, BeatmapParseError
from . import beatanalyzer
from . import kerminal
from . import beatmenu

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

songs = []

for root, dirs, files in os.walk("./songs"):
    for file in files:
        if file.endswith((".kaiko", ".ka", ".osu")):
            filepath = os.path.join(root, file)
            songs.append((file, BeatMenuPlay(filepath)))

menu = beatmenu.menu_tree([
    ("play", lambda: beatmenu.menu_tree(songs)),
    ("settings", None),
])

with kerminal.prepare_pyaudio() as manager:
    with menu:
        while True:
            result = beatmenu.explore(menu)
            if hasattr(result, 'execute'):
                result.execute(manager)
            elif result is None:
                break
