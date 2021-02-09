import sys
import contextlib
from .kerminal import *
from .beatmap import *
from .beatsheet import *
from .beatanalyzer import *


class KAIKO:
    def __init__(self, filename):
        self.filename = filename

    @contextlib.contextmanager
    def connect(self, kerminal):
        self.kerminal = kerminal

        self.beatmap = BeatmapDraft.read(self.filename)
        self.game = BeatmapPlayer(self.beatmap)

        with self.game.connect(self.kerminal) as main:
            yield main


filename = sys.argv[1]
self = KAIKO(filename)
Kerminal.execute(self, settings="debug_config")

print()
show_analyze(self.beatmap.settings.performance_tolerance, self.game.perfs)
