import sys
from .knock import *
from .beatmap import *
from .beatsheet import *
from .beatanalyzer import *

filename = sys.argv[1]

beatmap = BeatmapDraft.read(filename)
game = KAIKOGame(beatmap)
KnockConsole.run(game, settings="debug_config")

print()
show_analyze(beatmap.settings.performance_tolerance, game.perfs)
