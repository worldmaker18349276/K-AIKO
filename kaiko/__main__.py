import sys
from .knock import *
from .beatmap import *
from .beatsheet import *
from .beatanalyzer import *

filename = sys.argv[1]

beatmap = BeatmapDraft.read(filename)
game = KAIKOGame(beatmap)
console = KnockConsole()
console.settings.debug_timeit = True
console.run(game)

print()
show_analyze(beatmap.settings.performance_tolerance, game.perfs)
