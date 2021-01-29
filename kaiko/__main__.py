import sys
from .kerminal import *
from .beatmenu import *
from .beatanalyzer import *

filename = sys.argv[1]
self = KAIKO(filename)
Kerminal.execute(self, settings="debug_config")

print()
show_analyze(self.beatmap.settings.performance_tolerance, self.game.perfs)
