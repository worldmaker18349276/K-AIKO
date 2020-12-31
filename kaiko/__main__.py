import sys
from .knock import *
from .beatmap import *
from .beatanalyzer import *

filename = sys.argv[1]

if filename.endswith((".k-aiko", ".kaiko", ".ka")):
    beatmap = K_AIKO_STD_FORMAT.read(filename)
elif filename.endswith(".osu"):
    beatmap = OSU_FORMAT.read(filename)
else:
    raise ValueError(f"unknown file extension: {filename}")

field = PlayField(beatmap)
console = KnockConsole()
console.settings.debug_timeit = True
console.run(field)

print()
show_analyze(beatmap.settings.performance_tolerance, field.events)
