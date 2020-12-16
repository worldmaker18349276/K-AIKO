import sys
from .knock import *
from .beatmap import *

filename = sys.argv[1]

if filename.endswith((".k-aiko", ".kaiko", ".ka")):
    beatmap = K_AIKO_STD_FORMAT.read(filename)
elif filename.endswith(".osu"):
    beatmap = OSU_FORMAT.read(filename)
else:
    raise ValueError(f"unknown file extension: {filename}")

field = PlayField(beatmap)
console = KnockConsole()
console.run(field)

print()
field.judger.report(field.events)
