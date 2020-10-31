import sys
from .knock import *
from .beatmap import *

filename = sys.argv[1]

with open(filename) as file:
    if filename.endswith((".k-aiko", ".kaiko", ".ka")):
        beatmap = K_AIKO_STD_FORMAT.read(file)
    elif filename.endswith(".osu"):
        beatmap = OSU_FORMAT.read(file)
    else:
        raise ValueError(f"unknown file extension: {filename}")

console = KnockConsole()
console.run(beatmap)

# print()
# for event in beatmap.events:
#     print(event)
