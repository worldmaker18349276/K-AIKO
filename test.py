from knock import *
from beatmap import *

with open("蛋餅好朋友 [normal].ka") as file:
    sheet = BeatmapStdSheet()
    exec(file.read(), dict(), dict(sheet=sheet))
    beatmap = Beatmap(sheet.audio, sheet.events)

    console = KnockConsole()
    console.play(beatmap)
    print()
    for event in beatmap.events:
        print(event)
