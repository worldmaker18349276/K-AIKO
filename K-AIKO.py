#!/usr/bin/env python3

import sys
from knock import *
from beatmap import *

if __name__ == "__main__":
    filename = sys.argv[1]

    with open(filename) as file:
        sheet = BeatSheetStd()
        exec(file.read(), dict(), dict(sheet=sheet))
        beatmap = Beatmap(sheet.audio, sheet.events)

        console = KnockConsole()
        console.play(beatmap)

        # print()
        # for event in beatmap.events:
        #     print(event)
