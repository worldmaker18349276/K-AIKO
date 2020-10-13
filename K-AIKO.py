#!/usr/bin/env python3

import sys
from knock import *
from beatmap import *

if __name__ == "__main__":
    filename = sys.argv[1]

    with open(filename) as file:
        sheet = BeatSheetStd()
        if filename.endswith(".kaiko"):
            sheet.load(file.read())
        elif filename.endswith(".osu"):
            sheet.load_from_osu(file.read())
        beatmap = Beatmap(sheet.audio, sheet.events)

        console = KnockConsole()
        console.play(beatmap)

        # print()
        # for event in beatmap.events:
        #     print(event)
