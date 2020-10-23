#!/usr/bin/env python3

import sys
from knock import *
from beatmap import *

if __name__ == "__main__":
    filename = sys.argv[1]

    with open(filename) as file:
        beatmap = Beatmap()
        beatmap.read(file.read())
        console = KnockConsole()
        console.play(beatmap)

        # print()
        # for event in beatmap.events:
        #     print(event)
