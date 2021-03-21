import sys
import traceback
import os
import contextlib
import psutil
import appdirs
from .beatmap import BeatmapPlayer
from .beatsheet import BeatmapDraft, BeatmapParseError
from . import beatanalyzer
from . import kerminal
from . import beatmenu

class BeatMenuPlay:
    def __init__(self, filepath):
        self.filepath = filepath

    @contextlib.contextmanager
    def execute(self, manager):
        try:
            beatmap = BeatmapDraft.read(self.filepath)

        except BeatmapParseError:
            print(f"failed to read beatmap {self.filepath}")

        else:
            game = BeatmapPlayer(beatmap)
            game.execute(manager)

            print()
            beatanalyzer.show_analyze(beatmap.settings.performance_tolerance, game.perfs)

def main():
    try:
        # print logo
        print("\n"
            "  ██▀ ▄██▀   ▄██████▄ ▀██████▄ ██  ▄██▀ █▀▀▀▀▀▀█\n"
            "  ▀ ▄██▀  ▄▄▄▀█    ██    ██    ██▄██▀   █ ▓▓▓▓ █\n"
            "  ▄██▀██▄ ▀▀▀▄███████    ██    ███▀██▄  █ ▓▓▓▓ █\n"
            "  █▀   ▀██▄  ██    ██ ▀██████▄ ██   ▀██▄█▄▄▄▄▄▄█\n"
            "\n"
            "\n"
            "  🎧  Use headphones for the best experience 🎝 \n"
            "\n"
            , flush=True)

        # load PyAudio
        print("\x1b[94m🛠\x1b[0m  Loading PyAudio...", flush=True)
        print("\x1b[2m")
        with kerminal.prepare_pyaudio() as manager:
            print("\x1b[0m", flush=True)

            # play given beatmap
            if len(sys.argv) > 1:
                filepath = sys.argv[1]
                BeatMenuPlay(filepath).execute(manager)
                return

            # load user data
            user_data_dir = appdirs.user_data_dir("K-AIKO", psutil.Process().username())
            user_songs_dir = os.path.join(user_data_dir, "songs")
            if not os.path.isdir(user_data_dir) or True:
                # start up
                print(f"\x1b[92m🗀\x1b[0m  preparing your profile...")
                os.makedirs(user_data_dir, exist_ok=True)
                os.makedirs(user_songs_dir, exist_ok=True)
                print(f"\x1b[92m🗀\x1b[0m  your data will be stored in \x1b[1mfile://{user_data_dir}\x1b[21m")
                print(flush=True)

            # load songs
            print(f"\x1b[94m🛠\x1b[0m  Loading songs from \x1b[1mfile://{user_songs_dir}\x1b[21m...")

            songs = []

            for root, dirs, files in os.walk(user_songs_dir):
                for file in files:
                    if file.endswith((".kaiko", ".ka", ".osu")):
                        filepath = os.path.join(root, file)
                        songs.append((file, BeatMenuPlay(filepath)))

            menu = beatmenu.menu_tree([
                ("play", lambda: beatmenu.menu_tree(songs)),
                ("settings", None),
            ])

            print(flush=True)

            # enter menu
            print("\x1b[93m💡\x1b[0m  Use "
                  "\x1b[1mup\x1b[21m/"
                  "\x1b[1mdown\x1b[21m/"
                  "\x1b[1menter\x1b[21m/"
                  "\x1b[1mesc\x1b[21m"
                  " keys to select options.")
            print(flush=True)

            with menu:
                while True:
                    result = beatmenu.explore(menu)
                    if hasattr(result, 'execute'):
                        result.execute(manager)
                    elif result is None:
                        break

    except KeyboardInterrupt:
        pass

    except:
        # print error
        print("\x1b[31m", end="")
        traceback.print_exc(file=sys.stdout)
        print("\x1b[0m", end="")

if __name__ == '__main__':
    main()
