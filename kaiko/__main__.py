import sys
import traceback
import os
from typing import Tuple
import contextlib
import psutil
import appdirs
from .beatmap import BeatmapPlayer
from . import cfg
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

class KAIKOTheme(metaclass=cfg.Configurable):
    data_icon: str = "\x1b[92m🗀\x1b[0m "
    info_icon: str = "\x1b[94m🛠\x1b[0m "
    hint_icon: str = "\x1b[93m💡\x1b[0m "

    verb: Tuple[str, str] = ("\x1b[2m", "\x1b[0m")
    emph: Tuple[str, str] = ("\x1b[1m", "\x1b[21m")
    warn: Tuple[str, str] = ("\x1b[31m", "\x1b[0m")

def main(theme=None):
    # load theme
    theme = GameplaySettings()
    if theme is not None:
        cfg.config_read(open(theme, 'r'), main=theme)

    data_icon = theme.data_icon
    info_icon = theme.info_icon
    hint_icon = theme.hint_icon
    verb = theme.verb
    emph = theme.emph
    warn = theme.warn

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
        print(f"{info_icon} Loading PyAudio...")
        print()
        print(verb[0], end="", flush=True)
        with kerminal.prepare_pyaudio() as manager:
            print(verb[1], end="")
            print(flush=True)

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
                print(f"{data_icon} preparing your profile...")
                os.makedirs(user_data_dir, exist_ok=True)
                os.makedirs(user_songs_dir, exist_ok=True)
                print(f"{data_icon} your data will be stored in {('file://'+user_data_dir).join(emph)}")
                print(flush=True)

            # load songs
            print(f"{info_icon} Loading songs from {('file://'+user_songs_dir).join(emph)}...")

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
            print(f"{hint_icon} Use {'up'.join(emph)}/{'down'.join(emph)}/"
                  f"{'enter'.join(emph)}/{'esc'.join(emph)} keys to select options.")
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
        print(warn[0], end="")
        traceback.print_exc(file=sys.stdout)
        print(warn[1], end="")

if __name__ == '__main__':
    main()
