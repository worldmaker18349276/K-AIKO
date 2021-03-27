import sys
import traceback
import os
import zipfile
from typing import Tuple
import contextlib
import psutil
import appdirs
from . import datanodes as dn
from . import cfg
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

def print_pyaudio_info(manager):
    import pyaudio

    print()

    print("portaudio version:")
    print("  " + pyaudio.get_portaudio_version_text())
    print()

    print("available devices:")
    apis_list = [manager.get_host_api_info_by_index(i)['name'] for i in range(manager.get_host_api_count())]

    table = []
    for index in range(manager.get_device_count()):
        info = manager.get_device_info_by_index(index)

        ind = str(index)
        name = info['name']
        api = apis_list[info['hostApi']]
        freq = str(info['defaultSampleRate']/1000)
        chin = str(info['maxInputChannels'])
        chout = str(info['maxOutputChannels'])

        table.append((ind, name, api, freq, chin, chout))

    ind_len   = max(len(entry[0]) for entry in table)
    name_len  = max(len(entry[1]) for entry in table)
    api_len   = max(len(entry[2]) for entry in table)
    freq_len  = max(len(entry[3]) for entry in table)
    chin_len  = max(len(entry[4]) for entry in table)
    chout_len = max(len(entry[5]) for entry in table)

    for ind, name, api, freq, chin, chout in table:
        print(f"  {ind:>{ind_len}}. {name:{name_len}}  by  {api:{api_len}}"
              f"  ({freq:>{freq_len}} kHz, in: {chin:>{chin_len}}, out: {chout:>{chout_len}})")

    print()

    default_input_device_index = manager.get_default_input_device_info()['index']
    default_output_device_index = manager.get_default_output_device_info()['index']
    print(f"default input device: {default_input_device_index}")
    print(f"default output device: {default_output_device_index}")

class KAIKOTheme(metaclass=cfg.Configurable):
    data_icon: str = "\x1b[92m🗀\x1b[0m "
    info_icon: str = "\x1b[94m🛠\x1b[0m "
    hint_icon: str = "\x1b[93m💡\x1b[0m "

    verb: Tuple[str, str] = ("\x1b[2m", "\x1b[0m")
    emph: Tuple[str, str] = ("\x1b[1m", "\x1b[21m")
    warn: Tuple[str, str] = ("\x1b[31m", "\x1b[0m")

def main(theme=None):
    # load theme
    settings = KAIKOTheme()
    if theme is not None:
        cfg.config_read(open(theme, 'r'), main=settings)

    data_icon = settings.data_icon
    info_icon = settings.info_icon
    hint_icon = settings.hint_icon
    verb = settings.verb
    emph = settings.emph
    warn = settings.warn

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

            print_pyaudio_info(manager)

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
            if not os.path.isdir(user_data_dir):
                # start up
                print(f"{data_icon} preparing your profile...")
                os.makedirs(user_data_dir, exist_ok=True)
                os.makedirs(user_songs_dir, exist_ok=True)
                print(f"{data_icon} your data will be stored in {('file://'+user_data_dir).join(emph)}")
                print(flush=True)

            # load songs
            print(f"{info_icon} Loading songs from {('file://'+user_songs_dir).join(emph)}...")

            def load_songs():
                songs = []

                for root, dirs, files in os.walk(user_songs_dir):
                    for file in files:
                        if file.endswith(".osz"):
                            filepath = os.path.join(root, file)
                            distpath, _ = os.path.splitext(filepath)
                            if os.path.isdir(distpath):
                                continue
                            print(f"{info_icon} Unzip file {('file://'+filepath).join(emph)}...")
                            os.makedirs(distpath)
                            zf = zipfile.ZipFile(filepath, 'r')
                            zf.extractall(path=distpath)

                for root, dirs, files in os.walk(user_songs_dir):
                    for file in files:
                        if file.endswith((".kaiko", ".ka", ".osu")):
                            filepath = os.path.join(root, file)
                            songs.append((file, BeatMenuPlay(filepath)))

                return songs

            songs = load_songs()
            songs_mtime = os.stat(user_songs_dir).st_mtime

            def play_menu():
                nonlocal songs, songs_mtime
                if songs_mtime != os.stat(user_songs_dir).st_mtime:
                    songs = load_songs()
                    songs_mtime = os.stat(user_songs_dir).st_mtime
                return beatmenu.menu_tree(songs)

            menu = beatmenu.menu_tree([
                ("play", play_menu),
                ("settings", None),
            ])

            if len(songs) == 0:
                print("{info_icon} There is no song in the folder yet!")

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
