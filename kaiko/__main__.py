import sys
import os
import contextlib
import traceback
import zipfile
import psutil
import appdirs
from . import datanodes as dn
from . import cfg
from . import tui
from . import kerminal
from . import beatcmd
from .beatmap import BeatmapPlayer
from .beatsheet import BeatmapDraft, BeatmapParseError
from . import beatanalyzer


def print_logo():
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
    data_icon: str = "\x1b[92m🗀\x1b[m "
    info_icon: str = "\x1b[94m🛠\x1b[m "
    hint_icon: str = "\x1b[93m💡\x1b[m "

    verb: str = "2"
    emph: str = "1"
    warn: str = "31"

class KAIKOGame:
    def __init__(self, theme, data_dir, songs_dir, manager):
        self.theme = theme
        self.data_dir = data_dir
        self.songs_dir = songs_dir
        self.manager = manager
        self._beatmaps = []
        self.songs_mtime = None

    @classmethod
    @contextlib.contextmanager
    def init(clz, theme_path=None):
        # print logo
        print_logo()

        # load theme
        theme = KAIKOTheme()
        if theme_path is not None:
            cfg.config_read(open(theme_path, 'r'), main=theme)

        # load user data
        data_dir = appdirs.user_data_dir("K-AIKO", psutil.Process().username())
        songs_dir = os.path.join(data_dir, "songs")

        if not os.path.isdir(data_dir):
            # start up
            print(f"{theme.data_icon} preparing your profile...")
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(songs_dir, exist_ok=True)
            print(f"{theme.data_icon} your data will be stored in "
                  f"{tui.add_attr('file://'+data_dir, theme.emph)}")
            print(flush=True)

        # load PyAudio
        print(f"{theme.info_icon} Loading PyAudio...")
        print()

        ctxt = kerminal.prepare_pyaudio()

        print(f"\x1b[{theme.verb}m", end="", flush=True)
        try:
            manager = ctxt.__enter__()
            print_pyaudio_info(manager)
        finally:
            print("\x1b[m", flush=True)

        try:
            yield clz(theme, data_dir, songs_dir, manager)
        except:
            ctxt.__exit__(*sys.exc_info())
        else:
            ctxt.__exit__(None, None, None)

    def reload(self):
        info_icon = self.theme.info_icon
        emph = self.theme.emph
        songs_dir = self.songs_dir

        print(f"{info_icon} Loading songs from {tui.add_attr('file://'+songs_dir, emph)}...")

        self._beatmaps = []

        for root, dirs, files in os.walk(songs_dir):
            for file in files:
                if file.endswith(".osz"):
                    filepath = os.path.join(root, file)
                    distpath, _ = os.path.splitext(filepath)
                    if os.path.isdir(distpath):
                        continue
                    print(f"{info_icon} Unzip file {tui.add_attr('file://'+filepath, emph)}...")
                    os.makedirs(distpath)
                    zf = zipfile.ZipFile(filepath, 'r')
                    zf.extractall(path=distpath)

        for root, dirs, files in os.walk(songs_dir):
            for file in files:
                if file.endswith((".kaiko", ".ka", ".osu")):
                    filepath = os.path.relpath(os.path.join(root, file), songs_dir)
                    self._beatmaps.append(filepath)

        if len(self._beatmaps) == 0:
            print("{info_icon} There is no song in the folder yet!")
        print(flush=True)

        self.songs_mtime = os.stat(songs_dir).st_mtime

    @property
    def beatmaps(self):
        if self.songs_mtime != os.stat(self.songs_dir).st_mtime:
            self.reload()

        return list(self._beatmaps)

    def play(self, beatmap):
        if not os.path.isabs(beatmap):
            beatmap = os.path.join(self.songs_dir, beatmap)
        return KAIKOPlay(beatmap)

    def exit(self):
        print("bye~")
        raise KeyboardInterrupt

    def menu(self):
        def play(beatmap:self.beatmaps):
            return self.play(beatmap)

        return beatcmd.Promptable({
            "play": play,
            "reload": self.reload,
            "settings": lambda:None,
            "exit": self.exit,
        })

class KAIKOPlay:
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
        with KAIKOGame.init() as game:
            # play given beatmap
            if len(sys.argv) > 1:
                filepath = sys.argv[1]
                game.play(filepath).execute(game.manager)
                return

            # load songs
            game.reload()

            # prompt
            while True:
                result = beatcmd.prompt(game.menu())
                if hasattr(result, 'execute'):
                    result.execute(game.manager)

    except KeyboardInterrupt:
        pass

    except:
        # print error
        print("\x1b[31m", end="")
        traceback.print_exc(file=sys.stdout)
        print(f"\x1b[m", end="")

if __name__ == '__main__':
    main()
