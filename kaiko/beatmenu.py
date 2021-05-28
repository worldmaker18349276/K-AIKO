import sys
import os
import contextlib
import traceback
import zipfile
import shutil
import psutil
from pathlib import Path
import appdirs
import pyaudio
from . import datanodes as dn
from . import cfg
from . import tui
from . import kerminal
from . import beatshell
from . import biparser
from .beatshell import BeatShellSettings
from .beatmap import BeatmapPlayer, GameplaySettings
from .beatsheet import BeatSheet, BeatmapParseError
from . import beatanalyzer


def print_pyaudio_info(manager):
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

class KAIKOMenuSettings(cfg.Configurable):
    logo: str = """

  ██▀ ▄██▀   ▄██████▄ ▀██████▄ ██  ▄██▀ █▀▀▀▀▀▀█
  ▀ ▄██▀  ▄▄▄▀█    ██    ██    ██▄██▀   █ ▓▓▓▓ █
  ▄██▀██▄ ▀▀▀▄███████    ██    ███▀██▄  █ ▓▓▓▓ █
  █▀   ▀██▄  ██    ██ ▀██████▄ ██   ▀██▄█▄▄▄▄▄▄█


  🎧  Use headphones for the best experience 🎝 

"""

    data_icon: str = "\x1b[92m🗀 \x1b[m"
    info_icon: str = "\x1b[94m🛠 \x1b[m"
    hint_icon: str = "\x1b[93m💡 \x1b[m"

    verb_attr: str = "2"
    emph_attr: str = "1"
    warn_attr: str = "31"

    best_screen_size: int = 80

class KAIKOSettings(cfg.Configurable):
    menu = KAIKOMenuSettings
    shell = BeatShellSettings
    gameplay = GameplaySettings


class KAIKOMenu:
    def __init__(self, settings, username, data_dir, songs_dir, manager):
        self.settings = settings
        self._username = username
        self._data_dir = data_dir
        self._songs_dir = songs_dir
        self.manager = manager
        self._beatmaps = []
        self.songs_mtime = None

    @beatshell.function_command
    def username(self):
        return self._username

    @beatshell.function_command
    def data_dir(self):
        return self._data_dir

    @beatshell.function_command
    def config_file(self):
        return self._data_dir / "config.py"

    @beatshell.function_command
    def songs_dir(self):
        return self._songs_dir

    @beatshell.function_command
    def beatmaps(self):
        if self.songs_mtime != os.stat(str(self._songs_dir)).st_mtime:
            self.reload()

        return self._beatmaps

    @beatshell.function_command
    def intro(self):
        print(
"""Beat shell is a user friendly commandline shell for playing K-AIKO.

Just type a command followed by any arguments, and press \x1b[1mEnter\x1b[m to execute!

 \x1b[2mbeating prompt\x1b[m
   \x1b[2m│\x1b[m    \x1b[2mthe command you want to execute\x1b[m
   \x1b[2m│\x1b[m     \x1b[2m╱\x1b[m     \x1b[2m╭──\x1b[m \x1b[2margument of command\x1b[m
\x1b[36m⠶⠠⣊⠄⠴\x1b[m\x1b[38;5;252m❯ \x1b[m\x1b[94msay\x1b[m \x1b[92m'Welcome\x1b[2m⌴\x1b[m\x1b[92mto\x1b[2m⌴\x1b[m\x1b[92mK-AIKO!'\x1b[m \x1b[7;2m \x1b[m
Welcome to K-AIKO!    \x1b[2m│\x1b[m         \x1b[2m╰─\x1b[m \x1b[2mbeating caret\x1b[m
     \x1b[2m╲\x1b[m                \x1b[2m╰───\x1b[m \x1b[2mquoted whitespace look like this!\x1b[m
   \x1b[2moutput of command\x1b[m
""")

    @beatshell.function_command
    def reload(self):
        """Reload your songs.

        usage: \x1b[94mreload\x1b[m
        """

        info_icon = self.settings.menu.info_icon
        emph_attr = self.settings.menu.emph_attr
        songs_dir = self._songs_dir

        print(f"{info_icon} Loading songs from {tui.add_attr(songs_dir.as_uri(), emph_attr)}...")

        self._beatmaps = []

        for file in songs_dir.iterdir():
            if file.is_file() and file.suffix == ".osz":
                distpath = file.parent / file.stem
                if distpath.exists():
                    continue
                print(f"{info_icon} Unzip file {tui.add_attr(file.as_uri(), emph_attr)}...")
                distpath.mkdir()
                zf = zipfile.ZipFile(str(file), 'r')
                zf.extractall(path=str(distpath))
                file.unlink()

        for song in songs_dir.iterdir():
            if song.is_dir():
                for beatmap in song.iterdir():
                    if beatmap.suffix in (".kaiko", ".ka", ".osu"):
                        self._beatmaps.append(beatmap)

        if len(self._beatmaps) == 0:
            print(f"{info_icon} There is no song in the folder yet!")
        print(flush=True)

        self.songs_mtime = os.stat(str(songs_dir)).st_mtime

    @beatshell.function_command
    def add(self, beatmap):
        """Add beatmap/beatmapset to your songs folder.

        usage: \x1b[94madd\x1b[m \x1b[92m{beatmap}\x1b[m
                        ╲
              Path, the path to the
             beatmap you want to add.
             You can drop the file to
           the terminal to paste its path.
        """

        info_icon = self.settings.menu.info_icon
        emph_attr = self.settings.menu.emph_attr
        warn_attr = self.settings.menu.warn_attr
        songs_dir = self._songs_dir

        if not beatmap.exists():
            print(tui.add_attr(f"File not found: {str(beatmap)}", warn_attr))
            return

        print(f"{info_icon} Adding new song from {tui.add_attr(beatmap.as_uri(), emph_attr)}...")

        distpath = songs_dir / beatmap.name
        n = 1
        while distpath.exists():
            distpath = songs_dir / (beatmap.name + f" ({n})")
            n += 1
        if n != 1:
            print(f"{info_icon} Name conflict! rename to {tui.add_attr(distpath.name, emph_attr)}")

        if beatmap.is_file():
            shutil.copy(str(beatmap), str(songs_dir))
        elif beatmap.is_dir():
            shutil.copytree(str(beatmap), str(distpath))
        else:
            print(tui.add_attr(f"Not a file: {str(beatmap)}", warn_attr))
            return

        self.reload()

    @add.arg_parser("beatmap")
    def _add_beatmap_parser(self):
        return beatshell.PathParser()

    @beatshell.function_command
    def remove(self, beatmap):
        """Remove beatmap/beatmapset in your songs folder.

        usage: \x1b[94mremove\x1b[m \x1b[92m{beatmap}\x1b[m
                           ╲
                 Path, the path to the
               beatmap you want to remove.
        """

        info_icon = self.settings.menu.info_icon
        emph_attr = self.settings.menu.emph_attr
        warn_attr = self.settings.menu.warn_attr
        songs_dir = self._songs_dir

        beatmap_path = songs_dir / beatmap
        if beatmap_path.is_file():
            print(f"{info_icon} Removing the beatmap at {tui.add_attr(beatmap_path.as_uri(), emph_attr)}...")
            beatmap_path.unlink()
            self.reload()

        elif beatmap_path.is_dir():
            print(f"{info_icon} Removing the beatmapset at {tui.add_attr(beatmap_path.as_uri(), emph_attr)}...")
            shutil.rmtree(str(beatmap_path))
            self.reload()

        else:
            print(tui.add_attr(f"Not a file: {str(beatmap)}", warn_attr))

    @remove.arg_parser("beatmap")
    def _remove_beatmap_parser(self):
        songs_dir = self._songs_dir

        options = []
        for song in songs_dir.iterdir():
            options.append(os.path.join(str(song.relative_to(songs_dir)), ""))
            if song.is_dir():
                for beatmap in song.iterdir():
                    if beatmap.suffix in (".kaiko", ".ka", ".osu"):
                        options.append(str(beatmap.relative_to(songs_dir)))

        return beatshell.OptionParser(options)

    @beatshell.function_command
    def play(self, beatmap):
        """Let's beat with the song!

        usage: \x1b[94mplay\x1b[m \x1b[92m{beatmap}\x1b[m
                         ╲
               Path, the path to the
              beatmap you want to play.
              Only the beatmaps in your
             songs folder can be accessed.
        """

        return KAIKOPlay(self._songs_dir / beatmap, self.settings.gameplay)

    @play.arg_parser("beatmap")
    def _play_beatmap_parser(self):
        return beatshell.OptionParser([str(beatmap.relative_to(self._songs_dir)) for beatmap in self.beatmaps()])

    @beatshell.subcommand
    @property
    def config(self):
        return ConfigCommand(self.settings, self.config_file())

    @beatshell.function_command
    def say(self, message, escape=False):
        """Say something and I will echo.

        usage: \x1b[94msay\x1b[m \x1b[92m{message}\x1b[m [\x1b[95m--escape\x1b[m \x1b[92m{ESCAPE}\x1b[m]\x1b[m
                      ╱                    ╲
            text, the message               ╲
             to be printed.          bool, use backslash escapes
                                    or not; the default is False.
        """

        if escape:
            print(beatshell.echo_str(message))
        else:
            print(message)

    @say.arg_parser("message")
    def _say_message_parser(self):
        return beatshell.RawParser(expected="It should be some text,"
                                            " indicating the message to be printed.")

    @say.arg_parser("escape")
    def _say_escape_parser(self, message):
        return beatshell.LiteralParser(bool, default=False,
                                       expected="It should be bool,"
                                                " indicating whether to use backslash escapes;"
                                                " the default is False.")

    @beatshell.function_command
    def exit(self):
        print("bye~")
        raise KeyboardInterrupt

    @staticmethod
    def fit_screen(width, delay=1.0):
        import time

        @dn.datanode
        def fit():
            size = yield
            current_width = 0

            if size.columns < width:
                t = time.time()

                print("The screen size seems too small.")
                print(f"Can you adjust the screen size to (or bigger than) {width}?")
                print("Or you can try to fit the line below.")
                print("━"*width)

                while current_width < width or time.time() < t+delay:
                    if current_width != size.columns:
                        current_width = size.columns
                        t = time.time()
                        if current_width < width - 5:
                            hint = "(too small!)"
                        elif current_width < width:
                            hint = "(very close!)"
                        elif current_width == width:
                            hint = "(perfect!)"
                        else:
                            hint = "(great!)"
                        print(f"\r\x1b[KCurrent width: {current_width} {hint}", end="", flush=True)

                    size = yield

                print("\nThanks!\n")

                # sleep
                t = time.time()
                while time.time() < t+delay:
                    yield

        dn.exhaust(dn.pipe(dn.terminal_size(), fit()), dt=0.1)

    @classmethod
    @contextlib.contextmanager
    def init(clz):
        username = psutil.Process().username()
        data_dir = Path(appdirs.user_data_dir("K-AIKO", username))

        # load settings
        config_path = data_dir / "config.py"
        settings = KAIKOSettings.read(config_path)

        data_icon = settings.menu.data_icon
        info_icon = settings.menu.info_icon
        verb_attr = settings.menu.verb_attr
        emph_attr = settings.menu.emph_attr

        # print logo
        print(settings.menu.logo, flush=True)

        # fit screen size
        clz.fit_screen(settings.menu.best_screen_size)

        # load user data
        songs_dir = data_dir / "songs"

        if not data_dir.exists():
            # start up
            print(f"{data_icon} preparing your profile...")
            data_dir.mkdir(exist_ok=True)
            songs_dir.mkdir(exist_ok=True)
            print(f"{data_icon} your data will be stored in "
                  f"{tui.add_attr(data_dir.as_uri(), emph_attr)}")
            print(flush=True)

        # load PyAudio
        print(f"{info_icon} Loading PyAudio...")
        print()

        print(f"\x1b[{verb_attr}m", end="", flush=True)
        try:
            manager = pyaudio.PyAudio()
            print_pyaudio_info(manager)
        finally:
            print("\x1b[m", flush=True)

        try:
            yield clz(settings, username, data_dir, songs_dir, manager)
        finally:
            manager.terminate()

    @staticmethod
    def main():
        try:
            with KAIKOMenu.init() as game:
                info_icon = game.settings.menu.info_icon
                emph_attr = game.settings.menu.emph_attr

                # load songs
                game.reload()

                # execute given command
                if len(sys.argv) > 1:
                    res = beatshell.RootCommand(game).build(sys.argv[1:])()
                    if hasattr(res, 'execute'):
                        res.execute(game.manager)
                    return

                # tips
                print(f"{info_icon} Use {tui.add_attr('Tab', emph_attr)} to autocomplete command.")
                print(f"{info_icon} If you need help, press {tui.add_attr('Alt+Enter', emph_attr)}.")
                print()

                # prompt
                history = []
                while True:
                    result = beatshell.prompt(game, history)

                    if hasattr(result, 'execute'):
                        result.execute(game.manager)

                    elif isinstance(result, list):
                        for item in result:
                            print(item)

                    elif result is not None:
                        print(result)

        except KeyboardInterrupt:
            pass

        except:
            # print error
            print("\x1b[31m", end="")
            traceback.print_exc(file=sys.stdout)
            print(f"\x1b[m", end="")

class ConfigCommand:
    def __init__(self, settings, path):
        self.settings = settings
        self.path = path

    @beatshell.function_command
    def reload(self):
        self.settings = KAIKOSettings.read(self.path)

    @beatshell.function_command
    def show(self):
        print(str(self.settings))

    @beatshell.function_command
    def get(self, field):
        return self.settings.get(field)

    @beatshell.function_command
    def has(self, field):
        return self.settings.has(field)

    @beatshell.function_command
    def unset(self, field):
        self.settings.unset(field)

    @beatshell.function_command
    def set(self, field, value):
        self.settings.set(field, value)

    @get.arg_parser("field")
    @has.arg_parser("field")
    @unset.arg_parser("field")
    @set.arg_parser("field")
    def _field_parser(self):
        return FieldParser(type(self.settings))

    @set.arg_parser("value")
    def _set_value_parser(self, field):
        annotation = type(self.settings).get_configurable_fields()[field]
        default = self.settings.get(field)
        return beatshell.LiteralParser(annotation, default)

class FieldParser(beatshell.ArgumentParser):
    def __init__(self, config_type):
        self.config_type = config_type
        self.biparser = cfg.FieldBiparser(config_type)

    def parse(self, token):
        try:
            return self.biparser.decode(token)[0]
        except biparser.DecodeError:
            raise beatshell.TokenParseError("No such field")

    def suggest(self, token):
        try:
            self.biparser.decode(token)
        except biparser.DecodeError as e:
            sugg = beatshell.fit(token, [token[:e.index] + ex for ex in e.expected])
        else:
            sugg = []

        return sugg


class KAIKOPlay:
    def __init__(self, filepath, settings):
        self.filepath = filepath
        self.settings = settings

    @contextlib.contextmanager
    def execute(self, manager):
        try:
            beatmap = BeatSheet.read(str(self.filepath))

        except BeatmapParseError:
            print(f"failed to read beatmap {str(self.filepath)}")
            print("\x1b[31m", end="")
            traceback.print_exc(file=sys.stdout)
            print(f"\x1b[m", end="")

        else:
            game = BeatmapPlayer(beatmap, self.settings)
            game.execute(manager)

            print()
            beatanalyzer.show_analyze(beatmap.settings.difficulty.performance_tolerance, game.perfs)

