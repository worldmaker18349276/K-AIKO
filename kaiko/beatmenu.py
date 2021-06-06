import sys
import os
import random
import queue
import contextlib
from typing import Tuple, Optional
import traceback
import zipfile
import shutil
import psutil
import pkgutil
from pathlib import Path
import appdirs
import pyaudio
from . import datanodes as dn
from . import config as cfg
from . import wcbuffers as wcb
from . import beatshell
from . import biparsers as bp
from .engines import Mixer, MixerSettings
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
    def __init__(self, config, username, data_dir, songs_dir, manager):
        self._config = config
        self._username = username
        self._data_dir = data_dir
        self._songs_dir = songs_dir
        self.manager = manager
        self._beatmaps = []
        self.songs_mtime = None
        self._current_bgm = None
        self._bgm_repeat = False
        self.bgm_queue = queue.Queue()

    @staticmethod
    def main():
        try:
            with KAIKOMenu.init() as game:
                info_icon = game.settings.menu.info_icon
                hint_icon = game.settings.menu.hint_icon
                emph_attr = game.settings.menu.emph_attr
                dt = 0.01

                # fit screen size
                game.fit_screen(game.settings.menu.best_screen_size)

                # load songs
                game.reload()

                # execute given command
                if len(sys.argv) > 1:
                    result = beatshell.RootCommand(game).build(sys.argv[1:])()
                    game.run_result(result, dt)
                    return

                # load mixer
                with game.load_bgm(game.manager) as bgm_knot:
                    # tips
                    print(f"{hint_icon} Use {wcb.add_attr('Tab', emph_attr)} to autocomplete command.")
                    print(f"{hint_icon} If you need help, press {wcb.add_attr('Alt+Enter', emph_attr)}.")
                    print()

                    # prompt
                    history = []
                    while True:
                        # parse command
                        prompt_knot, prompt = beatshell.prompt(game, history)
                        dn.exhaust(prompt_knot, dt, interruptible=True, sync_to=bgm_knot)
                        result = prompt.result()

                        # execute result
                        game.run_result(result, dt, bgm_knot)

        except KeyboardInterrupt:
            pass

        except:
            # print error
            print("\x1b[31m", end="")
            traceback.print_exc(file=sys.stdout)
            print(f"\x1b[m", end="")

    @classmethod
    @contextlib.contextmanager
    def init(clz):
        username = psutil.Process().username()
        data_dir = Path(appdirs.user_data_dir("K-AIKO", username))

        # load settings
        config_path = data_dir / "config.py"
        config = cfg.Configuration(KAIKOSettings)
        if config_path.exists():
            config.read(config_path)
        settings = config.current

        data_icon = settings.menu.data_icon
        info_icon = settings.menu.info_icon
        verb_attr = settings.menu.verb_attr
        emph_attr = settings.menu.emph_attr

        # print logo
        print(settings.menu.logo, flush=True)

        # load user data
        songs_dir = data_dir / "songs"

        if not data_dir.exists():
            # start up
            print(f"{data_icon} Prepare your profile...")
            data_dir.mkdir(exist_ok=True)
            songs_dir.mkdir(exist_ok=True)
            if not config_path.exists():
                config.write(config_path)

            (data_dir / "samples/").mkdir(exist_ok=True)
            resources = ["samples/soft.wav",
                         "samples/loud.wav",
                         "samples/incr.wav",
                         "samples/rock.wav",
                         "samples/disk.wav"]
            for rspath in resources:
                print(f"{data_icon} Load resource {rspath}...")
                data = pkgutil.get_data("kaiko", rspath)
                open(data_dir / rspath, 'wb').write(data)

            print(f"{data_icon} Your data will be stored in "
                  f"{wcb.add_attr(data_dir.as_uri(), emph_attr)}")
            print(flush=True)

        # load PyAudio
        print(f"{info_icon} Load PyAudio...")
        print()

        print(f"\x1b[{verb_attr}m", end="", flush=True)
        try:
            manager = pyaudio.PyAudio()
            print_pyaudio_info(manager)
        finally:
            print("\x1b[m", flush=True)

        try:
            yield clz(config, username, data_dir, songs_dir, manager)
        finally:
            manager.terminate()

    @beatshell.function_command
    def exit(self):
        print("bye~")
        raise KeyboardInterrupt

    def run_result(self, result, dt, bgm_knot=None):
        if hasattr(result, 'execute'):
            self.bgm_stop()
            result.execute(self.manager)
            self.bgm_start()

        elif isinstance(result, dn.DataNode):
            dn.exhaust(result, dt, interruptible=True, sync_to=bgm_knot)

        elif isinstance(result, list):
            for item in result:
                print(item)

        elif result is not None:
            print(result)

    def load_bgm(self, manager):
        warn_attr = self.settings.menu.warn_attr
        device = self.settings.gameplay.mixer.output_device

        try:
            settings = MixerSettings()
            settings.output_device = device
            if device == -1: device = manager.get_default_output_device_info()['index']
            device_info = manager.get_device_info_by_index(device)
            settings.output_samplerate = int(device_info['defaultSampleRate'])
            settings.output_channels = min(2, device_info['maxOutputChannels'])

            knot, mixer = Mixer.create(settings, manager)

        except Exception:
            print(f"\x1b[{warn_attr}m", end="")
            print("Fail to load mixer")
            traceback.print_exc(file=sys.stdout)
            print(f"\x1b[m", end="")

            return dn.DataNode.wrap(lambda _:None)

        else:
            return dn.pipe(knot, dn.interval(self._bgm_rountine(mixer), dt=0.1))

    @dn.datanode
    def _bgm_rountine(self, mixer):
        current_bgm = None

        yield
        while True:
            while not self.bgm_queue.empty():
                current_bgm = self.bgm_queue.get()
            self._current_bgm = current_bgm

            if self._current_bgm is None:
                yield
                continue

            current_song, (start, end) = current_bgm

            with mixer.play(current_song, start=start, end=end) as bgm_key:
                while self.bgm_queue.empty():
                    if bgm_key.is_finalized():
                        if self._bgm_repeat:
                            self.bgm_queue.put(current_bgm)
                        else:
                            songs = self.get_songs()
                            songs.remove(current_song)
                            if songs:
                                self.bgm_queue.put((random.choice(songs), (None, None)))

                        break

                    yield

    def get_song(self, beatmap):
        beatmap = BeatSheet.read(str(self._songs_dir / beatmap), metadata_only=True)
        if beatmap.audio is None:
            return None
        return os.path.join(beatmap.root, beatmap.audio)

    def get_songs(self):
        songs = set()
        for beatmap in self._beatmaps:
            try:
                song = self.get_song(beatmap)
            except BeatmapParseError:
                pass
            else:
                if song is not None:
                    songs.add(song)

        return list(songs)

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

    # properties

    @property
    def settings(self):
        return self._config.current

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
    def current_bgm(self):
        return self._current_bgm and self._current_bgm[0]

    # bgm

    @beatshell.function_command
    def bgm_stop(self):
        self.bgm_queue.put(None)

    @beatshell.function_command
    def bgm_start(self, beatmap=None, clip:Tuple[Optional[float], Optional[float]]=(None, None)):
        warn_attr = self.settings.menu.warn_attr

        if beatmap is None:
            if self._current_bgm is not None:
                return

            songs = self.get_songs()
            if not songs:
                print(f"{data_icon} There is no song in the folder yet!")
                return

            self.bgm_queue.put((random.choice(songs), clip))
            return

        try:
            song = self.get_song(beatmap)

        except BeatmapParseError:
            print(wcb.add_attr(f"Fail to read beatmap", warn_attr))

        else:
            if song is None:
                print(wcb.add_attr(f"This beatmap has no song", warn_attr))

            self.bgm_queue.put((song, clip))

    @beatshell.function_command
    def bgm_repeat(self, repeat:bool):
        self._bgm_repeat = repeat

    # beatmaps

    @beatshell.function_command
    def reload(self):
        """Reload your songs.

        usage: \x1b[94mreload\x1b[m
        """

        data_icon = self.settings.menu.data_icon
        emph_attr = self.settings.menu.emph_attr
        songs_dir = self._songs_dir

        print(f"{data_icon} Load songs from {wcb.add_attr(songs_dir.as_uri(), emph_attr)}...")

        self._beatmaps = []

        for file in songs_dir.iterdir():
            if file.is_file() and file.suffix == ".osz":
                distpath = file.parent / file.stem
                if distpath.exists():
                    continue
                print(f"{data_icon} Unzip file {wcb.add_attr(file.as_uri(), emph_attr)}...")
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
            print(f"{data_icon} There is no song in the folder yet!")
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

        data_icon = self.settings.menu.data_icon
        emph_attr = self.settings.menu.emph_attr
        warn_attr = self.settings.menu.warn_attr
        songs_dir = self._songs_dir

        if not beatmap.exists():
            print(wcb.add_attr(f"File not found: {str(beatmap)}", warn_attr))
            return
        if not beatmap.is_file() and not beatmap.is_dir():
            print(wcb.add_attr(f"Not a file or directory: {str(beatmap)}", warn_attr))
            return

        print(f"{data_icon} Add new song from {wcb.add_attr(beatmap.as_uri(), emph_attr)}...")

        distpath = songs_dir / beatmap.name
        n = 1
        while distpath.exists():
            distpath = songs_dir / (beatmap.name + f" ({n})")
            n += 1
        if n != 1:
            print(f"{data_icon} Name conflict! rename to {wcb.add_attr(distpath.name, emph_attr)}")

        if beatmap.is_file():
            shutil.copy(str(beatmap), str(songs_dir))
        elif beatmap.is_dir():
            shutil.copytree(str(beatmap), str(distpath))

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

        data_icon = self.settings.menu.data_icon
        emph_attr = self.settings.menu.emph_attr
        warn_attr = self.settings.menu.warn_attr
        songs_dir = self._songs_dir

        beatmap_path = songs_dir / beatmap
        if beatmap_path.is_file():
            print(f"{data_icon} Remove the beatmap at {wcb.add_attr(beatmap_path.as_uri(), emph_attr)}...")
            beatmap_path.unlink()
            self.reload()

        elif beatmap_path.is_dir():
            print(f"{data_icon} Remove the beatmapset at {wcb.add_attr(beatmap_path.as_uri(), emph_attr)}...")
            shutil.rmtree(str(beatmap_path))
            self.reload()

        else:
            print(wcb.add_attr(f"Not a file: {str(beatmap)}", warn_attr))

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

        return KAIKOPlay(self._data_dir, self._songs_dir / beatmap, self.settings.gameplay)

    @bgm_start.arg_parser("beatmap")
    @play.arg_parser("beatmap")
    def _play_beatmap_parser(self):
        return BeatmapParser(self._beatmaps, self._songs_dir)

    # audio

    @beatshell.function_command
    def audio_input(self, device, samplerate=None, channels=None, format=None):
        warn_attr = self.settings.menu.warn_attr

        pa_device = device
        pa_samplerate = samplerate
        pa_channels = channels

        if pa_device == -1:
            pa_device = self.manager.get_default_input_device_info()['index']
        if pa_samplerate is None:
            pa_samplerate = self.settings.gameplay.detector.input_samplerate
        if pa_channels is None:
            pa_channels = self.settings.gameplay.detector.input_channels

        pa_format = {
            'f4': pyaudio.paFloat32,
            'i4': pyaudio.paInt32,
            'i2': pyaudio.paInt16,
            'i1': pyaudio.paInt8,
            'u1': pyaudio.paUInt8,
        }[format or self.settings.gameplay.detector.input_format]

        try:
            self.manager.is_format_supported(pa_samplerate,
                input_device=pa_device, input_channels=pa_channels, input_format=pa_format)

        except ValueError as e:
            info = e.args[0]
            print(wcb.add_attr(info, warn_attr))

        else:
            self.settings.gameplay.detector.input_device = device
            if samplerate is not None:
                self.settings.gameplay.detector.input_samplerate = samplerate
            if channels is not None:
                self.settings.gameplay.detector.input_channels = channels
            if format is not None:
                self.settings.gameplay.detector.input_format = format

    @beatshell.function_command
    def audio_output(self, device, samplerate=None, channels=None, format=None):
        warn_attr = self.settings.menu.warn_attr

        pa_device = device
        pa_samplerate = samplerate
        pa_channels = channels

        if pa_device == -1:
            pa_device = self.manager.get_default_output_device_info()['index']
        if pa_samplerate is None:
            pa_samplerate = self.settings.gameplay.mixer.output_samplerate
        if pa_channels is None:
            pa_channels = self.settings.gameplay.mixer.output_channels

        pa_format = {
            'f4': pyaudio.paFloat32,
            'i4': pyaudio.paInt32,
            'i2': pyaudio.paInt16,
            'i1': pyaudio.paInt8,
            'u1': pyaudio.paUInt8,
        }[format or self.settings.gameplay.mixer.output_format]

        try:
            self.manager.is_format_supported(pa_samplerate,
                output_device=pa_device, output_channels=pa_channels, output_format=pa_format)

        except ValueError as e:
            info = e.args[0]
            print(wcb.add_attr(info, warn_attr))

        else:
            self.settings.gameplay.mixer.output_device = device
            if samplerate is not None:
                self.settings.gameplay.mixer.output_samplerate = samplerate
            if channels is not None:
                self.settings.gameplay.mixer.output_channels = channels
            if format is not None:
                self.settings.gameplay.mixer.output_format = format

    @audio_input.arg_parser("device")
    def _audio_input_device_parser(self):
        return PyAudioDeviceParser(self.manager, True)

    @audio_output.arg_parser("device")
    def _audio_output_device_parser(self):
        return PyAudioDeviceParser(self.manager, False)

    @audio_input.arg_parser("samplerate")
    @audio_output.arg_parser("samplerate")
    def _audio_samplerate_parser(self, device, **__):
        options = [44100, 48000, 88200, 96000, 32000, 22050, 11025, 8000]
        return beatshell.OptionParser({str(rate): rate for rate in options})

    @audio_input.arg_parser("channels")
    @audio_output.arg_parser("channels")
    def _audio_channels_parser(self, device, **__):
        return beatshell.OptionParser({'2': 2, '1': 1})

    @audio_input.arg_parser("format")
    @audio_output.arg_parser("format")
    def _audio_format_parser(self, device, **__):
        return beatshell.OptionParser(['f4', 'i4', 'i2', 'i1', 'u1'])

    # config

    @beatshell.subcommand
    @property
    def config(self):
        return ConfigCommand(self._config, self.config_file())


class ConfigCommand:
    def __init__(self, config, path):
        self.config = config
        self.path = path

    @beatshell.function_command
    def reload(self):
        data_icon = self.config.current.menu.data_icon
        emph_attr = self.config.current.menu.emph_attr

        print(f"{data_icon} Load configuration from {wcb.add_attr(self.path.as_uri(), emph_attr)}...")
        print()

        self.config.read(self.path)

    @beatshell.function_command
    def save(self):
        data_icon = self.config.current.menu.data_icon
        emph_attr = self.config.current.menu.emph_attr

        print(f"{data_icon} Save configuration to {wcb.add_attr(self.path.as_uri(), emph_attr)}...")
        print()

        self.config.write(self.path)

    @beatshell.function_command
    def show(self):
        print(str(self.config))

    @beatshell.function_command
    def get(self, field):
        return self.config.get(field)

    @beatshell.function_command
    def has(self, field):
        return self.config.has(field)

    @beatshell.function_command
    def unset(self, field):
        self.config.unset(field)

    @beatshell.function_command
    def set(self, field, value):
        self.config.set(field, value)

    @get.arg_parser("field")
    @has.arg_parser("field")
    @unset.arg_parser("field")
    @set.arg_parser("field")
    def _field_parser(self):
        return FieldParser(self.config.config_type)

    @set.arg_parser("value")
    def _set_value_parser(self, field):
        annotation = self.config.config_type.get_configurable_fields()[field]
        default = self.config.get(field)
        return beatshell.LiteralParser(annotation, default)

    @beatshell.function_command
    def rename(self, profile):
        warn_attr = self.config.current.menu.warn_attr

        if "\n" in profile or "\r" in profile:
            print(wcb.add_attr("Invalid profile name.", warn_attr))
            return

        if profile in self.config.profiles:
            print(wcb.add_attr("This profile name already exists.", warn_attr))
            return

        self.config.name = profile

    @beatshell.function_command
    def new(self, profile, clone=None):
        warn_attr = self.config.current.menu.warn_attr

        if "\n" in profile or "\r" in profile:
            print(wcb.add_attr("Invalid profile name.", warn_attr))
            return

        if profile == self.config.name or profile in self.config.profiles:
            print(wcb.add_attr("This profile name already exists.", warn_attr))
            return

        self.config.new(profile, clone)

    @rename.arg_parser("profile")
    @new.arg_parser("profile")
    def _new_profile_parser(self):
        return beatshell.RawParser()

    @new.arg_parser("clone")
    def _new_clone_parser(self, profile):
        options = list(self.config.profiles.keys())
        options.insert(0, self.config.name)
        return beatshell.OptionParser(options)

    @beatshell.function_command
    def use(self, profile):
        self.config.use(profile)

    @beatshell.function_command
    def delete(self, profile):
        del self.config.profiles[profile]

    @use.arg_parser("profile")
    @delete.arg_parser("profile")
    def _profile_parser(self):
        return beatshell.OptionParser(list(self.config.profiles.keys()))

class FieldParser(beatshell.ArgumentParser):
    def __init__(self, config_type):
        self.config_type = config_type
        self.biparser = cfg.FieldBiparser(config_type)

    def parse(self, token):
        try:
            return self.biparser.decode(token)[0]
        except bp.DecodeError:
            raise beatshell.TokenParseError("No such field")

    def suggest(self, token):
        try:
            self.biparser.decode(token)
        except bp.DecodeError as e:
            sugg = beatshell.fit(token, [token[:e.index] + ex for ex in e.expected])
        else:
            sugg = []

        return sugg

class PyAudioDeviceParser(beatshell.ArgumentParser):
    def __init__(self, manager, is_input):
        self.manager = manager
        self.is_input = is_input
        self.options = ["-1"]
        for index in range(manager.get_device_count()):
            self.options.append(str(index))

    def parse(self, token):
        if token not in self.options:
            raise beatshell.TokenParseError("Invalid device index")
        return int(token)

    def suggest(self, token):
        return [val + "\000" for val in beatshell.fit(token, self.options)]

    def info(self, token):
        value = int(token)
        if value == -1:
            if self.is_input:
                value = self.manager.get_default_input_device_info()['index']
            else:
                value = self.manager.get_default_output_device_info()['index']

        device_info = self.manager.get_device_info_by_index(value)

        name = device_info['name']
        api = self.manager.get_host_api_info_by_index(device_info['hostApi'])['name']
        freq = device_info['defaultSampleRate']/1000
        ch_in = device_info['maxInputChannels']
        ch_out = device_info['maxOutputChannels']

        return f"{name} by {api} ({freq} kHz, in: {ch_in}, out: {ch_out})"


class BeatmapParser(beatshell.ArgumentParser):
    def __init__(self, beatmaps, songs_dir):
        self.beatmaps = beatmaps
        self.songs_dir = songs_dir

        self.options = [str(beatmap.relative_to(self.songs_dir)) for beatmap in self.beatmaps]
        self.expected = beatshell.expected_options(self.options)

    def parse(self, token):
        if token not in self.options:
            expected = self.expected
            raise beatshell.TokenParseError("Invalid value" + "\n" + self.expected)

        return token

    def suggest(self, token):
        return [val + "\000" for val in beatshell.fit(token, self.options)]

    def info(self, token):
        try:
            beatmap = BeatSheet.read(str(self.songs_dir / token), metadata_only=True)
        except BeatmapParseError:
            return None
        else:
            return beatmap.info.strip()


class KAIKOPlay:
    def __init__(self, data_dir, filepath, settings):
        self.data_dir = data_dir
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
            game = BeatmapPlayer(self.data_dir, beatmap, self.settings)
            game.execute(manager)

            print()
            beatanalyzer.show_analyze(beatmap.settings.difficulty.performance_tolerance, game.perfs)

