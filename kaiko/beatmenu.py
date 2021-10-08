import sys
import os
import re
import time
import random
import threading
import queue
import contextlib
import dataclasses
from typing import Tuple, Optional
import traceback
import zipfile
import getpass
import shutil
import pkgutil
from pathlib import Path
import appdirs
import pyaudio
from . import datanodes as dn
from . import config as cfg
from . import wcbuffers as wcb
from . import biparsers as bp
from . import commands as cmd
from . import engines
from . import beatshell
from . import beatmaps
from . import beatsheets
from . import beatanalyzer


def echo_str(escaped_str):
    r"""Interpret a string like bash's echo.
    It interprets the following backslash-escaped characters into:
        \a     alert (bell)
        \b     backspace
        \c     suppress further output
        \e     escape character
        \f     form feed
        \n     new line
        \r     carriage return
        \t     horizontal tab
        \v     vertical tab
        \\     backslash
        \0NNN  the character whose ASCII code is NNN (octal).  NNN can be 0 to 3 octal digits
        \xHH   the eight-bit character whose value is HH (hexadecimal).  HH can be one or two hex digits

    Parameters
    ----------
    escaped_str : str
        The string to be interpreted.

    Returns
    -------
    interpreted_str : str
        The interpreted string.
    """
    regex = r"\\c.*|\\[\\abefnrtv]|\\0[0-7]{0,3}|\\x[0-9a-fA-F]{1,2}|."

    escaped = {
        r"\\": "\\",
        r"\a": "\a",
        r"\b": "\b",
        r"\e": "\x1b",
        r"\f": "\f",
        r"\n": "\n",
        r"\r": "\r",
        r"\t": "\t",
        r"\v": "\v",
        }

    def repl(match):
        matched = match.group(0)

        if matched.startswith("\\c"):
            return ""
        elif matched in escaped:
            return escaped[matched]
        elif matched.startswith("\\0"):
            return chr(int(matched[2:] or "0", 8))
        elif matched.startswith("\\x"):
            return chr(int(matched[2:], 16))
        else:
            return matched

    return re.sub(regex, repl, escaped_str)

def fit_screen(logger, width, delay):
    r"""Guide user to adjust screen size.

    Parameters
    ----------
    logger : KAIKOLogger
    width : int
        The required width of screen.
    delay : float
        The delay time before and after completion.

    Returns
    -------
    fit_task : dn.DataNode
        The datanode to manage this process.
    """
    skip_event = threading.Event()

    skip = dn.input(lambda a: skip_event.set() if a[1] == '\x1b' else None)

    @dn.datanode
    def fit():
        size = yield
        current_width = 0

        if size.columns < width:
            t = time.time()

            logger.print("Your screen size seems too small.")
            logger.print(f"Can you adjust the width to (or bigger than) {width}?")
            logger.print(f"Or {logger.emph('Esc')} to skip this process.")
            logger.print("You can try to fit the line below.")
            logger.print("━"*width, flush=True)

            while current_width < width or time.time() < t+delay:
                if skip_event.is_set():
                    logger.print()
                    break

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
                    logger.print(f"\r\x1b[KCurrent width: {current_width} {hint}", end="", flush=True)

                size = yield

            else:
                logger.print()
                logger.print("Thanks!")

            logger.print("You can adjust the screen size at any time.\n", flush=True)

            # sleep
            t = time.time()
            while time.time() < t+delay:
                yield

    return dn.pipe(skip, dn.terminal_size(), fit())

def print_pyaudio_info(manager, print):
    r"""Print out some information of PyAudio.

    Loading PyAudio will cause PortAudio to print something to the terminal, which
    cannot be turned off by PyAudio, so why not print more information to make it
    more hacky.

    Parameters
    ----------
    manager : PyAudio
        The manager of PyAudio.
    print : function
        The print function.
    """
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

class DevicesSettings(cfg.Configurable):
    mixer = engines.MixerSettings
    detector = engines.DetectorSettings
    renderer = engines.RendererSettings
    controller = engines.ControllerSettings

class KAIKOSettings(cfg.Configurable):
    menu = KAIKOMenuSettings
    devices = DevicesSettings
    shell = beatshell.BeatShellSettings
    gameplay = beatmaps.GameplaySettings


class KAIKOLogger:
    def __init__(self, config):
        self.config = config
        self.level = 1

    @contextlib.contextmanager
    def verb(self):
        verb_attr = self.config.current.menu.verb_attr
        level = self.level
        self.level = 0
        try:
            print(f"\x1b[{verb_attr}m", end="", flush=True)
            yield
        finally:
            self.level = level
            print("\x1b[m", flush=True)

    @contextlib.contextmanager
    def warn(self):
        warn_attr = self.config.current.menu.warn_attr
        level = self.level
        self.level = 2
        try:
            print(f"\x1b[{warn_attr}m", end="", flush=True)
            yield
        finally:
            self.level = level
            print("\x1b[m", flush=True)

    def emph(self, msg):
        return wcb.add_attr(msg, self.config.current.menu.emph_attr)

    def print(self, msg="", prefix=None, end="\n", flush=False):
        if prefix is None:
            print(msg, end=end, flush=flush)
        elif prefix == "data":
            print(self.config.current.menu.data_icon + " " + msg, end=end, flush=flush)
        elif prefix == "info":
            print(self.config.current.menu.info_icon + " " + msg, end=end, flush=flush)
        elif prefix == "hint":
            print(self.config.current.menu.hint_icon + " " + msg, end=end, flush=flush)


@dataclasses.dataclass
class KAIKOUser:
    username: str
    config_file: Path
    data_dir: Path
    songs_dir: Path


class KAIKOMenu:
    def __init__(self, config, user, manager, logger):
        r"""Constructor.

        Parameters
        ----------
        config : KAIKOSettings
        user : KAIKOUser
        manager : PyAudio
        logger : KAIKOLogger
        """
        self._config = config
        self.user = user
        self.manager = manager
        self.logger = logger
        self.beatmap_manager = BeatmapManager(user, logger)
        self.bgm_controller = KAIKOBGMController(config, logger, self.beatmap_manager)

    @staticmethod
    def main():
        dt = 0.01
        with KAIKOMenu.init() as menu:
            dn.exhaust(menu.run(), dt=dt, interruptible=True)

    @classmethod
    @contextlib.contextmanager
    def init(clz):
        r"""Initialize KAIKOMenu within a context manager."""
        username = getpass.getuser()
        data_dir = Path(appdirs.user_data_dir("K-AIKO", username))

        # load settings
        config_file = data_dir / "config.py"
        config = cfg.Configuration(KAIKOSettings)
        if config_file.exists():
            config.read(config_file)
        settings = config.current

        logger = KAIKOLogger(config)

        # print logo
        logger.print(settings.menu.logo, flush=True)

        # load user data
        songs_dir = data_dir / "songs"

        if not data_dir.exists():
            # start up
            logger.print(f"Prepare your profile...", prefix="data")
            data_dir.mkdir(exist_ok=True)
            songs_dir.mkdir(exist_ok=True)
            if not config_file.exists():
                config.write(config_file)

            (data_dir / "samples/").mkdir(exist_ok=True)
            resources = ["samples/soft.wav",
                         "samples/loud.wav",
                         "samples/incr.wav",
                         "samples/rock.wav",
                         "samples/disk.wav"]
            for rspath in resources:
                logger.print(f"Load resource {rspath}...", prefix="data")
                data = pkgutil.get_data("kaiko", rspath)
                open(data_dir / rspath, 'wb').write(data)

            logger.print(f"Your data will be stored in {logger.emph(data_dir.as_uri())}", prefix="data")
            logger.print(flush=True)

        user = KAIKOUser(username, config_file, data_dir, songs_dir)

        # load PyAudio
        logger.print(f"Load PyAudio...", prefix="info")
        logger.print()

        with logger.verb():
            manager = pyaudio.PyAudio()
            print_pyaudio_info(manager, logger.print)

        try:
            yield clz(config, user, manager, logger)

        except KeyboardInterrupt:
            pass

        except:
            # print error
            print("\x1b[31m", end="")
            print(traceback.format_exc(), end="")
            print(f"\x1b[m", end="")

        finally:
            manager.terminate()

    @dn.datanode
    def run(self):
        r"""Run KAIKOMenu."""
        logger = self.logger

        yield

        if not sys.stdout.isatty():
            raise ValueError("please connect to interactive terminal device.")

        # fit screen size
        screen_size = self.settings.menu.best_screen_size
        fit_delay = 1.0
        with fit_screen(logger, screen_size, fit_delay) as fit_task:
            yield from fit_task.join((yield))

        # load songs
        self.reload()

        # execute given command
        if len(sys.argv) > 1:
            command = cmd.RootCommandParser(self).build(sys.argv[1:])
            with self.execute(command) as command_task:
                yield from command_task.join((yield))
            return

        # load bgm
        bgm_task = self.bgm_controller.load_bgm(self.manager)

        # tips
        confirm_key = self.settings.shell.input.confirm_key
        help_key = self.settings.shell.input.help_key
        tab_key, _, _ = self.settings.shell.input.autocomplete_keys
        logger.print(f"Type command and press {logger.emph(confirm_key)} to execute.", prefix="hint")
        logger.print(f"Use {logger.emph(tab_key)} to autocomplete command.", prefix="hint")
        logger.print(f"If you need help, press {logger.emph(help_key)}.", prefix="hint")
        logger.print()

        # prompt
        repl_task = self.repl()
        with dn.pipe(repl_task, bgm_task) as task:
            yield from task.join((yield))

    @dn.datanode
    def repl(self):
        r"""Start REPL."""
        input = beatshell.BeatInput(self)
        while True:
            # parse command
            with input.prompt(self.settings.devices, self.settings.shell) as prompt_task:
                yield from prompt_task.join((yield))

            # execute result
            if isinstance(input.result, beatshell.InputError):
                with self.logger.warn():
                    self.logger.print(input.result.value)
                input.prev_session()
            else:
                with self.execute(input.result.value) as command_task:
                    yield from command_task.join((yield))
                input.new_session()

    @dn.datanode
    def execute(self, command):
        r"""Execute a command.
        If it returns executable object (an object has method `execute`), call
        `result.execute(manager)`; if it returns a DataNode, exhaust it; otherwise,
        print repr of result.

        Parameters
        ----------
        command : function
            The command.
        """
        result = command()

        if hasattr(result, 'execute'):
            has_bgm = bool(self.bgm_controller._current_bgm)
            if has_bgm:
                self.bgm.off()
            with result.execute(self.manager) as command_task:
                yield from command_task.join((yield))
            if has_bgm:
                self.bgm.on()

        elif isinstance(result, dn.DataNode):
            with result:
                yield from result.join((yield))

        elif result is not None:
            yield
            self.logger.print(repr(result))

    @property
    def settings(self):
        r"""Current settings."""
        return self._config.current

    # beatmaps

    @cmd.function_command
    def play(self, beatmap):
        """Let's beat with the song!
        
        usage: \x1b[94mplay\x1b[m \x1b[92m{beatmap}\x1b[m
                         ╲
               Path, the path to the
              beatmap you want to play.
              Only the beatmaps in your
             songs folder can be accessed.
        """

        if not self.beatmap_manager.is_beatmap(beatmap):
            with self.logger.warn():
                self.logger.print("Not a beatmap.")
                return

        return KAIKOPlay(self.user.data_dir, self.user.songs_dir / beatmap,
                         self.settings.devices, self.settings.gameplay, self.logger)

    @play.arg_parser("beatmap")
    def _play_beatmap_parser(self):
        return self.beatmap_manager.make_parser(self.bgm_controller)

    @cmd.function_command
    def reload(self):
        """Reload your songs."""
        self.beatmap_manager.reload()

    @cmd.function_command
    def add(self, beatmap):
        """Add beatmap/beatmapset to your songs folder.
        
        usage: \x1b[94madd\x1b[m \x1b[92m{beatmap}\x1b[m
                        ╲
              Path, the path to the
             beatmap you want to add.
             You can drop the file to
           the terminal to paste its path.
        """

        self.beatmap_manager.add(beatmap)

    @add.arg_parser("beatmap")
    def _add_beatmap_parser(self):
        return cmd.PathParser()

    @cmd.function_command
    def remove(self, beatmap):
        """Remove beatmap/beatmapset in your songs folder.
        
        usage: \x1b[94mremove\x1b[m \x1b[92m{beatmap}\x1b[m
                           ╲
                 Path, the path to the
               beatmap you want to remove.
        """

        self.beatmap_manager.remove(beatmap)

    @remove.arg_parser("beatmap")
    def _remove_beatmap_parser(self):
        options = []
        for song, beatmapset in self.beatmap_manager._beatmaps.items():
            options.append(os.path.join(str(song), ""))
            for beatmap in beatmapset:
                options.append(str(beatmap))

        return cmd.OptionParser(options)

    @cmd.function_command
    def beatmaps(self):
        """Your beatmaps."""
        if not self.beatmap_manager.is_uptodate():
            self.reload()

        for beatmapset in self.beatmap_manager._beatmaps.values():
            for beatmap in beatmapset:
                self.logger.print("• " + str(beatmap))

    @cmd.subcommand
    @property
    def bgm(self):
        """Background music."""
        return BGMCommand(self.bgm_controller, self.beatmap_manager, self.logger)

    # devices

    @cmd.subcommand
    @property
    def devices(self):
        """Devices."""
        return DevicesCommand(self._config, self.logger, self.manager)

    # config

    @cmd.subcommand
    @property
    def config(self):
        """Configuration."""
        return ConfigCommand(self._config, self.logger, self.user.config_file)

    # system

    @cmd.function_command
    def me(self):
        """About user."""
        logger = self.logger

        logger.print(f"username: {logger.emph(self.user.username)}")
        logger.print(f"config file: {logger.emph(self.user.config_file.as_uri())}")
        logger.print(f"data directory: {logger.emph(self.user.data_dir.as_uri())}")
        logger.print(f"songs directory: {logger.emph(self.user.songs_dir.as_uri())}")

    @cmd.function_command
    def say(self, message, escape=False):
        """Say something to... yourself.
        
        usage: \x1b[94msay\x1b[m \x1b[92m{message}\x1b[m [\x1b[95m--escape\x1b[m \x1b[92m{ESCAPE}\x1b[m]
                      ╱                    ╲
            text, the message               ╲
             to be printed.          bool, use backslash escapes
                                    or not; the default is False.
        """

        if escape:
            self.logger.print(echo_str(message))
        else:
            self.logger.print(message)

    @cmd.function_command
    def clean(self):
        """Clean screen."""
        self.logger.print("\x1b[2J\x1b[H")

    @say.arg_parser("message")
    def _say_message_parser(self):
        return cmd.RawParser(desc="It should be some text,"
                                  " indicating the message to be printed.")

    @say.arg_parser("escape")
    def _say_escape_parser(self, message):
        return cmd.LiteralParser(bool, default=False,
                                       desc="It should be bool,"
                                            " indicating whether to use backslash escapes;"
                                            " the default is False.")

    @cmd.function_command
    def exit(self):
        """Close K-AIKO."""
        self.logger.print("bye~")
        raise KeyboardInterrupt


class DevicesCommand:
    def __init__(self, config, logger, manager):
        self.config = config
        self.logger = logger
        self.manager = manager

    # audio

    @cmd.function_command
    def audio_input(self, device, samplerate=None, channels=None, format=None):
        """Configure audio input.
        
        usage: devices \x1b[94maudio_input\x1b[m \x1b[92m{device}\x1b[m \
[\x1b[95m--samplerate\x1b[m \x1b[92m{RATE}\x1b[m] \
[\x1b[95m--channel\x1b[m \x1b[92m{CHANNEL}\x1b[m] \
[\x1b[95m--format\x1b[m \x1b[92m{FORMAT}\x1b[m]
                                     ╱                     ╱                      ╲                    ╲
                           the index of input        the sample rate       the channel of audio    the data format
                            device, -1 is the       of recorded sound.      input: 1 for mono,    of recorded sound.
                             default device.                                 2 for stereo.
        """
        logger = self.logger

        pa_device = device
        pa_samplerate = samplerate
        pa_channels = channels

        if pa_device == -1:
            pa_device = self.manager.get_default_input_device_info()['index']
        if pa_samplerate is None:
            pa_samplerate = self.config.get('devices.detector.input_samplerate')
        if pa_channels is None:
            pa_channels = self.config.get('devices.detector.input_channels')

        pa_format = {
            'f4': pyaudio.paFloat32,
            'i4': pyaudio.paInt32,
            'i2': pyaudio.paInt16,
            'i1': pyaudio.paInt8,
            'u1': pyaudio.paUInt8,
        }[format or self.config.get('devices.detector.input_format')]

        try:
            self.manager.is_format_supported(pa_samplerate,
                input_device=pa_device, input_channels=pa_channels, input_format=pa_format)

        except ValueError as e:
            info = e.args[0]
            with logger.warn():
                logger.print(info)

        else:
            self.config.set('devices.detector.input_device', device)
            if samplerate is not None:
                self.config.set('devices.detector.input_samplerate', samplerate)
            if channels is not None:
                self.config.set('devices.detector.input_channels', channels)
            if format is not None:
                self.config.set('devices.detector.input_format', format)

    @cmd.function_command
    def audio_output(self, device, samplerate=None, channels=None, format=None):
        """Configure audio output.
        
        usage: devices \x1b[94maudio_output\x1b[m \x1b[92m{device}\x1b[m \
[\x1b[95m--samplerate\x1b[m \x1b[92m{RATE}\x1b[m] \
[\x1b[95m--channel\x1b[m \x1b[92m{CHANNEL}\x1b[m] \
[\x1b[95m--format\x1b[m \x1b[92m{FORMAT}\x1b[m]
                                      ╱                     ╱                      ╲                    ╲
                            the index of output       the sample rate       the channel of audio    the data format
                             device, -1 is the        of played sound.       output: 1 for mono,    of played sound.
                              default device.                                 2 for stereo.
        """
        logger = self.logger

        pa_device = device
        pa_samplerate = samplerate
        pa_channels = channels

        if pa_device == -1:
            pa_device = self.manager.get_default_output_device_info()['index']
        if pa_samplerate is None:
            pa_samplerate = self.config.get('devices.mixer.output_samplerate')
        if pa_channels is None:
            pa_channels = self.config.get('devices.mixer.output_channels')

        pa_format = {
            'f4': pyaudio.paFloat32,
            'i4': pyaudio.paInt32,
            'i2': pyaudio.paInt16,
            'i1': pyaudio.paInt8,
            'u1': pyaudio.paUInt8,
        }[format or self.config.get('devices.mixer.output_format')]

        try:
            self.manager.is_format_supported(pa_samplerate,
                output_device=pa_device, output_channels=pa_channels, output_format=pa_format)

        except ValueError as e:
            info = e.args[0]
            with logger.warn():
                logger.print(info)

        else:
            self.config.set('devices.mixer.output_device', device)
            if samplerate is not None:
                self.config.set('devices.mixer.output_samplerate', samplerate)
            if channels is not None:
                self.config.set('devices.mixer.output_channels', channels)
            if format is not None:
                self.config.set('devices.mixer.output_format', format)

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
        return cmd.OptionParser({str(rate): rate for rate in options})

    @audio_input.arg_parser("channels")
    @audio_output.arg_parser("channels")
    def _audio_channels_parser(self, device, **__):
        return cmd.OptionParser({'2': 2, '1': 1})

    @audio_input.arg_parser("format")
    @audio_output.arg_parser("format")
    def _audio_format_parser(self, device, **__):
        return cmd.OptionParser(['f4', 'i4', 'i2', 'i1', 'u1'])

    # terminal

    @cmd.function_command
    def terminal(self):
        """Show your terminal configuration."""

        term = os.environ.get('TERM', None)
        vte = os.environ.get('VTE_VERSION', None)
        uni = os.environ.get('UNICODE_VERSION', None)
        size = shutil.get_terminal_size()

        self.logger.print(f"terminal type: {term}")
        self.logger.print(f"VTE version: {vte}")
        self.logger.print(f"unicode version: {uni}")
        self.logger.print(f"terminal size: {size.columns}×{size.lines}")


class ConfigCommand:
    def __init__(self, config, logger, path):
        self.config = config
        self.logger = logger
        self.path = path

    # configuration

    @cmd.function_command
    def reload(self):
        """Reload configuration."""
        logger = self.logger

        logger.print(f"Load configuration from {logger.emph(self.path.as_uri())}...", prefix="data")
        logger.print()

        self.config.read(self.path)

    @cmd.function_command
    def show(self):
        """Show configuration."""
        self.logger.print(str(self.config))

    @cmd.function_command
    def save(self):
        """Save configuration."""
        logger = self.logger

        logger.print(f"Save configuration to {logger.emph(self.path.as_uri())}...", prefix="data")
        logger.print()

        self.config.write(self.path)

    @cmd.function_command
    def has(self, field):
        """Check whether this field is set in the configuration.
        
        usage: \x1b[94mconfig\x1b[m \x1b[94mhas\x1b[m \x1b[92m{field}\x1b[m
                            ╱
                     The field name.
        """
        return self.config.has(field)

    @cmd.function_command
    def get(self, field):
        """Get the value of this field in the configuration.
        
        usage: \x1b[94mconfig\x1b[m \x1b[94mget\x1b[m \x1b[92m{field}\x1b[m
                            ╱
                     The field name.
        """
        return self.config.get(field)

    @cmd.function_command
    def set(self, field, value):
        """Set this field in the configuration.
        
        usage: \x1b[94mconfig\x1b[m \x1b[94mset\x1b[m \x1b[92m{field}\x1b[m \x1b[92m{value}\x1b[m
                            ╱         ╲
                   The field name.   The value.
        """
        self.config.set(field, value)

    @cmd.function_command
    def unset(self, field):
        """Unset this field in the configuration.
        
        usage: \x1b[94mconfig\x1b[m \x1b[94munset\x1b[m \x1b[92m{field}\x1b[m
                              ╱
                       The field name.
        """
        self.config.unset(field)

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
        return cmd.LiteralParser(annotation, default)

    # profiles

    @cmd.function_command
    def use(self, profile):
        """Change the current configuration profile.
        
        usage: \x1b[94mconfig\x1b[m \x1b[94muse\x1b[m \x1b[92m{profile}\x1b[m
                              ╱
                     The profile name.
        """
        if profile == self.config.name:
            return
        self.config.use(profile)

    @cmd.function_command
    def rename(self, profile):
        """Rename current configuration profile.
        
        usage: \x1b[94mconfig\x1b[m \x1b[94mrename\x1b[m \x1b[92m{profile}\x1b[m
                                ╱
                      The profile name.
        """
        if not profile.isprintable():
            with self.logger.warn():
                self.logger.print("Invalid profile name.")
            return

        if profile in self.config.profiles:
            with self.logger.warn():
                self.logger.print("This profile name already exists.")
            return

        self.config.name = profile

    @cmd.function_command
    def new(self, profile, clone=None):
        """Make new configuration profile.
        
        usage: \x1b[94mconfig\x1b[m \x1b[94mnew\x1b[m \x1b[92m{profile}\x1b[m [\x1b[95m--clone\x1b[m \x1b[92m{PROFILE}\x1b[m]
                              ╱                    ╲
                     The profile name.      The profile to be cloned.
        """
        if not profile.isprintable():
            with self.logger.warn():
                self.logger.print("Invalid profile name.")
            return

        if profile == self.config.name or profile in self.config.profiles:
            with self.logger.warn():
                self.logger.print("This profile name already exists.")
            return

        self.config.new(profile, clone)

    @cmd.function_command
    def delete(self, profile):
        """Delete a configuration profile.
        
        usage: \x1b[94mconfig\x1b[m \x1b[94mdelete\x1b[m \x1b[92m{profile}\x1b[m
                                ╱
                       he profile name.
        """
        if profile == self.config.name:
            with self.logger.warn():
                self.logger.print("Cannot delete current profile.")
            return
        del self.config.profiles[profile]

    @rename.arg_parser("profile")
    @new.arg_parser("profile")
    def _new_profile_parser(self):
        return cmd.RawParser()

    @new.arg_parser("clone")
    @use.arg_parser("profile")
    @delete.arg_parser("profile")
    def _old_profile_parser(self, *_, **__):
        options = list(self.config.profiles.keys())
        options.insert(0, self.config.name)
        return cmd.OptionParser(options,
                                desc="It should be the name of the profile that exists in the configuration.")


class FieldParser(cmd.ArgumentParser):
    def __init__(self, config_type):
        self.config_type = config_type
        self.biparser = cfg.FieldBiparser(config_type)

    def parse(self, token):
        try:
            return self.biparser.decode(token)[0]
        except bp.DecodeError:
            raise cmd.CommandParseError("No such field")

    def suggest(self, token):
        try:
            self.biparser.decode(token)
        except bp.DecodeError as e:
            sugg = cmd.fit(token, [token[:e.index] + ex for ex in e.expected])
        else:
            sugg = []

        return sugg

class PyAudioDeviceParser(cmd.ArgumentParser):
    def __init__(self, manager, is_input):
        self.manager = manager
        self.is_input = is_input
        self.options = ["-1"]
        for index in range(manager.get_device_count()):
            self.options.append(str(index))

    def parse(self, token):
        if token not in self.options:
            raise cmd.CommandParseError("Invalid device index")
        return int(token)

    def suggest(self, token):
        return [val + "\000" for val in cmd.fit(token, self.options)]

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


@dataclasses.dataclass
class SongMetadata:
    root: str
    audio: str
    volume: float
    info: str
    preview: float

    @classmethod
    def from_beatmap(clz, beatmap):
        if beatmap.audio is None:
            return None
        return clz(root=beatmap.root, audio=beatmap.audio,
                   volume=beatmap.volume, info=beatmap.info, preview=beatmap.preview)

    @property
    def path(self):
        return os.path.join(self.root, self.audio)

class BeatmapManager:
    def __init__(self, user, logger):
        self.user = user
        self.logger = logger
        self._beatmaps = {}
        self._beatmaps_mtime = None

    def is_uptodate(self):
        return self._beatmaps_mtime == os.stat(str(self.user.songs_dir)).st_mtime

    def reload(self):
        logger = self.logger
        songs_dir = self.user.songs_dir

        logger.print(f"Load songs from {logger.emph(songs_dir.as_uri())}...", prefix="data")

        for file in songs_dir.iterdir():
            if file.is_file() and file.suffix == ".osz":
                distpath = file.parent / file.stem
                if distpath.exists():
                    continue
                logger.print(f"Unzip file {logger.emph(file.as_uri())}...", prefix="data")
                distpath.mkdir()
                zf = zipfile.ZipFile(str(file), 'r')
                zf.extractall(path=str(distpath))
                file.unlink()

        logger.print("Load beatmaps...", prefix="data")

        self._beatmaps_mtime = os.stat(str(songs_dir)).st_mtime
        self._beatmaps = {}

        for song in songs_dir.iterdir():
            if song.is_dir():
                beatmapset = []
                for beatmap in song.iterdir():
                    if beatmap.suffix in (".kaiko", ".ka", ".osu"):
                        beatmapset.append(beatmap.relative_to(songs_dir))
                if beatmapset:
                    self._beatmaps[song.relative_to(songs_dir)] = beatmapset

        if len(self._beatmaps) == 0:
            logger.print("There is no song in the folder yet!", prefix="data")
        logger.print(flush=True)

    def add(self, beatmap):
        logger = self.logger
        songs_dir = self.user.songs_dir

        if not beatmap.exists():
            with logger.warn():
                logger.print(f"File not found: {str(beatmap)}")
            return
        if not beatmap.is_file() and not beatmap.is_dir():
            with logger.warn():
                logger.print(f"Not a file or directory: {str(beatmap)}")
            return

        logger.print(f"Add a new song from {logger.emph(beatmap.as_uri())}...", prefix="data")

        distpath = songs_dir / beatmap.name
        n = 1
        while distpath.exists():
            n += 1
            distpath = songs_dir / f"{beatmap.stem} ({n}){beatmap.suffix}"
        if n != 1:
            logger.print(f"Name conflict! Rename to {logger.emph(distpath.name)}", prefix="data")

        if beatmap.is_file():
            shutil.copy(str(beatmap), str(songs_dir))
        elif beatmap.is_dir():
            shutil.copytree(str(beatmap), str(distpath))

        self.reload()

    def remove(self, beatmap):
        logger = self.logger
        songs_dir = self.user.songs_dir

        beatmap_path = songs_dir / beatmap
        if beatmap_path.is_file():
            logger.print(f"Remove the beatmap at {logger.emph(beatmap_path.as_uri())}...", prefix="data")
            beatmap_path.unlink()
            self.reload()

        elif beatmap_path.is_dir():
            logger.print(f"Remove the beatmapset at {logger.emph(beatmap_path.as_uri())}...", prefix="data")
            shutil.rmtree(str(beatmap_path))
            self.reload()

        else:
            with logger.warn():
                logger.print(f"Not a file: {str(beatmap)}")

    def is_beatmapset(self, path):
        return path in self._beatmaps

    def is_beatmap(self, path):
        return path.parent in self._beatmaps and path in self._beatmaps[path.parent]

    def get_beatmap_metadata(self, path):
        if not self.is_beatmap(path):
            raise ValueError(f"Not a beatmap: {str(path)}")

        filepath = self.user.songs_dir / path
        try:
            beatmap = beatsheets.BeatSheet.read(str(filepath), metadata_only=True)
        except beatsheets.BeatmapParseError:
            return None
        else:
            return beatmap

    def get_song(self, path):
        if self.is_beatmapset(path):
            path = self._beatmaps[path][0]
        beatmap = self.get_beatmap_metadata(path)
        return beatmap and SongMetadata.from_beatmap(beatmap)

    def get_songs(self):
        songs = [self.get_song(path) for path in self._beatmaps.keys()]
        return [song for song in songs if song is not None]

    def make_parser(self, bgm_controller=None):
        return BeatmapParser(self, bgm_controller)

class BeatmapParser(cmd.TreeParser):
    def __init__(self, beatmap_manager, bgm_controller):
        super().__init__(BeatmapParser.make_tree(beatmap_manager._beatmaps))
        self.beatmap_manager = beatmap_manager
        self.bgm_controller = bgm_controller

    @staticmethod
    def make_tree(beatmapsets):
        tree = {}
        for beatmapset_path, beatmapset in beatmapsets.items():
            subtree = {}
            subtree[""] = Path
            for beatmap_path in beatmapset:
                subtree[str(beatmap_path.relative_to(beatmapset_path))] = Path
            tree[os.path.join(str(beatmapset_path), "")] = subtree
        return tree

    def info(self, token):
        path = Path(token)

        song = self.beatmap_manager.get_song(path)
        if self.bgm_controller is not None and song is not None:
            self.bgm_controller.play(song, song.preview)

        if self.beatmap_manager.is_beatmap(path):
            beatmap = self.beatmap_manager.get_beatmap_metadata(path)
            return beatmap.info.strip() if beatmap is not None else None

class KAIKOBGMController:
    def __init__(self, config, logger, beatmap_manager):
        self.config = config
        self.logger = logger
        self._current_bgm = None
        self._action_queue = queue.Queue()
        self.beatmap_manager = beatmap_manager

    def load_bgm(self, manager):
        try:
            mixer_task, mixer = engines.Mixer.create(self.config.current.devices.mixer, manager)

        except Exception:
            with self.logger.warn():
                self.logger.print("Failed to load mixer")
                self.logger.print(traceback.format_exc(), end="")

            return dn.DataNode.wrap(lambda _:None)

        else:
            return dn.pipe(mixer_task, self._bgm_rountine(mixer))

    @dn.datanode
    def _bgm_rountine(self, mixer):
        self._current_bgm = None

        yield
        while True:
            if self._action_queue.empty():
                yield
                continue

            action = self._action_queue.get()
            while action:
                song, start = action
                self._current_bgm = song

                node = yield from dn.create_task(lambda event: mixer.load_sound(self._current_bgm.path, event))
                node = dn.DataNode.wrap(node)

                with mixer.play(node, start=start, volume=song.volume) as bgm_key:
                    while not bgm_key.is_finalized():
                        if self._action_queue.empty():
                            yield
                            continue

                        action = self._action_queue.get()
                        break

                    else:
                        next_song = self.random_song()
                        action = next_song, None

                self._current_bgm = None

    def random_song(self):
        songs = self.beatmap_manager.get_songs()
        if self._current_bgm is not None:
            songs.remove(self._current_bgm)
        return random.choice(songs) if songs else None

    def stop(self):
        self._action_queue.put(None)

    def play(self, song, start=None):
        self._action_queue.put((song, start))

class BGMCommand:
    def __init__(self, bgm_controller, beatmap_manager, logger):
        self.bgm_controller = bgm_controller
        self.beatmap_manager = beatmap_manager
        self.logger = logger

    @cmd.function_command
    def on(self):
        logger = self.logger

        if self.bgm_controller._current_bgm is not None:
            logger.print("now playing: " + self.bgm_controller._current_bgm.path)
            return

        song = self.bgm_controller.random_song()
        if song is None:
            logger.print("There is no song in the folder yet!", prefix="data")
            return

        logger.print("will play: " + song.path)
        self.bgm_controller.play(song)

    @cmd.function_command
    def off(self):
        self.bgm_controller.stop()

    @cmd.function_command
    def skip(self):
        if self.bgm_controller._current_bgm is not None:
            song = self.bgm_controller.random_song()
            self.logger.print("will play: " + song.path)
            self.bgm_controller.play(song)

    @cmd.function_command
    def play(self, beatmap, start:Optional[float]=None):
        logger = self.logger

        try:
            song = self.beatmap_manager.get_song(beatmap)
        except beatsheets.BeatmapParseError:
            with logger.warn():
                logger.print("Fail to read beatmap")
            return
        if song is None:
            with logger.warn():
                logger.print("This beatmap has no song")
            return

        self.bgm_controller.play(song, start)

    @play.arg_parser("beatmap")
    def _play_beatmap_parser(self):
        return self.beatmap_manager.make_parser()

    @cmd.function_command
    def now_playing(self):
        self.logger.print(self.bgm_controller._current_bgm.path)


class KAIKOPlay:
    def __init__(self, data_dir, filepath, devices_settings, settings, logger):
        self.data_dir = data_dir
        self.filepath = filepath
        self.devices_settings = devices_settings
        self.settings = settings
        self.logger = logger

    @dn.datanode
    def execute(self, manager):
        logger = self.logger

        try:
            beatmap = beatsheets.BeatSheet.read(str(self.filepath))

        except beatsheets.BeatmapParseError:
            with logger.warn():
                logger.print(f"Failed to read beatmap {str(self.filepath)}")
                logger.print(traceback.format_exc(), end="")

        else:
            game = beatmaps.BeatmapPlayer(self.data_dir, beatmap, self.devices_settings, self.settings)

            with game.execute(manager) as task:
                yield from task.join((yield))

            logger.print()
            beatanalyzer.show_analyze(beatmap.settings.difficulty.performance_tolerance, game.perfs)

