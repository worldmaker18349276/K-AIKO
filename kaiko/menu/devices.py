import os
import re
import time
import shutil
import pyaudio
import contextlib
import threading
import queue
from kaiko.utils import wcbuffers as wcb
from kaiko.utils import datanodes as dn
from kaiko.utils import config as cfg
from kaiko.utils import commands as cmd
from kaiko.utils import engines


def print_pyaudio_info(manager, logger):
    logger.print("portaudio version:")
    logger.print("  " + pyaudio.get_portaudio_version_text())
    logger.print()

    logger.print("available devices:")
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
        logger.print(f"  {ind:>{ind_len}}. {name:{name_len}}  by  {api:{api_len}}"
              f"  ({freq:>{freq_len}} kHz, in: {chin:>{chin_len}}, out: {chout:>{chout_len}})")

    logger.print()

    default_input_device_index = manager.get_default_input_device_info()['index']
    default_output_device_index = manager.get_default_output_device_info()['index']
    logger.print(f"default input device: {default_input_device_index}")
    logger.print(f"default output device: {default_output_device_index}")

@contextlib.contextmanager
def prepare_pyaudio(logger):
    r"""Prepare PyAudio and print out some information.

    Loading PyAudio will cause PortAudio to print something to the terminal, which
    cannot be turned off by PyAudio, so why not print more information to make it
    more hacky.

    Parameters
    ----------
    logger : KAIKOLogger

    Yields
    ------
    manager : PyAudio
    """
    with logger.verb():
        manager = pyaudio.PyAudio()

        logger.print()

        print_pyaudio_info(manager, logger)

    logger.print()

    try:
        yield manager
    finally:
        manager.terminate()

def fit_screen(logger):
    r"""Guide user to adjust screen size.

    Parameters
    ----------
    logger : KAIKOLogger

    Returns
    -------
    fit_task : dn.DataNode
        The datanode to manage this process.
    """
    width = logger.settings.best_screen_size
    delay = logger.settings.adjust_screen_delay

    skip_event = threading.Event()

    skip = dn.input(lambda a: skip_event.set() if a[1] == '\x1b' else None)

    @dn.datanode
    def fit():
        size = yield
        current_width = 0

        t = time.time()

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

class KAIKOMenuSettings(cfg.Configurable):
    data_icon: str = "\x1b[92m🗀 \x1b[m"
    info_icon: str = "\x1b[94m🛠 \x1b[m"
    hint_icon: str = "\x1b[93m💡 \x1b[m"

    verb_attr: str = "2"
    emph_attr: str = "1"
    warn_attr: str = "31"

    best_screen_size: int = 80
    adjust_screen_delay: float = 1.0

    editor: str = "nano"

class KAIKOLogger:
    def __init__(self, config=None):
        self.config = config
        self.level = 1

    @property
    def settings(self):
        return self.config.current.menu if self.config else KAIKOMenuSettings()

    def set_config(self, config):
        self.config = config

    @contextlib.contextmanager
    def verb(self):
        verb_attr = self.settings.verb_attr
        level = self.level
        self.level = 0
        try:
            print(f"\x1b[{verb_attr}m", end="", flush=True)
            yield
        finally:
            self.level = level
            print("\x1b[m", end="", flush=True)

    @contextlib.contextmanager
    def warn(self):
        warn_attr = self.settings.warn_attr
        level = self.level
        self.level = 2
        try:
            print(f"\x1b[{warn_attr}m", end="", flush=True)
            yield
        finally:
            self.level = level
            print("\x1b[m", end="", flush=True)

    def emph(self, msg):
        return wcb.add_attr(msg, self.settings.emph_attr)

    def print(self, msg="", prefix=None, end="\n", flush=False):
        if prefix is None:
            print(msg, end=end, flush=flush)
        elif prefix == "data":
            print(self.settings.data_icon + " " + msg, end=end, flush=flush)
        elif prefix == "info":
            print(self.settings.info_icon + " " + msg, end=end, flush=flush)
        elif prefix == "hint":
            print(self.settings.hint_icon + " " + msg, end=end, flush=flush)

@dn.datanode
def determine_unicode_version(logger):
    pattern = re.compile(r"\x1b\[(\d*);(\d*)R")
    channel = queue.Queue()

    def get_pos(arg):
        m = pattern.match(arg[1])
        if not m:
            return
        x = int(m.group(2) or "1") - 1
        channel.put(x)

    @dn.datanode
    def query_pos():
        previous_version = '4.1.0'
        wide_by_version = [
            ('5.1.0', '龼'),
            ('5.2.0', '🈯'),
            ('6.0.0', '🈁'),
            ('8.0.0', '🉐'),
            ('9.0.0', '🐹'),
            ('10.0.0', '🦖'),
            ('11.0.0', '🧪'),
            ('12.0.0', '🪐'),
            ('12.1.0', '㋿'),
            ('13.0.0', '🫕'),
        ]

        yield

        for version, wchar in wide_by_version:
            print(wchar, end="", flush=True)
            print("\x1b[6n", end="", flush=True)

            while True:
                yield
                try:
                    x = channel.get(False)
                except queue.Empty:
                    continue
                else:
                    break

            print(f"\twidth={x}", end="\n", flush=True)
            if x == 1:
                break
            elif x == 2:
                previous_version = version
                continue
            else:
                return

        return previous_version

    logger.print("Determine unicode version...", prefix="info")

    with logger.verb():
        query_task = query_pos()
        with dn.pipe(dn.input(get_pos), query_task) as task:
            yield from task.join((yield))
        version = query_task.result

    if version is None:
        with logger.warn():
            logger.print("Fail to determine unicode version")

    else:
        logger.print(f"Your unicode version is {logger.emph(version)}")
        logger.print("You can put this command into your settings file:", prefix="hint")
        logger.print(logger.emph(f"UNICODE_VERSION={version}; export UNICODE_VERSION"))

    return version

class DevicesCommand:
    def __init__(self, config, logger, manager):
        self.config = config
        self.logger = logger
        self.manager = manager

    # audio

    @cmd.function_command
    def audio(self):
        """Show your audio configuration."""

        logger = self.logger
        print_pyaudio_info(self.manager, logger)

        logger.print()

        device = self.config.current.devices.detector.input_device
        if device == -1:
            device = "default"
        samplerate = self.config.current.devices.detector.input_samplerate
        channels = self.config.current.devices.detector.input_channels
        format = self.config.current.devices.detector.input_format
        logger.print(f"current input device: {device} ({samplerate/1000} kHz, {channels} ch)")

        device = self.config.current.devices.mixer.output_device
        if device == -1:
            device = "default"
        samplerate = self.config.current.devices.mixer.output_samplerate
        channels = self.config.current.devices.mixer.output_channels
        format = self.config.current.devices.mixer.output_format
        logger.print(f"current output device: {device} ({samplerate/1000} kHz, {channels} ch)")

    @cmd.function_command
    def set_audio_input(self, device, samplerate=None, channels=None, format=None):
        """Configure audio input.

        usage: devices \x1b[94mset_audio_input\x1b[m \x1b[92m{device}\x1b[m \
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
    def set_audio_output(self, device, samplerate=None, channels=None, format=None):
        """Configure audio output.

        usage: devices \x1b[94mset_audio_output\x1b[m \x1b[92m{device}\x1b[m \
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

    @set_audio_input.arg_parser("device")
    def _set_audio_input_device_parser(self):
        return PyAudioDeviceParser(self.manager, True)

    @set_audio_output.arg_parser("device")
    def _set_audio_output_device_parser(self):
        return PyAudioDeviceParser(self.manager, False)

    @set_audio_input.arg_parser("samplerate")
    @set_audio_output.arg_parser("samplerate")
    def _audio_samplerate_parser(self, device, **__):
        options = [44100, 48000, 88200, 96000, 32000, 22050, 11025, 8000]
        return cmd.OptionParser({str(rate): rate for rate in options})

    @set_audio_input.arg_parser("channels")
    @set_audio_output.arg_parser("channels")
    def _audio_channels_parser(self, device, **__):
        return cmd.OptionParser({'2': 2, '1': 1})

    @set_audio_input.arg_parser("format")
    @set_audio_output.arg_parser("format")
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

    @cmd.function_command
    def fit_screen(self):
        """Fit your terminal screen."""

        return fit_screen(self.logger)

    @cmd.function_command
    def ucs_detect(self):
        """Determines the unicode version of your terminal."""

        return determine_unicode_version(self.logger)

    # engines

    @cmd.function_command
    def keylog(self):
        """Test your keyboard."""

        logger = self.logger

        logger.print(f"Press {logger.emph('Esc')} to end keylog.")
        logger.print()
        logger.print(f"[ <time>  ] {logger.emph('<keyname>')} (<keycode>)")

        stop_event = threading.Event()

        def handler(arg):
            _, time, keyname, keycode = arg
            logger.print(f"[{time:07.3f} s] {logger.emph(keyname)} ({repr(keycode)})")
            if keycode == '\x1b':
                stop_event.set()

        controller_task, controller = engines.Controller.create(self.config.current.devices.controller)
        controller.add_handler(handler)

        return dn.pipe(controller_task, dn.take(lambda _: not stop_event.is_set()))

    @cmd.function_command
    def knock(self):
        """Test knock detection."""
        settings = self.config.current.devices.detector
        ref_time = 0.0
        return KnockTest(ref_time, settings, self.logger)

class KnockTest:
    def __init__(self, ref_time, settings, logger):
        self.ref_time = ref_time
        self.settings = settings
        self.logger = logger
        self.hit_queue = queue.Queue()

    def execute(self, manager):
        self.logger.print("Press any key to stop detecting")

        detector_task, detector = engines.Detector.create(self.settings, manager, self.ref_time)
        detector.add_listener(self.hit_listener())
        exit_task = dn.input([])

        return dn.pipe(detector_task, self.show_hit(), exit_task)

    @dn.datanode
    def show_hit(self):
        while True:
            yield

            while not self.hit_queue.empty():
                time, strength = self.hit_queue.get()
                self.logger.print(f"[{time:07.3f}s] {strength:.4f}")

    @dn.datanode
    def hit_listener(self):
        while True:
            _, time, strength, detected = yield

            if detected:
                self.hit_queue.put((time, strength))

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
