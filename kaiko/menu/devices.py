import os
import shutil
import pyaudio
import contextlib
import threading
from kaiko.utils import wcbuffers as wcb
from kaiko.utils import datanodes as dn
from kaiko.utils import commands as cmd


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

    try:
        yield manager
    finally:
        manager.terminate()

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

    def fit_screen(self):
        r"""Guide user to adjust screen size.

        Returns
        -------
        fit_task : dn.DataNode
            The datanode to manage this process.
        """
        width = self.config.current.menu.best_screen_size
        delay = self.config.current.menu.adjust_screen_delay

        skip_event = threading.Event()

        skip = dn.input(lambda a: skip_event.set() if a[1] == '\x1b' else None)

        @dn.datanode
        def fit():
            size = yield
            current_width = 0

            if size.columns < width:
                t = time.time()

                self.print("Your screen size seems too small.")
                self.print(f"Can you adjust the width to (or bigger than) {width}?")
                self.print(f"Or {self.emph('Esc')} to skip this process.")
                self.print("You can try to fit the line below.")
                self.print("━"*width, flush=True)

                while current_width < width or time.time() < t+delay:
                    if skip_event.is_set():
                        self.print()
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
                        self.print(f"\r\x1b[KCurrent width: {current_width} {hint}", end="", flush=True)

                    size = yield

                else:
                    self.print()
                    self.print("Thanks!")

                self.print("You can adjust the screen size at any time.\n", flush=True)

                # sleep
                t = time.time()
                while time.time() < t+delay:
                    yield

        return dn.pipe(skip, dn.terminal_size(), fit())

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
