import os
import re
import time
import shutil
import traceback
import contextlib
import threading
import queue
import pyaudio
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
    r"""
    Fields
    ------
    data_icon : str
        The marker of data log.
    info_icon : str
        The marker of info log.
    hint_icon : str
        The marker of hint log.

    verb_attr : str
        The text attribute of verb log.
    emph_attr : str
        The text attribute of emph log.
    warn_attr : str
        The text attribute of warn log.

    best_screen_size : int
        The best screen size.
    adjust screen delay : float
        The delay time to complete the screen adjustment.

    editor : str
        The editor to edit long text.
    """
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
    def ask(self, prompt, default=True):
        yield
        hint = self.emph("y")+"/n" if default == True else "y/"+self.emph("n")
        self.print(f"{prompt} [{hint}]: ", end="", flush=True)

        while True:
            res_queue = queue.Queue()
            input_task = dn.input(lambda a: res_queue.put(a[1]))
            try:
                with dn.pipe(input_task, dn.take(lambda _: res_queue.empty())) as task:
                    yield from task.join((yield))
            finally:
                self.print(flush=True)
            res = res_queue.get() if not res_queue.empty() else ""

            if res == "\n":
                return default
            if res in ("y", "Y"):
                return True
            elif res in ("n", "N"):
                return False

            self.print(f"Please reply {self.emph('y')} or {self.emph('n')}: ", end="", flush=True)

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
    def test_mic(self, device):
        """Test audio input.

        usage: devices \x1b[94mtest_mic\x1b[m \x1b[92m{device}\x1b[m
                                      ╱
                            the index of input
                             device, -1 is the
                              default device.
        """
        return MicTest(device, self.logger)

    @cmd.function_command
    def test_speaker(self, device):
        """Test audio output.

        usage: devices \x1b[94mtest_speaker\x1b[m \x1b[92m{device}\x1b[m
                                      ╱
                            the index of output
                             device, -1 is the
                              default device.
        """
        return SpeakerTest(device, self.logger)

    @cmd.function_command
    def set_mic(self, device, rate=None, ch=None, len=None, fmt=None):
        """Configure audio input.
        
                                          the sample rate        the buffer length
                                         of recorded sound.       of input device.
                                                  ╲                         ╲
        usage: devices \x1b[94mset_mic\x1b[m \x1b[92m{device}\x1b[m \
[\x1b[95m--rate\x1b[m \x1b[92m{RATE}\x1b[m] \
[\x1b[95m--ch\x1b[m \x1b[92m{CH}\x1b[m] \
[\x1b[95m--len\x1b[m \x1b[92m{LEN}\x1b[m] \
[\x1b[95m--fmt\x1b[m \x1b[92m{FMT}\x1b[m]
                                 ╱                             ╱                          ╱
                       the index of input           the channel of audio         the data format
                        device, -1 is the            input: 1 for mono,         of recorded sound.
                         default device.               2 for stereo.
        """
        logger = self.logger

        pa_samplerate = rate
        pa_channels = ch
        pa_format = fmt

        if pa_samplerate is None:
            pa_samplerate = self.config.current.devices.detector.input_samplerate
        if pa_channels is None:
            pa_channels = self.config.current.devices.detector.input_channels
        if pa_format is None:
            pa_format = self.config.current.devices.detector.input_format

        try:
            engines.validate_input_device(self.manager, device, pa_samplerate, pa_channels, pa_format)

        except ValueError as e:
            info = e.args[0]
            with logger.warn():
                logger.print(info)

        else:
            self.config.current.devices.detector.input_device = device
            if rate is not None:
                self.config.current.devices.detector.input_samplerate = rate
            if ch is not None:
                self.config.current.devices.detector.input_channels = ch
            if len is not None:
                self.config.current.devices.detector.input_buffer_length = len
            if fmt is not None:
                self.config.current.devices.detector.input_format = fmt

    @cmd.function_command
    def set_speaker(self, device, rate=None, ch=None, len=None, fmt=None):
        """Configure audio output.
        
                                              the sample rate        the buffer length
                                              of played sound.       of output device.
                                                      ╲                         ╲
        usage: devices \x1b[94mset_speaker\x1b[m \x1b[92m{device}\x1b[m \
[\x1b[95m--rate\x1b[m \x1b[92m{RATE}\x1b[m] \
[\x1b[95m--ch\x1b[m \x1b[92m{CH}\x1b[m] \
[\x1b[95m--len\x1b[m \x1b[92m{LEN}\x1b[m] \
[\x1b[95m--fmt\x1b[m \x1b[92m{FMT}\x1b[m]
                                     ╱                             ╱                          ╱
                           the index of output          the channel of audio         the data format
                            device, -1 is the           output: 1 for mono,          of played sound.
                             default device.               2 for stereo.
        """
        logger = self.logger

        pa_samplerate = rate
        pa_channels = ch
        pa_format = fmt

        if pa_samplerate is None:
            pa_samplerate = self.config.current.devices.mixer.output_samplerate
        if pa_channels is None:
            pa_channels = self.config.current.devices.mixer.output_channels
        if pa_format is None:
            pa_format = self.config.current.devices.mixer.output_format

        try:
            engines.validate_output_device(self.manager, device, pa_samplerate, pa_channels, pa_format)

        except ValueError as e:
            info = e.args[0]
            with logger.warn():
                logger.print(info)

        else:
            self.config.current.devices.mixer.output_device = device
            if rate is not None:
                self.config.current.devices.mixer.output_samplerate = rate
            if ch is not None:
                self.config.current.devices.mixer.output_channels = ch
            if len is not None:
                self.config.current.devices.mixer.output_buffer_length = len
            if fmt is not None:
                self.config.current.devices.mixer.output_format = fmt

    @test_mic.arg_parser("device")
    @set_mic.arg_parser("device")
    def _set_mic_device_parser(self):
        return PyAudioDeviceParser(self.manager, True)

    @test_speaker.arg_parser("device")
    @set_speaker.arg_parser("device")
    def _set_speaker_device_parser(self):
        return PyAudioDeviceParser(self.manager, False)

    @set_mic.arg_parser("rate")
    @set_speaker.arg_parser("rate")
    def _audio_samplerate_parser(self, device, **__):
        options = [44100, 48000, 88200, 96000, 32000, 22050, 11025, 8000]
        return cmd.OptionParser({str(rate): rate for rate in options})

    @set_mic.arg_parser("ch")
    @set_speaker.arg_parser("ch")
    def _audio_channels_parser(self, device, **__):
        return cmd.OptionParser({'2': 2, '1': 1})

    @set_mic.arg_parser("len")
    @set_speaker.arg_parser("len")
    def _audio_buffer_length_parser(self, device, **__):
        return cmd.OptionParser({"512": 512, "1024": 1024, "2048": 2048})

    @set_mic.arg_parser("fmt")
    @set_speaker.arg_parser("fmt")
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
    def test_keyboard(self):
        """Test your keyboard."""

        logger = self.logger
        exit_key = 'Esc'

        logger.print(f"Press {logger.emph(exit_key)} to end test.", prefix="hint")
        logger.print()
        logger.print(f"[ <time>  ] {logger.emph('<keyname>')} (<keycode>)")

        stop_event = threading.Event()

        def handler(arg):
            _, time, keyname, keycode = arg
            logger.print(f"[{time:07.3f} s] {logger.emph(keyname)} ({repr(keycode)})")

        controller_task, controller = engines.Controller.create(self.config.current.devices.controller)
        controller.add_handler(handler)
        controller.add_handler(lambda _: stop_event.set(), exit_key)

        return dn.pipe(controller_task, dn.take(lambda _: not stop_event.is_set()))

    @cmd.function_command
    def test_knock(self):
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
        self.logger.print("Press any key to end test.", prefix="hint")
        self.logger.print()
        self.logger.print(f"[ <time>  ] │{self.logger.emph('<strength>')}│ (<value>)")

        detector_task, detector = engines.Detector.create(self.settings, manager, self.ref_time)
        detector.add_listener(self.hit_listener())
        exit_task = dn.input([])

        return dn.pipe(detector_task, self.show_hit(), exit_task)

    @dn.datanode
    def show_hit(self):
        ticks = " ▏▎▍▌▋▊▉█"
        nticks = len(ticks) - 1
        length = 10
        loud_attr = "1"
        while True:
            yield

            while not self.hit_queue.empty():
                time, strength = self.hit_queue.get()
                value = int(strength * length * nticks)
                level = "".join(ticks[min(nticks, max(0, value - i * nticks))] for i in range(length))
                level = level[:length//2] + wcb.add_attr(level[length//2:], loud_attr)
                self.logger.print(f"[{time:07.3f} s] │{level}│ ({strength:.5f})")

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

class SpeakerTest:
    def __init__(self, device, logger, tempo=120.0, delay=0.5):
        self.device = device
        self.logger = logger
        self.tempo = tempo
        self.delay = delay

    def execute(self, manager):
        device = self.device

        if device == -1:
            device = manager.get_default_output_device_info()['index']
        device_info = manager.get_device_info_by_index(device)

        samplerate = int(device_info['defaultSampleRate'])
        nchannels = min(device_info['maxOutputChannels'], 2)
        format = engines.MixerSettings.output_format

        try:
            engines.validate_output_device(manager, device, samplerate, nchannels, format)

        except ValueError:
            with self.logger.warn():
                self.logger.print(traceback.format_exc(), end="")
            return dn.DataNode.wrap([])

        else:
            info = PyAudioDeviceParser(manager, False).info(str(device))
            self.logger.print(f"Test output device {self.logger.emph(info)}...")
            return self.test_speaker(manager, device, samplerate, nchannels)

    def test_speaker(self, manager, device, samplerate, nchannels):
        settings = engines.MixerSettings()
        settings.output_device = device
        settings.output_samplerate = samplerate
        settings.output_channels = nchannels

        mixer_task, mixer = engines.Mixer.create(settings, manager)

        dt = 60.0/self.tempo
        t0 = self.delay
        click_task = dn.interval(producer=self.make_click(mixer, samplerate, nchannels), dt=dt, t0=t0)

        return dn.pipe(mixer_task, click_task)

    @dn.datanode
    def make_click(self, mixer, samplerate, nchannels):
        click = dn.pulse(samplerate=samplerate)
        yield

        for n in range(nchannels):
            sound = click[:,None] * [[m==n for m in range(nchannels)]]
            self.logger.print(f"Test channel {n}: ", end="", flush=True)
            yield
            for m in range(4):
                mixer.play(dn.DataNode.wrap([sound]))
                self.logger.print(".", end="", flush=True)
                yield
            self.logger.print(flush=True)
            yield

class MicTest:
    def __init__(self, device, logger, width=12, decay_time=0.1):
        self.device = device
        self.logger = logger
        self.width = width
        self.decay_time = decay_time

    def execute(self, manager):
        device = self.device

        if device == -1:
            device = manager.get_default_input_device_info()['index']
        device_info = manager.get_device_info_by_index(device)

        samplerate = int(device_info['defaultSampleRate'])
        channels = 1
        buffer_length = engines.DetectorSettings.input_buffer_length
        format = engines.DetectorSettings.input_format

        try:
            engines.validate_input_device(manager, device, samplerate, channels, format)

        except ValueError:
            with self.logger.warn():
                self.logger.print(traceback.format_exc(), end="")
            return dn.DataNode.wrap([])

        else:
            info = PyAudioDeviceParser(manager, True).info(str(device))
            self.logger.print(f"Test input device {self.logger.emph(info)}...")
            return self.test_mic(manager, device, samplerate)

    @dn.datanode
    def test_mic(self, manager, device, samplerate):
        channels = 1
        buffer_length = engines.DetectorSettings.input_buffer_length
        format = engines.DetectorSettings.input_format

        vol = dn.branch(self.draw_volume(samplerate, buffer_length))

        exit_task = dn.input([])
        mic_task = dn.record(manager, vol, samplerate=samplerate,
                                           buffer_shape=(buffer_length, channels),
                                           format=format, device=device)

        self.logger.print("Press any key to end testing.", prefix="hint")
        with dn.pipe(mic_task, exit_task) as task:
            yield from task.join((yield))

    @dn.datanode
    def draw_volume(self, samplerate, buffer_length):
        decay_time = self.decay_time
        tick0 = " "
        tick1 = "▮"
        width = self.width

        decay = buffer_length / samplerate / decay_time
        volume_of = lambda x: dn.power2db((x**2).mean(), scale=(1e-5, 1e6)) / 60.0

        vol = 0.0
        try:
            while True:
                data = yield
                vol = max(0.0, vol-decay, min(1.0, volume_of(data)))
                size = int(vol * width)
                self.logger.print("[" + tick1 * size + tick0 * (width-size) + "]\r", end="", flush=True)

        finally:
            self.logger.print()
