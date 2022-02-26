import sys
import os
import time
import shutil
import traceback
import contextlib
import threading
import queue
from ..utils import datanodes as dn
from ..utils import commands as cmd
from ..devices import terminals as term
from ..devices import audios as aud
from ..devices import engines


@contextlib.contextmanager
def prepare_pyaudio(logger):
    r"""Prepare PyAudio and print out some information.

    Loading PyAudio will cause PortAudio to print something to the terminal, which
    cannot be turned off by PyAudio, so why not print more information to make it
    more hacky.

    Parameters
    ----------
    logger : loggers.Logger

    Yields
    ------
    manager : PyAudio
    """

    verb_ctxt = logger.verb()
    verb_ctxt.__enter__()
    hit_except = False
    has_exited = False

    try:
        with aud.create_manager() as manager:
            logger.print()

            aud.print_pyaudio_info(manager)
            verb_ctxt.__exit__(None, None, None)
            has_exited = True

            logger.print()
            yield manager

    except:
        if has_exited:
            raise
        hit_except = True
        if not verb_ctxt.__exit__(*sys.exc_info()):
            raise

    finally:
        if not has_exited and not hit_except:
            verb_ctxt.__exit__(None, None, None)


def fit_screen(logger, terminal_settings):
    r"""Guide user to adjust screen size.

    Parameters
    ----------
    logger : loggers.Logger
    terminal_settings : terminals.TerminalSettings

    Returns
    -------
    fit_task : datanodes.DataNode
        The datanode to manage this process.
    """
    width = terminal_settings.best_screen_size
    delay = terminal_settings.adjust_screen_delay

    skip_event = threading.Event()

    skip = term.inkey(lambda a: skip_event.set() if a[1] == "\x1b" else None)

    @dn.datanode
    def fit():
        size = yield
        current_width = 0

        t = time.perf_counter()

        logger.print(f"Can you adjust the width to (or bigger than) {width}?")
        logger.print("Or [emph]Esc[/] to skip this process.")
        logger.print("You can try to fit the line below.")
        logger.print("━" * width, flush=True)

        while current_width < width or time.perf_counter() < t + delay:
            if skip_event.is_set():
                logger.print()
                break

            if current_width != size.columns:
                current_width = size.columns
                t = time.perf_counter()
                if current_width < width - 5:
                    hint = "(too small!)"
                elif current_width < width:
                    hint = "(very close!)"
                elif current_width == width:
                    hint = "(perfect!)"
                else:
                    hint = "(great!)"
                logger.clear_line()
                logger.print(
                    f"Current width: {current_width} {hint}", end="", flush=True
                )

            size = yield

        else:
            logger.print()
            logger.print("Thanks!")

        logger.print("You can adjust the screen size at any time.\n", flush=True)

        # sleep
        t = time.perf_counter()
        while time.perf_counter() < t + delay:
            yield

    return dn.pipe(skip, term.terminal_size(), fit())


@dn.datanode
def determine_unicode_version(logger):
    logger.print("[info/] Determine unicode version...")

    with logger.verb():
        version = yield from term.ucs_detect().join()

    if version is None:
        logger.print("[warn]Fail to determine unicode version[/]")

    else:
        logger.print(f"Your unicode version is [emph]{version}[/]")
        logger.print("[hint/] You can put this command into your settings file:")
        logger.print(f"[emph]UNICODE_VERSION={version}; export UNICODE_VERSION[/]")

    return version


class DevicesCommand:
    def __init__(self, config, logger, manager):
        self.config = config
        self.logger = logger
        self.manager = manager

    # audio

    @cmd.function_command
    def audio(self):
        """[rich]Show your audio configuration.

        usage: [cmd]devices[/] [cmd]audio[/]
        """

        logger = self.logger
        aud.print_pyaudio_info(self.manager)

        logger.print()

        device = self.config.current.devices.detector.input_device
        if device == -1:
            device = "default"
        samplerate = self.config.current.devices.detector.input_samplerate
        channels = self.config.current.devices.detector.input_channels
        format = self.config.current.devices.detector.input_format
        logger.print(
            f"current input device: {device} ({samplerate/1000} kHz, {channels} ch)"
        )

        device = self.config.current.devices.mixer.output_device
        if device == -1:
            device = "default"
        samplerate = self.config.current.devices.mixer.output_samplerate
        channels = self.config.current.devices.mixer.output_channels
        format = self.config.current.devices.mixer.output_format
        logger.print(
            f"current output device: {device} ({samplerate/1000} kHz, {channels} ch)"
        )

    @cmd.function_command
    def test_mic(self, device):
        """[rich]Test audio input.

        usage: [cmd]devices[/] [cmd]test_mic[/] [arg]{device}[/]
                                  ╱
                        The index of input
                         device, -1 is the
                          default device.
        """
        return MicTest(device, self.logger)

    @cmd.function_command
    def test_speaker(self, device):
        """[rich]Test audio output.

        usage: [cmd]devices[/] [cmd]test_speaker[/] [arg]{device}[/]
                                      ╱
                            The index of output
                             device, -1 is the
                              default device.
        """
        return SpeakerTest(device, self.logger)

    @cmd.function_command
    def set_mic(self, device, rate=None, ch=None, len=None, fmt=None):
        """[rich]Configure audio input.

                                          The sample rate        The buffer length
                                         of recorded sound.       of input device.
                                                  ╲                         ╲
        usage: [cmd]devices[/] [cmd]set_mic[/] [arg]{device}[/] [[[kw]--rate[/] [arg]{RATE}[/]]] [[[kw]--ch[/] [arg]{CH}[/]]] [[[kw]--len[/] [arg]{LEN}[/]]] [[[kw]--fmt[/] [arg]{FMT}[/]]]
                                 ╱                             ╱                          ╱
                       The index of input           The channel of audio         The data format
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
            logger.print("Validate input device...")
            aud.validate_input_device(
                self.manager, device, pa_samplerate, pa_channels, pa_format
            )
            logger.print("Success!")

        except:
            logger.print("[warn]Invalid configuration for mic.[/]")
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)

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
            self.config.set_change()

    @cmd.function_command
    def set_speaker(self, device, rate=None, ch=None, len=None, fmt=None):
        """[rich]Configure audio output.

                                              The sample rate        The buffer length
                                              of played sound.       of output device.
                                                      ╲                         ╲
        usage: [cmd]devices[/] [cmd]set_speaker[/] [arg]{device}[/] [[[kw]--rate[/] [arg]{RATE}[/]]] [[[kw]--ch[/] [arg]{CH}[/]]] [[[kw]--len[/] [arg]{LEN}[/]]] [[[kw]--fmt[/] [arg]{FMT}[/]]]
                                     ╱                             ╱                          ╱
                           The index of output          The channel of audio         The data format
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
            logger.print("Validate output device...")
            aud.validate_output_device(
                self.manager, device, pa_samplerate, pa_channels, pa_format
            )
            logger.print("Success!")

        except:
            logger.print("[warn]Invalid configuration for speaker.[/]")
            with logger.warn():
                logger.print(traceback.format_exc(), end="", markup=False)

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
            self.config.set_change()

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
        return cmd.OptionParser({"2": 2, "1": 1})

    @set_mic.arg_parser("len")
    @set_speaker.arg_parser("len")
    def _audio_buffer_length_parser(self, device, **__):
        return cmd.OptionParser({"512": 512, "1024": 1024, "2048": 2048})

    @set_mic.arg_parser("fmt")
    @set_speaker.arg_parser("fmt")
    def _audio_format_parser(self, device, **__):
        return cmd.OptionParser(["f4", "i4", "i2", "i1", "u1"])

    # terminal

    @cmd.function_command
    def terminal(self):
        """[rich]Show your terminal configuration.

        usage: [cmd]devices[/] [cmd]terminal[/]
        """

        term = os.environ.get("TERM", None)
        vte = os.environ.get("VTE_VERSION", None)
        uni = os.environ.get("UNICODE_VERSION", None)
        size = shutil.get_terminal_size()

        self.logger.print(f"terminal type: {term}")
        self.logger.print(f"VTE version: {vte}")
        self.logger.print(f"unicode version: {uni}")
        self.logger.print(f"terminal size: {size.columns}×{size.lines}")

    @cmd.function_command
    def fit_screen(self):
        """[rich]Fit your terminal screen.

        usage: [cmd]devices[/] [cmd]fit_screen[/]
        """

        return fit_screen(self.logger, self.config.current.devices.terminal)

    @cmd.function_command
    @dn.datanode
    def ucs_detect(self):
        """[rich]Determines the unicode version of your terminal.

        usage: [cmd]devices[/] [cmd]ucs_detect[/]
        """

        version = yield from determine_unicode_version(self.logger).join()
        if version is not None:
            os.environ["UNICODE_VERSION"] = version
            self.config.current.devices.terminal.unicode_version = version
            self.config.set_change()

    # engines

    @cmd.function_command
    @dn.datanode
    def test_keyboard(self):
        """[rich]Test your keyboard.

        usage: [cmd]devices[/] [cmd]test_keyboard[/]
        """

        logger = self.logger
        exit_key = "Esc"

        logger.print(f"[hint/] Press {logger.emph(exit_key)} to end test.")
        logger.print()
        logger.print("[[ <time>  ]] [emph]<keyname>[/] (<keycode>)", end="\r")

        stop_event = threading.Event()

        def handler(arg):
            _, time, keyname, keycode = arg
            logger.clear_line()
            logger.print(
                f"[[{time:07.3f} s]] {logger.emph(keyname)} ({logger.escape(repr(keycode))})"
            )
            logger.print("[[ <time>  ]] [emph]<keyname>[/] (<keycode>)", end="\r")

        controller_task, controller = engines.Controller.create(
            self.config.current.devices.controller, self.config.current.devices.terminal
        )
        controller.add_handler(handler)
        controller.add_handler(lambda _: stop_event.set(), exit_key)

        try:
            with controller_task:
                while not stop_event.is_set():
                    try:
                        controller_task.send(None)
                    except StopIteration:
                        return
                    yield
        finally:
            logger.print()

    @cmd.function_command
    def test_knock(self):
        """[rich]Test knock detection.

        usage: [cmd]devices[/] [cmd]test_knock[/]
        """
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
        self.logger.print("[hint/] Press any key to end test.")
        self.logger.print()

        detector_task, detector = engines.Detector.create(
            self.settings, manager, self.ref_time
        )
        detector.add_listener(self.hit_listener())

        @dn.datanode
        def exit_any():
            keycode = None
            while keycode is None:
                _, keycode = yield

        exit_task = term.inkey(exit_any())

        return dn.pipe(detector_task, self.show_hit(), exit_task)

    @dn.datanode
    def show_hit(self):
        ticks = " ▏▎▍▌▋▊▉█"
        nticks = len(ticks) - 1
        length = 10
        try:
            while True:
                self.logger.print(
                    "[[ <time>  ]] │[emph]<strength>[/]│ (<value>)", end="\r"
                )

                while self.hit_queue.empty():
                    yield

                time, strength = self.hit_queue.get()
                value = int(strength * length * nticks)
                level = "".join(
                    ticks[min(nticks, max(0, value - i * nticks))]
                    for i in range(length)
                )
                level = f"{level[:length//2]}[weight=bold]{level[length//2:]}[/]"
                self.logger.print(f"[[{time:07.3f} s]] │{level}│ ({strength:.5f})")

        finally:
            self.logger.print()

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
            raise cmd.CommandParseError(f"Invalid device index: {token}")
        return int(token)

    def suggest(self, token):
        return [val + "\000" for val in cmd.fit(token, self.options)]

    def info(self, token):
        value = int(token)
        if value == -1:
            if self.is_input:
                value = self.manager.get_default_input_device_info()["index"]
            else:
                value = self.manager.get_default_output_device_info()["index"]

        device_info = self.manager.get_device_info_by_index(value)

        name = device_info["name"]
        api = self.manager.get_host_api_info_by_index(device_info["hostApi"])["name"]
        freq = device_info["defaultSampleRate"] / 1000
        ch_in = device_info["maxInputChannels"]
        ch_out = device_info["maxOutputChannels"]

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
            device = manager.get_default_output_device_info()["index"]
        device_info = manager.get_device_info_by_index(device)

        samplerate = int(device_info["defaultSampleRate"])
        nchannels = min(device_info["maxOutputChannels"], 2)
        format = engines.MixerSettings.output_format

        try:
            self.logger.print("Validate output device...")
            aud.validate_output_device(manager, device, samplerate, nchannels, format)
            self.logger.print("Success!")

        except:
            self.logger.print("[warn]Invalid configuration for speaker.[/]")
            with self.logger.warn():
                self.logger.print(traceback.format_exc(), end="", markup=False)
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

        dt = 60.0 / self.tempo
        t0 = self.delay
        click_task = dn.interval(
            producer=self.make_click(mixer, samplerate, nchannels), dt=dt, t0=t0
        )

        return dn.pipe(mixer_task, click_task)

    @dn.datanode
    def make_click(self, mixer, samplerate, nchannels):
        click = dn.pulse(samplerate=samplerate)
        yield

        for n in range(nchannels):
            sound = click[:, None] * [[m == n for m in range(nchannels)]]
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
            device = manager.get_default_input_device_info()["index"]
        device_info = manager.get_device_info_by_index(device)

        samplerate = int(device_info["defaultSampleRate"])
        channels = 1
        buffer_length = engines.DetectorSettings.input_buffer_length
        format = engines.DetectorSettings.input_format

        try:
            self.logger.print("Validate input device...")
            aud.validate_input_device(manager, device, samplerate, channels, format)
            self.logger.print("Success!")

        except:
            self.logger.print("[warn]Invalid configuration for mic.[/]")
            with self.logger.warn():
                self.logger.print(traceback.format_exc(), end="", markup=False)
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

        @dn.datanode
        def exit_any():
            keycode = None
            while keycode is None:
                _, keycode = yield

        exit_task = term.inkey(exit_any())
        mic_task = aud.record(
            manager,
            vol,
            samplerate=samplerate,
            buffer_shape=(buffer_length, channels),
            format=format,
            device=device,
        )

        self.logger.print("[hint/] Press any key to end testing.")
        yield from dn.pipe(mic_task, exit_task).join()

    @dn.datanode
    def draw_volume(self, samplerate, buffer_length):
        decay_time = self.decay_time
        tick0 = " "
        tick1 = "▮"
        width = self.width

        decay = buffer_length / samplerate / decay_time
        volume_of = lambda x: dn.power2db((x ** 2).mean(), scale=(1e-5, 1e6)) / 60.0

        vol = 0.0
        try:
            while True:
                data = yield
                vol = max(0.0, vol - decay, min(1.0, volume_of(data)))
                size = int(vol * width)
                self.logger.print(
                    "[" + tick1 * size + tick0 * (width - size) + "]\r",
                    end="",
                    flush=True,
                    markup=False,
                )

        finally:
            self.logger.print()
