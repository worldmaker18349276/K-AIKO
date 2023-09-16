import sys
import os
import time
import shutil
import contextlib
from inspect import cleandoc
from pathlib import Path
import threading
import queue
from ..utils import config as cfg
from ..utils import datanodes as dn
from ..utils import commands as cmd
from ..utils import markups as mu
from ..utils import providers
from ..devices import terminals as term
from ..devices import audios as aud
from ..devices import clocks
from ..devices import engines
from .loggers import Logger
from .files import FileManager, RecognizedDirPath, UnmovablePath
from .profiles import ProfileManager
from pyaudio import PyAudio
import numpy


class DevicesDirPath(RecognizedDirPath, UnmovablePath):
    """The place to manage your devices"""

    def banner(self):
        """
        [color=bright_cyan]  ╭────────▫ ╭──────6[/]
        [color=bright_cyan]1─╯╭─────▫ │ │ ╔════7[/] This folder doesn't contain any meaningful file, but
        [color=bright_cyan]2──╯╭──▫─│─│─│─║────8[/] is used to manage your devices such as keyboard,
        [color=bright_cyan]3───╯╭─┆─⬡ ⬡ ⬡ ║ ╭──9[/] terminal, audios, etc.
        [color=bright_cyan]4────╯ └╌╌⬡ □ ─║─╯╭─0[/]
        [color=bright_cyan]5──━━━━━━──────╨──╯  [/] Use the command [cmd]show[/] to view the details of your device.
        """
        return cleandoc(self.banner.__doc__)


class DevicesSettings(cfg.Configurable):
    mixer = cfg.subconfig(engines.MixerSettings)
    detector = cfg.subconfig(engines.DetectorSettings)
    renderer = cfg.subconfig(engines.RendererSettings)
    controller = cfg.subconfig(engines.ControllerSettings)
    terminal = cfg.subconfig(term.TerminalSettings)


class DeviceManager:
    def __init__(self, cache_dir, resources_dir, settings):
        self.cache_dir = cache_dir
        self.resources_dir = resources_dir
        self.settings = settings
        self.audio_manager = None

    def set_settings(self, settings):
        self.settings = settings

    @dn.datanode
    def initialize(self):
        logger = providers.get(Logger)
        profile_manager = providers.get(ProfileManager)

        # check tty
        if not sys.stdout.isatty():
            raise RuntimeError("please connect to interactive terminal device.")

        # deterimine unicode version
        if (
            profile_manager.current.devices.terminal.unicode_version == "auto"
            and "UNICODE_VERSION" not in os.environ
        ):
            version = yield from self.determine_unicode_version().join()
            if version is not None:
                os.environ["UNICODE_VERSION"] = version
                profile_manager.current.devices.terminal.unicode_version = version
                profile_manager.set_as_changed()
            logger.print()

        # fit screen size
        size = shutil.get_terminal_size()
        width = profile_manager.current.devices.terminal.best_screen_size
        if size.columns < width:
            logger.print("[hint/] Your screen size seems too small.")

            yield from self.fit_screen().join()

        # load PyAudio
        @contextlib.contextmanager
        def ctxt():
            logger.print("[info/] Load PyAudio...")
            logger.print()

            with self.prepare_pyaudio() as audio_manager:
                self.audio_manager = audio_manager
                try:
                    yield self
                finally:
                    self.audio_manager = None

        return ctxt()

    def load_engines(self, *types, clock=None, monitoring_session=None):
        if clock is None:
            clock = clocks.Clock(0.0, 1.0)
            init_time = None
        else:
            init_time = 0.0

        tasks = []
        res = []

        for typ in types:
            if typ == "mixer":
                mixer_monitor = None
                if monitoring_session is not None:
                    path = self.cache_dir.abs / f"{monitoring_session}_mixer_perf.csv"
                    mixer_monitor = engines.Monitor(path)

                mixer_task, mixer = engines.Mixer.create(
                    self.settings.mixer.copy(),
                    self.audio_manager,
                    clock,
                    init_time,
                    mixer_monitor,
                )

                tasks.append(mixer_task)
                res.append(mixer)

            elif typ == "detector":
                detector_monitor = None
                if monitoring_session is not None:
                    path = (
                        self.cache_dir.abs / f"{monitoring_session}_detector_perf.csv"
                    )
                    detector_monitor = engines.Monitor(path)

                detector_task, detector = engines.Detector.create(
                    self.settings.detector.copy(),
                    self.audio_manager,
                    clock,
                    init_time,
                    detector_monitor,
                )

                tasks.append(detector_task)
                res.append(detector)

            elif typ == "renderer":
                renderer_monitor = None
                if monitoring_session is not None:
                    path = (
                        self.cache_dir.abs / f"{monitoring_session}_renderer_perf.csv"
                    )
                    renderer_monitor = engines.Monitor(path)

                renderer_task, renderer = engines.Renderer.create(
                    self.settings.renderer.copy(),
                    self.settings.terminal,
                    clock,
                    init_time,
                    renderer_monitor,
                )

                tasks.append(renderer_task)
                res.append(renderer)

            elif typ == "controller":
                controller_task, controller = engines.Controller.create(
                    self.settings.controller.copy(),
                    self.settings.terminal,
                    clock,
                    init_time,
                )

                tasks.append(controller_task)
                res.append(controller)

            else:
                raise ValueError(typ)

        return dn.pipe(*tasks), res

    def load_rich(self):
        profile_manager = providers.get(ProfileManager)
        devices_settings = profile_manager.current.devices
        return mu.RichParser(
            devices_settings.terminal.unicode_version,
            devices_settings.terminal.color_support,
        )

    @dn.datanode
    def load_sound(self, src):
        WAVEFORM_MAX_TIME = 30.0
        samplerate = self.settings.mixer.output_samplerate
        nchannels = self.settings.mixer.output_channels

        if isinstance(src, Path):
            if src.is_absolute():
                sound_path = src
            else:
                sound_path = (self.resources_dir.abs / src).resolve()

            try:
                resource = yield from aud.load_sound(
                    sound_path,
                    samplerate=samplerate,
                    channels=nchannels,
                ).join()

                return resource

            except aud.IOCancelled:
                raise

            except Exception as e:
                raise RuntimeError(f"Failed to load resource {sound_path!s}") from e

        elif isinstance(src, dn.Waveform):
            node = src.generate(
                samplerate=samplerate,
                channels=nchannels,
            )

            res = []
            try:
                yield from dn.pipe(
                    node,
                    dn.tspan(samplerate=samplerate, end=WAVEFORM_MAX_TIME),
                    res.append,
                ).join()
                return res

            except GeneratorExit:
                raise aud.IOCancelled(
                    f"The operation of generating sound {src!r} has been cancelled."
                )

            except Exception as e:
                raise RuntimeError(f"Failed to load resource {src!r}") from e

        else:
            raise TypeError

    @staticmethod
    def format_pyaudio_info(manager):
        info = aud.pyaudio_info(manager)
        res = []

        res.append("[color=bright_blue]portaudio version:[/]")
        res.append("  " + info["version"])
        res.append("")

        res.append("[color=bright_blue]available devices:[/]")

        table = []
        for device in info["device_infos"]:
            ind = str(device["index"])
            name = device["name"]
            api = info["api_infos"][device["hostApi"]]["name"]
            freq = str(device["defaultSampleRate"] / 1000)
            chin = str(device["maxInputChannels"])
            chout = str(device["maxOutputChannels"])

            table.append((ind, name, api, freq, chin, chout))

        ind_len = max(len(entry[0]) for entry in table)
        name_len = max(len(entry[1]) for entry in table)
        api_len = max(len(entry[2]) for entry in table)
        freq_len = max(len(entry[3]) for entry in table)
        chin_len = max(len(entry[4]) for entry in table)
        chout_len = max(len(entry[5]) for entry in table)

        for ind, name, api, freq, chin, chout in table:
            res.append(
                f"  [weight=dim]{ind:>{ind_len}}.[/]"
                f" [color=magenta][weight=bold]{name:{name_len}}[/][/]"
                f"  [weight=dim]by[/]  [color=magenta]{api:{api_len}}[/]"
                f"  ([color=cyan][weight=bold]{freq:>{freq_len}}[/] kHz[/],"
                f" [weight=dim]in:[/]"
                f" [color=bright_blue][weight={'dim' if chin=='0' else 'bold'}]{chin:>{chin_len}}[/] ch[/],"
                f" [weight=dim]out:[/]"
                f" [color=bright_red][weight={'dim' if chout=='0' else 'bold'}]{chout:>{chout_len}}[/] ch[/])"
            )

        res.append("")

        default_input_device = info["default_input"]["index"]
        default_output_device = info["default_output"]["index"]
        res.append(
            f"[color=bright_blue]default input device:[/]  "
            + DeviceManager.format_device_info(info, default_input_device, True)
        )
        res.append(
            f"[color=bright_blue]default output device:[/] "
            + DeviceManager.format_device_info(info, default_input_device, False)
        )
        return "\n".join(res)

    @staticmethod
    def format_device_info(info, index, is_input):
        if index == -1:
            index = (
                info["default_input"]["index"]
                if is_input
                else info["default_output"]["index"]
            )

        device = info["device_infos"][index]
        name = device["name"]
        api = info["api_infos"][device["hostApi"]]["name"]
        freq = device["defaultSampleRate"] / 1000
        chin = device["maxInputChannels"]
        chout = device["maxOutputChannels"]

        if is_input:
            return (
                f"[color=magenta][weight=bold]{name}[/][/]"
                f" [weight=dim]by[/]"
                f" [color=magenta]{api}[/]"
                f" ([color=cyan][weight=bold]{freq}[/] kHz[/],"
                f" [color=bright_blue][weight={'dim' if chin==0 else 'bold'}]{chin}[/] ch[/])"
            )
        else:
            return (
                f"[color=magenta][weight=bold]{name}[/][/]"
                f" [weight=dim]by[/]"
                f" [color=magenta]{api}[/]"
                f" ([color=cyan][weight=bold]{freq}[/] kHz[/],"
                f" [color=bright_red][weight={'dim' if chout==0 else 'bold'}]{chout}[/] ch[/])"
            )

    @contextlib.contextmanager
    def prepare_pyaudio(self):
        r"""Prepare PyAudio and print out some information.

        Loading PyAudio will cause PortAudio to print something to the terminal,
        which cannot be turned off by PyAudio, so why not print more information to
        make it more hacky.

        Yields
        ------
        audio_manager : PyAudio
        """
        logger = providers.get(Logger)

        verb_ctxt = logger.verb()
        verb_ctxt.__enter__()
        hit_except = False
        has_exited = False

        try:
            with aud.create_manager() as audio_manager:
                logger.print()

                info = DeviceManager.format_pyaudio_info(audio_manager)
                info = logger.renderer.render_plain(logger.rich.parse(info))
                logger.print(info, markup=False)
                verb_ctxt.__exit__(None, None, None)
                has_exited = True

                logger.print()
                yield audio_manager

        except:
            if has_exited:
                raise
            hit_except = True
            if not verb_ctxt.__exit__(*sys.exc_info()):
                raise

        finally:
            if not has_exited and not hit_except:
                verb_ctxt.__exit__(None, None, None)

    def fit_screen(self):
        r"""Guide user to adjust screen size.

        Returns
        -------
        fit_task : datanodes.DataNode
            The datanode to manage this process.
        """
        logger = providers.get(Logger)
        profile_manager = providers.get(ProfileManager)
        terminal_settings = profile_manager.current.devices.terminal

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
            logger.print("━" * width, flush=True, log=False)

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
                    logger.clear_line(log=False)
                    logger.print(
                        f"Current width: {current_width} {hint}",
                        end="",
                        flush=True,
                        log=False,
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
    def determine_unicode_version(self):
        logger = providers.get(Logger)

        logger.print("[info/] Determine unicode version...")

        with logger.verb():
            version = yield from term.ucs_detect().join()

        if version is None:
            logger.print("[warn]Fail to determine unicode version[/]")

        else:
            with logger.print_stack() as print:
                print(f"Your unicode version is [emph]{version}[/]")
                print("[hint/] You can put this command into your .bashrc file:")
                print(f"[emph]UNICODE_VERSION={version}; export UNICODE_VERSION[/]")

        return version


class DevicesCommand:
    # audio

    @cmd.function_command
    def show(self):
        """[rich]Show your device configuration.

        usage: [cmd]show[/]
        """

        logger = providers.get(Logger)
        file_manager = providers.get(FileManager)
        device_manager = providers.get(DeviceManager)
        profile_manager = providers.get(ProfileManager)
        devices_settings = profile_manager.current.devices

        with logger.print_stack() as print:
            print(
                f"[color=bright_blue]workspace:[/] {logger.format_uri(file_manager.root.abs)}"
            )
            print()

            term = logger.escape(os.environ.get("TERM", "unknown"))
            vte = logger.escape(os.environ.get("VTE_VERSION", "unknown"))
            uni = logger.escape(os.environ.get("UNICODE_VERSION", "unknown"))
            size = shutil.get_terminal_size()

            print(f"[color=bright_blue]  terminal type:[/] {term}")
            print(f"[color=bright_blue]    VTE version:[/] {vte}")
            print(f"[color=bright_blue]unicode version:[/] {uni}")
            print(f"[color=bright_blue]  terminal size:[/] {size.columns}×{size.lines}")

            template = "[color={}]██[/]"
            palette = [
                "black",
                "red",
                "green",
                "yellow",
                "blue",
                "magenta",
                "cyan",
                "white",
            ]

            print()
            print("[color=bright_blue]color palette:[/]")
            print(
                " "
                + "".join(map(template.format, palette))
                + "\n "
                + "".join(map(template.format, map("bright_".__add__, palette)))
            )

            print()

            print(DeviceManager.format_pyaudio_info(device_manager.audio_manager))

            print()

            info = aud.pyaudio_info(device_manager.audio_manager)
            device = devices_settings.detector.input_device
            print(
                f"[color=bright_blue]current input device:[/]  "
                + DeviceManager.format_device_info(info, device, True)
            )

            device = devices_settings.mixer.output_device
            print(
                f"[color=bright_blue]current output device:[/] "
                + DeviceManager.format_device_info(info, device, False)
            )

    @cmd.function_command
    def test_mic(self, device):
        """[rich]Test audio input.

        usage: [cmd]test_mic[/] [arg]{device}[/]
                          ╱
                The index of input
                 device, -1 is the
                  default device.
        """
        logger = providers.get(Logger)
        device_manager = providers.get(DeviceManager)
        return MicTest(device, logger, device_manager)

    @cmd.function_command
    def test_speaker(self, device):
        """[rich]Test audio output.

        usage: [cmd]test_speaker[/] [arg]{device}[/]
                              ╱
                    The index of output
                     device, -1 is the
                      default device.
        """
        logger = providers.get(Logger)
        device_manager = providers.get(DeviceManager)
        return SpeakerTest(device, logger, device_manager)

    @cmd.function_command
    def set_mic(self, device, rate=None, ch=None, len=None, fmt=None):
        """[rich]Configure audio input.

                                  The sample rate        The buffer length
                                 of recorded sound.       of input device.
                                          ╲                         ╲
        usage: [cmd]set_mic[/] [arg]{device}[/] [[[kw]--rate[/] [arg]{RATE}[/]]] [[[kw]--ch[/] [arg]{CH}[/]]] [[[kw]--len[/] [arg]{LEN}[/]]] [[[kw]--fmt[/] [arg]{FMT}[/]]]
                         ╱                             ╱                          ╱
               The index of input           The channel of audio         The data format
                device, -1 is the            input: 1 for mono,         of recorded sound.
                 default device.               2 for stereo.
        """
        logger = providers.get(Logger)

        pa_samplerate = rate
        pa_channels = ch
        pa_format = fmt

        profile_manager = providers.get(ProfileManager)
        audio_manager = providers.get(DeviceManager).audio_manager
        devices_settings = profile_manager.current.devices

        if pa_samplerate is None:
            pa_samplerate = devices_settings.detector.input_samplerate
        if pa_channels is None:
            pa_channels = devices_settings.detector.input_channels
        if pa_format is None:
            pa_format = devices_settings.detector.input_format

        try:
            logger.print("Validate input device...")
            aud.validate_input_device(
                audio_manager, device, pa_samplerate, pa_channels, pa_format
            )
            logger.print("Success!")

        except Exception as exc:
            logger.print("[warn]Invalid configuration for mic.[/]")
            logger.print_traceback(exc)

        else:
            devices_settings.detector.input_device = device
            if rate is not None:
                devices_settings.detector.input_samplerate = rate
            if ch is not None:
                devices_settings.detector.input_channels = ch
            if len is not None:
                devices_settings.detector.input_buffer_length = len
            if fmt is not None:
                devices_settings.detector.input_format = fmt
            profile_manager.set_as_changed()

    @cmd.function_command
    def set_speaker(self, device, rate=None, ch=None, len=None, fmt=None):
        """[rich]Configure audio output.

                                      The sample rate        The buffer length
                                      of played sound.       of output device.
                                              ╲                         ╲
        usage: [cmd]set_speaker[/] [arg]{device}[/] [[[kw]--rate[/] [arg]{RATE}[/]]] [[[kw]--ch[/] [arg]{CH}[/]]] [[[kw]--len[/] [arg]{LEN}[/]]] [[[kw]--fmt[/] [arg]{FMT}[/]]]
                             ╱                             ╱                          ╱
                   The index of output          The channel of audio         The data format
                    device, -1 is the           output: 1 for mono,          of played sound.
                     default device.               2 for stereo.
        """
        logger = providers.get(Logger)

        pa_samplerate = rate
        pa_channels = ch
        pa_format = fmt

        profile_manager = providers.get(ProfileManager)
        audio_manager = providers.get(DeviceManager).audio_manager
        devices_settings = profile_manager.current.devices

        if pa_samplerate is None:
            pa_samplerate = devices_settings.mixer.output_samplerate
        if pa_channels is None:
            pa_channels = devices_settings.mixer.output_channels
        if pa_format is None:
            pa_format = devices_settings.mixer.output_format

        try:
            logger.print("Validate output device...")
            aud.validate_output_device(
                audio_manager, device, pa_samplerate, pa_channels, pa_format
            )
            logger.print("Success!")

        except Exception as exc:
            logger.print("[warn]Invalid configuration for speaker.[/]")
            logger.print_traceback(exc)

        else:
            devices_settings.mixer.output_device = device
            if rate is not None:
                devices_settings.mixer.output_samplerate = rate
            if ch is not None:
                devices_settings.mixer.output_channels = ch
            if len is not None:
                devices_settings.mixer.output_buffer_length = len
            if fmt is not None:
                devices_settings.mixer.output_format = fmt
            profile_manager.set_as_changed()

    @test_mic.arg_parser("device")
    @set_mic.arg_parser("device")
    def _set_mic_device_parser(self):
        device_manager = providers.get(DeviceManager)
        return PyAudioDeviceParser(device_manager, True)

    @test_speaker.arg_parser("device")
    @set_speaker.arg_parser("device")
    def _set_speaker_device_parser(self):
        device_manager = providers.get(DeviceManager)
        return PyAudioDeviceParser(device_manager, False)

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

    # engines

    @cmd.function_command
    def test_knock(self):
        """[rich]Test knock detection.

        usage: [cmd]test_knock[/]
        """
        logger = providers.get(Logger)
        device_manager = providers.get(DeviceManager)
        profile_manager = providers.get(ProfileManager)
        settings = profile_manager.current.devices.detector
        return KnockTest(logger, device_manager)

    @cmd.function_command
    @dn.datanode
    def test_keyboard(self):
        """[rich]Test your keyboard.

        usage: [cmd]test_keyboard[/]
        """
        logger = providers.get(Logger)
        device_manager = providers.get(DeviceManager)

        yield from test_keyboard(logger, device_manager).join()

    @cmd.function_command
    def test_waveform(self, waveform):
        """[rich]Test waveform generater.

        It accepts a small set of python expressions with some additional
        properties: use template `{shape:arg}` to generate some common waveforms;
        use hashtag `#effect:arg1,arg2,...` to post-process the generated waveform.

        usage: [cmd]test_waveform[/] [arg]{waveform}[/]
                                ╱
                         The function of
                         output waveform.

        Available templates: sine, square, triangle, sawtooth, square_duty
        Available hashtags: tspan, clip, bandpass, gammatone
        """
        logger = providers.get(Logger)
        device_manager = providers.get(DeviceManager)
        return WaveformTest(waveform, logger, device_manager)

    @test_waveform.arg_parser("waveform")
    def _test_waveform_waveform_parser(self):
        return cmd.RawParser(
            default="2**(-abs({sawtooth:t}+1)/0.02)*{sine:t*1000.0}",
            desc="It should be an expression of waveform.",
        )

    @cmd.function_command
    def test_logger(self, message, markup=True):
        """[rich]Print something.

        usage: [cmd]test_logger[/] [arg]{message}[/] [[[kw]--markup[/] [arg]{MARKUP}[/]]]
                              ╱                    ╲
                    text, the message               ╲
                     to be printed.          bool, use markup or not;
                                                default is True.
        """
        logger = providers.get(Logger)

        try:
            logger.print(message, markup=markup)
        except mu.MarkupParseError as e:
            logger.print(f"[warn]{logger.escape(str(e))}[/]")

    @test_logger.arg_parser("message")
    def _test_logger_message_parser(self):
        return cmd.RawParser(
            desc="It should be some text, indicating the message to be printed."
        )

    @test_logger.arg_parser("markup")
    def _test_logger_escape_parser(self, message):
        return cmd.LiteralParser(
            bool,
            default=False,
            desc="It should be bool,"
            " indicating whether to use markup;"
            " the default is False.",
        )

    # terminal

    @cmd.function_command
    def fit_screen(self):
        """[rich]Fit your terminal screen.

        usage: [cmd]fit_screen[/]
        """
        device_manager = providers.get(DeviceManager)

        return device_manager.fit_screen()

    @cmd.function_command
    @dn.datanode
    def ucs_detect(self):
        """[rich]Determines the unicode version of your terminal.

        usage: [cmd]ucs_detect[/]
        """
        profile_manager = providers.get(ProfileManager)
        device_manager = providers.get(DeviceManager)

        version = yield from device_manager.determine_unicode_version().join()
        if version is not None:
            os.environ["UNICODE_VERSION"] = version
            profile_manager.current.devices.terminal.unicode_version = version
            profile_manager.set_as_changed()


class PyAudioDeviceParser(cmd.ArgumentParser):
    def __init__(self, device_manager, is_input):
        self.device_manager = device_manager
        self.is_input = is_input
        self.options = ["-1"]
        for index in range(device_manager.audio_manager.get_device_count()):
            self.options.append(str(index))

    def parse(self, token):
        if token not in self.options:
            raise cmd.CommandParseError(f"Invalid device index: {token}")
        return int(token)

    def suggest(self, token):
        return [val + "\000" for val in cmd.fit(token, self.options)]

    def info(self, token):
        index = int(token)
        info = aud.pyaudio_info(self.device_manager.audio_manager)
        info_str = DeviceManager.format_device_info(info, index, self.is_input)
        return "[rich]" + info_str + "[/]"


def exit_any():
    return term.inkey(dn.pipe(dn.take(lambda arg: arg[1] is None), lambda _: None))


def test_keyboard(logger, devices_manager):
    exit_key = "Esc"
    exit_key_mu = logger.escape(exit_key, type="all")

    logger.print(f"[hint/] Press [emph]{exit_key_mu}[/] to end test.")
    logger.print()
    title = (
        "[color=bright_blue][[ [emph]<time>[/]  ]][/] "
        "[emph]<keyname>[/]"
        " [color=bright_green]'[emph]<keycode>[/]'[/]"
    )
    logger.print(title, end="\r", log=False)

    stop_event = threading.Event()

    @dn.datanode
    def handler():
        try:
            while True:
                _, time, keyname, keycode = yield
                keyname = logger.escape(keyname.ljust(9), type="all")
                keycode = logger.escape(keycode, type="all")
                logger.clear_line(log=False)
                logger.print(
                    f"[color=bright_blue][[{time:07.3f} s]][/] "
                    f"{keyname}"
                    f" [color=bright_green]'{keycode}'[/]",
                    log=False,
                )
                logger.print(title, end="\r", log=False)
        finally:
            logger.print()

    clock = clocks.Clock(0.0, 1.0)
    engine_task, engines = devices_manager.load_engines("controller", clock=clock)
    (controller,) = engines

    controller.add_handler(handler())
    controller.add_handler(lambda _: stop_event.set(), exit_key)

    stop_task = dn.take(lambda _: not stop_event.is_set())
    return dn.pipe(stop_task, engine_task)


class WaveformTest:
    def __init__(self, waveform, logger, device_manager):
        self.waveform = waveform
        self.logger = logger
        self.device_manager = device_manager

    def execute(self):
        self.logger.print("[info/] Compile waveform...")

        try:
            node = dn.Waveform(self.waveform).generate(
                self.device_manager.settings.mixer.output_samplerate,
                self.device_manager.settings.mixer.output_channels,
                self.device_manager.settings.mixer.output_buffer_length,
            )

        except Exception as exc:
            self.logger.print("[warn]Fail to compile waveform.[/]")
            self.logger.print_traceback(exc)
            return dn.DataNode.wrap([])

        clock = clocks.Clock(0.0, 1.0)
        engine_task, engines = self.device_manager.load_engines("mixer", clock=clock)
        (mixer,) = engines

        mixer.play(node)

        self.logger.print(f"Play waveform...")
        self.logger.print("[hint/] Press any key to end test.")
        return dn.pipe(engine_task, exit_any())


class KnockTest:
    def __init__(self, logger, device_manager):
        self.logger = logger
        self.device_manager = device_manager
        self.hit_queue = queue.Queue()

    def execute(self):
        self.logger.print("[hint/] Press any key to end test.")
        self.logger.print()

        clock = clocks.Clock(0.0, 1.0)
        engine_task, engines = self.device_manager.load_engines("detector", clock=clock)
        (detector,) = engines

        detector.add_listener(self.hit_listener())

        return dn.pipe(engine_task, self.show_hit(), exit_any())

    @dn.datanode
    def show_hit(self):
        ticks = " ▏▎▍▌▋▊▉█"
        nticks = len(ticks) - 1
        length = 10
        try:
            while True:
                self.logger.print(
                    "[color=bright_blue][[ [emph]<time>[/]  ]][/] "
                    "│[emph]<strength>[/]│"
                    " [color=bright_green]([emph]<value>[/])[/]",
                    end="\r",
                    log=False,
                )

                while self.hit_queue.empty():
                    yield

                time, strength = self.hit_queue.get()
                value = int(strength * length * nticks)
                level = "".join(
                    ticks[min(nticks, max(0, value - i * nticks))]
                    for i in range(length)
                )
                self.logger.print(
                    f"[color=bright_blue][[{time:07.3f} s]][/] "
                    f"│{level[:length//2]}[weight=bold]{level[length//2:]}[/]│"
                    f" [color=bright_green]({strength:.5f})[/]",
                    log=False,
                )

        finally:
            self.logger.print()

    @dn.datanode
    def hit_listener(self):
        while True:
            _, time, ratio, strength, detected = yield

            if detected:
                self.hit_queue.put((time, strength))


class SpeakerTest:
    CLICK_WAVEFORM = "2**(-t/0.01)*{sine:t*1000.0}#tspan:0,0.1"
    CLICK_DELAY = 0.5

    def __init__(self, device, logger, device_manager):
        self.device = device
        self.logger = logger
        self.device_manager = device_manager

    def execute(self):
        device = self.device

        if device == -1:
            device = self.device_manager.audio_manager.get_default_output_device_info()[
                "index"
            ]
        device_info = self.device_manager.audio_manager.get_device_info_by_index(device)

        samplerate = int(device_info["defaultSampleRate"])
        nchannels = min(device_info["maxOutputChannels"], 2)
        format = engines.MixerSettings.output_format

        try:
            self.logger.print("Validate output device...")
            aud.validate_output_device(
                self.device_manager.audio_manager, device, samplerate, nchannels, format
            )
            self.logger.print("Success!")

        except Exception as exc:
            self.logger.print(
                f"[warn]Invalid configuration for speaker: {self.logger.escape(repr(exc))}[/]"
            )
            return dn.DataNode.wrap([])

        else:
            info = PyAudioDeviceParser(self.device_manager, False).info(str(device))
            self.logger.print(f"Test output device {info}...")
            self.logger.print("[hint/] Press any key to end testing.")
            return self.test_speaker(device, samplerate, nchannels)

    def test_speaker(self, device, samplerate, nchannels):
        buffer_length = engines.MixerSettings.output_buffer_length
        format = engines.MixerSettings.output_format

        click = dn.chunk(
            self.make_click(samplerate, nchannels),
            chunk_shape=(buffer_length, nchannels),
        )

        speaker_task = aud.play(
            self.device_manager.audio_manager,
            click,
            samplerate=samplerate,
            buffer_shape=(buffer_length, nchannels),
            format=format,
            device=device,
        )

        return dn.pipe(speaker_task, exit_any())

    @dn.datanode
    def make_click(self, samplerate, nchannels):
        delay = numpy.zeros((round(self.CLICK_DELAY * samplerate), nchannels))
        click = dn.Waveform(self.CLICK_WAVEFORM).generate(samplerate, 0)
        click = dn.collect(click).exhaust()

        yield

        for n in range(nchannels):
            sound = click[:, None] * [[m == n for m in range(nchannels)]]
            self.logger.print(f"Test channel {n}: ", end="", flush=True)
            yield delay
            for m in range(4):
                self.logger.print(".", end="", flush=True, log=False)
                yield sound
                yield delay
            self.logger.print(flush=True)


class MicTest:
    VOLUME_DECAY_TIME = 0.01
    INDICATOR_WIDTH = 12
    INDICATOR_TICK0 = " "
    INDICATOR_TICK1 = "▮"

    def __init__(self, device, logger, device_manager):
        self.device = device
        self.logger = logger
        self.device_manager = device_manager

    def execute(self):
        device = self.device

        if device == -1:
            device = self.device_manager.audio_manager.get_default_input_device_info()[
                "index"
            ]
        device_info = self.device_manager.audio_manager.get_device_info_by_index(device)

        samplerate = int(device_info["defaultSampleRate"])
        channels = 1
        buffer_length = engines.DetectorSettings.input_buffer_length
        format = engines.DetectorSettings.input_format

        try:
            self.logger.print("Validate input device...")
            aud.validate_input_device(
                self.device_manager.audio_manager, device, samplerate, channels, format
            )
            self.logger.print("Success!")

        except Exception as exc:
            self.logger.print(
                f"[warn]Invalid configuration for mic: {self.logger.escape(repr(exc))}[/]"
            )
            return dn.DataNode.wrap([])

        else:
            info = PyAudioDeviceParser(self.device_manager, True).info(str(device))
            self.logger.print(f"Test input device {info}...")
            self.logger.print("[hint/] Press any key to end testing.")
            return self.test_mic(device, samplerate)

    def test_mic(self, device, samplerate):
        channels = 1
        buffer_length = engines.DetectorSettings.input_buffer_length
        format = engines.DetectorSettings.input_format

        vol = dn.branch(self.draw_volume(samplerate, buffer_length))

        mic_task = aud.record(
            self.device_manager.audio_manager,
            vol,
            samplerate=samplerate,
            buffer_shape=(buffer_length, channels),
            format=format,
            device=device,
        )

        return dn.pipe(mic_task, exit_any())

    @dn.datanode
    def draw_volume(self, samplerate, buffer_length):
        decay_time = self.VOLUME_DECAY_TIME
        width = self.INDICATOR_WIDTH
        tick0 = self.INDICATOR_TICK0
        tick1 = self.INDICATOR_TICK1

        decay = buffer_length / samplerate / decay_time
        volume_of = lambda x: dn.power2db((x**2).mean(), scale=(1e-5, 1e6)) / 60.0

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
                    log=False,
                )

        finally:
            self.logger.print()
