import sys
import os
import time
import contextlib
from collections import OrderedDict
import queue
import signal
import numpy
import pyaudio
from . import cfg
from . import realtime_analysis as ra


class NodeScheduler(ra.DataNode):
    def __init__(self, preprocess, postprocess):
        super().__init__(self.proxy(preprocess, postprocess))
        self.mutations = queue.Queue() # == [(key, node, zindex), ...]

    def proxy(self, preprocess, postprocess):
        preprocess = ra.DataNode.wrap(preprocess)
        postprocess = ra.DataNode.wrap(postprocess)
        nodes = OrderedDict()

        with preprocess, postprocess:
            try:
                data = None

                while True:
                    data = yield data
                    data = preprocess.send(data)

                    while not self.mutations.empty():
                        key, node, zindex = self.mutations.get()
                        if key in nodes:
                            nodes[key][0].__exit__()
                            del nodes[key]
                        if node is not None:
                            node.__enter__()
                            zindex_func = zindex if hasattr(zindex, '__call__') else lambda z=zindex: z
                            nodes[key] = (node, zindex_func)

                    for key, (node, _) in sorted(nodes.items(), key=lambda item: item[1][1]()):
                        try:
                            data = node.send(data)
                        except StopIteration:
                            del nodes[key]

                    data = postprocess.send(data)

            finally:
                for node, _ in nodes.values():
                    node.__exit__()


class AudioMixer(NodeScheduler):
    def __init__(self, samplerate=44100, buffer_shape=1024, delay=0.0):
        super().__init__(self.get_preprocess(), self.get_postprocess())
        self.samplerate = samplerate
        self.buffer_shape = buffer_shape
        self.delay = delay
        self.time = delay

    @ra.DataNode.from_generator
    def get_preprocess(self):
        index = 0
        buffer_length = self.buffer_shape[0] if isinstance(self.buffer_shape, tuple) else self.buffer_shape
        yield
        while True:
            self.time = index * buffer_length / self.samplerate + self.delay
            yield numpy.zeros(self.buffer_shape, dtype=numpy.float32)
            index += 1

    @ra.DataNode.from_generator
    def get_postprocess(self):
        buffer = yield
        while True:
            buffer = yield buffer

    def play(self, node, samplerate, channels=None, volume=0.0, start=None, end=None, time=None, zindex=0, key=None):
        if key is None:
            key = node
        if channels is None:
            channels = self.buffer_shape[1] if isinstance(self.buffer_shape, tuple) else 0

        if start is not None or end is not None:
            node = ra.tslice(node, samplerate, start, end)
        node = ra.pipe(node, ra.rechannel(channels))
        if samplerate != self.samplerate:
            node = ra.pipe(node, ra.resample(ratio=(self.samplerate, samplerate)))
        if volume != 0:
            node = ra.pipe(node, lambda s: s * 10**(volume/20))
        node = ra.attach(node)

        self.add_effect(node, time=time, zindex=zindex, key=key)

    def add_effect(self, node, time=None, zindex=0, key=None):
        if key is None:
            key = node
        if time is not None:
            node = self._shift(node, time)
        node = ra.DataNode.wrap(node)
        self.mutations.put((key, node, zindex))

    def remove_effect(self, key):
        self.mutations.put((key, None, 0))

    @ra.DataNode.from_generator
    def _shift(self, node, time):
        offset = round((time - self.time) * self.samplerate)
        node = ra.DataNode.wrap(node)

        with node:
            try:
                if offset < 0:
                    data = numpy.zeros((-offset,) + self.buffer_shape[1:], dtype=numpy.float32)
                    data = node.send(data)
                    offset = 0

            except StopIteration:
                yield

            else:
                data = yield

                while data.shape[0] < offset:
                    offset = offset - data.shape[0]
                    data = yield data

                if offset > 0:
                    res, data = data[:offset], data[offset:]
                    data = node.send(data)
                    data = numpy.concatenate((res, data), axis=0)
                    data = yield data

                while True:
                    data = yield node.send(data)

class KnockDetector(NodeScheduler):
    def __init__(self, samplerate=44100, buffer_length=1024, channels=1, *,
                       pre_max, post_max, pre_avg, post_avg, wait, delta,
                       time_res, freq_res, delay, energy):
        super().__init__(self.get_preprocess(), self.get_postprocess())
        self.samplerate = samplerate
        self.buffer_length = buffer_length
        self.channels = channels

        self.pre_max = pre_max
        self.post_max = post_max
        self.pre_avg = pre_avg
        self.post_avg = post_avg
        self.wait = wait
        self.delta = delta

        self.time_res = time_res
        self.freq_res = freq_res
        self.delay = delay
        self.energy = energy

    @ra.DataNode.from_generator
    def get_preprocess(self):
        detector = self._detector()
        hop_length = round(self.samplerate*self.time_res)
        if self.buffer_length != hop_length:
            detector = ra.unchunk(detector, chunk_shape=(hop_length, self.channels))
        with detector:
            while True:
                detector.send((yield))

    @ra.DataNode.from_generator
    def get_postprocess(self):
        while True:
            yield

    @ra.DataNode.from_generator
    def _detector(self):
        pre_max = round(self.pre_max / self.time_res)
        post_max = round(self.post_max / self.time_res)
        pre_avg = round(self.pre_avg / self.time_res)
        post_avg = round(self.post_avg / self.time_res)
        wait = round(self.wait / self.time_res)
        delta = self.delta
        prepare = max(post_max, post_avg)
        hop_length = round(self.samplerate*self.time_res)
        win_length = round(self.samplerate/self.freq_res)

        window = ra.get_half_Hann_window(win_length)
        onset = ra.pipe(
            ra.frame(win_length=win_length, hop_length=hop_length),
            ra.power_spectrum(win_length=win_length, samplerate=self.samplerate, windowing=window, weighting=True),
            ra.onset_strength(1))
        picker = ra.pick_peak(pre_max, post_max, pre_avg, post_avg, wait, delta)

        with onset, picker:
            buffer = [0.0]*prepare
            index = -prepare
            data = yield
            while True:
                strength = onset.send(data)
                detected = picker.send(strength)
                buffer.append(strength)
                strength = buffer.pop(0)

                self.time = index * hop_length / self.samplerate + self.delay
                self.strength = strength / self.energy
                self.detected = detected

                data = yield

                index += 1

    def on_hit(self, func, time=None, duration=None, key=None):
        if key is None:
            key = func

        @ra.DataNode.from_generator
        def _listener():
            if time is None:
                time = self.time
            while True:
                yield
                if self.time < time:
                    continue
                if duration is not None and time + duration <= self.time:
                    return

                if self.detected:
                    finished = func(self.strength)
                    if finished:
                        return
        node = _listener()

        self.mutations.put((key, node, 0))

    def on_time(self, func, time=None, key=None):
        if key is None:
            key = func

        @ra.DataNode.from_generator
        def _timed():
            if time is None:
                time = self.time
            while True:
                yield
                if time <= self.time:
                    func()
                    return
        node = _timed()

        self.mutations.put((key, node, 0))

    def add_listener(self, node, key=None):
        if key is None:
            key = node
        node = ra.DataNode.wrap(node)
        self.mutations.put((key, node, 0))

    def remove_listener(self, key):
        self.mutations.put((key, None, 0))

class TerminalLine(NodeScheduler):
    def __init__(self, framerate, delay):
        super().__init__(self.get_preprocess(), self.get_postprocess())
        self.width = int(os.popen("stty size", 'r').read().split()[1])
        self.chars = [" "]*self.width
        self.framerate = framerate
        self.delay = delay
        self.time = delay

    @ra.DataNode.from_generator
    def get_preprocess(self):
        index = 0
        while True:
            self.time = index / self.framerate + self.delay
            self.clear()
            yield
            index += 1

    @ra.DataNode.from_generator
    def get_postprocess(self):
        yield
        while True:
            yield f"\r{str(self)}\r"

    def __str__(self):
        return "".join(self.chars)

    def clear(self):
        for i in range(self.width):
            self.chars[i] = " "

    def addstr(self, index, str, mask=slice(None, None, None)):
        if isinstance(index, float):
            index = round(index)
        for ch in str:
            if ch == "\t":
                index += 1
            elif ch == "\b":
                index -= 1
            else:
                if index in range(self.width)[mask]:
                    self.chars[index] = ch
                index += 1

    def add_callback(self, node, zindex=0, key=None):
        if key is None:
            key = node
        node = ra.DataNode.wrap(node)
        self.mutations.put((key, node, zindex))

    def remove_callback(self, key):
        self.mutations.put((key, None, 0))


@cfg.configurable
class KnockConsoleSettings:
    # input
    input_device: int = -1
    input_samplerate: int = 44100
    input_buffer_length: int = 512
    input_channels: int = 1
    input_format: str = 'f4'

    # output
    output_device: int = -1
    output_samplerate: int = 44100
    output_buffer_length: int = 512
    output_channels: int = 1
    output_format: str = 'f4'

    # detector
    detector_time_res: float = 0.0116099773 # hop_length = 512 if samplerate == 44100
    detector_freq_res: float = 21.5332031 # win_length = 512*4 if samplerate == 44100
    detector_pre_max: float = 0.03
    detector_post_max: float = 0.03
    detector_pre_avg: float = 0.03
    detector_post_avg: float = 0.03
    detector_wait: float = 0.03
    detector_delta: float = 5.48e-6

    # controls
    display_framerate: float = 160.0 # ~ 2 / detector_time_res
    display_delay: float = 0.0
    knock_delay: float = 0.0
    knock_energy: float = 1.0e-3
    sound_delay: float = 0.0

    # debug
    debug_timeit: bool = False

class KnockConsole:
    settings: KnockConsoleSettings = KnockConsoleSettings()

    def __init__(self, config=None):
        self.stopped = False
        if config is not None:
            cfg.config_read(open(config, 'r'), main=self.settings)

    @contextlib.contextmanager
    def get_output_stream(self, manager):
        samplerate = self.settings.output_samplerate
        buffer_length = self.settings.output_buffer_length
        sound_delay = self.settings.sound_delay
        channels = self.settings.output_channels
        format = self.settings.output_format
        device = self.settings.output_device

        mixer = AudioMixer(samplerate, (buffer_length, channels), sound_delay)
        node = ra.timeit(mixer, "mixer") if self.settings.debug_timeit else mixer
        with ra.play(manager, node,
                     samplerate=samplerate,
                     buffer_shape=(buffer_length, channels),
                     format=format,
                     device=device,
                     ) as output_stream:

            yield output_stream, mixer

    @contextlib.contextmanager
    def get_input_stream(self, manager):
        samplerate = self.settings.input_samplerate
        buffer_length = self.settings.input_buffer_length
        channels = self.settings.input_channels
        format = self.settings.input_format
        device = self.settings.input_device

        pre_max = self.settings.detector_pre_max
        post_max = self.settings.detector_post_max
        pre_avg = self.settings.detector_pre_avg
        post_avg = self.settings.detector_post_avg
        wait = self.settings.detector_wait
        delta = self.settings.detector_delta
        time_res = self.settings.detector_time_res
        freq_res = self.settings.detector_freq_res

        knock_delay = self.settings.knock_delay
        knock_energy = self.settings.knock_energy

        detector = KnockDetector(samplerate, buffer_length, channels,
                                 pre_max=pre_max,
                                 post_max=post_max,
                                 pre_avg=pre_avg,
                                 post_avg=post_avg,
                                 wait=wait,
                                 delta=delta,
                                 time_res=time_res,
                                 freq_res=freq_res,
                                 delay=knock_delay,
                                 energy=knock_energy,
                                 )
        node = ra.timeit(detector, "detector") if self.settings.debug_timeit else detector
        with ra.record(manager, node,
                       samplerate=samplerate,
                       buffer_shape=(buffer_length, channels),
                       format=format,
                       device=device,
                       ) as input_stream:

            yield input_stream, detector

    @contextlib.contextmanager
    def get_display_thread(self):
        screen = TerminalLine(self.settings.display_framerate, self.settings.display_delay)

        @ra.DataNode.from_generator
        def show():
            try:
                while True:
                    t, view = yield
                    print(view, end="", flush=True)
            finally:
                print()

        node = ra.timeit(screen, "screen") if self.settings.debug_timeit else screen
        node = ra.pipe(ra.interval(1/self.settings.display_framerate, node), show())

        with ra.thread(node) as display_thread:
            yield display_thread, screen

    def SIGINT_handler(self, sig, frame):
        self.stopped = True

    def run(self, knock_program):
        try:
            manager = pyaudio.PyAudio()

            # connect audio/video streams and interface
            with self.get_output_stream(manager) as (output_stream, mixer),\
                 self.get_input_stream(manager) as (input_stream, detector),\
                 self.get_display_thread() as (display_thread, screen):

                # connect interface and program
                with knock_program.connect(mixer, detector, screen) as loop:

                    # activate audio/video streams
                    output_stream.start_stream()
                    input_stream.start_stream()
                    display_thread.start()

                    # loop
                    for _ in loop:
                        if (self.stopped or
                            not output_stream.is_active() or
                            not input_stream.is_active() or
                            not display_thread.is_alive()):

                            break

                        signal.signal(signal.SIGINT, self.SIGINT_handler)

        finally:
            manager.terminate()

class KnockProgram:
    # connect(mixer, detector, screen) -> DataNode
    # pause(self)
    # resume(self)
    pass


def test_speaker(manager, samplerate=44100, buffer_length=1024, channels=1, format='f4', device=-1):
    buffer_shape = (buffer_length, channels)
    duration = 2.0+0.5*4*channels

    mixer = AudioMixer(samplerate=samplerate, buffer_shape=buffer_shape)
    click = ra.pulse(samplerate=samplerate)
    for n in range(channels):
        for m in range(4):
            mixer.play([click], samplerate=samplerate, delay=1.0+0.5*(4*n+m))

    print("testing...")
    with ra.play(manager, mixer, samplerate=samplerate,
                                 buffer_shape=buffer_shape,
                                 format=format, device=device) as output_stream:
        output_stream.start_stream()
        time.sleep(duration)
    print("finish!")

def test_mic(manager, samplerate=44100, buffer_length=1024, channels=1, format='f4', device=-1):
    duration = 8.0

    spec_width = 5
    win_length = 512*4
    decay_time = 0.01
    Dt = buffer_length / samplerate
    spec = ra.pipe(ra.frame(win_length, buffer_length),
                   ra.power_spectrum(win_length, samplerate=samplerate),
                   ra.draw_spectrum(spec_width, win_length=win_length, samplerate=samplerate, decay=Dt/decay_time),
                   lambda s: print(f" {s}\r", end="", flush=True))

    print("testing...")
    with ra.record(manager, spec, samplerate=samplerate,
                                  buffer_shape=(buffer_length, channels),
                                  format=format, device=device) as input_stream:
        input_stream.start_stream()
        time.sleep(duration)
    print()
    print("finish!")

def input_with_default(hint, default, type=None):
    default_str = str(default)
    value = input(hint + default_str + "\b"*len(default_str))
    if value:
        return type(value) if type is not None else value
    else:
        return default

def configure_audio(config_name=None):
    config = configparser.ConfigParser()
    config.read("default.kconfig")
    if isinstance(config_name, str):
        config.read(config_name)
    elif isinstance(config_name, (dict, configparser.ConfigParser)):
        config.read_dict(config_name)
    elif config_name is None:
        pass
    else:
        raise ValueError("invalid configuration", config_name)

    try:
        manager = pyaudio.PyAudio()

        print()

        print("portaudio version:")
        print("  " + pyaudio.get_portaudio_version_text())

        print("available devices:")
        apis_list = [manager.get_host_api_info_by_index(i)['name'] for i in range(manager.get_host_api_count())]
        for index in range(manager.get_device_count()):
            info = manager.get_device_info_by_index(index)

            name = info['name']
            api = apis_list[info['hostApi']]
            freq = info['defaultSampleRate']/1000
            ch_in = info['maxInputChannels']
            ch_out = info['maxOutputChannels']

            print(f"  {index}. {name} by {api} ({freq} kHz, in: {ch_in}, out: {ch_out})")

        default_input_device_index = manager.get_default_input_device_info()['index']
        default_output_device_index = manager.get_default_output_device_info()['index']
        print(f"default input device: {default_input_device_index}")
        print(f"default output device: {default_output_device_index}")

        print()
        print("[output]")
        samplerate = input_with_default("samplerate = ", config.getint('output', 'samplerate'), int)
        buffer_length = input_with_default("buffer_length = ", config.getint('output', 'buffer_length'), int)
        channels = input_with_default("channels = ", config.getint('output', 'channels'), int)
        format = input_with_default("format = ", config.get('output', 'format'))
        device = input_with_default("device = ", config.getint('output', 'device'), int)
        test_speaker(manager, samplerate=samplerate,
                              buffer_length=buffer_length,
                              channels=channels,
                              format=format, device=device)

        print()
        print("[input]")
        samplerate = input_with_default("samplerate = ", config.getint('input', 'samplerate'), int)
        buffer_length = input_with_default("buffer_length = ", config.getint('input', 'buffer_length'), int)
        channels = input_with_default("channels = ", config.getint('input', 'channels'), int)
        format = input_with_default("format = ", config.get('input', 'format'))
        device = input_with_default("device = ", config.getint('input', 'device'), int)
        test_mic(manager, samplerate=samplerate,
                          buffer_length=buffer_length,
                          channels=channels,
                          format=format, device=device)

    finally:
        manager.terminate()

# manager.is_format_supported(rate,
#     input_device=None, input_channels=None, input_format=None,
#     output_device=None, output_channels=None, output_format=None)

# devices selector: device, samplerate, channels, format, buffer_length
# device:
#     device_index
# samplerate:
#     44100, 48000, 88200, 96000, 32000, 22050, 11025, 8000
# channels:
#     1, 2
# formats:
#     paFloat32, paInt32, paInt16, paInt8, paUInt8
# buffer_length:
#     1024, 512, 2048

# delta = noise_power * 20
# knock_volume = Dt / knock_max_energy

