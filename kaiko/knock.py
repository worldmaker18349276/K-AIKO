import sys
import os
import time
import functools
import contextlib
from collections import OrderedDict
import queue
import signal
import numpy
import pyaudio
import audioread
from . import cfg
from . import datanodes as dn


class TerminalLine:
    def __init__(self):
        self.width = int(os.popen("stty size", 'r').read().split()[1])
        self.chars = [" "]*self.width

    def display(self):
        return "\r" + "".join(self.chars) + "\r"

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

        self.sound_delay = self.settings.sound_delay
        self.knock_delay = self.settings.knock_delay
        self.knock_energy = self.settings.knock_energy
        self.display_delay = self.settings.display_delay

    def _play(self, manager, node):
        samplerate = self.settings.output_samplerate
        buffer_length = self.settings.output_buffer_length
        nchannels = self.settings.output_channels
        format = self.settings.output_format
        device = self.settings.output_device

        stream = dn.play(manager, node,
                         samplerate=samplerate,
                         buffer_shape=(buffer_length, nchannels),
                         format=format,
                         device=device,
                         )

        return stream

    def _record(self, manager, node):
        samplerate = self.settings.input_samplerate
        buffer_length = self.settings.input_buffer_length
        nchannels = self.settings.input_channels
        format = self.settings.input_format
        device = self.settings.input_device

        stream = dn.record(manager, node,
                           samplerate=samplerate,
                           buffer_shape=(buffer_length, nchannels),
                           format=format,
                           device=device,
                           )

        return stream

    def _display(self, node):
        framerate = self.settings.display_framerate

        @dn.datanode
        def show():
            try:
                while True:
                    time, view = yield
                    if view:
                        print(view, end="", flush=True)
            finally:
                print()

        thread = dn.thread(dn.pipe(dn.interval(1/framerate, node), show()))

        return thread

    @dn.datanode
    def _output_node(self):
        samplerate = self.settings.output_samplerate
        buffer_length = self.settings.output_buffer_length
        nchannels = self.settings.output_channels

        sched = dn.schedule(self.output_queue)
        index = 0
        with sched:
            yield
            while True:
                time = index * buffer_length / samplerate + self.sound_delay
                data = numpy.zeros((buffer_length, nchannels), dtype=numpy.float32)
                time, data = sched.send((time, data))
                yield data
                index += 1

    def _input_node(self):
        samplerate = self.settings.input_samplerate
        buffer_length = self.settings.input_buffer_length
        nchannels = self.settings.input_channels

        time_res = self.settings.detector_time_res
        freq_res = self.settings.detector_freq_res
        hop_length = round(samplerate*time_res)
        win_length = round(samplerate/freq_res)

        pre_max  = round(self.settings.detector_pre_max  / time_res)
        post_max = round(self.settings.detector_post_max / time_res)
        pre_avg  = round(self.settings.detector_pre_avg  / time_res)
        post_avg = round(self.settings.detector_post_avg / time_res)
        wait     = round(self.settings.detector_wait     / time_res)
        delta    =       self.settings.detector_delta
        prepare = max(post_max, post_avg)

        sched = dn.schedule(self.input_queue)

        window = dn.get_half_Hann_window(win_length)
        onset = dn.pipe(
            dn.frame(win_length=win_length, hop_length=hop_length),
            dn.power_spectrum(win_length=win_length,
                              samplerate=samplerate,
                              windowing=window,
                              weighting=True),
            dn.onset_strength(1))
        picker = dn.pick_peak(pre_max, post_max, pre_avg, post_avg, wait, delta)

        @dn.datanode
        def input_node():
            with sched, onset, picker:
                data = yield
                buffer = [(self.knock_delay, 0.0, False)]*prepare
                index = 0
                while True:
                    strength = onset.send(data)
                    detected = picker.send(strength)
                    time = index * hop_length / samplerate + self.knock_delay
                    strength = strength / self.knock_energy

                    buffer.append((time, strength, detected))
                    sched.send(buffer.pop(0))
                    data = yield

                    index += 1

        input_node = input_node()
        if buffer_length != hop_length:
            input_node = dn.unchunk(input_node, chunk_shape=(hop_length, nchannels))

        return input_node

    @dn.datanode
    def _display_node(self):
        framerate = self.settings.display_framerate

        sched = dn.schedule(self.display_queue)
        index = 0
        screen = TerminalLine()
        with sched:
            yield
            while True:
                time = index / framerate + self.display_delay
                screen.clear()
                sched.send((time, screen))
                yield screen.display()
                index += 1

    def add_effect(self, node, time=None, zindex=0, key=None):
        if key is None:
            key = object()
        if time is not None:
            node = self._shift(node, time)
        node = dn.DataNode.wrap(node)
        self.output_queue.put((key, node, zindex))
        return key

    def remove_effect(self, key):
        self.output_queue.put((key, None, 0))

    @dn.datanode
    def _shift(self, node, start_time):
        node = dn.DataNode.wrap(node)

        samplerate = self.settings.output_samplerate
        buffer_length = self.settings.output_buffer_length
        nchannels = self.settings.output_channels

        with node:
            time, data = yield
            offset = round((start_time - time) * samplerate)

            while offset < 0:
                length = min(-offset, buffer_length)
                dummy = numpy.zeros((length, nchannels), dtype=numpy.float32)
                node.send(dummy)
                offset -= length

            while 0 < offset:
                if data.shape[0] < offset:
                    offset -= data.shape[0]
                else:
                    data1, data2 = data[:offset], data[offset:]
                    data2 = node.send(data2)
                    data = numpy.concatenate((data1, data2), axis=0)
                    offset = 0

                time, data = yield time, data

            while True:
                time, data = yield time, node.send(data)

    @functools.lru_cache(maxsize=32)
    def load(self, filepath):
        with audioread.audio_open(filepath) as file:
            samplerate = file.samplerate
        node = dn.load(filepath)
        if samplerate != self.settings.output_samplerate:
            node = dn.pipe(node, dn.resample(ratio=(self.settings.output_samplerate, samplerate)))
        with node as filenode:
            sound = list(filenode)
        return sound

    def play(self, node, samplerate=None, channels=None, volume=0.0, start=None, end=None, time=None, zindex=0, key=None):
        if channels is None:
            channels = self.settings.output_channels
        if isinstance(node, str):
            node = dn.DataNode.wrap(self.load(node))
            samplerate = None

        if start is not None or end is not None:
            node = dn.tslice(node, samplerate, start, end)
        node = dn.pipe(node, dn.rechannel(channels))
        if samplerate is not None and samplerate != self.settings.output_samplerate:
            node = dn.pipe(node, dn.resample(ratio=(self.settings.output_samplerate, samplerate)))
        if volume != 0:
            node = dn.pipe(node, lambda s: s * 10**(volume/20))
        node = dn.attach(node)

        return self.add_effect(node, time=time, zindex=zindex, key=key)

    def add_listener(self, node, key=None):
        if key is None:
            key = object()
        node = dn.branch(node)
        self.input_queue.put((key, node, 0))
        return key

    def remove_listener(self, key):
        self.input_queue.put((key, None, 0))

    def on_hit(self, func, time=None, duration=None, key=None):
        return self.add_listener(self._hit_listener(func, time, duration))

    @dn.datanode
    def _hit_listener(self, func, start_time, duration):
        time, strength, detected = yield
        if start_time is None:
            start_time = time

        while time < start_time:
            time, strength, detected = yield

        while duration is None or time < start_time + duration:
            if detected:
                finished = func(strength)
                if finished:
                    return

            time, strength, detected = yield

    def add_renderer(self, node, zindex=0, key=None):
        if key is None:
            key = object()
        node = dn.branch(node)
        self.display_queue.put((key, node, zindex))
        return key

    def remove_renderer(self, key):
        self.display_queue.put((key, None, 0))

    def SIGINT_handler(self, sig, frame):
        self.stopped = True

    def run(self, knock_program):
        debug_timeit = self.settings.debug_timeit

        try:
            manager = pyaudio.PyAudio()

            # make interfaces
            self.output_queue = queue.Queue()
            self.input_queue = queue.Queue()
            self.display_queue = queue.Queue()

            output_node = self._output_node()
            input_node = self._input_node()
            display_node = self._display_node()

            # wrap interfaces with debuger
            with dn.timeit(output_node , "   mixer", debug_timeit) as output_node,\
                 dn.timeit(input_node  , "detector", debug_timeit) as input_node,\
                 dn.timeit(display_node, "renderer", debug_timeit) as display_node:

                # connect audio/video streams and interfaces
                with self._play(manager, output_node) as output_stream,\
                     self._record(manager, input_node) as input_stream,\
                     self._display(display_node) as display_thread:

                    # connect interfaces and program
                    with knock_program.connect(self) as loop:

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


def test_speaker(manager, samplerate=44100, buffer_length=1024, channels=1, format='f4', device=-1):
    buffer_shape = (buffer_length, channels)
    duration = 2.0+0.5*4*channels

    mixer = AudioMixer(samplerate=samplerate, buffer_shape=buffer_shape)
    click = dn.pulse(samplerate=samplerate)
    for n in range(channels):
        for m in range(4):
            mixer.play([click], samplerate=samplerate, delay=1.0+0.5*(4*n+m))

    print("testing...")
    with dn.play(manager, mixer, samplerate=samplerate,
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
    spec = dn.pipe(dn.frame(win_length, buffer_length),
                   dn.power_spectrum(win_length, samplerate=samplerate),
                   dn.draw_spectrum(spec_width, win_length=win_length, samplerate=samplerate, decay=Dt/decay_time),
                   lambda s: print(f" {s}\r", end="", flush=True))

    print("testing...")
    with dn.record(manager, spec, samplerate=samplerate,
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

