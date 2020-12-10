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
        if config is not None:
            cfg.config_read(open(config, 'r'), main=self.settings)

        self.sound_delay = self.settings.sound_delay
        self.knock_delay = self.settings.knock_delay
        self.knock_energy = self.settings.knock_energy
        self.display_delay = self.settings.display_delay

        self._SIGINT = False

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

    def _get_output_node(self):
        samplerate = self.settings.output_samplerate
        buffer_length = self.settings.output_buffer_length
        nchannels = self.settings.output_channels

        self.effect_queue = queue.Queue()
        effect_node = dn.schedule(self.effect_queue)

        @dn.datanode
        def output_node():
            index = 0
            with effect_node:
                yield
                while True:
                    time = index * buffer_length / samplerate + self.sound_delay
                    data = numpy.zeros((buffer_length, nchannels), dtype=numpy.float32)
                    time, data = effect_node.send((time, data))
                    yield data
                    index += 1

        output_node = output_node()

        if self.settings.debug_timeit:
            return dn.timeit(output_node, " output")
        else:
            return contextlib.nullcontext(output_node)

    def _get_input_node(self):
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

        self.listener_queue = queue.Queue()
        listener_node = dn.schedule(self.listener_queue)

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
            with listener_node, onset, picker:
                data = yield
                buffer = [(self.knock_delay, 0.0)]*prepare
                index = 0
                while True:
                    strength = onset.send(data)
                    detected = picker.send(strength)
                    time = index * hop_length / samplerate + self.knock_delay
                    strength = strength / self.knock_energy

                    buffer.append((time, strength))
                    time, strength = buffer.pop(0)

                    listener_node.send((time, strength, detected))
                    data = yield

                    index += 1

        input_node = input_node()
        if buffer_length != hop_length:
            input_node = dn.unchunk(input_node, chunk_shape=(hop_length, nchannels))

        if self.settings.debug_timeit:
            return dn.timeit(input_node, "  input")
        else:
            return contextlib.nullcontext(input_node)

    def _get_display_node(self):
        framerate = self.settings.display_framerate

        self.renderer_queue = queue.Queue()
        renderer_node = dn.schedule(self.renderer_queue)

        @dn.datanode
        def display_node():
            index = 0
            screen = TerminalLine()
            with renderer_node:
                yield
                while True:
                    time = index / framerate + self.display_delay
                    screen.clear()
                    renderer_node.send((time, screen))
                    yield screen.display()
                    index += 1

        display_node = display_node()

        if self.settings.debug_timeit:
            return dn.timeit(display_node, "display")
        else:
            return contextlib.nullcontext(display_node)

    def _SIGINT_handler(self, sig, frame):
        self._SIGINT = True

    def run(self, knock_program):
        try:
            manager = pyaudio.PyAudio()

            # initialize interfaces
            with self._get_output_node() as output_node,\
                 self._get_input_node() as input_node,\
                 self._get_display_node() as display_node:

                # connect audio/video streams and interfaces
                with self._play(manager, output_node) as output_stream,\
                     self._record(manager, input_node) as input_stream,\
                     self._display(display_node) as display_thread:

                    # connect to program
                    with knock_program.connect(self) as loop:

                        # activate audio/video streams
                        output_stream.start_stream()
                        input_stream.start_stream()
                        display_thread.start()

                        # loop
                        for _ in loop:
                            if self._SIGINT:
                                break

                            if not output_stream.is_active():
                                raise RuntimeError("output stream is down")
                            if not input_stream.is_active():
                                raise RuntimeError("input stream is down")
                            if not display_thread.is_alive():
                                raise RuntimeError("display thread is down")

                            signal.signal(signal.SIGINT, self._SIGINT_handler)

        finally:
            manager.terminate()

    def add_effect(self, node, time=None, zindex=0, key=None):
        if key is None:
            key = object()
        node = self._timed_effect(node, time)
        self.effect_queue.put((key, node, zindex))
        return key

    def remove_effect(self, key):
        self.effect_queue.put((key, None, 0))

    @dn.datanode
    def _timed_effect(self, node, start_time):
        node = dn.DataNode.wrap(node)

        samplerate = self.settings.output_samplerate
        buffer_length = self.settings.output_buffer_length
        nchannels = self.settings.output_channels

        with node:
            time, data = yield
            offset = round((start_time - time) * samplerate) if start_time is not None else 0

            while offset < 0:
                length = min(-offset, buffer_length)
                dummy = numpy.zeros((length, nchannels), dtype=numpy.float32)
                node.send(dummy)
                offset += length

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
    def load_sound(self, filepath):
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
            node = dn.DataNode.wrap(self.load_sound(node))
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
        self.listener_queue.put((key, node, 0))
        return key

    def remove_listener(self, key):
        self.listener_queue.put((key, None, 0))

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
        self.renderer_queue.put((key, node, zindex))
        return key

    def remove_renderer(self, key):
        self.renderer_queue.put((key, None, 0))

