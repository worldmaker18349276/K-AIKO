import sys
import os
import time
import functools
import unicodedata
import contextlib
from collections import OrderedDict
import threading
import shutil
import signal
import numpy
import pyaudio
import audioread
from . import cfg
from . import datanodes as dn
from . import tui


@contextlib.contextmanager
def nullcontext(contextmanager):
    with contextmanager as context:
        yield context

class AudioMixer:
    def __init__(self, settings, sound_delay):
        self.settings = settings
        self.sound_delay = sound_delay
        self.effects_scheduler = dn.Scheduler()

    @dn.datanode
    def get_output_node(self):
        samplerate = self.settings.output_samplerate
        buffer_length = self.settings.output_buffer_length
        nchannels = self.settings.output_channels

        index = 0
        with self.effects_scheduler:
            yield
            while True:
                time = index * buffer_length / samplerate + self.sound_delay
                data = numpy.zeros((buffer_length, nchannels), dtype=numpy.float32)
                time, data = self.effects_scheduler.send((time, data))
                yield data
                index += 1

    @contextlib.contextmanager
    def get_output_stream(self, manager):
        samplerate = self.settings.output_samplerate
        buffer_length = self.settings.output_buffer_length
        nchannels = self.settings.output_channels
        format = self.settings.output_format
        device = self.settings.output_device

        output_node = self.get_output_node()

        if self.settings.debug_timeit:
            output_ctxt = dn.timeit(output_node, " output")
        else:
            output_ctxt = nullcontext(output_node)

        with output_ctxt as output_node:
            stream_ctxt = dn.play(manager, output_node,
                                  samplerate=samplerate,
                                  buffer_shape=(buffer_length, nchannels),
                                  format=format,
                                  device=device,
                                  )

            with stream_ctxt as stream:
                yield stream

    def add_effect(self, node, zindex=(0,)):
        return self.effects_scheduler.add_node(node, zindex=zindex)

    def remove_effect(self, key):
        return self.effects_scheduler.remove_node(key)

class KnockDetector:
    def __init__(self, settings, knock_delay, knock_energy):
        self.settings = settings
        self.knock_delay = knock_delay
        self.knock_energy = knock_energy
        self.listeners_scheduler = dn.Scheduler()

    @dn.datanode
    def get_input_node(self):
        samplerate = self.settings.input_samplerate

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

        window = dn.get_half_Hann_window(win_length)
        onset = dn.pipe(
            dn.frame(win_length=win_length, hop_length=hop_length),
            dn.power_spectrum(win_length=win_length,
                              samplerate=samplerate,
                              windowing=window,
                              weighting=True),
            dn.onset_strength(1))
        picker = dn.pick_peak(pre_max, post_max, pre_avg, post_avg, wait, delta)

        with self.listeners_scheduler, onset, picker:
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

                self.listeners_scheduler.send((time, strength, detected))
                data = yield

                index += 1

    @contextlib.contextmanager
    def get_input_stream(self, manager):
        samplerate = self.settings.input_samplerate
        buffer_length = self.settings.input_buffer_length
        nchannels = self.settings.input_channels
        format = self.settings.input_format
        device = self.settings.input_device

        input_node = self.get_input_node()
        time_res = self.settings.detector_time_res
        hop_length = round(samplerate*time_res)
        if buffer_length != hop_length:
            input_node = dn.unchunk(input_node, chunk_shape=(hop_length, nchannels))
        if self.settings.debug_timeit:
            input_ctxt = dn.timeit(input_node, "  input")
        else:
            input_ctxt = nullcontext(input_node)

        with input_ctxt as input_node:
            stream_ctxt = dn.record(manager, input_node,
                                    samplerate=samplerate,
                                    buffer_shape=(buffer_length, nchannels),
                                    format=format,
                                    device=device,
                                    )

            with stream_ctxt as stream:
                yield stream

    def add_listener(self, node):
        node = dn.branch(node)
        return self.listeners_scheduler.add_node(node, (0,))

    def remove_listener(self, key):
        self.listeners_scheduler.remove_node(key)

class TerminalRenderer:
    def __init__(self, settings, display_delay):
        self.settings = settings
        self.display_delay = display_delay
        self.drawers_scheduler = dn.Scheduler()
        self.SIGWINCH_event = threading.Event()

    @dn.datanode
    def get_display_node(self):
        framerate = self.settings.display_framerate
        rows = self.settings.display_rows
        columns = self.settings.display_columns
        width = 0
        height = 0

        def SIGWINCH_handler(sig, frame):
            self.SIGWINCH_event.set()
        signal.signal(signal.SIGWINCH, SIGWINCH_handler)
        self.SIGWINCH_event.set()

        index = 0
        with self.drawers_scheduler:
            yield
            while True:
                if self.SIGWINCH_event.is_set():
                    self.SIGWINCH_event.clear()
                    size = shutil.get_terminal_size()
                    width = size.columns if columns == -1 else min(columns, size.columns)
                    height = size.lines if rows == -1 else min(rows, size.lines)

                time = index / framerate + self.display_delay
                view = [[" "]*width for _ in range(height)]
                _, view = self.drawers_scheduler.send((time, view))
                yield "\r" + "\n".join(map("".join, view)) + "\r" + (f"\x1b[{height-1}A" if height > 1 else "")
                index += 1

    @contextlib.contextmanager
    def get_display_thread(self):
        framerate = self.settings.display_framerate

        @dn.datanode
        def show():
            try:
                while True:
                    current_time, view = yield
                    if view and not self.SIGWINCH_event.is_set():
                        print(view, end="", flush=True)
            finally:
                print()

        display_node = self.get_display_node()
        display_ctxt = dn.timeit(display_node, "display") if self.settings.debug_timeit else nullcontext(display_node)
        with display_ctxt as display_node:
            with dn.interval(display_node, show(), 1/framerate) as thread:
                yield thread

    def add_drawer(self, node, zindex=(0,)):
        return self.drawers_scheduler.add_node(node, zindex=zindex)

    def remove_drawer(self, key):
        self.drawers_scheduler.remove_node(key)


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
    display_rows: int = 1
    display_columns: int = -1
    knock_delay: float = 0.0
    knock_energy: float = 1.0e-3
    sound_delay: float = 0.0

    # debug
    debug_timeit: bool = False

class KnockConsole:
    settings: KnockConsoleSettings = KnockConsoleSettings()

    def __init__(self, settings=None):
        if isinstance(settings, str):
            cfg.config_read(open(settings, 'r'), main=self.settings)

    @property
    def time(self):
        return time.time() - self._ref_time

    def run(self, knock_program):
        self.mixer = AudioMixer(self.settings, self.settings.sound_delay)
        self.detector = KnockDetector(self.settings, self.settings.knock_delay, self.settings.knock_energy)
        self.renderer = TerminalRenderer(self.settings, self.settings.display_delay)

        try:
            manager = pyaudio.PyAudio()

            # initialize audio/video streams
            with self.mixer.get_output_stream(manager) as output_stream,\
                 self.detector.get_input_stream(manager) as input_stream,\
                 self.renderer.get_display_thread() as display_thread:

                # activate audio/video streams
                self._ref_time = time.time()
                output_stream.start_stream()
                input_stream.start_stream()
                display_thread.start()

                # connect to program
                with knock_program.connect(self) as loop:
                    SIGINT_event = threading.Event()
                    def SIGINT_handler(sig, frame):
                        SIGINT_event.set()
                    signal.signal(signal.SIGINT, SIGINT_handler)

                    for _ in loop:
                        if SIGINT_event.is_set():
                            break

                        if (not output_stream.is_active()
                            or not input_stream.is_active()
                            or not display_thread.is_alive()):
                            break

                        signal.signal(signal.SIGINT, SIGINT_handler)

        finally:
            manager.terminate()

    def add_effect(self, node, time=None, zindex=(0,)):
        if time is not None:
            node = self._timed_effect(node, time)
        return self.mixer.add_effect(node, zindex)

    def remove_effect(self, key):
        self.mixer.remove_effect(key)

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
                node.send((time+offset/samplerate, dummy))
                offset += length

            while 0 < offset:
                if data.shape[0] < offset:
                    offset -= data.shape[0]
                else:
                    data1, data2 = data[:offset], data[offset:]
                    _, data2 = node.send((time+offset/samplerate, data2))
                    data = numpy.concatenate((data1, data2), axis=0)
                    offset = 0

                time, data = yield time, data

            while True:
                time, data = yield node.send((time, data))

    def play(self, node, samplerate=None, channels=None, volume=0.0, start=None, end=None, time=None, zindex=(0,)):
        if isinstance(node, str):
            node = dn.DataNode.wrap(self.load_sound(node))
            samplerate = None
        if channels is None:
            channels = self.settings.output_channels

        if start is not None or end is not None:
            node = dn.tslice(node, samplerate, start, end)
        node = dn.pipe(node, dn.rechannel(channels))
        if samplerate is not None and samplerate != self.settings.output_samplerate:
            node = dn.pipe(node, dn.resample(ratio=(self.settings.output_samplerate, samplerate)))
        if volume != 0:
            node = dn.pipe(node, lambda s: s * 10**(volume/20))
        node = dn.pair(lambda t:t, dn.attach(node))

        return self.add_effect(node, time=time, zindex=zindex)

    def add_listener(self, node):
        return self.detector.add_listener(node)

    def remove_listener(self, key):
        self.detector.remove_listener(key)

    def on_hit(self, func, time=None, duration=None):
        return self.add_listener(self._hit_listener(func, time, duration))

    @dn.datanode
    @staticmethod
    def _hit_listener(func, start_time, duration):
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

    def add_drawer(self, node, zindex=(0,)):
        return self.renderer.add_drawer(node, zindex)

    def add_text(self, text_node, y=0, x=0, ymask=slice(None,None), xmask=slice(None,None), zindex=(0,)):
        return self.renderer.add_drawer(self._text_drawer(text_node, y, x, ymask, xmask), zindex)

    def add_pad(self, pad_node, ymask=slice(None,None), xmask=slice(None,None), zindex=(0,)):
        return self.renderer.add_drawer(self._text_drawer(pad_node, ymask, xmask), zindex)

    def remove_drawer(self, key):
        self.renderer.remove_drawer(key)

    @dn.datanode
    @staticmethod
    def _text_drawer(text_node, y=0, x=0, ymask=slice(None,None), xmask=slice(None,None)):
        text_node = dn.DataNode.wrap(text_node)
        with text_node:
            time, view = yield
            while True:
                rows = len(view)
                columns = len(view[0]) if view else 0

                text = text_node.send((time, range(-y, rows-y)[ymask], range(-x, columns-x)[xmask]))
                view, y, x = tui.addtext(view, y, x, text, ymask=ymask, xmask=xmask)

                time, view = yield time, view

    @dn.datanode
    @staticmethod
    def _pad_drawer(pad_node, ymask=slice(None,None), xmask=slice(None,None)):
        pad_node = dn.DataNode.wrap(pad_node)
        with pad_node:
            time, view = yield
            while True:
                subview, y, x = tui.newpad(view, ymask=ymask, xmask=xmask)
                _, subview = pad_node.send((time, subview))
                view, yran, xran = tui.addpad(view, y, x, subview)

                time, view = yield time, view
