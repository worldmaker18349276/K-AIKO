import time
import itertools
import functools
import re
import contextlib
import queue
import threading
import signal
import numpy
import pyaudio
import audioread
from . import cfg
from . import datanodes as dn
from . import tui


@contextlib.contextmanager
def nullcontext(value):
    yield value

@contextlib.contextmanager
def prepare_pyaudio():
    try:
        manager = pyaudio.PyAudio()
        yield manager
    finally:
        manager.terminate()


class MixerSettings(metaclass=cfg.Configurable):
    output_device: int = -1
    output_samplerate: int = 44100
    output_buffer_length: int = 512*4
    output_channels: int = 1
    output_format: str = 'f4'

    sound_delay: float = 0.0

    debug_timeit: bool = False

class Mixer:
    def __init__(self, effects_scheduler, samplerate, buffer_length, nchannels):
        self.effects_scheduler = effects_scheduler
        self.samplerate = samplerate
        self.buffer_length = buffer_length
        self.nchannels = nchannels

    @staticmethod
    def get_node(scheduler, settings, manager, ref_time):
        samplerate = settings.output_samplerate
        buffer_length = settings.output_buffer_length
        nchannels = settings.output_channels
        format = settings.output_format
        device = settings.output_device
        sound_delay = settings.sound_delay
        debug_timeit = settings.debug_timeit

        @dn.datanode
        def _node():
            index = 0
            with scheduler:
                yield
                while True:
                    time = index * buffer_length / samplerate + sound_delay - ref_time
                    data = numpy.zeros((buffer_length, nchannels), dtype=numpy.float32)
                    data = scheduler.send((data, time))
                    yield data
                    index += 1

        output_node = _node()
        if debug_timeit:
            output_node = dn.timeit(output_node, lambda msg: print(" output: " + msg))

        return dn.play(manager, output_node,
                      samplerate=samplerate,
                      buffer_shape=(buffer_length, nchannels),
                      format=format,
                      device=device,
                      )

    @classmethod
    def create(clz, settings, manager, ref_time=0.0):
        samplerate = settings.output_samplerate
        buffer_length = settings.output_buffer_length
        nchannels = settings.output_channels

        scheduler = dn.Scheduler()
        output_node = clz.get_node(scheduler, settings, manager, ref_time)
        return output_node, clz(scheduler, samplerate, buffer_length, nchannels)

    def add_effect(self, node, time=None, zindex=(0,)):
        if time is not None:
            node = self.delay(node, time)
        return self.effects_scheduler.add_node(node, zindex=zindex)

    def remove_effect(self, key):
        return self.effects_scheduler.remove_node(key)

    @dn.datanode
    def delay(self, node, start_time):
        node = dn.DataNode.wrap(node)

        samplerate = self.samplerate
        buffer_length = self.buffer_length
        nchannels = self.nchannels

        with node:
            data, time = yield
            offset = round((start_time - time) * samplerate) if start_time is not None else 0

            while offset < 0:
                length = min(-offset, buffer_length)
                dummy = numpy.zeros((length, nchannels), dtype=numpy.float32)
                node.send((dummy, time+offset/samplerate))
                offset += length

            while 0 < offset:
                if data.shape[0] < offset:
                    offset -= data.shape[0]
                else:
                    data1, data2 = data[:offset], data[offset:]
                    data2 = node.send((data2, time+offset/samplerate))
                    data = numpy.concatenate((data1, data2), axis=0)
                    offset = 0

                data, time = yield data

            while True:
                data, time = yield node.send((data, time))

    def resample(self, node, samplerate=None, channels=None, volume=0.0, start=None, end=None):
        if channels is None:
            channels = self.nchannels

        if start is not None or end is not None:
            node = dn.tslice(node, samplerate, start, end)
        node = dn.pipe(node, dn.rechannel(channels))
        if samplerate is not None and samplerate != self.samplerate:
            node = dn.pipe(node, dn.resample(ratio=(self.samplerate, samplerate)))
        if volume != 0:
            node = dn.pipe(node, lambda s: s * 10**(volume/20))

        return node

    @functools.lru_cache(maxsize=32)
    def load_sound(self, filepath):
        return dn.load_sound(filepath, samplerate=self.samplerate)

    def play(self, node, samplerate=None, channels=None, volume=0.0, start=None, end=None, time=None, zindex=(0,)):
        if isinstance(node, str):
            node = dn.DataNode.wrap(self.load_sound(node))
            samplerate = None

        node = self.resample(node, samplerate, channels, volume, start, end)

        node = dn.pipe(lambda a:a[0], dn.attach(node))
        return self.add_effect(node, time=time, zindex=zindex)


class DetectorSettings(metaclass=cfg.Configurable):
    # input
    input_device: int = -1
    input_samplerate: int = 44100
    input_buffer_length: int = 512
    input_channels: int = 1
    input_format: str = 'f4'

    detector_time_res: float = 0.0116099773 # hop_length = 512 if samplerate == 44100
    detector_freq_res: float = 21.5332031 # win_length = 512*4 if samplerate == 44100
    detector_pre_max: float = 0.03
    detector_post_max: float = 0.03
    detector_pre_avg: float = 0.03
    detector_post_avg: float = 0.03
    detector_wait: float = 0.03
    detector_delta: float = 5.48e-6

    knock_delay: float = 0.0
    knock_energy: float = 1.0e-3

    debug_timeit: bool = False

class Detector:
    def __init__(self, listeners_scheduler):
        self.listeners_scheduler = listeners_scheduler

    @staticmethod
    def get_node(scheduler, settings, manager, ref_time):
        samplerate = settings.input_samplerate
        buffer_length = settings.input_buffer_length
        nchannels = settings.input_channels
        format = settings.input_format
        device = settings.input_device

        time_res = settings.detector_time_res
        freq_res = settings.detector_freq_res
        hop_length = round(samplerate*time_res)
        win_length = round(samplerate/freq_res)

        pre_max  = round(settings.detector_pre_max  / time_res)
        post_max = round(settings.detector_post_max / time_res)
        pre_avg  = round(settings.detector_pre_avg  / time_res)
        post_avg = round(settings.detector_post_avg / time_res)
        wait     = round(settings.detector_wait     / time_res)
        delta    =       settings.detector_delta

        knock_delay = settings.knock_delay
        knock_energy = settings.knock_energy

        debug_timeit = settings.debug_timeit

        @dn.datanode
        def _node():
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

            with scheduler, onset, picker:
                data = yield
                buffer = [(knock_delay, 0.0)]*prepare
                index = 0
                while True:
                    strength = onset.send(data)
                    detected = picker.send(strength)
                    time = index * hop_length / samplerate + knock_delay - ref_time
                    strength = strength / knock_energy

                    buffer.append((time, strength))
                    time, strength = buffer.pop(0)

                    scheduler.send((None, time, strength, detected))
                    data = yield

                    index += 1

        input_node = _node()
        if buffer_length != hop_length:
            input_node = dn.unchunk(input_node, chunk_shape=(hop_length, nchannels))
        if debug_timeit:
            input_node = dn.timeit(input_node, lambda msg: print("  input: " + msg))

        return dn.record(manager, input_node,
                        samplerate=samplerate,
                        buffer_shape=(buffer_length, nchannels),
                        format=format,
                        device=device,
                        )

    @classmethod
    def create(clz, settings, manager, ref_time=0.0):
        scheduler = dn.Scheduler()
        input_node = clz.get_node(scheduler, settings, manager, ref_time)
        return input_node, clz(scheduler)

    def add_listener(self, node):
        return self.listeners_scheduler.add_node(node, (0,))

    def remove_listener(self, key):
        self.listeners_scheduler.remove_node(key)

    def on_hit(self, func, time=None, duration=None):
        return self.add_listener(self._hit_listener(func, time, duration))

    @dn.datanode
    @staticmethod
    def _hit_listener(func, start_time, duration):
        _, time, strength, detected = yield
        if start_time is None:
            start_time = time

        while time < start_time:
            _, time, strength, detected = yield

        while duration is None or time < start_time + duration:
            if detected:
                finished = func(strength)
                if finished:
                    return

            _, time, strength, detected = yield


class RendererSettings(metaclass=cfg.Configurable):
    display_framerate: float = 160.0 # ~ 2 / detector_time_res
    display_delay: float = 0.0
    display_columns: int = -1

    debug_timeit: bool = False

class Renderer:
    def __init__(self, drawers_scheduler, msg_queue):
        self.drawers_scheduler = drawers_scheduler
        self.msg_queue = msg_queue

    @staticmethod
    def get_node(scheduler, msg_queue, settings, ref_time):
        framerate = settings.display_framerate
        display_delay = settings.display_delay
        columns = settings.display_columns
        debug_timeit = settings.debug_timeit

        @dn.datanode
        def _node():
            width = 0

            size_node = dn.terminal_size()

            index = 0
            with scheduler, size_node:
                yield
                while True:
                    size = size_node.send(None)
                    width = size.columns if columns == -1 else min(columns, size.columns)

                    time = index / framerate + display_delay - ref_time
                    view = tui.newwin1(width)
                    view = scheduler.send((view, time, width))

                    msg = []
                    while not msg_queue.empty():
                        msg.append(msg_queue.get())

                    msg = "\n" + "".join(msg) + "\n" if msg else ""
                    yield msg + "\r" + "".join(view) + "\r"
                    index += 1

        display_node = _node()
        if debug_timeit:
            display_node = dn.timeit(display_node, lambda msg: print("display: " + msg))
        return dn.interval(display_node, dn.show(), 1/framerate)

    @classmethod
    def create(clz, settings, ref_time=0.0):
        scheduler = dn.Scheduler()
        msg_queue = queue.Queue()
        display_node = clz.get_node(scheduler, msg_queue, settings, ref_time)
        return display_node, clz(scheduler, msg_queue)

    def message(self, msg):
        self.msg_queue.put(msg)

    def add_drawer(self, node, zindex=(0,)):
        return self.drawers_scheduler.add_node(node, zindex=zindex)

    def remove_drawer(self, key):
        self.drawers_scheduler.remove_node(key)

    def add_text(self, text_node, x=0, xmask=slice(None,None), zindex=(0,)):
        return self.add_drawer(self._text_drawer(text_node, x, xmask), zindex)

    def add_pad(self, pad_node, xmask=slice(None,None), zindex=(0,)):
        return self.add_drawer(self._pad_drawer(pad_node, xmask), zindex)

    @staticmethod
    @dn.datanode
    def _text_drawer(text_node, x=0, xmask=slice(None,None)):
        text_node = dn.DataNode.wrap(text_node)
        with text_node:
            view, time, width = yield
            while True:
                text = text_node.send((time, range(-x, width-x)[xmask]))
                view, _ = tui.addtext1(view, width, x, text, xmask=xmask)

                view, time, width = yield view

    @staticmethod
    @dn.datanode
    def _pad_drawer(pad_node, xmask=slice(None,None)):
        pad_node = dn.DataNode.wrap(pad_node)
        with pad_node:
            view, time, width = yield
            while True:
                subview, x, subwidth = tui.newpad1(view, width, xmask=xmask)
                subview = pad_node.send(((time, subwidth), subview))
                view, xran = tui.addpad1(view, width, x, subview, subwidth)

                view, time, width = yield view


class ControllerSettings(metaclass=cfg.Configurable):
    pass

class Controller:
    def __init__(self, handlers_scheduler):
        self.handlers_scheduler = handlers_scheduler

    @staticmethod
    def get_node(scheduler, settings, ref_time):
        @dn.datanode
        def _node():
            with scheduler:
                while True:
                    time, key = yield
                    time_ = time - ref_time
                    scheduler.send((None, time_, key))

        return dn.input(_node())

    @classmethod
    def create(clz, settings, ref_time=0.0):
        scheduler = dn.Scheduler()
        node = clz.get_node(scheduler, settings, ref_time)
        return node, clz(scheduler)

    def add_handler(self, node, key=None):
        if key is None:
            return self.handlers_scheduler.add_node(node, (0,))
        else:
            if isinstance(key, str):
                key = re.compile(re.escape(key))
            return self.handlers_scheduler.add_node(self._filter_node(node, key), (0,))

    def remove_handler(self, key):
        self.handlers_scheduler.remove_node(key)

    @dn.datanode
    def _filter_node(self, node, regex):
        node = dn.DataNode.wrap(node)
        with node:
            while True:
                _, t, key = yield
                if regex.fullmatch(key):
                    node.send((None, t, key))


class ClockSettings(metaclass=cfg.Configurable):
    tickrate: float = 60.0
    clock_delay: float = 0.0

class Clock:
    def __init__(self, coroutines_scheduler):
        self.coroutines_scheduler = coroutines_scheduler

    @staticmethod
    def get_node(scheduler, settings, ref_time):
        tickrate = settings.tickrate
        clock_delay = settings.clock_delay

        @dn.datanode
        def _node():
            index = 0
            with scheduler:
                yield
                while True:
                    time = index / tickrate + clock_delay - ref_time
                    scheduler.send((None, time))
                    yield
                    index += 1

        return dn.interval(consumer=_node(), dt=1/tickrate)

    @classmethod
    def create(clz, settings, ref_time=0.0):
        scheduler = dn.Scheduler()
        node = clz.get_node(scheduler, settings, ref_time)
        return node, clz(scheduler)

    def add_coroutine(self, node, time=None):
        if time is not None:
            node = self.schedule(node, time)
        return self.coroutines_scheduler.add_node(node, (0,))

    @dn.datanode
    def schedule(self, node, start_time):
        node = dn.DataNode.wrap(node)

        with node:
            _, time = yield

            while time < start_time:
                _, time = yield

            while True:
                node.send((None, time))
                _, time = yield

    def remove_coroutine(self, key):
        self.coroutines_scheduler.remove_node(key)

