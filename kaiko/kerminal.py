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

class KerminalMixer:
    def __init__(self, effects_scheduler, samplerate, buffer_length, nchannels):
        self.effects_scheduler = effects_scheduler
        self.samplerate = samplerate
        self.buffer_length = buffer_length
        self.nchannels = nchannels

    @staticmethod
    @contextlib.contextmanager
    def get_node(scheduler, settings, ref_time):
        samplerate = settings.output_samplerate
        buffer_length = settings.output_buffer_length
        nchannels = settings.output_channels
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

        output_ctxt = dn.timeit(_node(), " output") if debug_timeit else nullcontext(_node())
        with output_ctxt as output_node:
            yield output_node

    @classmethod
    @contextlib.contextmanager
    def create(clz, settings, ref_time=0.0):
        samplerate = settings.output_samplerate
        buffer_length = settings.output_buffer_length
        nchannels = settings.output_channels

        scheduler = dn.Scheduler()
        with clz.get_node(scheduler, settings, ref_time) as output_node:
            yield output_node, clz(scheduler, samplerate, buffer_length, nchannels)

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

class KerminalDetector:
    def __init__(self, listeners_scheduler):
        self.listeners_scheduler = listeners_scheduler

    @staticmethod
    @contextlib.contextmanager
    def get_node(scheduler, settings, ref_time):
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

        _node = _node()

        if buffer_length != hop_length:
            _node = dn.unchunk(_node, chunk_shape=(hop_length, nchannels))
        input_ctxt = dn.timeit(_node, "  input") if debug_timeit else nullcontext(_node)
        with input_ctxt as input_node:
            yield input_node

    @classmethod
    @contextlib.contextmanager
    def create(clz, settings, ref_time=0.0):
        scheduler = dn.Scheduler()
        with clz.get_node(scheduler, settings, ref_time) as input_node:
            yield input_node, clz(scheduler)

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

class KerminalRenderer:
    def __init__(self, drawers_scheduler, msg_queue):
        self.drawers_scheduler = drawers_scheduler
        self.msg_queue = msg_queue

    @staticmethod
    @contextlib.contextmanager
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

        display_ctxt = dn.timeit(_node(), "display") if debug_timeit else nullcontext(_node())
        with display_ctxt as display_node:
            yield display_node

    @classmethod
    @contextlib.contextmanager
    def create(clz, settings, ref_time=0.0):
        scheduler = dn.Scheduler()
        msg_queue = queue.Queue()
        with clz.get_node(scheduler, msg_queue, settings, ref_time) as display_node:
            yield display_node, clz(scheduler, msg_queue)

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

class KerminalController:
    def __init__(self, handlers_scheduler):
        self.handlers_scheduler = handlers_scheduler

    @staticmethod
    @contextlib.contextmanager
    def get_node(scheduler, settings, ref_time):
        @dn.datanode
        def _node():
            with scheduler:
                while True:
                    time, key = yield
                    time_ = time - ref_time
                    scheduler.send((None, time_, key))

        yield _node()

    @classmethod
    @contextlib.contextmanager
    def create(clz, settings, ref_time=0.0):
        scheduler = dn.Scheduler()
        with clz.get_node(scheduler, settings, ref_time) as node:
            yield node, clz(scheduler)

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

class KerminalClock:
    def __init__(self, coroutines_scheduler):
        self.coroutines_scheduler = coroutines_scheduler

    @staticmethod
    @contextlib.contextmanager
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

        yield _node()

    @classmethod
    @contextlib.contextmanager
    def create(clz, settings, ref_time=0.0):
        scheduler = dn.Scheduler()
        with clz.get_node(scheduler, settings, ref_time) as node:
            yield node, clz(scheduler)

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

class Kerminal:
    def __init__(self, clock, mixer, detector, renderer, controller):
        self.clock = clock
        self.mixer = mixer
        self.detector = detector
        self.renderer = renderer
        self.controller = controller

    @classmethod
    @contextlib.contextmanager
    def create(clz, settings, ref_time=0.0):
        with KerminalClock.create(settings, ref_time) as (tick_node, clock),\
             KerminalMixer.create(settings, ref_time) as (audioout_node, mixer),\
             KerminalDetector.create(settings, ref_time) as (audioin_node, detector),\
             KerminalRenderer.create(settings, ref_time) as (textout_node, renderer),\
             KerminalController.create(settings, ref_time) as (textin_node, controller):

            # initialize kerminal
            kerminal = clz(clock, mixer, detector, renderer, controller)
            yield tick_node, audioout_node, audioin_node, textout_node, textin_node, kerminal


def until_interrupt(dt=0.005):
    SIGINT_event = threading.Event()
    def SIGINT_handler(sig, frame):
        SIGINT_event.set()
    signal.signal(signal.SIGINT, SIGINT_handler)

    while True:
        yield

        if SIGINT_event.wait(dt):
            break

@contextlib.contextmanager
def prepare_pyaudio():
    try:
        manager = pyaudio.PyAudio()
        yield manager
    finally:
        manager.terminate()

@cfg.configurable
class KerminalSettings:
    # input
    input_device: int = -1
    input_samplerate: int = 44100
    input_buffer_length: int = 512
    input_channels: int = 1
    input_format: str = 'f4'

    # output
    output_device: int = -1
    output_samplerate: int = 44100
    output_buffer_length: int = 512*4
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
    display_columns: int = -1
    knock_delay: float = 0.0
    knock_energy: float = 1.0e-3
    sound_delay: float = 0.0
    tickrate: float = 60.0
    clock_delay: float = 0.0

    # debug
    debug_timeit: bool = False

class KerminalLaucher:
    @classmethod
    def execute(clz, knockable, settings=None):
        if isinstance(settings, str):
            with open(settings, 'r') as file:
                settings = KerminalSettings()
                cfg.config_read(file, main=settings)

        with prepare_pyaudio() as manager:
            # connect to knockable
            with knockable.connect(settings) as (tick_node, audioout_node, audioin_node, textout_node, textin_node, main):

                # initialize audio/text streams
                with clz.get_tick_thread(tick_node, settings) as tick_thread,\
                     clz.get_audioout_stream(audioout_node, manager, settings) as audioout_stream,\
                     clz.get_audioin_stream(audioin_node, manager, settings) as audioin_stream,\
                     clz.get_textout_thread(textout_node, settings) as textout_thread,\
                     clz.get_textin_thread(textin_node, settings) as textin_thread:

                    # activate audio/text streams
                    tick_thread.start()
                    audioout_stream.start_stream()
                    audioin_stream.start_stream()
                    textout_thread.start()
                    textin_thread.start()

                    main.start()

                    for _ in until_interrupt():
                        if (not main.is_alive()
                            or not tick_thread.is_alive()
                            or not audioout_stream.is_active()
                            or not audioin_stream.is_active()
                            or not textout_thread.is_alive()
                            or not textin_thread.is_alive()):
                            break

    @staticmethod
    @contextlib.contextmanager
    def get_tick_thread(node, settings):
        tickrate = settings.tickrate

        with dn.interval(consumer=node, dt=1/tickrate) as thread:
            yield thread

    @staticmethod
    @contextlib.contextmanager
    def get_audioout_stream(node, manager, settings):
        samplerate = settings.output_samplerate
        buffer_length = settings.output_buffer_length
        nchannels = settings.output_channels
        format = settings.output_format
        device = settings.output_device

        stream_ctxt = dn.play(manager, node,
                              samplerate=samplerate,
                              buffer_shape=(buffer_length, nchannels),
                              format=format,
                              device=device,
                              )

        with stream_ctxt as stream:
            yield stream

    @staticmethod
    @contextlib.contextmanager
    def get_audioin_stream(node, manager, settings):
        samplerate = settings.input_samplerate
        buffer_length = settings.input_buffer_length
        nchannels = settings.input_channels
        format = settings.input_format
        device = settings.input_device

        stream_ctxt = dn.record(manager, node,
                                samplerate=samplerate,
                                buffer_shape=(buffer_length, nchannels),
                                format=format,
                                device=device,
                                )

        with stream_ctxt as stream:
            yield stream

    @staticmethod
    @contextlib.contextmanager
    def get_textout_thread(node, settings):
        framerate = settings.display_framerate

        with dn.interval(node, dn.show(), 1/framerate) as thread:
            yield thread

    @staticmethod
    @contextlib.contextmanager
    def get_textin_thread(node, settings):
        with dn.input(node) as thread:
            yield thread
