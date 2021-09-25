import time
import itertools
import functools
import re
import contextlib
from typing import Dict
import queue
import threading
import signal
import numpy
import pyaudio
import audioread
from . import config as cfg
from . import datanodes as dn
from . import wcbuffers as wcb


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


class MixerSettings(cfg.Configurable):
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
                    try:
                        data = scheduler.send((data, time))
                    except StopIteration:
                        return
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
                try:
                    node.send((dummy, time+offset/samplerate))
                except StopIteration:
                    return
                offset += length

            while 0 < offset:
                if data.shape[0] < offset:
                    offset -= data.shape[0]
                else:
                    data1, data2 = data[:offset], data[offset:]
                    try:
                        data2 = node.send((data2, time+offset/samplerate))
                    except StopIteration:
                        return
                    data = numpy.concatenate((data1, data2), axis=0)
                    offset = 0

                data, time = yield data

            while True:
                try:
                    data, time = yield node.send((data, time))
                except StopIteration:
                    return

    def resample(self, node, samplerate=None, channels=None, volume=0.0, start=None, end=None):
        if start is not None or end is not None:
            node = dn.tslice(node, samplerate or self.samplerate, start, end)
        if channels is not None and channels != self.nchannels:
            node = dn.pipe(node, dn.rechannel(self.nchannels))
        if samplerate is not None and samplerate != self.samplerate:
            node = dn.pipe(node, dn.resample(ratio=(self.samplerate, samplerate)))
        if volume != 0:
            node = dn.pipe(node, lambda s: s * 10**(volume/20))

        return node

    @functools.lru_cache(maxsize=32)
    def load_sound(self, filepath):
        return dn.load_sound(filepath, channels=self.nchannels, samplerate=self.samplerate)

    def play(self, node, samplerate=None, channels=None, volume=0.0, start=None, end=None, time=None, zindex=(0,)):
        if isinstance(node, str):
            node = dn.DataNode.wrap(self.load_sound(node))
            samplerate = None
            channels = None

        node = self.resample(node, samplerate, channels, volume, start, end)

        node = dn.pipe(lambda a:a[0], dn.attach(node))
        return self.add_effect(node, time=time, zindex=zindex)


class DetectorSettings(cfg.Configurable):
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
                    try:
                        strength = onset.send(data)
                        detected = picker.send(strength)
                    except StopIteration:
                        return
                    time = index * hop_length / samplerate + knock_delay - ref_time
                    strength = strength / knock_energy

                    buffer.append((time, strength))
                    time, strength = buffer.pop(0)

                    try:
                        scheduler.send((None, time, strength, detected))
                    except StopIteration:
                        return
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


def pt_walk(text, width, x=0, tabsize=8):
    r"""Predict the position after print the given text in the terminal (GNOME terminal).

    Parameters
    ----------
    text : str
        The string to print.
    width : int
        The width of terminal.
    x : int, optional
        The initial position before printing.
    tabsize : int, optional
        The tab size of terminal.

    Returns
    -------
    x : int
    y : int
        The final position after printing.
    """
    y = 0

    for ch, w in wcb.parse_attr(text):
        if ch == "\t":
            if tabsize > 0 and x < width:
                x = min((x+1) // -tabsize * -tabsize, width-1)

        elif ch == "\b":
            x = max(min(x, width-1)-1, 0)

        elif ch == "\r":
            x = 0

        elif ch == "\n":
            y += 1
            x = 0

        elif ch == "\v":
            y += 1

        elif ch == "\f":
            y += 1

        elif ch == "\x00":
            pass

        elif ch[0] == "\x1b":
            pass

        else:
            x += w
            if x > width:
                y += 1
                x = w

    return x, y

class RendererSettings(cfg.Configurable):
    display_framerate: float = 160.0 # ~ 2 / detector_time_res
    display_delay: float = 0.0
    display_columns: int = -1

    debug_timeit: bool = False

class Renderer:
    def __init__(self, drawers_scheduler):
        self.drawers_scheduler = drawers_scheduler

    @staticmethod
    def get_node(scheduler, settings, ref_time):
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
                    try:
                        size = size_node.send(None)
                    except StopIteration:
                        return
                    width = size.columns if columns == -1 else min(columns, size.columns)

                    time = index / framerate + display_delay - ref_time
                    view = wcb.newwin1(width)
                    msg = None
                    try:
                        view, msg = scheduler.send(((view, msg), time, width))
                    except StopIteration:
                        return

                    if msg is None:
                        res_text = "\r" + "".join(view) + "\r"
                    elif msg == "":
                        res_text = "\r\x1b[J" + "".join(view) + "\r"
                    else:
                        _, y = pt_walk(msg, width, 0)
                        res_text = "\r\x1b[J" + "".join(view) + f"\n{msg}\x1b[{y+1}A\r"

                    yield res_text
                    index += 1

        display_node = _node()
        if debug_timeit:
            display_node = dn.timeit(display_node, lambda msg: print("display: " + msg))
        return dn.show(display_node, 1/framerate, hide_cursor=True)

    @classmethod
    def create(clz, settings, ref_time=0.0):
        scheduler = dn.Scheduler()
        display_node = clz.get_node(scheduler, settings, ref_time)
        return display_node, clz(scheduler)

    def add_drawer(self, node, zindex=(0,)):
        return self.drawers_scheduler.add_node(node, zindex=zindex)

    def remove_drawer(self, key):
        self.drawers_scheduler.remove_node(key)

    def add_message(self, msg, zindex=(0,)):
        return self.add_drawer(self._msg_drawer(msg), zindex)

    def add_text(self, text_node, x=0, xmask=slice(None,None), zindex=(0,)):
        return self.add_drawer(self._text_drawer(text_node, x, xmask), zindex)

    def add_pad(self, pad_node, xmask=slice(None,None), zindex=(0,)):
        return self.add_drawer(self._pad_drawer(pad_node, xmask), zindex)

    @staticmethod
    @dn.datanode
    def _msg_drawer(msg):
        (view, premsg), _, _ = yield
        msg = msg if premsg is None else premsg + msg
        yield (view, msg)

    @staticmethod
    @dn.datanode
    def _text_drawer(text_node, x=0, xmask=slice(None,None)):
        text_node = dn.DataNode.wrap(text_node)
        with text_node:
            (view, msg), time, width = yield
            while True:
                try:
                    text = text_node.send((time, range(-x, width-x)[xmask]))
                except StopIteration:
                    return
                view, _ = wcb.addtext1(view, width, x, text, xmask=xmask)

                (view, msg), time, width = yield (view, msg)

    @staticmethod
    @dn.datanode
    def _pad_drawer(pad_node, xmask=slice(None,None)):
        pad_node = dn.DataNode.wrap(pad_node)
        with pad_node:
            (view, msg), time, width = yield
            while True:
                subview, x, subwidth = wcb.newpad1(view, width, xmask=xmask)
                try:
                    subview = pad_node.send(((time, subwidth), subview))
                except StopIteration:
                    return
                view, xran = wcb.addpad1(view, width, x, subview, subwidth)

                (view, msg), time, width = yield (view, msg)


class ControllerSettings(cfg.Configurable):
    keycodes: Dict[str, str] = {
        "\x1b"       : "Esc",
        "\x1b\x1b"   : "Alt_Esc",

        "\n"         : "Enter",
        "\x1b\n"     : "Alt_Enter",

        "\x7f"       : "Backspace",
        "\x08"       : "Ctrl_Backspace",
        "\x1b\x7f"   : "Alt_Backspace",
        "\x1b\x08"   : "Ctrl_Alt_Backspace",

        "\t"         : "Tab",
        "\x1b[Z"     : "Shift_Tab",
        "\x1b\t"     : "Alt_Tab",
        "\x1b\x1b[Z" : "Alt_Shift_Tab",

        "\x1b[A"     : "Up",
        "\x1b[1;2A"  : "Shift_Up",
        "\x1b[1;3A"  : "Alt_Up",
        "\x1b[1;4A"  : "Alt_Shift_Up",
        "\x1b[1;5A"  : "Ctrl_Up",
        "\x1b[1;6A"  : "Ctrl_Shift_Up",
        "\x1b[1;7A"  : "Ctrl_Alt_Up",
        "\x1b[1;8A"  : "Ctrl_Alt_Shift_Up",

        "\x1b[B"     : "Down",
        "\x1b[1;2B"  : "Shift_Down",
        "\x1b[1;3B"  : "Alt_Down",
        "\x1b[1;4B"  : "Alt_Shift_Down",
        "\x1b[1;5B"  : "Ctrl_Down",
        "\x1b[1;6B"  : "Ctrl_Shift_Down",
        "\x1b[1;7B"  : "Ctrl_Alt_Down",
        "\x1b[1;8B"  : "Ctrl_Alt_Shift_Down",

        "\x1b[C"     : "Right",
        "\x1b[1;2C"  : "Shift_Right",
        "\x1b[1;3C"  : "Alt_Right",
        "\x1b[1;4C"  : "Alt_Shift_Right",
        "\x1b[1;5C"  : "Ctrl_Right",
        "\x1b[1;6C"  : "Ctrl_Shift_Right",
        "\x1b[1;7C"  : "Ctrl_Alt_Right",
        "\x1b[1;8C"  : "Ctrl_Alt_Shift_Right",

        "\x1b[D"     : "Left",
        "\x1b[1;2D"  : "Shift_Left",
        "\x1b[1;3D"  : "Alt_Left",
        "\x1b[1;4D"  : "Alt_Shift_Left",
        "\x1b[1;5D"  : "Ctrl_Left",
        "\x1b[1;6D"  : "Ctrl_Shift_Left",
        "\x1b[1;7D"  : "Ctrl_Alt_Left",
        "\x1b[1;8D"  : "Ctrl_Alt_Shift_Left",

        "\x1b[F"     : "End",
        "\x1b[1;2F"  : "Shift_End",
        "\x1b[1;3F"  : "Alt_End",
        "\x1b[1;4F"  : "Alt_Shift_End",
        "\x1b[1;5F"  : "Ctrl_End",
        "\x1b[1;6F"  : "Ctrl_Shift_End",
        "\x1b[1;7F"  : "Ctrl_Alt_End",
        "\x1b[1;8F"  : "Ctrl_Alt_Shift_End",

        "\x1b[H"     : "Home",
        "\x1b[1;2H"  : "Shift_Home",
        "\x1b[1;3H"  : "Alt_Home",
        "\x1b[1;4H"  : "Alt_Shift_Home",
        "\x1b[1;5H"  : "Ctrl_Home",
        "\x1b[1;6H"  : "Ctrl_Shift_Home",
        "\x1b[1;7H"  : "Ctrl_Alt_Home",
        "\x1b[1;8H"  : "Ctrl_Alt_Shift_Home",

        "\x1b[2~"    : "Insert",
        "\x1b[2;2~"  : "Shift_Insert",
        "\x1b[2;3~"  : "Alt_Insert",
        "\x1b[2;4~"  : "Alt_Shift_Insert",
        "\x1b[2;5~"  : "Ctrl_Insert",
        "\x1b[2;6~"  : "Ctrl_Shift_Insert",
        "\x1b[2;7~"  : "Ctrl_Alt_Insert",
        "\x1b[2;8~"  : "Ctrl_Alt_Shift_Insert",

        "\x1b[3~"    : "Delete",
        "\x1b[3;2~"  : "Shift_Delete",
        "\x1b[3;3~"  : "Alt_Delete",
        "\x1b[3;4~"  : "Alt_Shift_Delete",
        "\x1b[3;5~"  : "Ctrl_Delete",
        "\x1b[3;6~"  : "Ctrl_Shift_Delete",
        "\x1b[3;7~"  : "Ctrl_Alt_Delete",
        "\x1b[3;8~"  : "Ctrl_Alt_Shift_Delete",

        "\x1b[5~"    : "PageUp",
        "\x1b[5;2~"  : "Shift_PageUp",
        "\x1b[5;3~"  : "Alt_PageUp",
        "\x1b[5;4~"  : "Alt_Shift_PageUp",
        "\x1b[5;5~"  : "Ctrl_PageUp",
        "\x1b[5;6~"  : "Ctrl_Shift_PageUp",
        "\x1b[5;7~"  : "Ctrl_Alt_PageUp",
        "\x1b[5;8~"  : "Ctrl_Alt_Shift_PageUp",

        "\x1b[6~"    : "PageDown",
        "\x1b[6;2~"  : "Shift_PageDown",
        "\x1b[6;3~"  : "Alt_PageDown",
        "\x1b[6;4~"  : "Alt_Shift_PageDown",
        "\x1b[6;5~"  : "Ctrl_PageDown",
        "\x1b[6;6~"  : "Ctrl_Shift_PageDown",
        "\x1b[6;7~"  : "Ctrl_Alt_PageDown",
        "\x1b[6;8~"  : "Ctrl_Alt_Shift_PageDown",
    }

class Controller:
    def __init__(self, handlers_scheduler):
        self.handlers_scheduler = handlers_scheduler

    @staticmethod
    def get_node(scheduler, settings, ref_time):
        keycodes = settings.keycodes
        @dn.datanode
        def _node():
            with scheduler:
                while True:
                    time, keycode = yield

                    if keycode in keycodes:
                        keyname = keycodes[keycode]
                    elif keycode.isprintable():
                        keyname = "PRINTABLE"
                    else:
                        keyname = None

                    time_ = time - ref_time
                    try:
                        scheduler.send((None, time_, keyname, keycode))
                    except StopIteration:
                        return

        return dn.input(_node())

    @classmethod
    def create(clz, settings, ref_time=0.0):
        scheduler = dn.Scheduler()
        node = clz.get_node(scheduler, settings, ref_time)
        return node, clz(scheduler)

    def add_handler(self, node, key=None):
        if key is None:
            return self.handlers_scheduler.add_node(dn.DataNode.wrap(node), (0,))
        else:
            return self.handlers_scheduler.add_node(self._filter_node(node, key), (0,))

    def remove_handler(self, key):
        self.handlers_scheduler.remove_node(key)

    @dn.datanode
    def _filter_node(self, node, key):
        node = dn.DataNode.wrap(node)
        with node:
            while True:
                _, t, keyname, keycode = yield
                if key == keyname:
                    try:
                        node.send((None, t, keyname, keycode))
                    except StopIteration:
                        return


class ClockSettings(cfg.Configurable):
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
                    try:
                        scheduler.send((None, time))
                    except StopIteration:
                        return
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
                try:
                    node.send((None, time))
                except StopIteration:
                    return
                _, time = yield

    def remove_coroutine(self, key):
        self.coroutines_scheduler.remove_node(key)

