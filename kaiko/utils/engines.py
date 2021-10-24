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

def validate_input_device(manager, device, samplerate, channels, format):
    if device == -1:
        device = manager.get_default_input_device_info()['index']

    format = {
        'f4': pyaudio.paFloat32,
        'i4': pyaudio.paInt32,
        'i2': pyaudio.paInt16,
        'i1': pyaudio.paInt8,
        'u1': pyaudio.paUInt8,
    }[format]

    manager.is_format_supported(samplerate,
        input_device=device, input_channels=channels, input_format=format)

def validate_output_device(manager, device, samplerate, channels, format):
    if device == -1:
        device = manager.get_default_output_device_info()['index']

    format = {
        'f4': pyaudio.paFloat32,
        'i4': pyaudio.paInt32,
        'i2': pyaudio.paInt16,
        'i1': pyaudio.paInt8,
        'u1': pyaudio.paUInt8,
    }[format]

    manager.is_format_supported(samplerate,
        output_device=device, output_channels=channels, output_format=format)


class MixerSettings(cfg.Configurable):
    r"""
    Fields
    ------
    output_device : int
        The index of output device, or -1 for default device.
    output_samplerate : int
        The samplerate of output device.
    output_buffer_length : int
        The buffer length of output device.
        Note that too large will affect the reaction time, but too small will affect the efficiency.
    output_channels : int
        The number of channels of output device.
    output_format : str
        The data format of output device.  The valid formats are 'f4', 'i4', 'i2', 'i1', 'u1'.

    sound_delay : float
        The delay of clock of the mixer.

    debug_timeit : bool
        Whether or not to record the execution time of the mixer.
        This is used for debugging.
    """
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
    def get_task(scheduler, settings, manager, ref_time):
        samplerate = settings.output_samplerate
        buffer_length = settings.output_buffer_length
        nchannels = settings.output_channels
        format = settings.output_format
        device = settings.output_device
        debug_timeit = settings.debug_timeit

        @dn.datanode
        def _node():
            index = 0
            with scheduler:
                yield
                while True:
                    time = index * buffer_length / samplerate + settings.sound_delay - ref_time
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
        task = clz.get_task(scheduler, settings, manager, ref_time)
        return task, clz(scheduler, samplerate, buffer_length, nchannels)

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
    def load_sound(self, filepath, stop_event=None):
        return dn.load_sound(filepath, channels=self.nchannels, samplerate=self.samplerate, stop_event=stop_event)

    def play(self, node, samplerate=None, channels=None, volume=0.0, start=None, end=None, time=None, zindex=(0,)):
        if isinstance(node, str):
            node = dn.DataNode.wrap(self.load_sound(node))
            samplerate = None
            channels = None

        node = self.resample(node, samplerate, channels, volume, start, end)

        node = dn.pipe(lambda a:a[0], dn.attach(node))
        return self.add_effect(node, time=time, zindex=zindex)


class DetectorSettings(cfg.Configurable):
    r"""
    Fields
    ------
    input_device : int
        The index of input device, or -1 for default device.
    input_samplerate : int
        The samplerate of input device.
    input_buffer_length : int
        The buffer length of input device.
        Note that too large will affect the reaction time, but too small will affect the efficiency.
    input_channels : int
        The number of channels of input device.
    input_format : str
        The data format of input device.  The valid formats are 'f4', 'i4', 'i2', 'i1', 'u1'.

    knock_delay : float
        The delay of clock of the detector.
    knock_energy : float
        The reference volume of the detector.

    debug_timeit : bool
        Whether or not to record the execution time of the detector.
        This is used for debugging.
    """
    input_device: int = -1
    input_samplerate: int = 44100
    input_buffer_length: int = 512
    input_channels: int = 1
    input_format: str = 'f4'

    class detect(cfg.Configurable):
        time_res: float = 0.0116099773 # hop_length = 512 if samplerate == 44100
        freq_res: float = 21.5332031 # win_length = 512*4 if samplerate == 44100
        pre_max: float = 0.03
        post_max: float = 0.03
        pre_avg: float = 0.03
        post_avg: float = 0.03
        wait: float = 0.03
        delta: float = 5.48e-6 # ~ noise_power * 20

    knock_delay: float = 0.0
    knock_energy: float = 1.0e-3 # ~ Dt / knock_max_energy

    debug_timeit: bool = False

class Detector:
    def __init__(self, listeners_scheduler):
        self.listeners_scheduler = listeners_scheduler

    @staticmethod
    def get_task(scheduler, settings, manager, ref_time):
        samplerate = settings.input_samplerate
        buffer_length = settings.input_buffer_length
        nchannels = settings.input_channels
        format = settings.input_format
        device = settings.input_device

        time_res = settings.detect.time_res
        hop_length = round(samplerate*time_res)

        debug_timeit = settings.debug_timeit

        input_node = Detector._detect_node(scheduler, ref_time, settings)
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

    @staticmethod
    @dn.datanode
    def _detect_node(scheduler, ref_time, settings):
        samplerate = settings.input_samplerate

        time_res = settings.detect.time_res
        freq_res = settings.detect.freq_res
        hop_length = round(samplerate*time_res)
        win_length = round(samplerate/freq_res)

        pre_max  = round(settings.detect.pre_max  / time_res)
        post_max = round(settings.detect.post_max / time_res)
        pre_avg  = round(settings.detect.pre_avg  / time_res)
        post_avg = round(settings.detect.post_avg / time_res)
        wait     = round(settings.detect.wait     / time_res)
        delta    =       settings.detect.delta

        prepare = max(post_max, post_avg)

        window = dn.get_half_Hann_window(win_length)
        onset = dn.pipe(
            dn.frame(win_length=win_length, hop_length=hop_length),
            dn.power_spectrum(win_length=win_length,
                              samplerate=samplerate,
                              windowing=window,
                              weighting=True),
            dn.onset_strength(1))
        delay = dn.delay((index * hop_length / samplerate + settings.knock_delay - ref_time, 0.0)
                         for index in range(-prepare, 0))
        picker = dn.pick_peak(pre_max, post_max, pre_avg, post_avg, wait, delta)

        with scheduler, onset, delay, picker:
            data = yield
            index = 0
            while True:
                try:
                    strength = onset.send(data)
                except StopIteration:
                    return

                time = index * hop_length / samplerate + settings.knock_delay - ref_time
                normalized_strength = strength / settings.knock_energy
                try:
                    time, normalized_strength = delay.send((time, normalized_strength))
                except StopIteration:
                    return

                try:
                    detected = picker.send(strength)
                except StopIteration:
                    return

                try:
                    scheduler.send((None, time, normalized_strength, detected))
                except StopIteration:
                    return
                data = yield

                index += 1

    @classmethod
    def create(clz, settings, manager, ref_time=0.0):
        scheduler = dn.Scheduler()
        task = clz.get_task(scheduler, settings, manager, ref_time)
        return task, clz(scheduler)

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

def print_below(text, width, height, tabsize=8):
    r"""Print the given text below the current position (GNOME terminal).

    Parameters
    ----------
    text : str
        The string to print.
    width : int
        The width of terminal.
    height : int
        The height of terminal.
    tabsize : int, optional
        The tab size of terminal.

    Returns
    -------
    res : str
    """
    if height == 1:
        return ""

    x = 0
    y = 0

    res = []
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

        if y < height-1:
            res.append(ch)
        else:
            y = height-2
            break

    return "\n" + "".join(res) + f"\x1b[m\x1b[{y+1}A"

class RendererSettings(cfg.Configurable):
    r"""
    Fields
    ------
    display_framerate : float
        The framerate of the renderer.
    display_delay : float
        The delay of clock of the renderer.
    resize_delay : float
        The delay time to redraw display after resizing.

    debug_timeit : bool
        Whether or not to record the execution time of the renderer.
        This is used for debugging.
    """
    display_framerate: float = 160.0 # ~ 2 / detect.time_res
    display_delay: float = 0.0
    resize_delay: float = 0.5

    debug_timeit: bool = False

class Renderer:
    def __init__(self, drawers_scheduler):
        self.drawers_scheduler = drawers_scheduler

    @staticmethod
    def get_task(scheduler, settings, ref_time):
        framerate = settings.display_framerate
        debug_timeit = settings.debug_timeit
        resize_delay = settings.resize_delay

        @dn.datanode
        def _node():
            size_node = dn.terminal_size()

            curr_msg = ""
            index = 0
            width = 0
            resize_time = None
            with scheduler, size_node:
                shown = yield
                while True:
                    time = index / framerate + settings.display_delay - ref_time

                    try:
                        size = size_node.send(None)
                    except StopIteration:
                        return

                    if resize_time is not None:
                        if size.columns < width:
                            resize_time = time
                        if time < resize_time + resize_delay:
                            yield "\r\x1b[Kresizing...\r"
                            index += 1
                            width = size.columns
                            continue

                    elif size.columns < width:
                        resize_time = time
                        yield "\n\x1b[Jresizing...\r"
                        index += 1
                        width = size.columns
                        continue

                    width = size.columns
                    view = wcb.newwin1(width)
                    msg = ""
                    try:
                        view, msg = scheduler.send(((view, msg), time, width))
                    except StopIteration:
                        return

                    # track changes of the message
                    if resize_time is None and curr_msg == msg:
                        res_text = "\r\x1b[K" + "".join(view).rstrip() + "\r"
                    elif msg == "":
                        res_text = "\r\x1b[J" + "".join(view).rstrip() + "\r"
                    else:
                        res_text = "\r\x1b[J" + "".join(view).rstrip() + print_below(msg, width, size.lines) + "\r"

                    if resize_time is not None:
                        res_text = "\x1b[2J\x1b[H" + res_text

                    shown = yield res_text
                    if shown:
                        curr_msg = msg
                        resize_time = None
                    index += 1

        display_node = _node()
        if debug_timeit:
            display_node = dn.timeit(display_node, lambda msg: print("display: " + msg))
        return dn.show(display_node, 1/framerate, hide_cursor=True)

    @classmethod
    def create(clz, settings, ref_time=0.0):
        scheduler = dn.Scheduler()
        task = clz.get_task(scheduler, settings, ref_time)
        return task, clz(scheduler)

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
        while True:
            (view, premsg), _, _ = yield (view, premsg+msg)

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
    r"""
    Fields
    ------
    keycodes : dict
        The maps from keycodes to keynames.
    """
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
    def get_task(scheduler, settings, ref_time):
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
                        keyname = repr(keycode)

                    time_ = time - ref_time
                    try:
                        scheduler.send((None, time_, keyname, keycode))
                    except StopIteration:
                        return

        return dn.input(_node())

    @classmethod
    def create(clz, settings, ref_time=0.0):
        scheduler = dn.Scheduler()
        task = clz.get_task(scheduler, settings, ref_time)
        return task, clz(scheduler)

    def add_handler(self, node, keyname=None):
        if keyname is None:
            return self.handlers_scheduler.add_node(dn.DataNode.wrap(node), (0,))
        else:
            return self.handlers_scheduler.add_node(self._filter_node(node, keyname), (0,))

    def remove_handler(self, key):
        self.handlers_scheduler.remove_node(key)

    @dn.datanode
    def _filter_node(self, node, name):
        node = dn.DataNode.wrap(node)
        with node:
            while True:
                _, t, keyname, keycode = yield
                if name == keyname:
                    try:
                        node.send((None, t, keyname, keycode))
                    except StopIteration:
                        return


class DevicesSettings(cfg.Configurable):
    mixer = MixerSettings
    detector = DetectorSettings
    renderer = RendererSettings
    controller = ControllerSettings
