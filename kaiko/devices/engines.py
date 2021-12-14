import time
import bisect
import functools
import contextlib
import numpy
import audioread
from ..utils import config as cfg
from ..utils import datanodes as dn
from ..utils import markups as mu
from . import terminals as term
from . import audios as aud
from . import loggers as log


class Monitor:
    def __init__(self, path, N=10):
        self.path = path
        self.N = N

        # state
        self.count = None
        self.eff = None
        self.avg = None
        self.best = None
        self.worst = None

        # statistics
        self.total_avg = None
        self.total_dev = None
        self.total_eff = None

    @dn.datanode
    def monitoring(self, node):
        if hasattr(time, 'thread_time'):
            get_time = time.thread_time
        elif hasattr(time, 'clock_gettime'):
            get_time = lambda: time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)
        else:
            get_time = time.perf_counter

        N = self.N
        start = prev = 0.0
        stop = numpy.inf
        self.count = 0
        total = 0.0
        total2 = 0.0
        spend_N = [0.0]*N
        recent_N = [0.0]*N
        best_N = [numpy.inf]*N
        worst_N = [-numpy.inf]*N

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.path, 'w') as file:
            with node:
                try:
                    data = yield

                    start = stop = prev = time.perf_counter()

                    while True:

                        t0 = get_time()
                        data = node.send(data)
                        t = get_time() - t0
                        stop = time.perf_counter()
                        spend = stop - prev
                        prev = stop
                        print(f"{spend}\t{t}", file=file)

                        self.count += 1
                        total += t
                        total2 += t**2
                        spend_N.insert(0, spend)
                        spend_N.pop()
                        recent_N.insert(0, t)
                        recent_N.pop()
                        bisect.insort_left(best_N, t)
                        best_N.pop()
                        bisect.insort(worst_N, t)
                        worst_N.pop(0)
                        self.avg = sum(recent_N)/N
                        self.eff = sum(recent_N)/sum(spend_N)
                        self.best = sum(best_N)/N
                        self.worst = sum(worst_N)/N

                        data = yield data

                except StopIteration:
                    return

                finally:
                    stop = time.perf_counter()

                    if self.count > 0:
                        self.total_avg = total/self.count
                        self.total_dev = (total2/self.count - self.total_avg**2)**0.5
                        self.total_eff = total/(stop - start)

    def __str__(self):
        if self.count is None:
            return f"UNINITIALIZED"

        if self.total_avg is None:
            return f"count={self.count}"

        if self.best is None or self.best == float('inf'):
            return f"count={self.count}, avg={self.total_avg*1000:5.3f}±{self.total_dev*1000:5.3f}ms ({self.total_eff: >6.1%})"

        return (f"count={self.count}, avg={self.total_avg*1000:5.3f}±{self.total_dev*1000:5.3f}ms"
                f" ({self.best*1000:5.3f}ms ~ {self.worst*1000:5.3f}ms) ({self.total_eff: >6.1%})")


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
    """
    output_device: int = -1
    output_samplerate: int = 44100
    output_buffer_length: int = 512*4
    output_channels: int = 1
    output_format: str = 'f4'

    sound_delay: float = 0.0

class Mixer:
    def __init__(self, effects_scheduler, samplerate, buffer_length, nchannels, monitor):
        self.effects_scheduler = effects_scheduler
        self.samplerate = samplerate
        self.buffer_length = buffer_length
        self.nchannels = nchannels
        self.monitor = monitor

    @staticmethod
    def get_task(scheduler, settings, manager, ref_time, monitor):
        samplerate = settings.output_samplerate
        buffer_length = settings.output_buffer_length
        nchannels = settings.output_channels
        format = settings.output_format
        device = settings.output_device

        output_node = Mixer._mix_node(scheduler, settings, ref_time)
        if monitor:
            output_node = monitor.monitoring(output_node)

        return aud.play(manager, output_node,
                        samplerate=samplerate,
                        buffer_shape=(buffer_length, nchannels),
                        format=format,
                        device=device,
                        )

    @staticmethod
    @dn.datanode
    def _mix_node(scheduler, settings, ref_time):
        samplerate = settings.output_samplerate
        buffer_length = settings.output_buffer_length
        nchannels = settings.output_channels

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

    @classmethod
    def create(clz, settings, manager, ref_time=0.0, monitor=None):
        samplerate = settings.output_samplerate
        buffer_length = settings.output_buffer_length
        nchannels = settings.output_channels

        scheduler = dn.Scheduler()
        task = clz.get_task(scheduler, settings, manager, ref_time, monitor)
        return task, clz(scheduler, samplerate, buffer_length, nchannels, monitor)

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
            node = dn.pipe(node, dn.rechannel(self.nchannels, channels))
        if samplerate is not None and samplerate != self.samplerate:
            node = dn.pipe(node, dn.resample(ratio=(self.samplerate, samplerate)))
        if volume != 0:
            node = dn.pipe(node, lambda s: s * 10**(volume/20))

        return node

    @functools.lru_cache(maxsize=32)
    def load_sound(self, filepath, stop_event=None):
        return aud.load_sound(filepath, channels=self.nchannels, samplerate=self.samplerate, stop_event=stop_event)

    def play(self, node, samplerate=None, channels=None, volume=0.0, start=None, end=None, time=None, zindex=(0,)):
        node = self.resample(node, samplerate, channels, volume, start, end)
        node = dn.pipe(lambda a:a[0], dn.attach(node))
        return self.add_effect(node, time=time, zindex=zindex)

    def play_file(self, path, volume=0.0, start=None, end=None, time=None, zindex=(0,)):
        meta = aud.AudioMetadata.read(path)
        node = aud.load(path)

        sliced_node = dn.tslice(node, meta.samplerate, start, end)
        # initialize before attach; it will seek to the starting frame
        sliced_node.__enter__()
        sliced_node = self.resample(sliced_node, meta.samplerate, meta.channels, volume)
        effect_node = dn.pipe(lambda a:a[0], dn.attach(sliced_node))
        return self.add_effect(effect_node, time=time, zindex=zindex)


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

class Detector:
    def __init__(self, listeners_scheduler, monitor):
        self.listeners_scheduler = listeners_scheduler
        self.monitor = monitor

    @staticmethod
    def get_task(scheduler, settings, manager, ref_time, monitor):
        samplerate = settings.input_samplerate
        buffer_length = settings.input_buffer_length
        nchannels = settings.input_channels
        format = settings.input_format
        device = settings.input_device

        time_res = settings.detect.time_res
        hop_length = round(samplerate*time_res)

        input_node = Detector._detect_node(scheduler, ref_time, settings)
        if buffer_length != hop_length:
            input_node = dn.unchunk(input_node, chunk_shape=(hop_length, nchannels))
        if monitor:
            input_node = monitor.monitoring(input_node)

        return aud.record(manager, input_node,
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
    def create(clz, settings, manager, ref_time=0.0, monitor=None):
        scheduler = dn.Scheduler()
        task = clz.get_task(scheduler, settings, manager, ref_time, monitor)
        return task, clz(scheduler, monitor)

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


@functools.lru_cache(maxsize=32)
def to_range(start, stop, width):
    # range of slice without clamp
    if start is None:
        start = 0
    elif start < 0:
        start = width+start
    else:
        start = start

    if stop is None:
        stop = width
    elif stop < 0:
        stop = width+stop
    else:
        stop = stop

    return range(start, stop)

class RichBar:
    def __init__(self, terminal_settings):
        self.markups = []
        self.rich = mu.RichBarRenderer(terminal_settings.unicode_version, terminal_settings.color_support)

    def add_markup(self, markup, mask=slice(None,None), shift=0):
        self.markups.append((markup, mask, shift))

    def draw(self, width):
        buffer = [" "]*width
        xran = range(width)

        for markup, mask, shift in self.markups:
            if mask.start is None:
                x = shift
            elif mask.start >= 0:
                x = shift + mask.start
            else:
                x = shift + mask.start + width

            self.rich._render(buffer, markup, x=x, width=width, xmask=xran[mask], attrs=())

        return "".join(buffer).rstrip()

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
    """
    display_framerate: float = 160.0 # ~ 2 / detect.time_res
    display_delay: float = 0.0
    resize_delay: float = 0.5

class Renderer:
    def __init__(self, drawers_scheduler, monitor):
        self.drawers_scheduler = drawers_scheduler
        self.monitor = monitor

    @staticmethod
    def get_task(scheduler, settings, term_settings, ref_time, monitor):
        framerate = settings.display_framerate

        display_node = Renderer._resize_node(Renderer._render_node(scheduler, term_settings), settings, term_settings, ref_time)
        if monitor:
            display_node = monitor.monitoring(display_node)
        return term.show(display_node, 1/framerate, hide_cursor=True)

    @staticmethod
    @dn.datanode
    def _render_node(scheduler, term_settings):
        rich = mu.RichTextRenderer(term_settings.unicode_version, term_settings.color_support)
        clear_line = rich.render(rich.clear_line().expand())
        clear_below = rich.render(rich.clear_below().expand())
        width = 0
        msgs = []
        curr_msgs = list(msgs)
        with scheduler:
            shown, resized, time, size = yield
            while True:
                width = size.columns
                view = RichBar(term_settings)
                try:
                    view, msgs = scheduler.send(((view, msgs), time, width))
                except StopIteration:
                    return
                view_str = view.draw(width)

                # track changes of the message
                if not resized and curr_msgs == msgs:
                    res_text = f"{clear_line}{view_str}\r"
                elif not msgs:
                    res_text = f"{clear_below}{view_str}\r"
                else:
                    msg_text = rich.render_less(mu.Group((mu.Text("\n"), *msgs)), size)
                    res_text = f"{clear_below}{view_str}\r{msg_text}"

                shown, resized, time, size = yield res_text
                if shown:
                    curr_msgs = list(msgs)

    @staticmethod
    @dn.datanode
    def _resize_node(render_node, settings, term_settings, ref_time):
        framerate = settings.display_framerate

        rich = mu.RichTextRenderer(term_settings.unicode_version, term_settings.color_support)
        clear_line = rich.render(rich.clear_line().expand())
        clear_screen = rich.render(rich.clear_screen().expand())
        size_node = term.terminal_size()

        index = -1
        width = 0
        resize_time = 0.0
        resized = False
        with render_node, size_node:
            shown = yield
            while True:
                index += 1
                time = index / framerate + settings.display_delay - ref_time

                try:
                    size = size_node.send(None)
                except StopIteration:
                    return

                if size.columns < width:
                    resize_time = time
                    resized = True
                if resized and time < resize_time + settings.resize_delay:
                    yield f"{clear_line}resizing...\r"
                    width = size.columns
                    continue

                width = size.columns
                try:
                    res_text = render_node.send((shown, resized, time, size))
                except StopIteration:
                    return
                if resized:
                    res_text = f"{clear_screen}{res_text}"

                shown = yield res_text
                if shown:
                    resized = False

    @classmethod
    def create(clz, settings, term_settings, ref_time=0.0, monitor=None):
        scheduler = dn.Scheduler()
        task = clz.get_task(scheduler, settings, term_settings, ref_time, monitor)
        return task, clz(scheduler, monitor)

    def add_drawer(self, node, zindex=(0,)):
        return self.drawers_scheduler.add_node(node, zindex=zindex)

    def remove_drawer(self, key):
        self.drawers_scheduler.remove_node(key)

    def add_message(self, msg, zindex=(0,)):
        return self.add_drawer(self._msg_drawer(msg), zindex)

    def add_text(self, text_node, xmask=slice(None,None), zindex=(0,)):
        return self.add_drawer(self._text_drawer(text_node, xmask), zindex)

    @staticmethod
    @dn.datanode
    def _msg_drawer(msg):
        (view, msgs), _, _ = yield
        msgs.append(msg)
        yield (view, msgs)

    @staticmethod
    @dn.datanode
    def _text_drawer(text_node, xmask=slice(None,None)):
        text_node = dn.DataNode.wrap(text_node)
        with text_node:
            (view, msg), time, width = yield
            while True:
                xran = to_range(xmask.start, xmask.stop, width)

                try:
                    res = text_node.send((time, xran))
                except StopIteration:
                    return

                if res is not None:
                    xshift, text = res
                    view.add_markup(text, xmask, xshift)

                (view, msg), time, width = yield (view, msg)


class ControllerSettings(cfg.Configurable):
    pass

class Controller:
    def __init__(self, handlers_scheduler):
        self.handlers_scheduler = handlers_scheduler

    @staticmethod
    def get_task(scheduler, settings, term_settings, ref_time):
        return term.inkey(Controller._control_node(scheduler, settings, term_settings, ref_time))

    @staticmethod
    @dn.datanode
    def _control_node(scheduler, settings, term_settings, ref_time):
        keycodes = term_settings.keycodes

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

    @classmethod
    def create(clz, settings, term_settings, ref_time=0.0):
        scheduler = dn.Scheduler()
        task = clz.get_task(scheduler, settings, term_settings, ref_time)
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
    terminal = term.TerminalSettings
    logger = log.LoggerSettings
