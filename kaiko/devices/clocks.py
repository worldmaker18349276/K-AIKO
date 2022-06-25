import contextlib
import dataclasses
import threading
import queue
import time
from ..utils import datanodes as dn


class ClockOperation:
    pass


@dataclasses.dataclass(frozen=True)
class ClockSpeed(ClockOperation):
    time: float
    ratio: float


@dataclasses.dataclass(frozen=True)
class ClockSkip(ClockOperation):
    time: float
    offset: float


@dataclasses.dataclass(frozen=True)
class ClockDelay(ClockOperation):
    delay: float


@dataclasses.dataclass(frozen=True)
class ClockStop(ClockOperation):
    pass


class Clock:
    def __init__(self, offset, ratio):
        self.offset = offset
        self.ratio = ratio
        self.action_queues = {}
        self.lock = threading.RLock()
        self.is_stopped = False

    @staticmethod
    @dn.datanode
    def _tick(action_queue, offset, ratio, delay):
        action = None

        time = yield
        last_time = time
        last_tick = offset + last_time * ratio
        last_ratio = ratio
        while True:
            # find unhandled action
            if action is None and not action_queue.empty():
                action = action_queue.get()

            if action is None:
                action_time = None
            elif hasattr(action, "time"):
                action_time = action.time - delay
            else:
                action_time = last_time

            # update last_time, last_tick, last_ratio
            curr_time = time if action is None else min(action_time, time)
            if last_time < curr_time:
                time_slice = slice(offset + last_time * ratio, offset + curr_time * ratio)
                last_tick = max(last_tick, time_slice.start)
                last_ratio = 0 if last_tick > time_slice.stop else ratio
                last_tick = max(last_tick, time_slice.stop)
                last_time = curr_time

            # update offset, ratio
            if action is not None and action_time <= time:
                if isinstance(action, ClockStop):
                    return
                if isinstance(action, ClockSpeed):
                    offset, ratio = offset + action_time * (ratio - action.ratio), action.ratio
                elif isinstance(action, ClockSkip):
                    offset += action.offset
                elif isinstance(action, ClockDelay):
                    offset += action.delay * ratio
                    delay += action.delay
                else:
                    raise TypeError

                action = None
                continue

            time = yield (last_tick, last_ratio)

    @staticmethod
    @dn.datanode
    def _tick_slice(action_queue, offset, ratio, delay):
        action = None

        time_slice = yield
        slices_map = []
        last_ratio = ratio
        last_time = time_slice.start
        last_tick = offset + last_time * ratio
        while True:
            # find unhandled action
            if action is None and not action_queue.empty():
                action = action_queue.get()

            if action is None:
                action_time = None
            elif hasattr(action, "time"):
                action_time = action.time - delay
            else:
                action_time = last_time

            # update last_time, last_tick, last_ratio
            curr_time = time_slice.stop if action is None else min(action_time, time_slice.stop)
            if last_time < curr_time:
                tick_slice = slice(offset + last_time * ratio, offset + curr_time * ratio)

                if last_tick < tick_slice.start:
                    slices_map.append(
                        (
                            slice(last_time, last_time),
                            slice(last_tick, tick_slice.start),
                            float("inf"),
                        )
                    )
                    last_tick = tick_slice.start

                if last_tick > tick_slice.start:
                    cut_time = (last_tick - offset) / ratio if ratio != 0 else curr_time
                    cut_time = min(max(cut_time, last_time), curr_time)
                    slices_map.append(
                        (slice(last_time, cut_time), slice(last_tick, last_tick), 0)
                    )
                    last_ratio = 0
                    last_time = cut_time
                    tick_slice = slice(last_tick, tick_slice.stop)

                if last_time < curr_time:
                    slices_map.append((slice(last_time, curr_time), tick_slice, ratio))
                    last_ratio = ratio
                    last_time = curr_time
                    last_tick = tick_slice.stop

            # update offset, ratio
            if action is not None and action_time <= time_slice.stop:
                if isinstance(action, ClockStop):
                    return
                if isinstance(action, ClockSpeed):
                    offset, ratio = offset + action_time * (ratio - action.ratio), action.ratio
                elif isinstance(action, ClockSkip):
                    offset += action.offset
                elif isinstance(action, ClockDelay):
                    offset += action.delay * ratio
                    delay += action.delay
                else:
                    raise TypeError

                action = None
                continue

            time_slice = yield slices_map
            slices_map = []

    @contextlib.contextmanager
    def tick(self, key, delay=0.0):
        action_queue = queue.Queue()
        tick_node = self._tick(action_queue, self.offset, self.ratio, delay=delay)
        with self.lock:
            if self.is_stopped:
                raise ValueError("clock has already stopped")
            if key in self.action_queues:
                raise ValueError(f"already register: {key}")
            self.action_queues[key] = action_queue
        try:
            yield tick_node
        finally:
            with self.lock:
                del self.action_queues[key]

    @contextlib.contextmanager
    def tick_slice(self, key, delay=0.0):
        action_queue = queue.Queue()
        tick_node = self._tick_slice(action_queue, self.offset, self.ratio, delay=delay)
        with self.lock:
            if self.is_stopped:
                raise ValueError("clock has already stopped")
            if key in self.action_queues:
                raise ValueError(f"already register: {key}")
            self.action_queues[key] = action_queue
        try:
            yield tick_node
        finally:
            with self.lock:
                del self.action_queues[key]

    def speed(self, time, ratio):
        action = ClockSpeed(time, ratio)
        with self.lock:
            if self.is_stopped:
                return
            for action_queue in self.action_queues.values():
                action_queue.put(action)
            self.offset, self.ratio = self.offset + time * (self.ratio - ratio), ratio

    def skip(self, time, offset):
        action = ClockSkip(time, offset)
        with self.lock:
            if self.is_stopped:
                return
            for action_queue in self.action_queues.values():
                action_queue.put(action)
            self.offset += offset

    def delay(self, key, delay):
        action = ClockDelay(delay)
        with self.lock:
            if self.is_stopped:
                return
            action_queue = self.action_queues[key]
            action_queue.put(action)

    def stop(self):
        action = ClockStop()
        with self.lock:
            if self.is_stopped:
                return
            for action_queue in self.action_queues.values():
                action_queue.put(action)
            self.is_stopped = True


class Metronome(Clock):
    def __init__(self, tempo):
        super().__init__(0.0, tempo / 60)

    def tempo(self, time, offset, tempo):
        with self.lock:
            tick0 = offset * tempo / 60
            tick = self.offset + time * self.ratio
            self.skip(time, (tick0 - tick) % 1)
            self.speed(time, tempo / 60)

