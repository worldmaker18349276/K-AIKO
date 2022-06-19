import dataclasses
import queue
from ..utils import datanodes as dn


class ClockAction:
    pass


@dataclasses.dataclass(frozen=True)
class ClockPause(ClockAction):
    time: float


@dataclasses.dataclass(frozen=True)
class ClockResume(ClockAction):
    time: float


@dataclasses.dataclass(frozen=True)
class ClockSkip(ClockAction):
    time: float
    delta: float


@dataclasses.dataclass(frozen=True)
class ClockStop(ClockAction):
    time: float


class Clock:
    def __init__(self):
        self.action_queue = queue.Queue()

    @dn.datanode
    def clock(self, offset, ratio):
        ratio0 = ratio

        waited = None
        action = None

        time = yield
        last_time = time
        last_tick = offset + last_time * ratio
        while True:
            # find unhandled action
            if waited is None and not self.action_queue.empty():
                waited = self.action_queue.get()
            if waited is not None and waited.time <= time:
                action, waited = waited, None
            else:
                action = None

            # no action -> yield value
            if action is None:
                last_time = time
                if last_tick <= offset + last_time * ratio:
                    last_tick = offset + last_time * ratio
                    time = yield (last_tick, ratio)
                else:
                    time = yield (last_tick, 0)
                continue

            # update state
            if isinstance(action, ClockStop):
                return
            if isinstance(action, ClockResume):
                offset, ratio = offset + action.time * (ratio - ratio0), ratio0
            elif isinstance(action, ClockPause):
                offset, ratio = offset + action.time * ratio, 0
            elif isinstance(action, ClockSkip):
                offset += action.delta
            else:
                raise TypeError

            last_tick = max(last_tick, offset + last_time * ratio)
            last_time = max(last_time, action.time)
            last_tick = max(last_tick, offset + last_time * ratio)

    @dn.datanode
    def clock_slice(self, offset, ratio):
        ratio0 = ratio

        waited = None
        action = None

        time_slice = yield
        slices_map = []
        last_time = time_slice.start
        last_tick = offset + last_time * ratio
        while True:
            # find unhandled action
            if waited is None and not self.action_queue.empty():
                waited = self.action_queue.get()
            if waited is not None and waited.time <= time_slice.stop:
                action, waited = waited, None
            else:
                action = None

            # slice
            time = time_slice.stop if action is None else action.time
            if last_time < time:
                tick_slice = slice(offset + last_time * ratio, offset + time * ratio)

                if last_tick < tick_slice.start:
                    slices_map.append(
                        (
                            slice(last_time, last_time),
                            slice(last_tick, tick_slice.start),
                            ratio,
                        )
                    )
                    last_tick = tick_slice.start

                if last_tick > tick_slice.start:
                    cut_time = (last_tick - offset) / ratio if ratio != 0 else time
                    cut_time = min(max(cut_time, last_time), time)
                    slices_map.append(
                        (slice(last_time, cut_time), slice(last_tick, last_tick), 0)
                    )
                    last_time = cut_time
                    tick_slice = slice(last_tick, offset + time * ratio)

                if last_time < time:
                    slices_map.append((slice(last_time, time), tick_slice, ratio))
                    last_time = time
                    last_tick = tick_slice.stop

            # no action -> yield value
            if action is None:
                time_slice = yield slices_map
                slices_map = []
                continue

            # update state
            if isinstance(action, ClockStop):
                return
            if isinstance(action, ClockResume):
                offset, ratio = offset + action.time * (ratio - ratio0), ratio0
            elif isinstance(action, ClockPause):
                offset, ratio = offset + action.time * ratio, 0
            elif isinstance(action, ClockSkip):
                offset += action.delta
            else:
                raise TypeError

    def resume(self, time):
        self.action_queue.put(ClockResume(time))

    def pause(self, time):
        self.action_queue.put(ClockPause(time))

    def skip(self, time, delta):
        self.action_queue.put(ClockSkip(time, delta))

    def stop(self, time):
        self.action_queue.put(ClockStop(time))


@dataclasses.dataclass
class Metronome:
    offset: float
    tempo: float

    def time(self, beat):
        r"""Convert beat to time (in seconds).

        Parameters
        ----------
        beat : int or Fraction or float

        Returns
        -------
        time : float
        """
        return self.offset + beat * 60 / self.tempo

    def beat(self, time):
        r"""Convert time (in seconds) to beat.

        Parameters
        ----------
        time : float

        Returns
        -------
        beat : float
        """
        return (time - self.offset) * self.tempo / 60

    def dtime(self, beat, length):
        r"""Convert length to time difference (in seconds).

        Parameters
        ----------
        beat : int or Fraction or float
        length : int or Fraction or float

        Returns
        -------
        dtime : float
        """
        return self.time(beat + length) - self.time(beat)


