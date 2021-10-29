import sys
import os
import time
import re
import contextlib
import queue
import threading
import signal
import shutil
import termios
import select
import tty
from . import datanodes as dn


@dn.datanode
def ucs_detect():
    pattern = re.compile(r"\x1b\[(\d*);(\d*)R")
    channel = queue.Queue()

    def get_pos(arg):
        m = pattern.match(arg[1])
        if not m:
            return
        x = int(m.group(2) or "1") - 1
        channel.put(x)

    @dn.datanode
    def query_pos():
        previous_version = '4.1.0'
        wide_by_version = [
            ('5.1.0', '龼'),
            ('5.2.0', '🈯'),
            ('6.0.0', '🈁'),
            ('8.0.0', '🉐'),
            ('9.0.0', '🐹'),
            ('10.0.0', '🦖'),
            ('11.0.0', '🧪'),
            ('12.0.0', '🪐'),
            ('12.1.0', '㋿'),
            ('13.0.0', '🫕'),
        ]

        yield

        for version, wchar in wide_by_version:
            print(wchar, end="", flush=True)
            print("\x1b[6n", end="", flush=True)

            while True:
                yield
                try:
                    x = channel.get(False)
                except queue.Empty:
                    continue
                else:
                    break

            print(f"\twidth={x}", end="\n", flush=True)
            if x == 1:
                break
            elif x == 2:
                previous_version = version
                continue
            else:
                return

        return previous_version

    query_task = query_pos()
    with dn.pipe(inkey(get_pos), query_task) as task:
        yield from task.join((yield))
    return query_task.result

@dn.datanode
def terminal_size():
    resize_event = threading.Event()
    def SIGWINCH_handler(sig, frame):
        resize_event.set()
    resize_event.set()
    signal.signal(signal.SIGWINCH, SIGWINCH_handler)

    yield
    while True:
        if resize_event.is_set():
            resize_event.clear()
            size = shutil.get_terminal_size()
        yield size

@contextlib.contextmanager
def inkey_ctxt(stream, raw=False):
    fd = stream.fileno()
    old_attrs = termios.tcgetattr(fd)
    old_blocking = os.get_blocking(fd)

    try:
        tty.setcbreak(fd, termios.TCSANOW)
        if raw:
            tty.setraw(fd, termios.TCSANOW)
        os.set_blocking(fd, False)

        yield

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        os.set_blocking(fd, old_blocking)

@dn.datanode
def inkey(node, stream=None, raw=False):
    node = dn.DataNode.wrap(node)
    dt = 0.01

    if stream is None:
        stream = sys.stdin
    fd = stream.fileno()

    def run(stop_event):
        ref_time = time.time()
        while True:
            ready, _, _ = select.select([fd], [], [], dt)
            if stop_event.is_set():
                break
            if fd not in ready:
                continue

            data = stream.read()

            try:
                node.send((time.time()-ref_time, data))
            except StopIteration:
                return

    with inkey_ctxt(stream, raw):
        with node:
            with dn.create_task(run) as task:
                yield from task.join((yield))

@contextlib.contextmanager
def show_ctxt(stream, hide_cursor=False, end="\n"):
    hide_cursor = hide_cursor and stream == sys.stdout

    try:
        if hide_cursor:
            stream.write("\x1b[?25l")

        yield

    finally:
        if hide_cursor:
            stream.write("\x1b[?25h")
        stream.write(end)
        stream.flush()

@dn.datanode
def show(node, dt, t0=0, stream=None, hide_cursor=False, end="\n"):
    node = dn.DataNode.wrap(node)
    if stream is None:
        stream = sys.stdout

    def run(stop_event):
        ref_time = time.time()

        # stream.write("\n")
        # dropped = 0
        shown = False
        i = -1
        while True:
            try:
                view = node.send(shown)
            except StopIteration:
                break
            shown = False
            i += 1

            delta = ref_time+t0+i*dt - time.time()
            if delta < 0:
                # dropped += 1
                continue
            if stop_event.wait(delta):
                break

            # stream.write(f"\x1b[A(spend:{(dt-delta)/dt:.3f}, drop:{dropped})\n")
            stream.write(view)
            stream.flush()
            shown = True

    with show_ctxt(stream, hide_cursor, end):
        with node:
            with dn.create_task(run) as task:
                yield from task.join((yield))
