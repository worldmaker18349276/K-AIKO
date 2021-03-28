import os
import time
import contextlib
import queue
import threading
from . import cfg
from . import datanodes as dn
from . import tui

default_keymap = {
    "\x1b[B": 'NEXT',
    "\x1b[A": 'PREV',
    "\n": 'ENTER',
    "\x1b": 'EXIT',
}

def explore(menu_tree, keymap=default_keymap, sep="❯ ", framerate=60.0):
    prompts = []
    result = None

    @dn.datanode
    def input_handler(menu_tree):
        nonlocal prompts, result
        try:
            prompts = menu_tree.send(None)
        except StopIteration:
            return

        while True:
            _, key = yield
            if key not in keymap:
                continue

            try:
                res = menu_tree.send(keymap[key])
            except StopIteration:
                return

            if isinstance(res, list):
                prompts = res
            else:
                result = res
                return

    input_knot = dn.input(input_handler(menu_tree))

    @dn.datanode
    def prompt_node():
        size_node = dn.terminal_size()
        headers = [ # game of life - Blocker
            "⠶⠦⣚⠀⠶",
            "⢎⣀⡛⠀⠶",
            "⢖⣄⠻⠀⠶",
            "⠖⠐⡩⠂⠶",
            "⠶⠀⡭⠲⠶",
            "⠶⠀⣬⠉⡱",
            "⠶⠀⣦⠙⠵",
            "⠶⠠⣊⠄⠴",
            ]
        period = 1/8

        with size_node:
            yield
            ind = 0
            while True:
                header = headers[int(ind/framerate/period) % len(headers)] + sep

                try:
                    size = size_node.send(None)
                except StopIteration:
                    return
                width = size.columns

                view = tui.newwin1(width)
                tui.addtext1(view, width, 0, header + sep.join(prompts))
                yield "\r" + "".join(view) + "\r"
                ind += 1

    display_knot = dn.interval(prompt_node(), dn.show(hide_cursor=True), 1/framerate)

    menu_knot = dn.pipe(input_knot, display_knot)
    dn.exhaust(menu_knot, dt=0.01, interruptible=True)
    return result

@dn.datanode
def menu_tree(items):
    index = 0
    length = len(items)
    if length == 0:
        return

    prompt, func = items[index]

    action = yield

    while True:
        if action is None:
            pass

        elif action == 'NEXT':
            index = min(index+1, length-1)

        elif action == 'PREV':
            index = max(index-1, 0)

        elif action == 'ENTER':
            if func is None:
                # None -> no action
                pass

            elif hasattr(func, 'execute'):
                # executable -> suspend to execute
                action = yield func
                continue

            elif hasattr(func, '__call__'):
                # datanode function -> forward action to submenu
                with func() as node:
                    action = None
                    while True:
                        try:
                            res = node.send(action)
                        except StopIteration:
                            break
                        res = res if hasattr(res, 'execute') else [prompt, *res]
                        action = yield res

            else:
                raise ValueError(f"unknown function: {repr(func)}")

        elif action == 'EXIT':
            break

        prompt, func = items[index]
        action = yield [prompt]

