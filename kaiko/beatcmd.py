import threading
from . import datanodes as dn
from . import tui


default_keymap = {
    "Backspace"        : "\x7f",
    "Delete"           : "\x1b[3~",
    "Left"             : "\x1b[D",
    "Right"            : "\x1b[C",
    "Home"             : "\x1b[H",
    "End"              : "\x1b[F",
    "Up"               : "\x1b[A",
    "Down"             : "\x1b[B",
    "PageUp"           : "\x1b[5~",
    "PageDown"         : "\x1b[6~",
    "Tab"              : "\t",
    "Esc"              : "\x1b",
    "Enter"            : "\n",
    "Shift+Tab"        : "\x1b[Z",
    "Ctrl+Backspace"   : "\x08",
    "Ctrl+Delete"      : "\x1b[3;5~",
    "Shift+Left"       : "\x1b[1;2D",
    "Shift+Right"      : "\x1b[1;2C",
    "Alt+Left"         : "\x1b[1;3D",
    "Alt+Right"        : "\x1b[1;3C",
    "Alt+Shift+Left"   : "\x1b[1;4D",
    "Alt+Shift+Right"  : "\x1b[1;4C",
    "Ctrl+Left"        : "\x1b[1;5D",
    "Ctrl+Right"       : "\x1b[1;5C",
    "Ctrl+Shift+Left"  : "\x1b[1;6D",
    "Ctrl+Shift+Right" : "\x1b[1;6C",
    "Alt+Ctrl+Left"    : "\x1b[1;7D",
    "Alt+Ctrl+Right"   : "\x1b[1;7C",
    "Alt+Ctrl+Shift+Left"  : "\x1b[1;8D",
    "Alt+Ctrl+Shift+Right" : "\x1b[1;8C",
}

class Beatstroke:
    def __init__(self, text=None, pos=0, suggestions=None):
        self.text = text or []
        self.pos = pos
        self.suggestions = suggestions or []
        self.hint = ""
        self.event = threading.Event()
        self.keymap = default_keymap

    @dn.datanode
    def input_handler(self):
        while True:
            _, key = yield
            self.event.set()

            self.hint = ""

            # suggestions
            while key == self.keymap["Tab"]:
                original_text = list(self.text)
                original_pos = self.pos

                search_text = "".join(self.text)
                for suggestion in self.suggestions:
                    if suggestion.startswith(search_text):
                        self.text = list(suggestion)
                        self.pos = len(self.text)

                        _, key = yield
                        self.event.set()

                        if key == self.keymap["Esc"]:
                            self.text = original_text
                            self.pos = original_pos

                            _, key = yield
                            self.event.set()
                            break

                        elif key != self.keymap["Tab"]:
                            break

                else:
                    self.text = original_text
                    self.pos = original_pos

                    _, key = yield
                    self.event.set()

            # edit
            if len(key) == 1 and key.isprintable():
                self.text[self.pos:self.pos] = [key]
                self.pos += 1

                # hint
                search_text = "".join(self.text)
                for suggestion in self.suggestions:
                    if suggestion.startswith(search_text):
                        self.hint = suggestion
                        break

            elif key == self.keymap["Backspace"]:
                if self.pos == 0:
                    continue
                self.pos -= 1
                del self.text[self.pos]

            elif key == self.keymap["Delete"]:
                if self.pos >= len(self.text):
                    continue
                del self.text[self.pos]

            elif key == self.keymap["Left"]:
                if self.pos == 0:
                    continue
                self.pos -= 1

            elif key == self.keymap["Right"]:
                if self.pos >= len(self.text):
                    continue
                self.pos += 1

            elif key == self.keymap["Home"]:
                self.pos = 0

            elif key == self.keymap["End"]:
                self.pos = len(self.text)

class Beatline:
    def __init__(self, seg, framerate, offset=0.0, tempo=130.0):
        self.seg = seg
        self.framerate = framerate
        self.offset = offset
        self.tempo = tempo
        self.headers = [ # game of life - Blocker
            "\x1b[96;1m⠶⠦⣚⠀⠶\x1b[m\x1b[38;5;255m❯\x1b[m ",
            "\x1b[96;1m⢎⣀⡛⠀⠶\x1b[m\x1b[38;5;255m❯\x1b[m ",
            "\x1b[36m⢖⣄⠻⠀⠶\x1b[m\x1b[38;5;254m❯\x1b[m ",
            "\x1b[36m⠖⠐⡩⠂⠶\x1b[m\x1b[38;5;254m❯\x1b[m ",
            "\x1b[96m⠶⠀⡭⠲⠶\x1b[m\x1b[38;5;253m❯\x1b[m ",
            "\x1b[36m⠶⠀⣬⠉⡱\x1b[m\x1b[38;5;253m❯\x1b[m ",
            "\x1b[36m⠶⠀⣦⠙⠵\x1b[m\x1b[38;5;252m❯\x1b[m ",
            "\x1b[36m⠶⠠⣊⠄⠴\x1b[m\x1b[38;5;252m❯\x1b[m ",

            "\x1b[96m⠶⠦⣚⠀⠶\x1b[m\x1b[38;5;251m❯\x1b[m ",
            "\x1b[36m⢎⣀⡛⠀⠶\x1b[m\x1b[38;5;251m❯\x1b[m ",
            "\x1b[36m⢖⣄⠻⠀⠶\x1b[m\x1b[38;5;250m❯\x1b[m ",
            "\x1b[36m⠖⠐⡩⠂⠶\x1b[m\x1b[38;5;250m❯\x1b[m ",
            "\x1b[96m⠶⠀⡭⠲⠶\x1b[m\x1b[38;5;249m❯\x1b[m ",
            "\x1b[36m⠶⠀⣬⠉⡱\x1b[m\x1b[38;5;249m❯\x1b[m ",
            "\x1b[36m⠶⠀⣦⠙⠵\x1b[m\x1b[38;5;248m❯\x1b[m ",
            "\x1b[36m⠶⠠⣊⠄⠴\x1b[m\x1b[38;5;248m❯\x1b[m ",

            "\x1b[96m⠶⠦⣚⠀⠶\x1b[m\x1b[38;5;247m❯\x1b[m ",
            "\x1b[36m⢎⣀⡛⠀⠶\x1b[m\x1b[38;5;247m❯\x1b[m ",
            "\x1b[36m⢖⣄⠻⠀⠶\x1b[m\x1b[38;5;246m❯\x1b[m ",
            "\x1b[36m⠖⠐⡩⠂⠶\x1b[m\x1b[38;5;246m❯\x1b[m ",
            "\x1b[96m⠶⠀⡭⠲⠶\x1b[m\x1b[38;5;245m❯\x1b[m ",
            "\x1b[36m⠶⠀⣬⠉⡱\x1b[m\x1b[38;5;245m❯\x1b[m ",
            "\x1b[36m⠶⠀⣦⠙⠵\x1b[m\x1b[38;5;244m❯\x1b[m ",
            "\x1b[36m⠶⠠⣊⠄⠴\x1b[m\x1b[38;5;244m❯\x1b[m ",

            "\x1b[96m⠶⠦⣚⠀⠶\x1b[m\x1b[38;5;243m❯\x1b[m ",
            "\x1b[36m⢎⣀⡛⠀⠶\x1b[m\x1b[38;5;243m❯\x1b[m ",
            "\x1b[36m⢖⣄⠻⠀⠶\x1b[m\x1b[38;5;242m❯\x1b[m ",
            "\x1b[36m⠖⠐⡩⠂⠶\x1b[m\x1b[38;5;242m❯\x1b[m ",
            "\x1b[96m⠶⠀⡭⠲⠶\x1b[m\x1b[38;5;241m❯\x1b[m ",
            "\x1b[36m⠶⠀⣬⠉⡱\x1b[m\x1b[38;5;241m❯\x1b[m ",
            "\x1b[36m⠶⠀⣦⠙⠵\x1b[m\x1b[38;5;240m❯\x1b[m ",
            "\x1b[36m⠶⠠⣊⠄⠴\x1b[m\x1b[38;5;240m❯\x1b[m ",
            ]

    @dn.datanode
    def output_handler(self):
        size_node = dn.terminal_size()

        with size_node:
            yield
            t = self.offset/(60/self.tempo)
            tr = 0
            while True:
                if self.seg.event.is_set():
                    self.seg.event.clear()
                    tr = t // -1 * -1

                ind = int(t / 4 * len(self.headers)) % len(self.headers)
                header = self.headers[ind]

                # cursor
                if t-tr < 0 or (t-tr) % 1 < 0.3:
                    cursor_pos = self.seg.pos
                    if ind == 0 or ind == 1:
                        cursor_wrapper = "\x1b[7;1m", "\x1b[m"
                    else:
                        cursor_wrapper = "\x1b[7;2m", "\x1b[m"
                else:
                    cursor_wrapper = None

                # input
                input_text = "".join(self.seg.text)
                input_hint = "".join(f"\x1b[2m{ch}\x1b[m" for ch in self.seg.hint) if self.seg.hint else ""

                # size
                try:
                    size = size_node.send(None)
                except StopIteration:
                    return
                width = size.columns

                # draw
                view = tui.newwin1(width)
                view, x = tui.addtext1(view, width, 0, header)
                tui.addtext1(view, width, x, input_hint)
                tui.addtext1(view, width, x, input_text)
                if cursor_wrapper:
                    _, cursor_x = tui.textrange1(x, input_text[:cursor_pos])
                    cursor_ran = tui.select1(view, width, slice(cursor_x, cursor_x+1))
                    view[cursor_ran.start] = view[cursor_ran.start].join(cursor_wrapper)

                yield "\r" + "".join(view) + "\r"
                t += 1/self.framerate/(60/self.tempo)

def prompt(framerate=60.0, suggestions=[]):
    seg = Beatstroke(suggestions=["play", "say", "settings", "exit"])
    input_knot = dn.input(seg.input_handler())
    prompt = Beatline(seg, framerate)
    display_knot = dn.interval(prompt.output_handler(), dn.show(hide_cursor=True), 1/framerate)

    menu_knot = dn.pipe(input_knot, display_knot)
    dn.exhaust(menu_knot, dt=0.01, interruptible=True)


