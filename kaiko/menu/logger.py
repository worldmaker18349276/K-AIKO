import contextlib
from kaiko.utils import datanodes as dn
from kaiko.utils import config as cfg
from kaiko.utils import markups as mu
from kaiko.utils import terminals as term


class KAIKOMenuSettings(cfg.Configurable):
    r"""
    Fields
    ------
    data_icon : str
        The template of data icon.
    info_icon : str
        The template of info icon.
    hint_icon : str
        The template of hint icon.

    verb : str
        The template of verb log.
    emph : str
        The template of emph log.
    warn : str
        The template of warn log.

    best_screen_size : int
        The best screen size.
    adjust screen delay : float
        The delay time to complete the screen adjustment.

    editor : str
        The editor to edit long text.
    """
    data_icon: str = "[color=bright_green][wide=🗀/][/]"
    info_icon: str = "[color=bright_blue][wide=🛠/][/]"
    hint_icon: str = "[color=bright_yellow][wide=💡/][/]"

    verb: str = f"{'─'*80}\n[weight=dim][slot/][/]{'─'*80}\n"
    emph: str = "[weight=bold][slot/][/]"
    warn: str = "[color=red][slot/][/]"

    best_screen_size: int = 80
    adjust_screen_delay: float = 1.0

    editor: str = "nano"

class KAIKOLogger:
    def __init__(self, terminal_settings=None, menu_settings=None):
        self.terminal_settings = terminal_settings
        self.menu_settings = menu_settings
        self.level = 1
        self.recompile_style()

    def recompile_style(self):
        terminal_settings = self.terminal_settings if self.terminal_settings else term.TerminalSettings()
        self.rich = mu.RichTextRenderer(terminal_settings.unicode_version, terminal_settings.color_support)

        menu_settings = self.menu_settings if self.menu_settings else KAIKOMenuSettings()
        self.rich.add_single_template("data", menu_settings.data_icon)
        self.rich.add_single_template("info", menu_settings.info_icon)
        self.rich.add_single_template("hint", menu_settings.hint_icon)
        self.rich.add_pair_template("verb", menu_settings.verb)
        self.rich.add_pair_template("emph", menu_settings.emph)
        self.rich.add_pair_template("warn", menu_settings.warn)

    def set_settings(self, terminal_settings=None, menu_settings=None):
        if terminal_settings is not None:
            self.terminal_settings = terminal_settings
        if menu_settings is not None:
            self.menu_settings = menu_settings
        self.recompile_style()

    @contextlib.contextmanager
    def verb(self):
        level = self.level
        template = self.rich.parse("[verb][slot/][/]", slotted=True)
        self.level = 0
        try:
            with self.rich.render_context(template, lambda text: print(text, end="", flush=True)):
                yield
        finally:
            self.level = level

    @contextlib.contextmanager
    def normal(self):
        level = self.level
        template = self.rich.parse("[reset][slot/][/]", slotted=True)
        self.level = 1
        try:
            with self.rich.render_context(template, lambda text: print(text, end="", flush=True)):
                yield
        finally:
            self.level = level

    @contextlib.contextmanager
    def warn(self):
        level = self.level
        template = self.rich.parse("[warn][slot/][/]", slotted=True)
        self.level = 2
        try:
            with self.rich.render_context(template, lambda text: print(text, end="", flush=True)):
                yield
        finally:
            self.level = level

    def escape(self, text):
        return mu.escape(text)

    def emph(self, text):
        return f"[emph]{self.escape(text)}[/]"

    def print(self, msg="", end="\n", flush=False, markup=True):
        if not markup:
            print(msg, end=end, flush=flush)
            return

        if isinstance(msg, str):
            msg = self.rich.parse(msg)
        print(self.rich.render(msg), end=end, flush=flush)

    def clear_line(self, flush=False):
        print(self.rich.render(self.rich.clear_line().expand()), end="", flush=flush)

    def clear(self, flush=False):
        print(self.rich.render(self.rich.clear_screen().expand()), end="", flush=flush)

    def print_code(self, content, title=None, is_changed=False):
        width = 80
        lines = content.split("\n")
        n = len(str(len(lines)-1))
        if title is not None:
            change_mark = "*" if is_changed else ""
            self.print(f"[weight=dim]{'─'*n}────{'─'*(max(0, width-n-4))}[/]")
            self.print(f" [emph]{self.escape(title)}[/]{change_mark}")
        self.print(f"[weight=dim]{'─'*n}──┬─{'─'*(max(0, width-n-4))}[/]")
        for i, line in enumerate(lines):
            self.print(f" [weight=dim]{i:>{n}d}[/] [weight=dim]│[/] [color=bright_white]{self.escape(line)}[/]")
        self.print(f"[weight=dim]{'─'*n}──┴─{'─'*(max(0, width-n-4))}[/]")

    def ask(self, prompt, default=True):
        @dn.datanode
        def _ask():
            hint = "[emph]y[/]/n" if default else "y/[emph]n[/]"
            self.print(f"{self.escape(prompt)} \[{hint}]", end="", flush=True)

            while True:
                try:
                    _, keycode = yield
                finally:
                    self.print(flush=True)

                if keycode == "\n":
                    return default
                if keycode in ("y", "Y"):
                    return True
                elif keycode in ("n", "N"):
                    return False

                self.print("Please reply [emph]y[/] or [emph]n[/]", end="", flush=True)

        return term.inkey(_ask())

