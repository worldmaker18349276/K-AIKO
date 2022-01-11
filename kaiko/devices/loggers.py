import contextlib
from ..utils import datanodes as dn
from ..utils import config as cfg
from ..utils import markups as mu
from . import terminals as term


class LoggerSettings(cfg.Configurable):
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

    verb_block : str
        The template of verb block log.
    warn_block : str
        The template of warn block log.
    """
    data_icon: str = "[color=bright_green][wide=🖿/][/]"
    info_icon: str = "[color=bright_blue][wide=🛠/][/]"
    hint_icon: str = "[color=bright_yellow][wide=🖈/][/]"

    verb: str = f"[weight=dim][slot/][/]"
    warn: str = f"[color=red][slot/][/]"
    emph: str = "[weight=bold][slot/][/]"

    verb_block: str = f"[weight=dim]{'╌'*80}\n[slot/]{'╌'*80}\n[/]"
    warn_block: str = f"[color=red]{'═'*80}\n[slot/]{'═'*80}\n[/]"

    @cfg.subconfig
    class shell(cfg.Configurable):
        r"""
        Fields
        ------
        quotation : str
            The replacement text for quotation marks.
        backslash : str
            The replacement text for backslashes.
        whitespace : str
            The replacement text for escaped whitespaces.
        typeahead : str
            The markup template for the type-ahead.

        token_unknown : str
            The markup template for the unknown token.
        token_unfinished : str
            The markup template for the unfinished token.
        token_command : str
            The markup template for the command token.
        token_keyword : str
            The markup template for the keyword token.
        token_argument : str
            The markup template for the argument token.
        token_highlight : str
            The markup template for the highlighted token.
        """
        quotation: str = "[weight=dim]'[/]"
        backslash: str = "[weight=dim]\\[/]"
        whitespace: str = "[weight=dim]⌴[/]"
        typeahead: str = "[weight=dim][slot/][/]"

        token_unknown: str = "[color=red][slot/][/]"
        token_unfinished: str = "[slot/]"
        token_command: str = "[color=bright_blue][slot/][/]"
        token_keyword: str = "[color=bright_magenta][slot/][/]"
        token_argument: str = "[color=bright_green][slot/][/]"
        token_highlight: str = "[underline][slot/][/]"

class Logger:
    def __init__(self, terminal_settings=None, logger_settings=None):
        self.level = 1
        self.terminal_settings = terminal_settings
        self.logger_settings = logger_settings
        self.recompile_style(terminal_settings, logger_settings)

    def recompile_style(self, terminal_settings=None, logger_settings=None):
        if terminal_settings is not None:
            self.terminal_settings = terminal_settings
        if logger_settings is not None:
            self.logger_settings = logger_settings

        terminal_settings = self.terminal_settings if self.terminal_settings else term.TerminalSettings()
        self.rich = mu.RichTextRenderer(terminal_settings.unicode_version, terminal_settings.color_support)

        logger_settings = self.logger_settings if self.logger_settings else LoggerSettings()
        self.rich.add_single_template("data", logger_settings.data_icon)
        self.rich.add_single_template("info", logger_settings.info_icon)
        self.rich.add_single_template("hint", logger_settings.hint_icon)
        self.rich.add_pair_template("verb", logger_settings.verb)
        self.rich.add_pair_template("emph", logger_settings.emph)
        self.rich.add_pair_template("warn", logger_settings.warn)

        self.rich.add_pair_template("unknown", logger_settings.shell.token_unknown)
        self.rich.add_pair_template("unfinished", logger_settings.shell.token_unfinished)
        self.rich.add_pair_template("cmd", logger_settings.shell.token_command)
        self.rich.add_pair_template("kw", logger_settings.shell.token_keyword)
        self.rich.add_pair_template("arg", logger_settings.shell.token_argument)
        self.rich.add_pair_template("highlight", logger_settings.shell.token_highlight)
        self.rich.add_single_template("ws", logger_settings.shell.whitespace)
        self.rich.add_single_template("qt", logger_settings.shell.quotation)
        self.rich.add_single_template("bs", logger_settings.shell.backslash)
        self.rich.add_pair_template("typeahead", logger_settings.shell.typeahead)

    @contextlib.contextmanager
    def verb(self):
        level = self.level
        verb_block = self.logger_settings.verb_block if self.logger_settings else LoggerSettings.verb_block
        template = self.rich.parse(verb_block, slotted=True)
        self.level = 0
        try:
            with self.rich.render_context(template, lambda text: print(text, end="", flush=True)):
                yield
        finally:
            self.level = level

    @contextlib.contextmanager
    def warn(self):
        level = self.level
        warn_block = self.logger_settings.warn_block if self.logger_settings else LoggerSettings.warn_block
        template = self.rich.parse(warn_block, slotted=True)
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

    def format_code(self, content, title=None, is_changed=False):
        width = 80
        lines = content.split("\n")
        n = len(str(len(lines)-1))
        res = []
        if title is not None:
            change_mark = "*" if is_changed else ""
            res.append(f"[weight=dim]{'─'*n}────{'─'*(max(0, width-n-4))}[/]")
            res.append(f" [emph]{self.escape(title)}[/]{change_mark}")
        res.append(f"[weight=dim]{'─'*n}──┬─{'─'*(max(0, width-n-4))}[/]")
        for i, line in enumerate(lines):
            res.append(f" [weight=dim]{i:>{n}d}[/] [weight=dim]│[/] [color=bright_white]{self.escape(line)}[/]")
        res.append(f"[weight=dim]{'─'*n}──┴─{'─'*(max(0, width-n-4))}[/]")
        return "\n".join(res)

    def format_dict(self, data, show_border=True):
        total_width = 80
        if any(self.rich.widthof(k) < 0 for k in data.keys()):
            raise ValueError("contain unprintable key")
        width = max((self.rich.widthof(k) for k in data.keys()), default=0)
        res = []
        for k, v in data.items():
            key = ' '*(width - self.rich.widthof(k)) + mu.escape(k)
            value = mu.escape(v)
            res.append(f"{key} [weight=dim]│[/] [emph]{value}[/]")
        if show_border:
            border = f"[weight=dim]{'─'*total_width}[/]"
            res.insert(0, border)
            res.append(border)
        return "\n".join(res)

    def ask(self, prompt, default=True):
        @dn.datanode
        def _ask():
            hint = "[emph]y[/]/n" if default else "y/[emph]n[/]"
            self.print(f"{self.escape(prompt)} [[{hint}]]", end="", flush=True)

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

