import contextlib
import dataclasses
import enum
import shutil
import numpy
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
    class syntax(cfg.Configurable):
        py_NoneType: str    = "[color=bright_cyan][slot/][/]"
        py_ellipsis: str    = "[color=bright_cyan][slot/][/]"
        py_bool: str        = "[color=bright_cyan][slot/][/]"
        py_int: str         = "[color=bright_cyan][slot/][/]"
        py_float: str       = "[color=bright_cyan][slot/][/]"
        py_complex: str     = "[color=bright_cyan][slot/][/]"
        py_bytes: str       = "[color=bright_cyan][slot/][/]"
        py_str: str         = "[color=bright_cyan][slot/][/]"
        py_punctuation: str = "[color=white][slot/][/]"
        py_argument: str    = "[color=bright_white][slot/][/]"
        py_class: str       = "[color=bright_blue][slot/][/]"
        py_attribute: str   = "[color=bright_blue][slot/][/]"

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
        self.rich = mu.RichParser(terminal_settings.unicode_version, terminal_settings.color_support)
        self.renderer = mu.RichTextRenderer(terminal_settings.unicode_version)

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

        self.rich.add_pair_template("py_NoneType", logger_settings.syntax.py_NoneType)
        self.rich.add_pair_template("py_ellipsis", logger_settings.syntax.py_ellipsis)
        self.rich.add_pair_template("py_bool", logger_settings.syntax.py_bool)
        self.rich.add_pair_template("py_int", logger_settings.syntax.py_int)
        self.rich.add_pair_template("py_float", logger_settings.syntax.py_float)
        self.rich.add_pair_template("py_complex", logger_settings.syntax.py_complex)
        self.rich.add_pair_template("py_bytes", logger_settings.syntax.py_bytes)
        self.rich.add_pair_template("py_str", logger_settings.syntax.py_str)
        self.rich.add_pair_template("py_punctuation", logger_settings.syntax.py_punctuation)
        self.rich.add_pair_template("py_argument", logger_settings.syntax.py_argument)
        self.rich.add_pair_template("py_class", logger_settings.syntax.py_class)
        self.rich.add_pair_template("py_attribute", logger_settings.syntax.py_attribute)

    @contextlib.contextmanager
    def verb(self):
        level = self.level
        verb_block = self.logger_settings.verb_block if self.logger_settings else LoggerSettings.verb_block
        template = self.rich.parse(verb_block, slotted=True)
        self.level = 0
        try:
            with self.renderer.render_context(template, lambda text: print(text, end="", flush=True)):
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
            with self.renderer.render_context(template, lambda text: print(text, end="", flush=True)):
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
        print(self.renderer.render(msg), end=end, flush=flush)

    def clear_line(self, flush=False):
        print(self.renderer.render(self.renderer.clear_line().expand()), end="", flush=flush)

    def clear(self, flush=False):
        print(self.renderer.render(self.renderer.clear_screen().expand()), end="", flush=flush)

    def format_code(self, content, title=None, is_changed=False):
        total_width = 80
        lines = content.split("\n")
        n = len(str(len(lines)-1))
        res = []
        if title is not None:
            change_mark = "*" if is_changed else ""
            res.append(f"[weight=dim]{'─'*n}────{'─'*(max(0, total_width-n-4))}[/]")
            res.append(f" [emph]{self.escape(title)}[/]{change_mark}")
        res.append(f"[weight=dim]{'─'*n}──┬─{'─'*(max(0, total_width-n-4))}[/]")
        for i, line in enumerate(lines):
            res.append(f" [weight=dim]{i:>{n}d}[/] [weight=dim]│[/] [color=bright_white]{self.escape(line)}[/]")
        res.append(f"[weight=dim]{'─'*n}──┴─{'─'*(max(0, total_width-n-4))}[/]")
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

    def format_value(self, value, multiline=True, indent=0):
        if type(value) in (type(None), type(...), bool, int, float, complex, bytes, str):
            return f"[py_{type(value).__name__}]{mu.escape(repr(value))}[/]"

        elif isinstance(value, enum.Enum):
            cls = type(value)
            cls_name = f"[py_class]{mu.escape(cls.__name__)}[/]"
            value_name = f"[py_attribute]{mu.escape(value.name)}[/]"
            return cls_name + "[py_punctuation].[/]" + value_name

        elif type(value) in (list, tuple, set, dict):
            if type(value) is list:
                opening, closing = "[py_punctuation][[[/]", "[py_punctuation]]][/]"
            elif type(value) is tuple:
                opening, closing = "[py_punctuation]([/]", "[py_punctuation])[/]"
            elif type(value) is set:
                opening, closing = "[py_punctuation]{[/]", "[py_punctuation]}[/]"
            elif type(value) is dict:
                opening, closing = "[py_punctuation]{[/]", "[py_punctuation]}[/]"
            else:
                assert False

            if not value:
                if type(value) is set:
                    return "[py_class]set[/][py_punctuation]([/][py_punctuation])[/]"
                else:
                    return opening + closing

            if type(value) is tuple and len(value) == 1:
                content = self.format_value(value[0], multiline=multiline, indent=indent)
                return opening + content + "[py_punctuation],[/]" + closing

            if not multiline:
                template = "%s%s%s"
                delimiter = "[py_punctuation],[/] "
            else:
                template = f"%s\n{' '*(indent+4)}%s\n{' '*indent}%s"
                delimiter = f"[py_punctuation],[/]\n{' '*(indent+4)}"
                indent = indent + 4

            if type(value) is dict:
                keys = [self.format_value(key, multiline=False) for key in value.keys()]
                subvalues = [
                    self.format_value(subvalue, multiline=multiline, indent=indent)
                    for subvalue in value.values()
                ]

                if multiline and all(type(key) is str for key in value.keys()):
                    widths = [self.rich.widthof(repr(key)) for key in value.keys()]
                    total_width = max(widths)
                    keys = [key + " "*(total_width - width) for width, key in zip(widths, keys)]
                
                content = delimiter.join(
                    key + "[py_punctuation]:[/]" + subvalue for key, subvalue in zip(keys, subvalues)
                )

            else:
                content = delimiter.join(
                    self.format_value(subvalue, multiline=multiline, indent=indent)
                    for subvalue in value
                )

            return template % (opening, content, closing)

        elif dataclasses.is_dataclass(value):
            cls = type(value)
            fields = dataclasses.fields(cls)
            cls_name = f"[py_class]{mu.escape(cls.__name__)}[/]"
            opening, closing = "[py_punctuation]([/]", "[py_punctuation])[/]"

            if not fields:
                return cls_name + opening + closing

            if not multiline:
                template = cls_name + opening + "%s" + closing
                delimiter = "[py_punctuation],[/] "
            else:
                template = cls_name + opening + f"\n{' '*(indent+4)}%s\n{' '*indent}" + closing
                delimiter = f"[py_punctuation],[/]\n{' '*(indent+4)}"
                indent = indent + 4

            content = delimiter.join(
                f"[py_argument]{mu.escape(field.name)}[/][py_punctuation]=[/]"
                + self.format_value(getattr(value, field.name), multiline=multiline, indent=indent)
                for field in fields
            )
            return template % content

        else:
            return mu.escape(repr(value))

    def print_scores(self, tol, perfs):
        grad_minwidth = 15
        stat_minwidth = 15
        scat_height = 7
        acc_height = 2
        time_margin = 0.1

        width = shutil.get_terminal_size().columns
        emax = tol*7
        start = min((perf.time for perf in perfs), default=0.0) - time_margin
        end   = max((perf.time for perf in perfs), default=0.0) + time_margin

        # grades infos
        grades = [perf.grade for perf in perfs if not perf.is_miss]
        miss_count = sum(perf.is_miss for perf in perfs)
        failed_count    = sum(not grade.is_wrong and abs(grade.shift) == 3 for grade in grades)
        bad_count       = sum(not grade.is_wrong and abs(grade.shift) == 2 for grade in grades)
        good_count      = sum(not grade.is_wrong and abs(grade.shift) == 1 for grade in grades)
        perfect_count   = sum(not grade.is_wrong and abs(grade.shift) == 0 for grade in grades)
        failed_wrong_count  = sum(grade.is_wrong and abs(grade.shift) == 3 for grade in grades)
        bad_wrong_count     = sum(grade.is_wrong and abs(grade.shift) == 2 for grade in grades)
        good_wrong_count    = sum(grade.is_wrong and abs(grade.shift) == 1 for grade in grades)
        perfect_wrong_count = sum(grade.is_wrong and abs(grade.shift) == 0 for grade in grades)
        accuracy = sum(2.0**(-abs(grade.shift)) for grade in grades) / len(perfs) if perfs else 0.0
        mistakes = sum(grade.is_wrong for grade in grades) / len(grades) if grades else 0.0

        grad_infos = [
            f"   miss: {   miss_count}",
            f" failed: { failed_count}+{ failed_wrong_count}",
            f"    bad: {    bad_count}+{    bad_wrong_count}",
            f"   good: {   good_count}+{   good_wrong_count}",
            f"perfect: {perfect_count}+{perfect_wrong_count}",
            "",
            "",
            f"accuracy: {accuracy:.1%}",
            f"mistakes: {mistakes:.2%}",
            "",
        ]

        # statistics infos
        errors = [(perf.time, perf.err) for perf in perfs if not perf.is_miss]
        misses = [perf.time for perf in perfs if perf.is_miss]
        err = sum(abs(err) for _, err in errors) / len(errors) if errors else 0.0
        ofs = sum(err for _, err in errors) / len(errors) if errors else 0.0
        dev = (sum((err-ofs)**2 for _, err in errors) / len(errors))**0.5 if errors else 0.0

        stat_infos = [
            f"err={err*1000:.3f} ms",
            f"ofs={ofs*1000:+.3f} ms",
            f"dev={dev*1000:.3f} ms",
        ]

        # timespan
        def minsec(sec):
            sec = round(sec)
            sgn = +1 if sec >= 0 else -1
            min, sec = divmod(abs(sec), 60)
            min *= sgn
            return f"{min}:{sec:02d}"

        timespan = f"╡{minsec(start)} ~ {minsec(end)}╞"

        # layout
        grad_width = max(grad_minwidth, len(timespan), max(len(info_str) for info_str in grad_infos))
        stat_width = max(stat_minwidth, max(len(info_str) for info_str in stat_infos))
        scat_width = width - grad_width - stat_width - 4

        grad_top = "═"*grad_width
        grad_bot = timespan.center(grad_width, "═")
        scat_top = scat_bot = "═"*scat_width
        stat_top = stat_bot = "═"*stat_width
        grad_infos = [info_str.ljust(grad_width) for info_str in grad_infos]
        stat_infos = [info_str.ljust(stat_width) for info_str in stat_infos]

        # discretize data
        dx = (end - start)/(scat_width*2-1)
        dy = 2*emax/(scat_height*4-1)
        data = numpy.zeros((scat_height*4+1, scat_width*2), dtype=int)
        for time, err in errors:
            i = round((err+emax)/dy)
            j = round((time-start)/dx)
            if i in range(scat_height*4) and j in range(scat_width*2):
                data[i,j] += 1
        for time in misses:
            j = round((time-start)/dx)
            if j in range(scat_width*2):
                data[-1,j] += 1

        braille_block = 2**numpy.array([0, 3, 1, 4, 2, 5, 6, 7]).reshape(1, 4, 1, 2)

        # plot scatter
        scat_data = (data[:-1,:] > 0).reshape(scat_height, 4, scat_width, 2)
        scat_code = 0x2800 + (scat_data * braille_block).sum(axis=(1, 3)).astype('i2')
        scat_graph = [line.tostring().decode('utf-16') for line in scat_code]
        miss_data = (data[-1,:] > 0).reshape(scat_width, 2)
        miss_code = (miss_data * [1, 2]).sum(axis=-1)
        miss_graph = "".join("─╾╼━"[code] for code in miss_code)

        # plot statistics
        stat_data = data[:-1,:].sum(axis=1)
        stat_level = numpy.linspace(0, numpy.max(stat_data), stat_width*2, endpoint=False)
        stat_data = (stat_level[None,:] < stat_data[:,None]).reshape(scat_height, 4, stat_width, 2)
        stat_code = 0x2800 + (stat_data * braille_block).sum(axis=(1, 3)).astype('i2')
        stat_graph = [line.tostring().decode('utf-16') for line in stat_code]

        # plot accuracies
        acc_weight = 2.0**numpy.array([-3, -2, -1, 0, -1, -2, -3])
        acc_data = (data[:-1,:].reshape(scat_height, 4, scat_width, 2).sum(axis=(1,3)) * acc_weight[:,None]).sum(axis=0)
        acc_data /= numpy.maximum(1, data.sum(axis=0).reshape(scat_width, 2).sum(axis=1))
        acc_level = numpy.arange(acc_height)*8
        acc_code = 0x2580 + numpy.clip(acc_data[None,:]*acc_height*8 - acc_level[::-1,None], 0, 8).astype('i2')
        acc_code[acc_code==0x2580] = ord(" ")
        acc_graph = [line.tostring().decode('utf-16') for line in acc_code]

        # print
        self.print("╒" + grad_top      + "╤" + scat_top      + "╤" + stat_top      + "╕", markup=False)
        self.print("│" + grad_infos[0] + "│" + scat_graph[0] + "│" + stat_graph[0] + "│", markup=False)
        self.print("│" + grad_infos[1] + "│" + scat_graph[1] + "│" + stat_graph[1] + "│", markup=False)
        self.print("│" + grad_infos[2] + "│" + scat_graph[2] + "│" + stat_graph[2] + "│", markup=False)
        self.print("│" + grad_infos[3] + "│" + scat_graph[3] + "│" + stat_graph[3] + "│", markup=False)
        self.print("│" + grad_infos[4] + "│" + scat_graph[4] + "│" + stat_graph[4] + "│", markup=False)
        self.print("│" + grad_infos[5] + "│" + scat_graph[5] + "│" + stat_graph[5] + "│", markup=False)
        self.print("│" + grad_infos[6] + "│" + scat_graph[6] + "│" + stat_graph[6] + "│", markup=False)
        self.print("│" + grad_infos[7] + "├" + miss_graph    + "┤" + stat_infos[0] + "│", markup=False)
        self.print("│" + grad_infos[8] + "│" + acc_graph[0]  + "│" + stat_infos[1] + "│", markup=False)
        self.print("│" + grad_infos[9] + "│" + acc_graph[1]  + "│" + stat_infos[2] + "│", markup=False)
        self.print("╘" + grad_bot      + "╧" + scat_bot      + "╧" + stat_bot      + "╛", markup=False)

    def ask(self, prompt, default=True):
        @dn.datanode
        def _ask():
            hint = "[emph]y[/]/n" if default else "y/[emph]n[/]"
            self.print(f"{self.escape(prompt)} [[{hint}]]", end="", flush=True)

            while True:
                keycode = None
                try:
                    while keycode is None:
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

