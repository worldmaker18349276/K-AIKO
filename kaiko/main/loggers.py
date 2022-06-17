from datetime import datetime
import contextlib
import traceback
import dataclasses
import enum
import re
import queue
from inspect import cleandoc
import shutil
import numpy
import difflib
from ..utils import datanodes as dn
from ..utils import config as cfg
from ..utils import markups as mu
from ..devices import terminals as term


class LoggerSettings(cfg.Configurable):
    r"""
    Fields
    ------
    data : single tag
        The template of data icon.
    info : single tag
        The template of info icon.
    hint : single tag
        The template of hint icon.
    music : single tag
        The template of music icon.

    emph : pair tag
        The template of emph log.
    verb : pair tag
        The template of verb log.
    warn : pair tag
        The template of warn log.

    verb_block : pair tag
        The template of verb block log.
    warn_block : pair tag
        The template of warn block log.

    codepoint : pair tag
        The template of codepoint representation.
    """
    data: str = "[color=bright_green][wide=🖿/][/]"
    info: str = "[color=bright_blue][wide=🛠/][/]"
    hint: str = "[color=bright_yellow][wide=🖈/][/]"
    music: str = "[color=bright_magenta][wide=🎜/][/]"

    emph: str = "[weight=bold][slot/][/]"
    verb: str = "[weight=dim][slot/][/]"
    warn: str = "[color=red][slot/][/]"

    verb_block: str = f"[weight=dim]{'╌'*80}\n[slot/]{'╌'*80}\n[/]"
    warn_block: str = f"[color=red]{'═'*80}\n[slot/]{'═'*80}\n[/]"

    codepoint: str = "[underline=on][slot/][/]"

    @cfg.subconfig
    class python(cfg.Configurable):
        r"""
        Fields
        ------
        py_none : pair tag
        py_ellipsis : pair tag
        py_bool : pair tag
        py_int : pair tag
        py_float : pair tag
        py_complex : pair tag
        py_bytes : pair tag
        py_str : pair tag
        py_punctuation : pair tag
        py_argument : pair tag
        py_class : pair tag
        py_attribute : pair tag
        """
        py_none: str = "[color=bright_cyan][slot/][/]"
        py_ellipsis: str = "[color=bright_cyan][slot/][/]"
        py_bool: str = "[color=bright_cyan][slot/][/]"
        py_int: str = "[color=bright_cyan][slot/][/]"
        py_float: str = "[color=bright_cyan][slot/][/]"
        py_complex: str = "[color=bright_cyan][slot/][/]"
        py_bytes: str = "[color=bright_cyan][slot/][/]"
        py_str: str = "[color=bright_cyan][slot/][/]"
        py_punctuation: str = "[color=white][slot/][/]"
        py_argument: str = "[color=bright_white][slot/][/]"
        py_class: str = "[color=bright_blue][slot/][/]"
        py_attribute: str = "[color=bright_blue][slot/][/]"

    @cfg.subconfig
    class shell(cfg.Configurable):
        r"""
        Fields
        ------
        qt : single tag
            The replacement text for quotation marks.
        bs : single tag
            The replacement text for backslashes.
        ws : single tag
            The replacement text for escaped whitespaces.

        unk : pair tag
            The markup template for the unknown token.
        cmd : pair tag
            The markup template for the command token.
        kw : pair tag
            The markup template for the keyword token.
        arg : pair tag
            The markup template for the argument token.
        """
        qt: str = "[weight=dim]'[/]"
        bs: str = "[weight=dim]\\[/]"
        ws: str = "[weight=dim]⌴[/]"

        unk: str = "[color=red][slot/][/]"
        cmd: str = "[color=bright_blue][slot/][/]"
        kw: str = "[color=bright_magenta][slot/][/]"
        arg: str = "[color=bright_green][slot/][/]"


class Logger:
    def __init__(self, terminal_settings=None, logger_settings=None):
        self.log_file = None
        self._stack = None
        self._redirect_queue = None
        self.recompile_style(terminal_settings, logger_settings)

    def recompile_style(self, terminal_settings, logger_settings):
        if terminal_settings is None:
            terminal_settings = term.TerminalSettings()
        if logger_settings is None:
            logger_settings = LoggerSettings()

        self.terminal_settings = terminal_settings
        self.logger_settings = logger_settings

        self.rich = mu.RichParser(
            terminal_settings.unicode_version, terminal_settings.color_support
        )
        self.renderer = mu.RichRenderer(terminal_settings.unicode_version)

        self._compile_settings(logger_settings)
        self._compile_settings(logger_settings.shell)
        self._compile_settings(logger_settings.python)

        self.verb_block = self.rich.parse(logger_settings.verb_block, slotted=True)
        self.warn_block = self.rich.parse(logger_settings.warn_block, slotted=True)

    def set_log_file(self, path):
        self.log_file = open(path, "a")
        self.log_file.write(f"\n\n\n[log_session={datetime.now()}/]\n")

    @staticmethod
    def _parse_tag_type(doc):
        res = {}

        doc = cleandoc(doc)

        m = re.search(r"Fields\n------\n", doc)
        if not m:
            return res
        doc = doc[m.end(0) :]

        while True:
            m = re.match(r"([0-9a-zA-Z_]+) : ([^\n]+)\n+((?:[ ]+[^\n]*(?:\n+|$))*)", doc)
            if not m:
                return res
            res[m.group(1)] = cleandoc(m.group(2)).strip()
            doc = doc[m.end(0) :]

    def _compile_settings(self, settings):
        res = Logger._parse_tag_type(type(settings).__doc__)
        for tag_name, tag_type in res.items():
            if tag_type == "single tag":
                self.rich.add_single_tag(tag_name, getattr(settings, tag_name))
            elif tag_type == "pair tag":
                self.rich.add_pair_tag(tag_name, getattr(settings, tag_name))
            else:
                assert False

    @contextlib.contextmanager
    def verb(self):
        try:
            self.log("[verb_block]")
            with self.renderer.render_context(
                self.verb_block, lambda text: print(text, end="", flush=True)
            ):
                yield
        finally:
            self.log("[/verb_block]")

    @contextlib.contextmanager
    def warn(self):
        try:
            self.log("[warn_block]")
            with self.renderer.render_context(
                self.warn_block, lambda text: print(text, end="", flush=True)
            ):
                yield
        finally:
            self.log("[/warn_block]")

    @contextlib.contextmanager
    def stack(self, end="", log=True):
        if self._stack is not None:
            yield
            return

        try:
            self._stack = []
            yield
        finally:
            markup = mu.Group(tuple(self._stack))
            self._stack = None
            self.print(markup, end=end, flush=True, log=False)
            if log:
                self.log("\n" + markup.represent())

    @staticmethod
    @dn.datanode
    def _redirect_node(queue):
        (view, msgs, logs), _, _ = yield
        while True:
            while not queue.empty():
                logs.append(queue.get().expand())
            (view, msgs, logs), _, _ = yield (view, msgs, logs)

    @contextlib.contextmanager
    def popup(self, renderer):
        redirect_queue = queue.Queue()
        with renderer.add_drawer(self._redirect_node(redirect_queue)):
            try:
                self._redirect_queue = redirect_queue
                yield
            finally:
                self._redirect_queue = None

    @contextlib.contextmanager
    def mute(self):
        redirect_queue = queue.Queue()
        try:
            self._redirect_queue = redirect_queue
            yield
        finally:
            self._redirect_queue = None

    def backslashreplace(self, ch):
        if ch in "\a\r\n\t\b\v\f":
            return f"[codepoint]{repr(ch)[1:-1]}[/]"
        r = hex(ord(ch))[2:]
        if len(r) <= 2:
            r = rf"\x{r:0>2}"
        elif len(r) <= 4:
            r = rf"\u{r:0>4}"
        elif len(r) <= 8:
            r = rf"\U{r:0>8}"
        else:
            raise ValueError(ch)
        return f"[codepoint]{r}[/]"

    def escape(self, text, type="plain"):
        if type == "plain":
            text = mu.escape(text)
            text = "".join(
                ch
                if ch.isprintable() or ch in ["\n", "\t"]
                else self.backslashreplace(ch)
                for ch in text
            )
            return text

        elif type == "all":
            text = mu.escape(text)
            text = "".join(
                ch
                if ch.isprintable()
                else self.backslashreplace(ch)
                for ch in text
            )
            return text

        else:
            raise ValueError

    def as_uri(self, path, emph=True):
        if path.is_absolute():
            path = path.as_uri()
        else:
            path = str(path)
        if path.endswith("."):
            path = path + "/"
        path = self.escape(path, type="all")
        if emph:
            path = f"[emph]{path}[/]"
        return path

    def log(self, msg):
        if self.log_file is not None:
            self.log_file.write(f"[log={datetime.now()}/]{msg}\n")

    def print(self, msg="", end="\n", flush=False, markup=True, log=True):
        if not markup:
            msg = str(msg)

            if self._stack is not None:
                self._stack.append(mu.Text(msg + end))
                return

            if log:
                self.log(msg)

            if self._redirect_queue is not None:
                self._redirect_queue.put(mu.Text(msg + end))
                return

            print(msg, end=end, flush=flush)
            return

        if isinstance(msg, str):
            markup = self.rich.parse(msg, expand=False)
        elif isinstance(msg, mu.Markup):
            markup = msg
        else:
            assert TypeError(type(msg))

        if self._stack is not None:
            self._stack.append(mu.Group((markup, mu.Text(end))))
            return

        if msg and log:
            self.log(markup.represent())

        if self._redirect_queue is not None:
            self._redirect_queue.put(mu.Group((markup, mu.Text(end))))
            return

        print(self.renderer.render(markup.expand()), end=end, flush=flush)

    def print_traceback(self):
        with self.warn():
            self.print(traceback.format_exc(), end="", markup=False)

    def clear_line(self, flush=False, log=True):
        self.print(self.renderer.clear_line(), end="", flush=flush, log=log)

    def clear(self, bottom=False, flush=False, log=False):
        markup = self.renderer.clear_screen()
        if bottom:
            y = shutil.get_terminal_size().lines - 1
            markup = mu.Group((markup, mu.Move(x=0, y=y)))
        self.print(markup, end="", flush=flush, log=log)

    def format_code(self, content, marked=None, title=None, is_changed=False):
        total_width = 80
        lines = content.split("\n")
        n = len(str(content.count("\n") + 1))
        res = []
        if title is not None:
            title = self.escape(title, type='all')
            change_mark = "*" if is_changed else ""
            res.append(f"[weight=dim]{'─'*n}────{'─'*(max(0, total_width-n-4))}[/]")
            res.append(f" [emph]{title}[/]{change_mark}")
        res.append(f"[weight=dim]{'─'*n}──┬─{'─'*(max(0, total_width-n-4))}[/]")
        for i, line in enumerate(lines):
            if marked and marked[0] == i:
                line = (
                    self.escape(line[: marked[1]], type="all")
                    + "[color=red]◊[/]"
                    + self.escape(line[marked[1] :], type="all")
                )
            else:
                line = self.escape(line, type="all")
            res.append(
                f" [weight=dim]{i+1:>{n}d}[/] [weight=dim]│[/] [color=bright_white]{line}[/]"
            )
        res.append(f"[weight=dim]{'─'*n}──┴─{'─'*(max(0, total_width-n-4))}[/]")
        return "\n".join(res)

    def format_code_diff(self, old, new, title=None, is_changed=False):
        lines = difflib.Differ().compare(old.split("\n"), new.split("\n"))

        total_width = 80
        n = len(str(new.count("\n") + 1))
        res = []
        if title is not None:
            title = self.escape(title, type='all')
            change_mark = "*" if is_changed else ""
            res.append(f"[weight=dim]{'─'*n}────{'─'*(max(0, total_width-n-4))}[/]")
            res.append(f" [emph]{title}[/]{change_mark}")
        res.append(f"[weight=dim]{'─'*n}──┬─{'─'*(max(0, total_width-n-4))}[/]")
        i = 0
        for line in lines:
            ln = self.escape(line[2:], type="all")
            if line.startswith("  "):
                res.append(
                    f" [weight=dim]{i+1:>{n}d}[/] [weight=dim]│[/] [color=bright_white]{ln}[/]"
                )
                i += 1
            elif line.startswith("+ "):
                res.append(
                    f" [color=bright_blue]{i+1:>{n}d}[/] [weight=dim]│[/] [color=bright_blue]{ln}[/]"
                )
                i += 1
            elif line.startswith("- "):
                res.append(
                    f" [color=bright_red]{'-':>{n}s}[/] [weight=dim]│[/] [color=bright_red]{ln}[/]"
                )
            elif line.startswith("? "):
                pass
        res.append(f"[weight=dim]{'─'*n}──┴─{'─'*(max(0, total_width-n-4))}[/]")
        return "\n".join(res)

    def format_dict(self, data, show_border=True):
        total_width = 80
        if any(self.rich.widthof(k) < 0 for k in data.keys()):
            raise ValueError("contain non-printable key")
        width = max((self.rich.widthof(k) for k in data.keys()), default=0)
        res = []
        for k, v in data.items():
            key = mu.escape(k)
            value = self.escape(v)
            res.append(f"{key: >{width}} [weight=dim]│[/] [emph]{value}[/]")
        if show_border:
            border = f"[weight=dim]{'─'*total_width}[/]"
            res.insert(0, border)
            res.append(border)
        return "\n".join(res)

    def format_value(self, value, multiline=True, indent=0):
        if type(value) in (
            type(None),
            type(...),
            bool,
            int,
            float,
            complex,
            bytes,
            str,
        ):
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
                content = self.format_value(
                    value[0], multiline=multiline, indent=indent
                )
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
                    keys = [
                        key + " " * (total_width - width)
                        for width, key in zip(widths, keys)
                    ]

                content = delimiter.join(
                    key + "[py_punctuation]:[/]" + subvalue
                    for key, subvalue in zip(keys, subvalues)
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
                template = (
                    cls_name + opening + f"\n{' '*(indent+4)}%s\n{' '*indent}" + closing
                )
                delimiter = f"[py_punctuation],[/]\n{' '*(indent+4)}"
                indent = indent + 4

            content = delimiter.join(
                f"[py_argument]{mu.escape(field.name)}[/][py_punctuation]=[/]"
                + self.format_value(
                    getattr(value, field.name), multiline=multiline, indent=indent
                )
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
        emax = tol * 7
        start = min((perf.time for perf in perfs), default=0.0) - time_margin
        end = max((perf.time for perf in perfs), default=0.0) + time_margin

        # grades infos
        grades = [perf.grade for perf in perfs if not perf.is_miss]
        miss_count = sum(perf.is_miss for perf in perfs)
        failed_count = sum(
            not grade.is_wrong and abs(grade.shift) == 3 for grade in grades
        )
        bad_count = sum(
            not grade.is_wrong and abs(grade.shift) == 2 for grade in grades
        )
        good_count = sum(
            not grade.is_wrong and abs(grade.shift) == 1 for grade in grades
        )
        perfect_count = sum(
            not grade.is_wrong and abs(grade.shift) == 0 for grade in grades
        )
        failed_wrong_count = sum(
            grade.is_wrong and abs(grade.shift) == 3 for grade in grades
        )
        bad_wrong_count = sum(
            grade.is_wrong and abs(grade.shift) == 2 for grade in grades
        )
        good_wrong_count = sum(
            grade.is_wrong and abs(grade.shift) == 1 for grade in grades
        )
        perfect_wrong_count = sum(
            grade.is_wrong and abs(grade.shift) == 0 for grade in grades
        )
        accuracy = (
            sum(2.0 ** (-abs(grade.shift)) for grade in grades) / len(perfs)
            if perfs
            else 0.0
        )
        mistakes = (
            sum(grade.is_wrong for grade in grades) / len(grades) if grades else 0.0
        )

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
        dev = (
            (sum((err - ofs) ** 2 for _, err in errors) / len(errors)) ** 0.5
            if errors
            else 0.0
        )

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
        grad_width = max(
            grad_minwidth, len(timespan), max(len(info_str) for info_str in grad_infos)
        )
        stat_width = max(stat_minwidth, max(len(info_str) for info_str in stat_infos))
        scat_width = width - grad_width - stat_width - 4

        grad_top = "═" * grad_width
        grad_bot = timespan.center(grad_width, "═")
        scat_top = scat_bot = "═" * scat_width
        stat_top = stat_bot = "═" * stat_width
        grad_infos = [info_str.ljust(grad_width) for info_str in grad_infos]
        stat_infos = [info_str.ljust(stat_width) for info_str in stat_infos]

        # discretize data
        dx = (end - start) / (scat_width * 2 - 1)
        dy = 2 * emax / (scat_height * 4 - 1)
        data = numpy.zeros((scat_height * 4 + 1, scat_width * 2), dtype=int)
        for time, err in errors:
            i = round((err + emax) / dy)
            j = round((time - start) / dx)
            if i in range(scat_height * 4) and j in range(scat_width * 2):
                data[i, j] += 1
        for time in misses:
            j = round((time - start) / dx)
            if j in range(scat_width * 2):
                data[-1, j] += 1

        braille_block = 2 ** numpy.array([0, 3, 1, 4, 2, 5, 6, 7]).reshape(1, 4, 1, 2)

        # plot scatter
        scat_data = (data[:-1, :] > 0).reshape(scat_height, 4, scat_width, 2)
        scat_code = 0x2800 + (scat_data * braille_block).sum(axis=(1, 3)).astype("i2")
        scat_graph = [line.tostring().decode("utf-16") for line in scat_code]
        miss_data = (data[-1, :] > 0).reshape(scat_width, 2)
        miss_code = (miss_data * [1, 2]).sum(axis=-1)
        miss_graph = "".join("─╾╼━"[code] for code in miss_code)

        # plot statistics
        stat_data = data[:-1, :].sum(axis=1)
        stat_level = numpy.linspace(
            0, numpy.max(stat_data), stat_width * 2, endpoint=False
        )
        stat_data = (stat_level[None, :] < stat_data[:, None]).reshape(
            scat_height, 4, stat_width, 2
        )
        stat_code = 0x2800 + (stat_data * braille_block).sum(axis=(1, 3)).astype("i2")
        stat_graph = [line.tostring().decode("utf-16") for line in stat_code]

        # plot accuracies
        acc_weight = 2.0 ** numpy.array([-3, -2, -1, 0, -1, -2, -3])
        acc_data = (
            data[:-1, :].reshape(scat_height, 4, scat_width, 2).sum(axis=(1, 3))
            * acc_weight[:, None]
        ).sum(axis=0)
        acc_data /= numpy.maximum(
            1, data.sum(axis=0).reshape(scat_width, 2).sum(axis=1)
        )
        acc_level = numpy.arange(acc_height) * 8
        acc_code = 0x2580 + numpy.clip(
            acc_data[None, :] * acc_height * 8 - acc_level[::-1, None], 0, 8
        ).astype("i2")
        acc_code[acc_code == 0x2580] = ord(" ")
        acc_graph = [line.tostring().decode("utf-16") for line in acc_code]

        # print
        with self.stack():
            self.print("╒" + grad_top + "╤" + scat_top + "╤" + stat_top + "╕", markup=False)
            self.print(
                "│" + grad_infos[0] + "│" + scat_graph[0] + "│" + stat_graph[0] + "│",
                markup=False,
            )
            self.print(
                "│" + grad_infos[1] + "│" + scat_graph[1] + "│" + stat_graph[1] + "│",
                markup=False,
            )
            self.print(
                "│" + grad_infos[2] + "│" + scat_graph[2] + "│" + stat_graph[2] + "│",
                markup=False,
            )
            self.print(
                "│" + grad_infos[3] + "│" + scat_graph[3] + "│" + stat_graph[3] + "│",
                markup=False,
            )
            self.print(
                "│" + grad_infos[4] + "│" + scat_graph[4] + "│" + stat_graph[4] + "│",
                markup=False,
            )
            self.print(
                "│" + grad_infos[5] + "│" + scat_graph[5] + "│" + stat_graph[5] + "│",
                markup=False,
            )
            self.print(
                "│" + grad_infos[6] + "│" + scat_graph[6] + "│" + stat_graph[6] + "│",
                markup=False,
            )
            self.print(
                "│" + grad_infos[7] + "├" + miss_graph + "┤" + stat_infos[0] + "│",
                markup=False,
            )
            self.print(
                "│" + grad_infos[8] + "│" + acc_graph[0] + "│" + stat_infos[1] + "│",
                markup=False,
            )
            self.print(
                "│" + grad_infos[9] + "│" + acc_graph[1] + "│" + stat_infos[2] + "│",
                markup=False,
            )
            self.print("╘" + grad_bot + "╧" + scat_bot + "╧" + stat_bot + "╛", markup=False)

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
