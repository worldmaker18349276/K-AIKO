import re
import contextlib
import enum
import dataclasses
from typing import Union, Sequence
import wcwidth

# pair: [tag=param]...[/]
# single: [tag=param/]
# escape: [[, ]]

# basic:
#   csi: [csi=2A/]  =>  "\x1b[2A"
#   sgr: [sgr=2;3]...[/]  =>  "\x1b[2;3m...\x1b[m"
# template:
#   pair: [color=green]>>> [weight=bold][slot/][/]
#   single: [color=red]!!![/]


class MarkupParseError(Exception):
    pass


def loc_at(text, index):
    line = text.count("\n", 0, index)
    last_ln = text.rfind("\n", 0, index)
    col = index - (last_ln + 1)
    return f"{line}:{col}"


def parse_markup(markup_str, tags, props={}):
    stack = [(Group, [])]

    for match in re.finditer(
        r"(?P<text>([^\[]|\[\[)+)|(?P<tag>\[[^\]]*\])", markup_str
    ):
        tag = match.group("tag")
        text = match.group("text")

        if text is not None:
            # process escapes
            raw = text.replace("[[", "[").replace("]]", "]")
            stack[-1][1].append(Text(raw))
            continue

        if tag == "[/]":  # [/]
            if len(stack) <= 1:
                loc = loc_at(markup_str, match.start("tag"))
                raise MarkupParseError(f"parse failed at {loc}, too many closing tag")
            markup_type, children, *param = stack.pop()
            markup = markup_type(tuple(children), *param)
            stack[-1][1].append(markup)
            continue

        match_single = re.match(r"^\[(\w+)(?:=(.*))?/\]$", tag)  # [tag=param/]
        if match_single:
            name = match_single.group(1)
            param_str = match_single.group(2)
            if name not in tags or not issubclass(tags[name], Single):
                loc = loc_at(markup_str, match.start("tag"))
                raise MarkupParseError(f"parse failed at {loc}, unknown tag [{name}/]")
            param = tags[name].parse(param_str)
            if name in props:
                param += props[name]
            stack[-1][1].append(tags[name](*param))
            continue

        match_pair = re.match("^\[(\w+)(?:=(.*))?\]$", tag)  # [tag=param]
        if match_pair:
            name = match_pair.group(1)
            param_str = match_pair.group(2)
            if name not in tags or not issubclass(tags[name], Pair):
                loc = loc_at(markup_str, match.start("tag"))
                raise MarkupParseError(f"parse failed at {loc}, unknown tag [{name}]")
            param = tags[name].parse(param_str)
            if name in props:
                param += props[name]
            stack.append((tags[name], [], *param))
            continue

        loc = loc_at(markup_str, match.start("tag"))
        raise MarkupParseError(f"parse failed at {loc}, invalid tag {tag}")

    for i in range(len(stack) - 1, 0, -1):
        markup_type, children, *param = stack[i]
        markup = markup_type(tuple(children), *param)
        stack[i - 1][1].append(markup)
    markup_type, children, *param = stack[0]
    markup = markup_type(tuple(children), *param)
    return markup


def escape(text):
    return text.replace("[", "[[")


class Markup:
    def _represent(self):
        raise NotImplementedError

    def represent(self):
        return "".join(self._represent())

    def expand(self):
        return self

    def traverse(self, markup_type, func):
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class Text(Markup):
    string: str

    def _represent(self):
        for ch in self.string:
            yield ("[[" if ch == "[" else ch)

    def traverse(self, markup_type, func):
        if isinstance(self, markup_type):
            return func(self)
        else:
            return self


@dataclasses.dataclass(frozen=True)
class Group(Markup):
    children: Sequence[Markup]

    def _represent(self):
        for child in self.children:
            yield from child._represent()

    def expand(self):
        return dataclasses.replace(
            self, children=tuple(child.expand() for child in self.children)
        )

    def traverse(self, markup_type, func):
        children = []
        modified = False
        for child in self.children:
            child_ = child.traverse(markup_type, func)
            children.append(child_)
            modified = modified or child_ is not child

        return (
            self
            if not modified
            else dataclasses.replace(self, children=tuple(children))
        )


class Tag(Markup):
    @classmethod
    def parse(cls, param):
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class Single(Tag):
    # name

    @property
    def param(self):
        raise NotImplementedError

    def _represent(self):
        param = self.param
        param_str = f"={param}" if param is not None else ""
        yield f"[{self.name}{param_str}/]"

    def traverse(self, markup_type, func):
        if isinstance(self, markup_type):
            return func(self)
        else:
            return self


@dataclasses.dataclass(frozen=True)
class Pair(Tag):
    # name
    children: Sequence[Markup]

    @property
    def param(self):
        raise NotImplementedError

    def _represent(self):
        param = self.param
        param_str = f"={param}" if param is not None else ""
        yield f"[{self.name}{param_str}]"
        for child in self.children:
            yield from child._represent()
        yield f"[/]"

    def expand(self):
        return dataclasses.replace(
            self, children=tuple(child.expand() for child in self.children)
        )

    def traverse(self, markup_type, func):
        if isinstance(self, markup_type):
            return func(self)
        else:
            children = []
            modified = False
            for child in self.children:
                child_ = child.traverse(markup_type, func)
                children.append(child_)
                modified = modified or child_ is not child

            return (
                self
                if not modified
                else dataclasses.replace(self, children=tuple(children))
            )


# template
@dataclasses.dataclass(frozen=True)
class SingleTemplate(Single):
    # name
    # _template

    @classmethod
    def parse(cls, param):
        if param is not None:
            raise MarkupParseError("no parameter is needed for template tag")
        return ()

    @property
    def param(self):
        return None

    def expand(self):
        return self._template.expand()


@dataclasses.dataclass(frozen=True)
class Slot(Single):
    name = "slot"

    @classmethod
    def parse(cls, param):
        if param is not None:
            raise MarkupParseError(f"no parameter is needed for tag [{cls.name}/]")
        return ()

    @property
    def param(self):
        return None


@dataclasses.dataclass(frozen=True)
class PairTemplate(Pair):
    # name
    # _template

    @classmethod
    def parse(cls, param):
        if param is not None:
            raise MarkupParseError("no parameter is needed for template tag")
        return ()

    @property
    def param(self):
        return None

    def expand(self):
        return replace_slot(self._template, Group(self.children)).expand()


def replace_slot(template, markup):
    injected = False

    def inject_once(slot):
        nonlocal injected
        if injected:
            return Group([])
        injected = True
        return markup

    return template.traverse(Slot, inject_once)


def make_single_template(name, template, tags, props={}):
    temp = parse_markup(template, tags=tags, props=props)
    cls = type(name.capitalize(), (SingleTemplate,), dict(name=name, _template=temp))
    cls = dataclasses.dataclass(frozen=True)(cls)
    return cls


def make_pair_template(name, template, tags, props={}):
    temp = parse_markup(template, tags=dict(tags, slot=Slot), props=props)
    cls = type(name.capitalize(), (PairTemplate,), dict(name=name, _template=temp))
    cls = dataclasses.dataclass(frozen=True)(cls)
    return cls


# ctrl code
@dataclasses.dataclass(frozen=True)
class CSI(Single):
    name = "csi"
    code: str

    @classmethod
    def parse(cls, param):
        if param is None:
            raise MarkupParseError(f"missing parameter for tag [{cls.name}/]")
        return (param,)

    @property
    def param(self):
        return self.code

    @property
    def ansi_code(self):
        return f"\x1b[{self.code}"

    def __str__(self):
        return self.ansi_code


@dataclasses.dataclass(frozen=True)
class Move(Single):
    name = "move"
    x: int
    y: int

    @classmethod
    def parse(cls, param):
        if param is None:
            raise MarkupParseError(f"missing parameter for tag [{cls.name}/]")
        try:
            x, y = tuple(int(n) for n in param.split(","))
        except ValueError:
            raise MarkupParseError(f"invalid parameter for tag [{cls.name}/]: {param}")
        return x, y

    @property
    def param(self):
        return f"{self.x},{self.y}"

    def expand(self):
        res = []
        if self.x > 0:
            res.append(CSI(f"{self.x}C"))
        elif self.x < 0:
            res.append(CSI(f"{-self.x}D"))
        if self.y > 0:
            res.append(CSI(f"{self.y}B"))
        elif self.y < 0:
            res.append(CSI(f"{-self.y}A"))
        return Group(tuple(res))

    def __str__(self):
        res = ""
        if self.x > 0:
            res += f"\x1b[{self.x}C"
        elif self.x < 0:
            res += f"\x1b[{-self.x}D"
        if self.y > 0:
            res += f"\x1b[{self.y}B"
        elif self.y < 0:
            res += f"\x1b[{-self.y}A"
        return res


@dataclasses.dataclass(frozen=True)
class Pos(Single):
    name = "pos"
    x: int
    y: int

    @classmethod
    def parse(cls, param):
        if param is None:
            raise MarkupParseError(f"missing parameter for tag [{cls.name}/]")
        try:
            x, y = tuple(int(n) for n in param.split(","))
        except ValueError:
            raise MarkupParseError(f"invalid parameter for tag [{cls.name}/]: {param}")
        return x, y

    @property
    def param(self):
        return f"{self.x},{self.y}"

    def expand(self):
        return CSI(f"{self.y+1};{self.x+1}H")

    def __str__(self):
        return f"\x1b[{self.y+1};{self.x+1}H"


@dataclasses.dataclass(frozen=True)
class Scroll(Single):
    name = "scroll"
    x: int

    @classmethod
    def parse(cls, param):
        if param is None:
            raise MarkupParseError(f"missing parameter for tag [{cls.name}/]")
        try:
            x = int(param)
        except ValueError:
            raise MarkupParseError(f"invalid parameter for tag [{cls.name}/]: {param}")
        return (x,)

    @property
    def param(self):
        return str(self.x)

    def expand(self):
        if self.x > 0:
            return CSI(f"{self.x}T")
        elif self.x < 0:
            return CSI(f"{-self.x}S")
        else:
            return Group(())

    def __str__(self):
        if self.x > 0:
            return f"\x1b[{self.x}T"
        elif self.x < 0:
            return f"\x1b[{-self.x}S"
        else:
            return ""


class ClearRegion(enum.Enum):
    to_right = "0K"
    to_left = "1K"
    line = "2K"
    to_end = "0J"
    to_beginning = "1J"
    screen = "2J"


@dataclasses.dataclass(frozen=True)
class Clear(Single):
    name = "clear"
    region: ClearRegion

    @classmethod
    def parse(cls, param):
        if param is None:
            raise MarkupParseError(f"missing parameter for tag [{cls.name}/]")
        if all(param != region.name for region in ClearRegion):
            raise MarkupParseError(f"invalid parameter for tag [{cls.name}/]: {param}")
        region = ClearRegion[param]

        return (region,)

    @property
    def param(self):
        return self.region.name

    def expand(self):
        return CSI(f"{self.region.value}")

    def __str__(self):
        return f"\x1b[{self.region.value}"


# attr code
@dataclasses.dataclass(frozen=True)
class SGR(Pair):
    name = "sgr"
    attr: tuple

    @classmethod
    def parse(cls, param):
        if param is None:
            return cls((), ())

        try:
            attr = tuple(int(n or "0") for n in param.split(";"))
        except ValueError:
            raise MarkupParseError(f"invalid parameter for tag [{cls.name}]: {param}")
        return attr

    @property
    def param(self):
        if not self.attr:
            return None
        return ";".join(map(str, self.attr))

    @property
    def ansi_code(self):
        if self.attr:
            return f"\x1b[{';'.join(map(str, self.attr))}m"
        else:
            return ""

    def __str__(self):
        return self.ansi_code


@dataclasses.dataclass(frozen=True)
class SimpleAttr(Pair):
    option: str

    @property
    def param(self):
        return self.option

    @classmethod
    def parse(cls, param):
        if param is None:
            param = next(iter(cls._options.keys()))
        if param not in cls._options:
            raise MarkupParseError(f"invalid parameter for tag [{cls.name}]: {param}")
        return (param,)

    def expand(self):
        return SGR(
            tuple(child.expand() for child in self.children),
            (self._options[self.option],),
        )

    def __str__(self):
        return f"\x1b[{self._options[self.option]}m"


@dataclasses.dataclass(frozen=True)
class Reset(SimpleAttr):
    name = "reset"
    _options = {"on": 0}


@dataclasses.dataclass(frozen=True)
class Weight(SimpleAttr):
    name = "weight"
    _options = {"bold": 1, "dim": 2, "normal": 22}


@dataclasses.dataclass(frozen=True)
class Italic(SimpleAttr):
    name = "italic"
    _options = {"on": 3, "off": 23}


@dataclasses.dataclass(frozen=True)
class Underline(SimpleAttr):
    name = "underline"
    _options = {"on": 4, "double": 21, "off": 24}


@dataclasses.dataclass(frozen=True)
class Strike(SimpleAttr):
    name = "strike"
    _options = {"on": 9, "off": 29}


@dataclasses.dataclass(frozen=True)
class Blink(SimpleAttr):
    name = "blink"
    _options = {"on": 5, "off": 25}


@dataclasses.dataclass(frozen=True)
class Invert(SimpleAttr):
    name = "invert"
    _options = {"on": 7, "off": 27}


# colors
colors_16 = [
    (0x00, 0x00, 0x00),
    (0x80, 0x00, 0x00),
    (0x00, 0x80, 0x00),
    (0x80, 0x80, 0x00),
    (0x00, 0x00, 0x80),
    (0x80, 0x00, 0x80),
    (0x00, 0x80, 0x80),
    (0xC0, 0xC0, 0xC0),
    (0x80, 0x80, 0x80),
    (0xFF, 0x00, 0x00),
    (0x00, 0xFF, 0x00),
    (0xFF, 0xFF, 0x00),
    (0x00, 0x00, 0xFF),
    (0xFF, 0x00, 0xFF),
    (0x00, 0xFF, 0xFF),
    (0xFF, 0xFF, 0xFF),
]
colors_256 = []
for r in (0, *range(95, 256, 40)):
    for g in (0, *range(95, 256, 40)):
        for b in (0, *range(95, 256, 40)):
            colors_256.append((r, g, b))
for gray in range(8, 248, 10):
    colors_256.append((gray, gray, gray))


class ColorSupport(enum.Enum):
    MONO = "mono"
    STANDARD = "standard"
    COLORS256 = "colors256"
    TRUECOLOR = "truecolor"


def sRGB(c1, c2):
    r1, g1, b1 = c1
    r2, g2, b2 = c2

    if r1 + r2 < 256:
        return 2 * (r1 - r2) ** 2 + 4 * (g1 - g2) ** 2 + 3 * (b1 - b2) ** 2
    else:
        return 3 * (r1 - r2) ** 2 + 4 * (g1 - g2) ** 2 + 2 * (b1 - b2) ** 2


def find_256color(code):
    code = min(colors_256, key=lambda c: sRGB(c, code))
    return colors_256.index(code) + 16


def find_16color(code):
    code = min(colors_16, key=lambda c: sRGB(c, code))
    return colors_16.index(code)


class Palette(enum.Enum):
    DEFAULT = "default"
    BLACK = "black"
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    BLUE = "blue"
    MAGENTA = "magenta"
    CYAN = "cyan"
    WHITE = "white"
    BRIGHT_BLACK = "bright_black"
    BRIGHT_RED = "bright_red"
    BRIGHT_GREEN = "bright_green"
    BRIGHT_YELLOW = "bright_yellow"
    BRIGHT_BLUE = "bright_blue"
    BRIGHT_MAGENTA = "bright_magenta"
    BRIGHT_CYAN = "bright_cyan"
    BRIGHT_WHITE = "bright_white"


color_names = [color.value for color in Palette]


@dataclasses.dataclass(frozen=True)
class Color(Pair):
    name = "color"
    _palette = {
        Palette.DEFAULT: 39,
        Palette.BLACK: 30,
        Palette.RED: 31,
        Palette.GREEN: 32,
        Palette.YELLOW: 33,
        Palette.BLUE: 34,
        Palette.MAGENTA: 35,
        Palette.CYAN: 36,
        Palette.WHITE: 37,
        Palette.BRIGHT_BLACK: 90,
        Palette.BRIGHT_RED: 91,
        Palette.BRIGHT_GREEN: 92,
        Palette.BRIGHT_YELLOW: 93,
        Palette.BRIGHT_BLUE: 94,
        Palette.BRIGHT_MAGENTA: 95,
        Palette.BRIGHT_CYAN: 96,
        Palette.BRIGHT_WHITE: 97,
    }
    rgb: Union[Palette, int]
    color_support: ColorSupport

    @classmethod
    def parse(cls, param):
        if param is None:
            param = "default"

        try:
            rgb = Palette(param) if param in color_names else int(param, 16)
        except ValueError:
            raise MarkupParseError(f"invalid parameter for tag [{cls.name}]: {param}")
        return (rgb,)

    @property
    def param(self):
        return self.rgb.value if isinstance(self.rgb, Palette) else f"{self.rgb:06x}"

    def expand(self):
        if isinstance(self.rgb, Palette):
            if self.color_support is not ColorSupport.MONO:
                return SGR(
                    tuple(child.expand() for child in self.children),
                    (self._palette[self.rgb],),
                )
            else:
                return Group(tuple(child.expand() for child in self.children))

        r = (self.rgb & 0xFF0000) >> 16
        g = (self.rgb & 0x00FF00) >> 8
        b = self.rgb & 0x0000FF
        if self.color_support is ColorSupport.TRUECOLOR:
            return SGR(
                tuple(child.expand() for child in self.children), (38, 2, r, g, b)
            )
        elif self.color_support is ColorSupport.COLORS256:
            c = find_256color((r, g, b))
            return SGR(tuple(child.expand() for child in self.children), (38, 5, c))
        elif self.color_support is ColorSupport.STANDARD:
            c = find_16color((r, g, b))
            if c < 8:
                c += 30
            else:
                c += 82
            return SGR(tuple(child.expand() for child in self.children), (c,))
        else:
            return Group(tuple(child.expand() for child in self.children))


@dataclasses.dataclass(frozen=True)
class BgColor(Pair):
    name = "bgcolor"
    _palette = {
        Palette.DEFAULT: 49,
        Palette.BLACK: 40,
        Palette.RED: 41,
        Palette.GREEN: 42,
        Palette.YELLOW: 43,
        Palette.BLUE: 44,
        Palette.MAGENTA: 45,
        Palette.CYAN: 46,
        Palette.WHITE: 47,
        Palette.BRIGHT_BLACK: 100,
        Palette.BRIGHT_RED: 101,
        Palette.BRIGHT_GREEN: 102,
        Palette.BRIGHT_YELLOW: 103,
        Palette.BRIGHT_BLUE: 104,
        Palette.BRIGHT_MAGENTA: 105,
        Palette.BRIGHT_CYAN: 106,
        Palette.BRIGHT_WHITE: 107,
    }
    rgb: Union[Palette, int]
    color_support: ColorSupport

    @classmethod
    def parse(cls, param):
        if param is None:
            param = "default"

        try:
            rgb = Palette(param) if param in color_names else int(param, 16)
        except ValueError:
            raise MarkupParseError(f"invalid parameter for tag [{cls.name}]: {param}")
        return (rgb,)

    @property
    def param(self):
        return self.rgb.value if isinstance(self.rgb, Palette) else f"{self.rgb:06x}"

    def expand(self):
        if isinstance(self.rgb, Palette):
            if self.color_support is not ColorSupport.MONO:
                return SGR(
                    tuple(child.expand() for child in self.children),
                    (self._palette[self.rgb],),
                )
            else:
                return Group(tuple(child.expand() for child in self.children))

        r = (self.rgb & 0xFF0000) >> 16
        g = (self.rgb & 0x00FF00) >> 8
        b = self.rgb & 0x0000FF
        if self.color_support is ColorSupport.TRUECOLOR:
            return SGR(
                tuple(child.expand() for child in self.children), (48, 2, r, g, b)
            )
        elif self.color_support is ColorSupport.COLORS256:
            c = find_256color((r, g, b))
            return SGR(tuple(child.expand() for child in self.children), (48, 5, c))
        elif self.color_support is ColorSupport.STANDARD:
            c = find_16color((r, g, b))
            if c < 8:
                c += 40
            else:
                c += 92
            return SGR(tuple(child.expand() for child in self.children), (c,))
        else:
            return Group(tuple(child.expand() for child in self.children))


# C0 control characters
@dataclasses.dataclass(frozen=True)
class ControlCharacter(Single):
    # character

    @classmethod
    def parse(cls, param):
        if param is not None:
            raise MarkupParseError(f"no parameter is needed for tag [{cls.name}/]")
        return ()

    @property
    def param(self):
        return None


@dataclasses.dataclass(frozen=True)
class BEL(ControlCharacter):
    name = "bel"
    character = "\a"


@dataclasses.dataclass(frozen=True)
class BS(ControlCharacter):
    name = "bs"
    character = "\b"


@dataclasses.dataclass(frozen=True)
class CR(ControlCharacter):
    name = "cr"
    character = "\r"


@dataclasses.dataclass(frozen=True)
class VT(ControlCharacter):
    name = "vt"
    character = "\v"


@dataclasses.dataclass(frozen=True)
class FF(ControlCharacter):
    name = "ff"
    character = "\f"


# others
@dataclasses.dataclass(frozen=True)
class Tab(ControlCharacter):
    name = "tab"
    character = "\t"


@dataclasses.dataclass(frozen=True)
class Newline(ControlCharacter):
    name = "nl"
    character = "\n"


@dataclasses.dataclass(frozen=True)
class Space(Single):
    name = "sp"
    character = " "

    @classmethod
    def parse(cls, param):
        if param is not None:
            raise MarkupParseError(f"no parameter is needed for tag [{cls.name}/]")
        return ()

    @property
    def param(self):
        return None

    def expand(self):
        return Text(self.character)


@dataclasses.dataclass(frozen=True)
class Wide(Single):
    name = "wide"
    char: str
    unicode_version: str

    @classmethod
    def parse(cls, param):
        if param is None:
            raise MarkupParseError(f"missing parameter for tag [{cls.name}/]")
        if len(param) != 1 or not param.isprintable():
            raise MarkupParseError(f"invalid parameter for tag [{cls.name}/]: {param}")
        return (param,)

    @property
    def param(self):
        return self.char

    def expand(self):
        w = wcwidth.wcwidth(self.char, self.unicode_version)
        if w == 1:
            return Text(self.char + " ")
        else:
            return Text(self.char)


# bar position
@dataclasses.dataclass(frozen=True)
class X(Single):
    name = "x"
    x: int

    @classmethod
    def parse(cls, param):
        if param is None:
            raise MarkupParseError(f"missing parameter for tag [{cls.name}]")
        try:
            x = int(param)
        except ValueError:
            raise MarkupParseError(f"invalid parameter for tag [{cls.name}]: {param}")
        return (x,)

    @property
    def param(self):
        return str(self.x)


@dataclasses.dataclass(frozen=True)
class DX(Single):
    name = "dx"
    dx: int

    @classmethod
    def parse(cls, param):
        if param is None:
            raise MarkupParseError(f"missing parameter for tag [{cls.name}]")
        try:
            dx = int(param)
        except ValueError:
            raise MarkupParseError(f"invalid parameter for tag [{cls.name}]: {param}")
        return (dx,)

    @property
    def param(self):
        return str(self.dx)


@dataclasses.dataclass(frozen=True)
class Restore(Pair):
    name = "restore"

    @classmethod
    def parse(cls, param):
        if param is not None:
            raise MarkupParseError(f"no parameter is needed for tag [{cls.name}]")
        return ()

    @property
    def param(self):
        return None


@dataclasses.dataclass(frozen=True)
class Mask(Pair):
    name = "mask"
    mask: slice

    @classmethod
    def parse(cls, param):
        try:
            start, stop = [int(p) if p else None for p in (param or ":").split(":")]
        except ValueError:
            raise MarkupParseError(f"invalid parameter for tag [{cls.name}]: {param}")
        return (slice(start, stop),)

    @property
    def param(self):
        return f"{self.mask.start if self.mask.start is not None else ''}:{self.mask.stop if self.mask.stop is not None else ''}"


@dataclasses.dataclass(frozen=True)
class Rich(Pair):
    name = "rich"

    @classmethod
    def parse(cls, param):
        if param is not None:
            raise MarkupParseError(f"no parameter is needed for tag [{cls.name}/]")
        return ()

    @property
    def param(self):
        return None

    def expand(self):
        return Group(self.children).expand()


class RichParser:
    default_tags = {
        Rich.name: Rich,
        Reset.name: Reset,
        Weight.name: Weight,
        Italic.name: Italic,
        Underline.name: Underline,
        Strike.name: Strike,
        Blink.name: Blink,
        Invert.name: Invert,
        Color.name: Color,
        BgColor.name: BgColor,
        Tab.name: Tab,
        Newline.name: Newline,
        Space.name: Space,
        Wide.name: Wide,
        X.name: X,
        DX.name: DX,
        Restore.name: Restore,
        Mask.name: Mask,
    }

    def __init__(self, unicode_version="auto", color_support=ColorSupport.TRUECOLOR):
        self.tags = dict(self.default_tags)
        self.unicode_version = unicode_version
        self.color_support = color_support

    @property
    def props(self):
        return {
            Wide.name: (self.unicode_version,),
            Color.name: (self.color_support,),
            BgColor.name: (self.color_support,),
        }

    def parse(self, markup_str, expand=True, slotted=False, root_tag=False):
        if root_tag and not markup_str.startswith(f"[{Rich.name}]"):
            return Text(markup_str)

        tags = self.tags if not slotted else dict(self.tags, slot=Slot)
        markup = parse_markup(markup_str, tags, self.props)
        if expand:
            markup = markup.expand()
        return markup

    def add_single_template(self, name, template):
        tag = make_single_template(name, template, self.tags, self.props)
        self.tags[tag.name] = tag
        return tag

    def add_pair_template(self, name, template):
        tag = make_pair_template(name, template, self.tags, self.props)
        self.tags[tag.name] = tag
        return tag

    def widthof(self, text):
        width = 0
        for ch in text:
            w = wcwidth.wcwidth(ch, self.unicode_version)
            if w == -1:
                return -1
            width += w
        return width


class RichRenderer:
    def __init__(self, unicode_version="auto"):
        self.unicode_version = unicode_version
        self.reset_default()

    def add_default(self, *attrs):
        for attr in attrs:
            if not isinstance(attr, SGR):
                raise TypeError(f"Invalid markup for default attribute: {type(attr)}")
            self.default_ansi_code += attr.ansi_code

    def reset_default(self):
        self.default_ansi_code = Reset((), "on").expand().ansi_code

    def clear_line(self):
        return Group((Clear(ClearRegion.line), CR()))

    def clear_below(self):
        return Group((Clear(ClearRegion.to_end), CR()))

    def clear_screen(self):
        return Group((Clear(ClearRegion.screen), Pos(0, 0)))

    def _render(self, markup, reopens=()):
        if isinstance(markup, Text):
            yield markup.string

        elif isinstance(markup, ControlCharacter):
            yield markup.character

        elif isinstance(markup, Group):
            for child in markup.children:
                yield from self._render(child, reopens)

        elif isinstance(markup, CSI):
            yield markup.ansi_code

        elif isinstance(markup, SGR):
            open = markup.ansi_code
            close = open and self.default_ansi_code

            if open:
                yield open
            for child in markup.children:
                yield from self._render(child, (open, *reopens))
            if close:
                yield close
            for reopen in reopens[::-1]:
                if reopen:
                    yield reopen

        else:
            raise TypeError(f"unknown markup type: {type(markup)}")

    def render(self, markup):
        return "".join(self._render(markup))

    def _render_context(self, markup, print, reopens=()):
        if isinstance(markup, Text):
            print(markup.string)

        elif isinstance(markup, ControlCharacter):
            yield markup.character

        elif isinstance(markup, CSI):
            print(markup.ansi_code)

        elif isinstance(markup, Slot):
            yield

        elif isinstance(markup, Group):
            if not markup.children:
                return
            child = markup.children[0]
            try:
                yield from self._render_context(child, print, reopens)
            finally:
                yield from self._render_context(
                    Group(markup.children[1:]), print, reopens
                )

        elif isinstance(markup, SGR):
            open = markup.ansi_code
            close = open and self.default_ansi_code

            if open:
                print(open)

            try:
                yield from self._render_context(
                    Group(markup.children), print, (open, *reopens)
                )
            finally:
                if close:
                    print(close)
                for reopen in reopens[::-1]:
                    if reopen:
                        print(reopen)

        else:
            raise TypeError(f"unknown markup type: {type(markup)}")

    @contextlib.contextmanager
    def render_context(self, markup, print):
        yield from self._render_context(markup, print)

    def _less(self, markup, size, pos=(0, 0), reopens=(), wrap=True):
        if pos is None:
            return None

        elif isinstance(markup, Text):
            x, y = pos
            for ch in markup.string:
                if ch == "\n":
                    y += 1
                    x = 0

                else:
                    w = wcwidth.wcwidth(ch, self.unicode_version)
                    if w == -1:
                        raise ValueError(f"unprintable character: {repr(ch)}")
                    x += w
                    if wrap and x > size.columns:
                        y += 1
                        x = w

                if y == size.lines:
                    return None
                if x <= size.columns:
                    yield ch
            return x, y

        elif isinstance(markup, Newline):
            x, y = pos
            y += 1
            x = 0

            if y == size.lines:
                return None
            if x <= size.columns:
                yield markup.character
            return x, y

        elif isinstance(markup, Group):
            for child in markup.children:
                pos = yield from self._less(child, size, pos, reopens, wrap=wrap)
            return pos

        elif isinstance(markup, SGR):
            open = markup.ansi_code
            close = open and self.default_ansi_code

            if open:
                yield open
            for child in markup.children:
                pos = yield from self._less(
                    child, size, pos, (open, *reopens), wrap=wrap
                )
            if close:
                yield close
            for reopen in reopens[::-1]:
                if reopen:
                    yield reopen
            return pos

        else:
            raise TypeError(f"unknown markup type: {type(markup)}")

    def render_less(self, markup, size, pos=(0, 0), wrap=True, restore=True):
        def _restore_pos(markup, size, pos, wrap):
            x0, y0 = pos
            pos = yield from self._less(markup, size, pos, wrap=wrap)
            x, y = pos or (None, size.lines - 1)
            if y > y0:
                yield f"\x1b[{y-y0}A"
            yield "\r"
            if x0 > 0:
                yield f"\x1b[{x0}C"

        if restore:
            markup = _restore_pos(markup, size, pos, wrap)
        return "".join(markup)

    def _render_bar_text(self, buffer, string, x, width, xmask, attrs):
        start = xmask.start
        stop = xmask.stop
        for ch in string:
            w = wcwidth.wcwidth(ch, self.unicode_version)
            if w == -1:
                raise ValueError(f"invalid string: {repr(ch)} in {repr(string)}")

            if x + w > stop:
                break

            if x < start:
                x += w
                continue

            # assert start <= x <= x+w <= stop

            if w == 0:
                x_ = x - 1
                if 0 <= x_ and buffer[x_] == "":
                    x_ -= 1
                if start <= x_:
                    buffer[x_] += ch

            elif w == 1:
                if 0 <= x - 1 and buffer[x] == "":
                    buffer[x - 1] = " "
                if x + 1 < width and buffer[x + 1] == "":
                    buffer[x + 1] = " "
                buffer[x] = (
                    ch if not attrs else f"\x1b[{';'.join(map(str, attrs))}m{ch}\x1b[m"
                )
                x += 1

            else:
                x_ = x + 1
                if 0 <= x - 1 and buffer[x] == "":
                    buffer[x - 1] = " "
                if x_ + 1 < width and buffer[x_ + 1] == "":
                    buffer[x_ + 1] = " "
                buffer[x] = (
                    ch if not attrs else f"\x1b[{';'.join(map(str, attrs))}m{ch}\x1b[m"
                )
                buffer[x_] = ""
                x += 2

        return x

    def _render_bar(self, buffer, markup, x, width, xmask, attrs):
        if isinstance(markup, Text):
            return self._render_bar_text(buffer, markup.string, x, width, xmask, attrs)

        elif isinstance(markup, Group):
            for child in markup.children:
                x = self._render_bar(buffer, child, x, width, xmask, attrs)
            return x

        elif isinstance(markup, SGR):
            attrs = (*attrs, *markup.attr)
            for child in markup.children:
                x = self._render_bar(buffer, child, x, width, xmask, attrs)
            return x

        elif isinstance(markup, X):
            return markup.x

        elif isinstance(markup, DX):
            return x + markup.dx

        elif isinstance(markup, Restore):
            x0 = x
            for child in markup.children:
                x = self._render_bar(buffer, child, x, width, xmask, attrs)
            return x0

        elif isinstance(markup, Mask):
            xmask = clamp(range(width)[markup.mask], xmask)
            for child in markup.children:
                x = self._render_bar(buffer, child, x, width, xmask, attrs)
            return x

        else:
            raise TypeError(f"unknown markup type: {type(markup)}")

    def render_bar(self, width, markup):
        buffer = [" "] * width
        self._render_bar(buffer, markup, 0, width, range(width), ())
        return "".join(buffer).rstrip()


def clamp(ran, mask):
    start = min(max(mask.start, ran.start), mask.stop)
    stop = max(min(mask.stop, ran.stop), mask.start)
    return range(start, stop)
