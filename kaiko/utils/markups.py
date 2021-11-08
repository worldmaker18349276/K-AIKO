import re
import ast
import enum
import dataclasses

# pair: [tag=param]...[/]
# single: [tag=param/]
# escape: \\, \n, \x3A (python's escapes), \[, \]

# basic:
#   csi: [csi=2A/]  =>  "\x1b[2A"
#   sgr: [sgr=2;3]...[/]  =>  "\x1b[2;3m...\x1b[m"
# template:
#   pair: [color=green]>>> [weight=bold][slot/][/]
#   single: [color=red]!!![/]

class MarkupParseError(Exception):
    pass

def parse_markup(markup, tags=[]):
    tags_dict = {tag.name: tag for tag in tags}
    stack = [Group([])]

    for match in re.finditer(r"(?P<tag>\[([^\]]*)\])|(?P<text>([^\[\\]|\\.)+)", markup):
        tag = match.group('tag')
        text = match.group('text')

        if text is not None:
            # process backslash escapes
            raw = text
            raw = re.sub(r"(?<!\\)(\\\\)*\\\[", r"\1[", raw) # \[ => [
            raw = re.sub(r"(?<!\\)(\\\\)*\\\]", r"\1]", raw) # \] => ]
            raw = re.sub(r"(?<!\\)(\\\\)*'", r"\1\\'", raw)  # ' => \'
            try:
                raw = ast.literal_eval("'" + raw + "'")
            except SyntaxError:
                raise MarkupParseError(f"invalid text: {repr(text)}")
            stack[-1].children.append(Text(raw))
            continue

        if tag == "/": # [/]
            if len(stack) <= 1:
                raise MarkupParseError(f"too many closing tag: [/]")
            stack.pop()
            continue

        match = re.match("^(\w+)(?:=(.*))?/$", tag) # [tag=param/]
        if match:
            name = match.group(1)
            param = match.group(2)
            if name not in tags_dict or not issubclass(tags_dict[name], Single):
                raise MarkupParseError(f"unknown tag: [{name}/]")
            res = tags_dict[name].parse(param)
            stack[-1].children.append(res)
            continue

        match = re.match("^(\w+)(?:=(.*))?$", tag) # [tag=param]
        if match:
            name = match.group(1)
            param = match.group(2)
            if name not in tags_dict or not issubclass(tags_dict[name], Pair):
                raise MarkupParseError(f"unknown tag: [{name}]")
            res = tags_dict[name].parse(param)
            stack[-1].children.append(res)
            stack.append(res)
            continue

        raise MarkupParseError(f"invalid tag: [{tag}]")

    return stack[0]

class Node:
    def represent(self):
        raise NotImplementedError

    def __str__(self):
        return "".join(self.represent())

    def expand(self):
        return self

@dataclasses.dataclass
class Text(Node):
    string: str

    @classmethod
    def parse(clz, param):
        raise ValueError("no parser for text")

    def represent(self):
        for ch in self.string:
            if ch == "\\":
                yield r"\\"
            elif ch == "[":
                yield r"\["
            else:
                yield repr(ch)[1:-1]

@dataclasses.dataclass
class Group(Node):
    children: list

    @classmethod
    def parse(clz, param):
        raise ValueError("no parser for group")

    def represent(self):
        for child in self.children:
            yield from child.represent()

    def expand(self):
        return dataclasses.replace(self, children=[child.expand() for child in self.children])

class Tag(Node):
    @classmethod
    def parse(clz, param):
        raise NotImplementedError

@dataclasses.dataclass
class Single(Tag):
    # name

    @property
    def param(self):
        raise NotImplementedError

    def represent(self):
        param = self.param
        param_str = f"={param}" if param is not None else ""
        yield f"[{self.name}{param_str}/]"

@dataclasses.dataclass
class Pair(Tag):
    # name
    children: list

    @property
    def param(self):
        raise NotImplementedError

    def represent(self):
        param = self.param
        param_str = f"={param}" if param is not None else ""
        yield f"[{self.name}{param_str}]"
        for child in self.children:
            yield from child.represent()
        yield f"[/]"

    def expand(self):
        return dataclasses.replace(self, children=[child.expand() for child in self.children])


def render_ansi(node, *reopens):
    if isinstance(node, Text):
        yield from node.string

    if isinstance(node, Group):
        for child in node.children:
            yield from render_ansi(child, *reopens)

    if isinstance(node, CSI):
        yield f"\x1b[{node.code}"

    elif isinstance(node, SGR):
        open = close = None
        if node.attr:
            open = f"\x1b[{';'.join(map(str, node.attr))}m"
            close = "\x1b[m"

        if open:
            yield open
        for child in node.children:
            yield from render_ansi(child, open, *reopens)
        if close:
            yield close
        for reopen in reopens[::-1]:
            if reopen:
                yield reopen

    else:
        raise TypeError(f"unknown node type: {type(node)}")

# ctrl code
@dataclasses.dataclass
class CSI(Single):
    name = "csi"
    code: str

    @classmethod
    def parse(clz, param):
        if param is None:
            raise MarkupParseError(f"missing parameter for tag [{clz.name}/]")
        return clz(param)

    @property
    def param(self):
        return self.code

@dataclasses.dataclass
class Move(Single):
    name = "move"
    x: int
    y: int

    @classmethod
    def parse(clz, param):
        if param is None:
            raise MarkupParseError(f"missing parameter for tag [{clz.name}/]")
        try:
            x, y = tuple(int(n) for n in param.split(","))
        except ValueError:
            raise MarkupParseError(f"invalid parameter for tag [{clz.name}/]: {param}")
        return clz(x, y)

    @property
    def param(self):
        return f"{self.x},{self.y}"

    def expand(self):
        res = []
        if x > 0:
            res.append(CSI(f"{x}C"))
        elif x < 0:
            res.append(CSI(f"{-x}D"))
        if y > 0:
            res.append(CSI(f"{y}B"))
        elif y < 0:
            res.append(CSI(f"{-y}A"))
        return Group(res)

@dataclasses.dataclass
class Pos(Single):
    name = "pos"
    x: int
    y: int

    @classmethod
    def parse(clz, param):
        if param is None:
            raise MarkupParseError(f"missing parameter for tag [{clz.name}/]")
        try:
            x, y = tuple(int(n) for n in param.split(","))
        except ValueError:
            raise MarkupParseError(f"invalid parameter for tag [{clz.name}/]: {param}")
        return clz(x, y)

    @property
    def param(self):
        return f"{self.x},{self.y}"

    def expand(self):
        return CSI(f"{self.y+1};{self.x+1}H")

@dataclasses.dataclass
class Scroll(Single):
    name = "scroll"
    x: int

    @classmethod
    def parse(clz, param):
        if param is None:
            raise MarkupParseError(f"missing parameter for tag [{clz.name}/]")
        try:
            x = int(param)
        except ValueError:
            raise MarkupParseError(f"invalid parameter for tag [{clz.name}/]: {param}")
        return clz(x)

    @property
    def param(self):
        return str(self.x)

    def expand(self):
        if x > 0:
            return CSI(f"{self.x}T")
        elif x < 0:
            return CSI(f"{-self.x}S")
        else:
            return Group([])

class ClearRegion(enum.Enum):
    to_right = "0K"
    to_left = "1K"
    line = "2K"
    to_end = "0J"
    to_beginning = "1J"
    screen = "2J"

@dataclasses.dataclass
class Clear(Single):
    region: ClearRegion

    def parse(self, param):
        if param is None:
            raise MarkupParseError(f"missing parameter for tag [{clz.name}/]")
        if all(param != region.name for region in ClearRegion):
            raise MarkupParseError(f"invalid parameter for tag [{clz.name}/]: {param}")
        region = ClearRegion[param]

        return clz(region)

    @property
    def param(self):
        return self.region.name

    def expand(self):
        return CSI(f"{self.region.value}")


# attr code
@dataclasses.dataclass
class SGR(Pair):
    name = "sgr"
    attr: tuple

    @classmethod
    def parse(clz, param):
        if param is None:
            return clz([], ())

        try:
            attr = tuple(int(n or "0") for n in param.split(";"))
        except ValueError:
            raise ValueError(f"invalid parameter for tag [{clz.name}]: {param}")
        return clz([], attr)

    @property
    def param(self):
        if not self.attr:
            return None
        return ';'.join(map(str, self.attr))

@dataclasses.dataclass
class SimpleAttr(Pair):
    param: str

    def parse(clz, param):
        if param is None:
            param = next(iter(clz._options.keys()))
        if param not in clz._options:
            raise ValueError(f"invalid parameter for tag [{clz.name}]: {param}")
        return clz([], param)

    def expand(self):
        yield SGR(self.children, (self._options[self.param],))

@dataclasses.dataclass
class Reset(SimpleAttr):
    name = "reset"
    _options = {"on": 0}

@dataclasses.dataclass
class Weight(SimpleAttr):
    name = "weight"
    _options = {"bold": 1, "dim": 2, "normal": 22}

@dataclasses.dataclass
class Italic(SimpleAttr):
    name = "italic"
    _options = {"on": 3, "off": 23}

@dataclasses.dataclass
class Underline(SimpleAttr):
    name = "underline"
    _options = {"on": 4, "double": 21, "off": 24}

@dataclasses.dataclass
class Strike(SimpleAttr):
    name = "strike"
    _options = {"on": 9, "off": 29}

@dataclasses.dataclass
class Blink(SimpleAttr):
    name = "blink"
    _options = {"on": 5, "off": 25}

@dataclasses.dataclass
class Invert(SimpleAttr):
    name = "invert"
    _options = {"on": 7, "off": 27}

@dataclasses.dataclass
class Color(SimpleAttr):
    name = "color"
    _options = {
        "default": 39,
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "white": 37,
        "bright_black": 90,
        "bright_red": 91,
        "bright_green": 92,
        "bright_yellow": 93,
        "bright_blue": 94,
        "bright_magenta": 95,
        "bright_cyan": 96,
        "bright_white": 97,
    }

@dataclasses.dataclass
class BgColor(SimpleAttr):
    name = "bgcolor"
    _options = {
        "default": 49,
        "black": 40,
        "red": 41,
        "green": 42,
        "yellow": 43,
        "blue": 44,
        "magenta": 45,
        "cyan": 46,
        "white": 47,
        "bright_black": 100,
        "bright_red": 101,
        "bright_green": 102,
        "bright_yellow": 103,
        "bright_blue": 104,
        "bright_magenta": 105,
        "bright_cyan": 106,
        "bright_white": 107,
    }


# template
@dataclasses.dataclass
class SingleTemplate(Single):
    # name
    # _template

    @classmethod
    def parse(clz, param):
        if param is not None:
            raise MarkupParseError("no parameter is needed for template tag")
        return clz()

    @property
    def param(self):
        return None

    def expand(self):
        return self._template

@dataclasses.dataclass
class Slot(Single):
    name = "slot"

    def parse(clz, param):
        if param is not None:
            raise MarkupParseError(f"no parameter is needed for tag [{clz.name}/]")

    @property
    def param(self):
        return None

def replace_slot(node, children):
    if isinstance(node, Slot):
        return Group(children)
    elif isinstance(node, (Single, Text)):
        return node
    elif isinstance(node, (Pair, Group)):
        return dataclasses.replace(node, children=[replace_slot(child, children) for child in node.children])
    else:
        raise TypeError(f"unknown node type {type(node)}")

@dataclasses.dataclass
class PairTemplate(Pair):
    # name
    # _template

    @classmethod
    def parse(clz, param):
        if param is not None:
            raise MarkupParseError("no parameter is needed for template tag")
        return clz([])

    @property
    def param(self):
        return None

    def expand(self):
        return replace_slot(self._template, self.children).expand()

def make_single_template(name, template, tags=[]):
    temp = parse_markup(template, tags=tags)
    return type(name + "Template", (SingleTemplate,), dict(name=name, _template=temp))

def make_pair_template(name, template, tags=[]):
    temp = parse_markup(template, tags=list(Slot, *tags))
    return type(name + "Template", (PairTemplate,), dict(name=name, _template=temp))

