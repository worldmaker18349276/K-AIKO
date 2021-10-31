import re

class Sequence:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def represent(self):
        raise NotImplementedError

    def construct(self, ctxt=()):
        raise NotImplementedError

    def __repr__(self):
        return "".join(self.represent())

    def __str__(self):
        return "".join(self.construct())

class RawSequence(Sequence):
    def __init__(self, string):
        self.buffer = list(string)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, key):
        return RawSequence(self.buffer[key])

    def represent(self):
        for ch in self.buffer:
            if ch == "[":
                yield "[bra/]"
            elif ch == "]":
                yield "[ket/]"
            elif not ch.isprintable():
                yield f"[chr={hex(ord(ch))}/]"
            else:
                yield ch

    def construct(self, ctxt=()):
        yield from self.buffer


class ControlSequence(Sequence):
    tag = "ctrl"

    def __init__(self, code):
        self.code = code

    @classmethod
    def parse_param(clz, param):
        if param is None:
            raise SequenceParseError(f"missing parameter for tag [{clz.tag}/]")
        return param

    def __len__(self):
        return 1

    def __getitem__(self, key):
        if isinstance(key, slice):
            return RawSequence("") if range(1)[key] else self

        if not isinstance(key, int):
            raise TypeError(f"buffer indices must be integers, not {type(key)}")

        if key != 0 and key != -1:
            raise IndexError("buffer index out of range")

        return self

    def represent(self):
        yield f"[{self.tag}={self.code}/]"

    def construct(self, ctxt=()):
        yield "\x1b[" + self.code

class MoveSequence(ControlSequence):
    tag = "move"

    def __init__(self, args):
        self.args = args

    @classmethod
    def parse_param(clz, param):
        if param is None:
            raise SequenceParseError(f"missing parameter for tag [{clz.tag}/]")
        try:
            args = tuple(int(n) for n in param.split(","))
        except ValueError:
            raise SequenceParseError(f"invalid parameter for tag [{clz.tag}/]: {param}")
        if len(args) != 2:
            raise SequenceParseError(f"invalid parameter for tag [{clz.tag}/]: {param}")
        return args

    def represent(self):
        yield f"[{self.tag}={','.join(map(str, self.args))}/]"

    def construct(self, ctxt=()):
        x, y = self.args
        if x > 0:
            yield f"\x1b[{x}C"
        elif x < 0:
            yield f"\x1b[{-x}D"
        if y > 0:
            yield f"\x1b[{y}B"
        elif y < 0:
            yield f"\x1b[{-y}A"

class PosSequence(ControlSequence):
    tag = "pos"

    def __init__(self, args):
        self.args = args

    @classmethod
    def parse_param(clz, param):
        if param is None:
            raise SequenceParseError(f"missing parameter for tag [{clz.tag}/]")
        try:
            args = tuple(int(n) for n in param.split(","))
        except ValueError:
            raise SequenceParseError(f"invalid parameter for tag [{clz.tag}/]: {param}")
        if len(args) != 2:
            raise SequenceParseError(f"invalid parameter for tag [{clz.tag}/]: {param}")
        return args

    def represent(self):
        yield f"[{self.tag}={','.join(map(str, self.args))}/]"

    def construct(self, ctxt=()):
        x, y = self.args
        yield f"\x1b[{y+1};{x+1}H"

class ScrollSequence(ControlSequence):
    tag = "scroll"

    def __init__(self, arg):
        self.arg = arg

    @classmethod
    def parse_param(clz, param):
        if param is None:
            raise SequenceParseError(f"missing parameter for tag [{clz.tag}/]")
        try:
            param = int(param)
        except ValueError:
            raise SequenceParseError(f"invalid parameter for tag [{clz.tag}/]: {param}")
        return param

    def represent(self):
        yield f"[{self.tag}={self.arg}/]"

    def construct(self, ctxt=()):
        if self.arg > 0:
            yield f"\x1b[{x}T"
        elif self.arg < 0:
            yield f"\x1b[{-x}S"

class ClearSequence(ControlSequence):
    tag = "clear"
    _options = {
        "to_right": "0K",
        "to_left": "1K",
        "line": "2K",
        "to_end": "0J",
        "to_beginning": "1J",
        "screen": "2J",
    }

    def __init__(self, option):
        self.option = option

    @classmethod
    def parse_param(clz, param):
        if param is None:
            raise SequenceParseError(f"missing parameter for tag [{clz.tag}/]")
        if param not in clz._options:
            raise SequenceParseError(f"invalid parameter for tag [{clz.tag}/]: {param}")
        return param

    def represent(self):
        yield f"[{self.tag}={self.option}/]"

    def construct(self, ctxt=()):
        yield "\x1b[" + self._options[self.option]


class AttributeSequence(Sequence):
    tag = "attr"

    def __init__(self, children, attr):
        self.children = children
        self.attr = attr

    def replace(self, children):
        return type(self)(children, self.attr)

    @classmethod
    def parse_param(clz, param):
        if param is None:
            raise SequenceParseError(f"missing parameter for tag [{clz.tag}/]")
        try:
            param = tuple(int(n or "0") for n in param.split(";"))
        except ValueError:
            raise SequenceParseError(f"invalid parameter for tag [{clz.tag}/]: {param}")
        return param

    def __len__(self):
        return sum(len(buffer) for buffer in self.children)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.slice(key)

        if not isinstance(key, int):
            raise TypeError(f"buffer indices must be integers, not {type(key)}")

        length = len(self)
        if key < 0:
            key = length + key
        if key not in range(length):
            raise IndexError("buffer index out of range")

        index = 0
        for buffer in self.children:
            index += len(buffer)
            if key < index:
                return self.replace([buffer[key-index]])

        raise RuntimeError("invalid state")

    def slice(self, mask):
        start, stop, step = mask.indices(len(self))
        if step != 1:
            raise ValueError("the step of the buffer slice should be 1")
        if stop < start:
            return RawSequence("")

        index = 0
        res = self.replace([])
        for buffer in self.children:
            index += len(buffer)
            if stop < index:
                res.children.append(buffer[start-index:stop-index])
                break
            elif start < index:
                res.children.append(buffer[start-index:])
        return res

    def _open_tag(self):
        if self.attr:
            yield f"[{self.tag}={';'.join(map(str, self.attr))}]"

    def _close_tag(self):
        if self.attr:
            yield f"[/{self.tag}]"

    def represent(self):
        yield from self._open_tag()

        for buffer in self.children:
            yield from buffer.represent()

        yield from self._close_tag()

    def construct(self, ctxt=()):
        opener = "" if self.attr == "" else f"\x1b[{';'.join(map(str, self.attr))}m"
        closer = "" if self.attr == "" else "\x1b[m"

        yield opener

        for buffer in self.children:
            if isinstance(buffer, AttributeSequence):
                yield from buffer.construct((*ctxt, opener))
                yield from ctxt
                yield opener

            else:
                yield from buffer.construct((*ctxt, opener))

        yield closer

class SimpleAttributeSequence(AttributeSequence):
    def __init__(self, children, option=True):
        self.option = option
        super().__init__(children, (self._attrs[option],))

    def replace(self, children):
        type(self)(children, self.option)

    @classmethod
    def parse_param(clz, param):
        if param is None:
            return next(iter(clz._attrs.keys()))
        if param not in clz._attrs:
            raise SequenceParseError(f"invalid parameter for tag [{clz.tag}/]: {param}")
        return param

    def _open_tag(self):
        if self.option == next(iter(self._attrs.keys())):
            yield f"[{self.tag}]"
        else:
            yield f"[{self.tag}={self.option}]"

    def _close_tag(self):
        yield f"[/{self.tag}]"

simple_attrs = {
    "normal": {"on": 0},
    "weight": {"bold": 1, "dim": 2, "normal": 21},
    "italic": {"on": 3, "off": 23},
    "underline": {"on": 4, "off": 24},
    "strike": {"on": 9, "off": 29},
    "blink": {"on": 5, "off": 25},
    "invert": {"on": 7, "off": 27},
    "hide": {"on": 8, "off": 28},
    "color": {
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
        },
    "bgcolor": {
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
        },
}


class Bra:
    tag = "bra"

    def __new__(self, param):
        return RawSequence("[")

    @classmethod
    def parse_param(clz, param):
        if param is not None:
            raise SequenceParseError(f"no parameter should be given for tag [{clz.tag}/]")
        return None

class Ket:
    tag = "ket"

    def __new__(self, param):
        return RawSequence("]")

    @classmethod
    def parse_param(clz, param):
        if param is not None:
            raise SequenceParseError(f"no parameter should be given for tag [{clz.tag}/]")
        return None

class Chr:
    tag = "chr"

    def __new__(self, param):
        return RawSequence(chr(param))

    @classmethod
    def parse_param(clz, param):
        if param is None:
            raise SequenceParseError(f"missing parameter for tag [{clz.tag}/]")
        try:
            if param.startswith("0x") or param.startswith("0X"):
                param = int(param, 16)
            elif param.startswith("0o") or param.startswith("0O"):
                param = int(param, 8)
            elif param.startswith("0b") or param.startswith("0B"):
                param = int(param, 2)
            else:
                param = int(param)
        except ValueError:
            raise SequenceParseError(f"invalid parameter for tag [{clz.tag}/]: {param}")
        return param

default_singles = {
    ControlSequence.tag: ControlSequence,
    MoveSequence.tag: MoveSequence,
    PosSequence.tag: PosSequence,
    ScrollSequence.tag: ScrollSequence,
    ClearSequence.tag: ClearSequence,
    Bra.tag: Bra,
    Ket.tag: Ket,
    Chr.tag: Chr,
}
default_pairs = {
    AttributeSequence.tag: AttributeSequence,
}
for name, attrs in simple_attrs.items():
    seq = type(name.capitalize() + "Sequence", (SimpleAttributeSequence,), {'tag': name, '_attrs': attrs})
    default_pairs[seq.tag] = seq

class SequenceParseError(Exception):
    pass

def parse_sequence(text, singles=default_singles, pairs=default_pairs):
    stack = [AttributeSequence([], "")]

    for match in re.finditer(r"\[([^\[\]]+)\]|([^\[]+)", text):
        tag = match.group(1)
        raw = match.group(2)

        if raw is not None:
            stack[-1].children.append(RawSequence(raw))
            continue

        match = re.match("^(\w+)(?:=(.*))?/$", tag)
        if match:
            name = match.group(1)
            param = match.group(2)
            if name not in singles:
                raise SequenceParseError(f"no such tag: [{name}/]")
            param = singles[name].parse_param(param)
            res = singles[name](param)
            stack[-1].children.append(res)
            continue

        match = re.match("^(\w+)(?:=(.*))?$", tag)
        if match:
            name = match.group(1)
            param = match.group(2)
            if name not in pairs:
                raise SequenceParseError(f"no such tag: [{name}]")

            param = pairs[name].parse_param(param)
            buffer = pairs[name]([], param)
            stack[-1].children.append(buffer)
            stack.append(buffer)
            continue

        match = re.match("^/(\w+)$", tag)
        if match:
            if len(stack) <= 1:
                raise SequenceParseError(f"too many closing tag: [/{name}]")
            if stack[-1].tag != match.group(1):
                raise SequenceParseError(f"mismatched tag: [/{name}]")
            stack.pop()
            continue

        if tag == "/":
            if len(stack) <= 1:
                raise SequenceParseError(f"too many closing tag: [/]")
            stack.pop()
            continue

        raise SequenceParseError(f"invalid syntax: [{tag}]")

    return stack[0]


