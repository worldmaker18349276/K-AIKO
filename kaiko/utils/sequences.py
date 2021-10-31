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

class AttributeSequence(Sequence):
    tag = "attr"

    def __init__(self, children, attr):
        self.children = children
        self.attr = attr

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
                return AttributeSequence([buffer[key-index]], self.attr)

        raise RuntimeError("invalid state")

    def slice(self, mask):
        start, stop, step = mask.indices(len(self))
        if step != 1:
            raise ValueError("the step of the buffer slice should be 1")
        if stop < start:
            return RawSequence("")

        index = 0
        res = AttributeSequence([], self.attr)
        for buffer in self.children:
            index += len(buffer)
            if stop < index:
                res.children.append(buffer[start-index:stop-index])
                break
            elif start < index:
                res.children.append(buffer[start-index:])
        return res

    def represent(self):
        if self.attr:
            yield f"[{self.tag}={';'.join(map(str, self.attr))}]"

        for buffer in self.children:
            yield from buffer.represent()

        if self.attr:
            yield "[/attr]"

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
    Bra.tag: Bra,
    Ket.tag: Ket,
    Chr.tag: Chr,
}
default_pairs = {
    AttributeSequence.tag: AttributeSequence,
}

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


