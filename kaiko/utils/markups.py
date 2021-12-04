import re
import ast
from typing import Sequence
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

def parse_markup(markup_str, tags):
    stack = [(Group, [])]

    for match in re.finditer(r"(?P<tag>\[[^\]]*\])|(?P<text>([^\[\\]|\\[\s\S])+)", markup_str):
        tag = match.group('tag')
        text = match.group('text')

        if text is not None:
            # process backslash escapes
            raw = text
            raw = re.sub(r"(?<!\\)((\\\\)*)\\\[", r"\1[", raw) # \[ => [
            raw = re.sub(r"(?<!\\)((\\\\)*)\\\]", r"\1]", raw) # \] => ]
            raw = re.sub(r"(?<!\\)((\\\\)*)'", r"\1\\'", raw)  # ' => \'
            raw = re.sub(r"(?<!\\)((\\\\)*)\r", r"\1\\r", raw)  # '\r' => \r
            try:
                raw = ast.literal_eval("'''" + raw + "'''")
            except SyntaxError:
                raise MarkupParseError(f"invalid text: {repr(text)}")
            stack[-1][1].append(Text(raw))
            continue

        if tag == "[/]": # [/]
            if len(stack) <= 1:
                raise MarkupParseError(f"too many closing tag: [/]")
            markup_type, children, *param = stack.pop()
            markup = markup_type(tuple(children), *param)
            stack[-1][1].append(markup)
            continue

        match = re.match("^\[(\w+)(?:=(.*))?/\]$", tag) # [tag=param/]
        if match:
            name = match.group(1)
            param_str = match.group(2)
            if name not in tags or not issubclass(tags[name], Single):
                raise MarkupParseError(f"unknown tag: [{name}/]")
            param = tags[name].parse(param_str)
            stack[-1][1].append(tags[name](*param))
            continue

        match = re.match("^\[(\w+)(?:=(.*))?\]$", tag) # [tag=param]
        if match:
            name = match.group(1)
            param_str = match.group(2)
            if name not in tags or not issubclass(tags[name], Pair):
                raise MarkupParseError(f"unknown tag: [{name}]")
            param = tags[name].parse(param_str)
            stack.append((tags[name], [], *param))
            continue

        raise MarkupParseError(f"invalid tag: {tag}")

    for i in range(len(stack)-1, 0, -1):
        markup_type, children, *param = stack[i]
        markup = markup_type(tuple(children), *param)
        stack[i-1][1].append(markup)
    markup_type, children, *param = stack[0]
    markup = markup_type(tuple(children), *param)
    return markup

def escape(text):
    return text.replace("[", "\\[").replace("\\", r"\\")

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
            if ch == "\\":
                yield r"\\"
            elif ch == "[":
                yield r"\["
            else:
                yield repr(ch)[1:-1]

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
        return dataclasses.replace(self, children=tuple(child.expand() for child in self.children))

    def traverse(self, markup_type, func):
        children = []
        modified = False
        for child in self.children:
            child_ = child.traverse(markup_type, func)
            children.append(child_)
            modified = modified or child_ is not child

        return self if not modified else dataclasses.replace(self, children=tuple(children))

class Tag(Markup):
    @classmethod
    def parse(clz, param):
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
        return dataclasses.replace(self, children=tuple(child.expand() for child in self.children))

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

            return self if not modified else dataclasses.replace(self, children=tuple(children))


# template
@dataclasses.dataclass(frozen=True)
class SingleTemplate(Single):
    # name
    # _template

    @classmethod
    def parse(clz, param):
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
    def parse(clz, param):
        if param is not None:
            raise MarkupParseError(f"no parameter is needed for tag [{clz.name}/]")
        return ()

    @property
    def param(self):
        return None

@dataclasses.dataclass(frozen=True)
class PairTemplate(Pair):
    # name
    # _template

    @classmethod
    def parse(clz, param):
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

def make_single_template(name, template, tags):
    temp = parse_markup(template, tags=tags)
    clz = type(name.capitalize(), (SingleTemplate,), dict(name=name, _template=temp))
    clz = dataclasses.dataclass(frozen=True)(clz)
    return clz

def make_pair_template(name, template, tags):
    temp = parse_markup(template, tags=dict(tags, slot=Slot))
    clz = type(name.capitalize(), (PairTemplate,), dict(name=name, _template=temp))
    clz = dataclasses.dataclass(frozen=True)(clz)
    return clz
