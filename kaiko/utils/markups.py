import re
import ast
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

def parse_markup(markup, tags):
    tags_dict = {tag.name: tag for tag in tags}
    stack = [Group([])]

    for match in re.finditer(r"(?P<tag>\[[^\]]*\])|(?P<text>([^\[\\]|\\[\s\S])+)", markup):
        tag = match.group('tag')
        text = match.group('text')

        if text is not None:
            # process backslash escapes
            raw = text
            raw = re.sub(r"(?<!\\)(\\\\)*\\\[", r"\1[", raw) # \[ => [
            raw = re.sub(r"(?<!\\)(\\\\)*\\\]", r"\1]", raw) # \] => ]
            raw = re.sub(r"(?<!\\)(\\\\)*'", r"\1\\'", raw)  # ' => \'
            try:
                raw = ast.literal_eval("'''" + raw + "'''")
            except SyntaxError:
                raise MarkupParseError(f"invalid text: {repr(text)}")
            stack[-1].children.append(Text(raw))
            continue

        if tag == "[/]": # [/]
            if len(stack) <= 1:
                raise MarkupParseError(f"too many closing tag: [/]")
            stack.pop()
            continue

        match = re.match("^\[(\w+)(?:=(.*))?/\]$", tag) # [tag=param/]
        if match:
            name = match.group(1)
            param = match.group(2)
            if name not in tags_dict or not issubclass(tags_dict[name], Single):
                raise MarkupParseError(f"unknown tag: [{name}/]")
            res = tags_dict[name].parse(param)
            stack[-1].children.append(res)
            continue

        match = re.match("^\[(\w+)(?:=(.*))?\]$", tag) # [tag=param]
        if match:
            name = match.group(1)
            param = match.group(2)
            if name not in tags_dict or not issubclass(tags_dict[name], Pair):
                raise MarkupParseError(f"unknown tag: [{name}]")
            res = tags_dict[name].parse(param)
            stack[-1].children.append(res)
            stack.append(res)
            continue

        raise MarkupParseError(f"invalid tag: {tag}")

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
        return self._template.expand()

@dataclasses.dataclass
class Slot(Single):
    name = "slot"

    @classmethod
    def parse(clz, param):
        if param is not None:
            raise MarkupParseError(f"no parameter is needed for tag [{clz.name}/]")
        return clz()

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

def make_single_template(name, template, tags):
    temp = parse_markup(template, tags=tags)
    return type(name.capitalize(), (SingleTemplate,), dict(name=name, _template=temp))

def make_pair_template(name, template, tags):
    temp = parse_markup(template, tags=[Slot, *tags])
    return type(name.capitalize(), (PairTemplate,), dict(name=name, _template=temp))

