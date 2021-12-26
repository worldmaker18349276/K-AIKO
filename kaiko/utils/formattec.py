import string
import itertools
import enum
import dataclasses
import typing
from typing import Dict, List, Set, Tuple, Union


class FormatError(Exception):
    def __init__(self, field, expected):
        self.field = field
        self.expected = expected

    def __str__(self):
        if self.field == "":
            return f"expected: {self.expected}"
        else:
            return f"expected: {self.expected} at {self.field}"


class Formattec:
    _string_formatter = string.Formatter()

    def __init__(self, func):
        self.func = func

    def format(self, value, **contexts):
        return self.func(value, "", **contexts)

    def context(self, override):
        def context_formatter(value, field, **contexts):
            return self.func(value, field, **override(**contexts))
        return Formattec(context_formatter)

    def validate(self, pred, expected="satisfy some condition"):
        def validate_formatter(value, field, **contexts):
            if not pred(value):
                raise FormatError(field, expected)
            return self.func(value, field, **contexts)
        return Formattec(validate_formatter)

    @staticmethod
    def string(string=""):
        def string_formatter(value, field, **contexts):
            return string
        return Formattec(string_formatter)

    def attempt(self):
        def attempt_formatter(value, field, **contexts):
            try:
                self.func(value, "", **contexts)
            except FormatError as e:
                if e.field == "":
                    raise e
                else:
                    raise FormatError(field, f"({e.expected} at {e.field})") from e
        return Formattec(attempt_formatter)

    def concat(self, *others):
        def concat_formatter(value, field, **contexts):
            return "".join(formatter.func(value, field, **contexts) for formatter in [self, *others])
        return Formattec(concat_formatter)

    def in_field(self, subfield, getter=None):
        if getter is None:
            getter = lambda value: Formattec._string_formatter.get_field(subfield, [value], {})
        def infield_formatter(value, field, **contexts):
            subvalue = getter(value)
            return self.func(subvalue, field + subfield, **contexts)
        return Formattec(infield_formatter)

    @staticmethod
    def template(format_str, **formatters):
        def template_formatter(value, field, **contexts):
            res = ""
            for prefix, name, spec, conv in Formattec._string_formatter.parse(format_str):
                if prefix is not None:
                    res += prefix
                if name is None:
                    continue
                if name[:1] not in ("", ".", "["):
                    field_value, _ = Formattec._string_formatter.get_field(name, [], contexts)
                    if conv is not None:
                        field_value = Formattec._string_formatter.convert_field(field_value, conv)
                    spec = spec.format(**contexts) if spec is not None else ""
                    res += Formattec._string_formatter.format_field(field_value, spec)
                    continue

                if conv is None or spec is not None:
                    raise ValueError
                res += formatters[conv].in_field(name).func(value, field, **contexts)

            return res
        return Formattec(template_formatter)

    def union(self, *others):
        formatters = [self, *others]
        def union_formatter(value, field, **contexts):
            expected = []
            for formatter in formatters:
                try:
                    return formatter.func(value, field, **contexts)
                except FormatError as e:
                    if e.field != field:
                        raise e
                    else:
                        expected.append(e.expected)
            else:
                raise FormatError(field, " or ".join(expected))
        return Formattec(union_formatter)

    def join(self, elems, multiline=False):
        if len(elems) == 0:
            return Formattec.string("")
        if multiline:
            sep = Formattec.template("{!sep}\n{indent}", sep=self)
        else:
            sep = Formattec.template("{!sep} ", sep=self)
        return elems[0].concat(*[sep + elem for elem in elems[1:]])

    def many(self, subfields=lambda i: (f"[{i}]", None)):
        def many_formatter(value, field, **contexts):
            res = ""
            for i in itertools.count():
                subfield, getter = subfields(i)
                try:
                    res += self.in_field(subfield, getter).func(value, field, **contexts)
                except FormatError as e:
                    if e.field != field:
                        raise e
                    else:
                        return res
        return Formattec(many_formatter)

    def sep_by(self, sep, subfields=lambda i: (f"[{i}]", None), multiline=False):
        if multiline:
            sep = Formattec.template("{!sep}\n{indent}", sep=sep)
        else:
            sep = Formattec.template("{!sep} ", sep=sep)
        return self.in_field(*subfields(0)) + (sep + self).many(subfields=lambda i: subfields(i+1))

    def between(self, opening, closing, indent="    ", multiline=False):
        if multiline:
            _indent = indent
            elem = self.context(lambda indent="", **kw: dict(indent=indent + _indent, **kw))
            return Formattec.template("{!opening}\n{indent}" + _indent + "{!elem}\n{indent}{!closing}",
                                      opening=opening, elem=elem, closing=closing)
        else:
            return opening + self + closing

    def __or__(self, other):
        return self.union(other)

    def __add__(self, other):
        return self.concat(other)


def _make_literal_formmatter(cls, func=repr):
    return Formattec(lambda value, field, **contexts: func(value)).validate(lambda value: type(value) is cls, cls.__name__)

none_formatter = _make_literal_formmatter(type(None))
bool_formatter = _make_literal_formmatter(bool)
int_formatter = _make_literal_formmatter(int)
float_formatter = _make_literal_formmatter(float)

def _complex_formatter(value):
    repr_value = repr(value)
    # remove parentheses
    if repr_value.startswith("(") and repr_value.endswith(")"):
        repr_value = repr_value[1:-1]
    return repr_value
complex_formatter = _make_literal_formmatter(complex, _complex_formatter)

def _bytes_formatter(value):
    # make sure it uses double quotation
    return 'b"' + repr(value + b'"')[2:-2].replace('"', r'\"').replace(r"\'", "'") + '"'
bytes_formatter = _make_literal_formmatter(bytes, _bytes_formatter)

def _str_formatter(value):
    # make sure it uses double quotation
    return '"' + repr(value + '"')[1:-2].replace('"', r'\"').replace(r"\'", "'") + '"'
str_formatter = _make_literal_formmatter(str, _str_formatter)

def _sstr_formatter(value):
    # make sure it uses single quotation
    return repr(value + '"')[:-2] + "'"
sstr_formatter = _make_literal_formmatter(str, _sstr_formatter)


# composite

def list_formatter(elem, multiline=False):
    empty = Formattec.string("[]").validate(lambda value: value == [], "empty list")
    nonempty = (
        elem.sep_by(Formattec.string(","), multiline=multiline)
            .between(Formattec.string("["), Formattec.string("]"), multiline=multiline)
            .validate(lambda value: type(value) is list, "list")
    )
    return empty | nonempty

def set_formatter(elem, multiline=False):
    empty = Formattec.string("set()").validate(lambda value: value == set(), "empty set")
    nonempty = (
        elem.sep_by(Formattec.string(","), multiline=multiline)
            .between(Formattec.string("{"), Formattec.string("}"), multiline=multiline)
            .in_field(".to_list()", lambda a: list(a))
            .validate(lambda value: type(value) is set, "set")
    )
    return empty | nonempty


def dict_formatter(key, value, multiline=False):
    empty = Formattec.string("{}").validate(lambda value: value == {}, "empty dict")
    nonempty = (
        Formattec.template("{[0]!key}:{[1]!value}", key=key, value=value)
            .sep_by(Formattec.string(","), multiline=multiline)
            .between(Formattec.string("{"), Formattec.string("}"), multiline=multiline)
            .in_field(".items()", lambda value: value.items())
            .validate(lambda value: type(value) is dict, "dict")
    )
    return empty | nonempty


def tuple_formatter(elems, multiline=False):
    if len(elems) == 0:
        return Formattec.string("()").validate(lambda value: value == (), "empty tuple")
    elif len(elems) == 1:
        return (
            Formattec.template("({[0]!elem},)", elem=elems[0])
                .validate(lambda value: type(value) is tuple and len(value) == 1, "singleton tuple")
        )
    else:
        return (
            Formattec.join(Formattec.string(","), elems, multiline=multiline)
                .between(Formattec.string("("), Formattec.string(")"), multiline=multiline)
                .validate(lambda value: type(value) is tuple, "tuple")
        )


def dataclass_formatter(cls, fields, multiline=False):
    if not fields:
        return Formattec.string(f"{cls.__name__}()").validate(lambda value: type(value) is cls, cls.__name__)
    else:
        items = [Formattec.template("%s={.%s!field}" % (key, key), field=field) for key, field in fields.items()]
        return (
            Formattec.join(Formattec.string(","), items, multiline=multiline)
                .between(Formattec.string(f"{cls.__name__}("), Formattec.string(")"), multiline=multiline)
                .validate(lambda value: type(value) is cls, cls.__name__)
        )


def union_formatter(options):
    return Formattec.union(*[option.attempt() for option in options])


def enum_formatter(cls):
    return _make_literal_formmatter(cls, lambda value: f"{cls.__name__}.{value.name}")


def get_args(type_hint):
    if hasattr(typing, 'get_args'):
        return typing.get_args(type_hint)
    else:
        return type_hint.__args__


def get_origin(type_hint):
    if hasattr(typing, 'get_origin'):
        return typing.get_origin(type_hint)
    else:
        origin = type_hint.__origin__
        if origin == List:
            origin = list
        elif origin == Tuple:
            origin = tuple
        elif origin == Set:
            origin = set
        elif origin == Dict:
            origin = dict
        else:
            raise ValueError
        return origin


def from_type_hint(type_hint, multiline=False):
    """Make Formatter from type hint.

    Parameters
    ----------
    type_hint : type or type hint
        The type to format.
    multiline : bool, optional

    Returns
    -------
    Formattec
        the formatter of the given type.
    """
    if type_hint is None:
        type_hint = type(None)

    if type_hint is type(None):
        return none_formatter

    elif type_hint is bool:
        return bool_formatter

    elif type_hint is int:
        return int_formatter

    elif type_hint is float:
        return float_formatter

    elif type_hint is complex:
        return complex_formatter

    elif type_hint is str:
        return str_formatter

    elif type_hint is bytes:
        return bytes_formatter

    elif isinstance(type_hint, type) and issubclass(type_hint, enum.Enum):
        return enum_formatter(type_hint)

    elif isinstance(type_hint, type) and dataclasses.is_dataclass(type_hint):
        fields = {field.name: from_type_hint(field.type, multiline)
                  for field in dataclasses.fields(type_hint)}
        return dataclass_formatter(type_hint, fields, multiline)

    elif get_origin(type_hint) is list:
        elem_hint, = get_args(type_hint)
        elem = from_type_hint(elem_hint, multiline)
        return list_formatter(elem, multiline)

    elif get_origin(type_hint) is set:
        elem_hint, = get_args(type_hint)
        elem = from_type_hint(elem_hint, multiline)
        return set_formatter(elem, multiline)

    elif get_origin(type_hint) is tuple:
        args = get_args(type_hint)
        if len(args) == 1 and args[0] == ():
            elems = []
        else:
            elems = [from_type_hint(arg, multiline) for arg in args]
        return tuple_formatter(elems, multiline)

    elif get_origin(type_hint) is dict:
        key_hint, value_hint = get_args(type_hint)
        key = from_type_hint(key_hint, False)
        value = from_type_hint(value_hint, multiline)
        return dict_formatter(key, value, multiline)

    elif get_origin(type_hint) is Union:
        options = [from_type_hint(arg, multiline) for arg in get_args(type_hint)]
        return union_formatter(options)

    else:
        raise ValueError(f"No formatter for type hint: {type_hint!r}")
