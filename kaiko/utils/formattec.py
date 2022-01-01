import re
import string
import enum
import dataclasses
import typing
from typing import Dict, List, Set, Tuple, Union


class FormatError(Exception):
    def __init__(self, value, expected):
        self.value = value
        self.expected = expected

    def __str__(self):
        return f"Invalid value {self.value}, expecting {self.expected}"


class Formattec:
    def __init__(self, func):
        self.func = func

    def format(self, value, **contexts):
        return "".join(self.func(value, **contexts))

    def context(self, override):
        def context_formatter(value, field, **contexts):
            yield from self.func(value, field, **override(**contexts))
        return Formattec(context_formatter)

    def validate(self, validater, expected=None):
        if expected is None:
            expected = f"satisfy {validater!r}"
        def validate_formatter(value, **contexts):
            if not validater(value):
                raise FormatError(value, expected)
            yield from self.func(value, **contexts)
        return Formattec(validate_formatter)

    @staticmethod
    def string(string=""):
        def string_formatter(value, **contexts):
            yield string
        return Formattec(string_formatter)

    def map(self, func):
        def func_formatter(value, **contexts):
            yield from self.func(func(value), **contexts)
        return Formattec(func_formatter)

    @staticmethod
    def bind(func):
        def bind_formatter(value, **contexts):
            yield from func(value).func(value, **contexts)
        return Formattec(bind_formatter)

    def concat(self, *others):
        def concat_formatter(value, **contexts):
            for other in [self, *others]:
                yield other.func(value, **contexts)
        return Formattec(concat_formatter)

    @staticmethod
    def template(format_str, **formatters):
        # Formattec.template("key={.key!value}", value=value_formatter)
        _string_formatter = string.Formatter()

        def template_formatter(value, **contexts):
            for prefix, name, spec, conv in _string_formatter.parse(format_str):
                if prefix is not None:
                    yield prefix
                if name is None:
                    continue
                if name[:1] not in ("", ".", "["):
                    field_value, _ = _string_formatter.get_field(name, [], contexts)
                    if conv is not None:
                        field_value = _string_formatter.convert_field(field_value, conv)
                    spec = spec.format(**contexts) if spec is not None else ""
                    yield _string_formatter.format_field(field_value, spec)
                    continue

                if conv is None or spec is not None:
                    raise ValueError
                field_value, _ = _string_formatter.get_field(name, [value], {})
                yield from formatters[conv].func(field_value, **contexts)
        return Formattec(template_formatter)

    def many(self):
        def many_formatter(value, **contexts):
            for subvalue in value:
                yield from self.func(subvalue, **contexts)
        return Formattec(many_formatter)

    def join(self, elems, multiline=False):
        sep = self
        if multiline:
            sep = Formattec.template("{!sep}\n{indent}", sep=sep)

        def join_formatter(value, **contexts):
            if len(value) != len(elems):
                raise FormatError(value, f"iterable with length {len(elems)}")
            is_first = True
            for subvalue, formatter in zip(value, elems):
                if not is_first:
                    yield from sep.func(value, **contexts)
                yield from formatter.func(subvalue, **contexts)
                is_first = False
        return Formattec(join_formatter)

    def sep_by(self, sep, multiline=False):
        if multiline:
            sep = Formattec.template("{!sep}\n{indent}", sep=sep)

        def sep_formatter(value, **contexts):
            is_first = True
            for subvalue in value:
                if not is_first:
                    yield from sep.func(value, **contexts)
                yield from self.func(subvalue, **contexts)
                is_first = False
        return Formattec(sep_formatter)

    def between(self, opening, closing, indent="    ", multiline=False):
        if multiline:
            _indent = indent
            opening = Formattec.template("{!opening}\n{indent}" + _indent, opening=opening)
            closing = Formattec.template("\n{indent}{!closing}", closing=closing)
            entry = self.context(lambda indent="", **kw: dict(indent=indent+_indent, **kw))
            return opening + entry + closing
        else:
            return opening + self + closing

    def __add__(self, other):
        return self.concat(other)


def _make_literal_formatter(cls, func=repr):
    def literal_formatter(value, **contexts):
        yield func(value)
    return Formattec(literal_formatter).validate(lambda value: type(value) is cls, cls.__name__)

none_formatter = _make_literal_formatter(type(None))
bool_formatter = _make_literal_formatter(bool)
int_formatter = _make_literal_formatter(int)
float_formatter = _make_literal_formatter(float)

def _complex_repr(value):
    repr_value = repr(value)
    # remove parentheses
    if repr_value.startswith("(") and repr_value.endswith(")"):
        repr_value = repr_value[1:-1]
    return repr_value
complex_formatter = _make_literal_formatter(complex, _complex_repr)

def _bytes_repr(value):
    # make sure it uses double quotation
    return 'b"' + repr(value + b'"')[2:-2].replace('"', r'\"').replace(r"\'", "'") + '"'
bytes_formatter = _make_literal_formatter(bytes, _bytes_repr)

def _str_repr(value):
    # make sure it uses double quotation
    return '"' + repr(value + '"')[1:-2].replace('"', r'\"').replace(r"\'", "'") + '"'
str_formatter = _make_literal_formatter(str, _str_repr)

def _sstr_repr(value):
    # make sure it uses single quotation
    return repr(value + '"')[:-2] + "'"
sstr_formatter = _make_literal_formatter(str, _sstr_repr)

def _mstr_repr(value):
    if not value.startswith("\n") or not value.endswith("\n"):
        raise FormatError(value, "string started and ended with newline")
    return '"""' + repr(value + '"')[1:-2].replace('"', r'\"').replace(r"\'", "'").replace(r"\n", "\n") + '"""'
mstr_formatter = _make_literal_formatter(str, _mstr_repr)

def _rmstr_repr(value):
    if not value.startswith("\n") or not value.endswith("\n"):
        raise FormatError(value, "string started and ended with newline")
    m = re.search(r'\x00|\r|"""|\\$', value)
    if m:
        raise FormatError(value, "string without '\\x00', '\\r', '\"\"\"' and single '\\'")
    return 'r"""' + value + '"""'
rmstr_formatter = _make_literal_formatter(str, _rmstr_repr)

# composite

def list_formatter(elem, multiline=False):
    empty = Formattec.string("[]")
    nonempty = (
        elem.sep_by(Formattec.string(", "), multiline=multiline)
            .between(Formattec.string("["), Formattec.string("]"), multiline=multiline)
    )
    return (
        Formattec.bind(lambda value: nonempty if value else empty)
            .validate(lambda value: type(value) is list, "list")
    )

def set_formatter(elem, multiline=False):
    empty = Formattec.string("set()")
    nonempty = (
        elem.sep_by(Formattec.string(", "), multiline=multiline)
            .between(Formattec.string("{"), Formattec.string("}"), multiline=multiline)
    )
    return (
        Formattec.bind(lambda value: nonempty if value else empty)
            .validate(lambda value: type(value) is set, "set")
    )

def dict_formatter(key, value, multiline=False):
    empty = Formattec.string("{}")
    nonempty = (
        Formattec.template("{[0]!key}:{[1]!value}", key=key, value=value)
            .sep_by(Formattec.string(", "), multiline=multiline)
            .between(Formattec.string("{"), Formattec.string("}"), multiline=multiline)
            .map(lambda value: value.items())
    )
    return (
        Formattec.bind(lambda value: nonempty if value else empty)
            .validate(lambda value: type(value) is dict, "dict")
    )


def tuple_formatter(elems, multiline=False):
    if len(elems) == 0:
        return Formattec.string("()").validate(lambda value: value == (), "empty tuple")
    elif len(elems) == 1:
        return (
            Formattec.template("({!elem},)", elem=elems[0])
                .validate(lambda value: type(value) is tuple and len(value) == 1, "singleton tuple")
        )
    else:
        return (
            Formattec.string(", ").join(elems, multiline=multiline)
                .between(Formattec.string("("), Formattec.string(")"), multiline=multiline)
                .validate(lambda value: type(value) is tuple, "tuple")
        )


def dataclass_formatter(cls, fields, multiline=False):
    if not fields:
        return Formattec.string(f"{cls.__name__}()").validate(lambda value: type(value) is cls, cls.__name__)
    else:
        items = [Formattec.template(key+"={!field}", field=field) for key, field in fields.items()]
        return (
            Formattec.string(", ").join(items, multiline=multiline)
                .between(Formattec.string(f"{cls.__name__}("), Formattec.string(")"), multiline=multiline)
                .map(lambda value: [getattr(value, key) for key in fields.keys()])
                .validate(lambda value: type(value) is cls, cls.__name__)
        )


def union_formatter(options):
    def union_formatter(value, **contexts):
        for type_hint, option_formatter in options.items():
            if has_type(value, type_hint):
                yield from option_formatter.func(value, **contexts)
                return
        else:
            raise FormatError(value, " or ".join(str(type_hint) for type_hint in options.keys()))
    return Formattec(union_formatter)


def enum_formatter(cls):
    return _make_literal_formatter(cls, lambda value: f"{cls.__name__}.{value.name}")


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


def has_type(value, type_hint):
    if type_hint is None:
        type_hint = type(None)

    if type_hint in [type(None), bool, int, float, complex, str, bytes]:
        return type(value) is type_hint

    elif isinstance(type_hint, type) and issubclass(type_hint, enum.Enum):
        return type(value) is type_hint

    elif isinstance(type_hint, type) and dataclasses.is_dataclass(type_hint):
        return type(value) is type_hint and all(has_type(getattr(value, field.name), field.type)
                                                for field in dataclasses.fields(type_hint))

    elif get_origin(type_hint) is list:
        elem_hint, = get_args(type_hint)
        return type(value) is list and all(has_type(value, elem_hint) for elem in value)

    elif get_origin(type_hint) is set:
        elem_hint, = get_args(type_hint)
        return type(value) is set and all(has_type(value, elem_hint) for elem in value)

    elif get_origin(type_hint) is tuple:
        args = get_args(type_hint)
        if len(args) == 1 and args[0] == ():
            args = []
        return (
            type(value) is tuple
            and len(value) == len(args)
            and all(has_type(elem, elem_hint) for elem, elem_hint in zip(value, args))
        )

    elif get_origin(type_hint) is dict:
        key_hint, value_hint = get_args(type_hint)
        return type(value) is dict and all(has_type(key, key_hint) and has_type(value, value_hint)
                                           for key, value in value.items())

    elif get_origin(type_hint) is Union:
        return any(has_type(value, option) for option in get_args(type_hint))

    else:
        raise ValueError(f"Invalid type hint: {type_hint!r}")


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
        options = {arg: from_type_hint(arg, multiline) for arg in get_args(type_hint)}
        return union_formatter(options)

    else:
        raise ValueError(f"No formatter for type hint: {type_hint!r}")
