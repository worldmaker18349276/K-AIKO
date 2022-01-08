import keyword
import ast
import enum
import dataclasses
import typing
from typing import Dict, List, Set, Tuple, Union
from . import parsec as pc


# literals

def validate_identifier(name):
    if not isinstance(name, str) or not str.isidentifier(name) or keyword.iskeyword(name):
        raise ValueError(f"Invalid identifier {name!r}")

def parse_identifiers(*tokens):
    for token in tokens:
        validate_identifier(token)
    return pc.tokens(tokens)

def _make_literal_parser(expr, desc):
    return pc.regex(expr).map(ast.literal_eval).desc(desc)

none_parser = _make_literal_parser(r"None", "None")
bool_parser = _make_literal_parser(r"False|True", "bool")
int_parser = _make_literal_parser(r"[-+]?(0|[1-9][0-9]*)(?![0-9\.\+eEjJ])", "int")
float_parser = _make_literal_parser(
    r"[-+]?([0-9]+\.[0-9]+(e[-+]?[0-9]+)?|[0-9]+[eE][-+]?[0-9]+)(?![0-9\+jJ])",
    "float",
)
complex_parser = _make_literal_parser(
    r"[-+]?({0}[-+])?{0}[jJ]".format(
        r"(0|[1-9][0-9]*|[0-9]+\.[0-9]+(e[-+]?[0-9]+)?|[0-9]+e[-+]?[0-9]+)"
    ),
    "complex",
)
bytes_parser = _make_literal_parser(
    r'b"('
    r'(?![\r\n\\"])[\x01-\x7f]'
    r'|\\[0-7]{1,3}'
    r'|\\x[0-9a-fA-F]{2}'
    r'|\\u[0-9a-fA-F]{4}'
    r'|\\U[0-9a-fA-F]{8}'
    r'|\\(?![xuUN])[\x01-\x7f]'
    r')*"',
    "bytes",
)
str_parser = _make_literal_parser(
    r'"('
    r'[^\r\n\\"\x00]'
    r'|\\[0-7]{1,3}'
    r'|\\x[0-9a-fA-F]{2}'
    r'|\\u[0-9a-fA-F]{4}'
    r'|\\U[0-9a-fA-F]{8}'
    r'|\\(?![xuUN\x00]).'
    r')*"',
    "str",
)
sstr_parser = _make_literal_parser(
    r"'("
    r"[^\r\n\\']"
    r"|\\[0-7]{1,3}"
    r"|\\x[0-9a-fA-F]{2}"
    r"|\\u[0-9a-fA-F]{4}"
    r"|\\U[0-9a-fA-F]{8}"
    r"|\\(?![xuUN])."
    r")*'",
    "str",
)
mstr_parser = _make_literal_parser(
    # always start/end with newline
    r'"""(?=\n)('
    r'(?!""")[^\\\x00]'
    r'|\\[0-7]{1,3}'
    r'|\\x[0-9a-fA-F]{2}'
    r'|\\u[0-9a-fA-F]{4}'
    r'|\\U[0-9a-fA-F]{8}'
    r'|\\(?![xuUN\x00]).'
    r')*(?<=\n)"""',
    "str",
)
rmstr_parser = _make_literal_parser(
    r'r"""(?=\n)('
    r'(?!""")[^\\\x00]'
    r'|\\[^\x00]'
    r')*(?<=\n)"""',
    "str",
)


# composite

def make_list_parser(elem):
    opening = pc.regex(r"\[\s*").desc("'['")
    comma = pc.regex(r"\s*,\s*").desc("','")
    closing = pc.regex(r"\s*\]").desc("']'")
    return (
        elem.sep_end_by(comma)
            .between(opening, closing)
            .map(list)
    )

def make_set_parser(elem):
    opening = pc.regex(r"\{\s*").desc("'{'")
    comma = pc.regex(r"\s*,\s*").desc("','")
    closing = pc.regex(r"\s*\}").desc("'}'")
    empty = pc.tokens(["set()"]).result([]).desc("'set()'")
    nonempty = (
        elem.sep_end_by1(comma)
            .between(opening, closing)
    )
    return (empty | nonempty).map(set)

def make_dict_parser(key, value):
    opening = pc.regex(r"\{\s*").desc("'{'")
    colon = pc.regex(r"\s*:\s*").desc("':'")
    comma = pc.regex(r"\s*,\s*").desc("','")
    closing = pc.regex(r"\s*\}").desc("'}'")
    item = colon.join((key, value))
    return (
        item.sep_end_by(comma)
            .between(opening, closing)
            .map(dict)
    )

def make_tuple_parser(elems):
    opening = pc.regex(r"\(\s*").desc("'('")
    comma = pc.regex(r"\s*,\s*").desc("','")
    closing = pc.regex(r"\s*\)").desc("')'")
    if len(elems) == 0:
        return (opening + closing).result(())
    elif len(elems) == 1:
        return (elems[0] << comma).between(opening, closing).map(lambda e: (e,))
    else:
        entries = comma.join(elems) << comma.optional()
        return entries.between(opening, closing).map(tuple)

def make_dataclass_parser(cls, fields):
    name = parse_identifiers(cls.__name__)
    opening = pc.regex(r"\(\s*").desc("'('")
    equal = pc.regex(r"\s*=\s*").desc("'='")
    comma = pc.regex(r"\s*,\s*").desc("','")
    closing = pc.regex(r"\s*\)").desc(")")
    if fields:
        items = [equal.join((parse_identifiers(key), field)) for key, field in fields.items()]
        entries = comma.join(items) << comma.optional()
    else:
        entries = pc.nothing(())
    return entries.between(name >> opening, closing).map(lambda a: cls(**dict(a)))

def make_union_parser(options):
    if len(options) == 0:
        raise ValueError("empty union")
    elif len(options) == 1:
        return options[0]
    else:
        return pc.choice(*[option.attempt() for option in options])

def make_enum_parser(cls):
    return (
        parse_identifiers(cls.__name__)
            .then(pc.tokens(["."]))
            .then(parse_identifiers(*[option.name for option in cls]))
            .map(lambda option: getattr(cls, option))
    )


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

def make_parser_from_type_hint(type_hint):
    """Make Parser from type hint.

    Parameters
    ----------
    type_hint : type or type hint
        The type to parse.

    Returns
    -------
    Parsec
        The parser of the given type.
    """
    if type_hint is None:
        type_hint = type(None)

    if type_hint is type(None):
        return none_parser

    elif type_hint is bool:
        return bool_parser

    elif type_hint is int:
        return int_parser

    elif type_hint is float:
        return float_parser

    elif type_hint is complex:
        return complex_parser

    elif type_hint is str:
        return str_parser

    elif type_hint is bytes:
        return bytes_parser

    elif isinstance(type_hint, type) and issubclass(type_hint, enum.Enum):
        return make_enum_parser(type_hint)

    elif isinstance(type_hint, type) and dataclasses.is_dataclass(type_hint):
        fields = {field.name: make_parser_from_type_hint(field.type)
                  for field in dataclasses.fields(type_hint)}
        return make_dataclass_parser(type_hint, fields)

    elif get_origin(type_hint) is list:
        elem_hint, = get_args(type_hint)
        elem = make_parser_from_type_hint(elem_hint)
        return make_list_parser(elem)

    elif get_origin(type_hint) is set:
        elem_hint, = get_args(type_hint)
        elem = make_parser_from_type_hint(elem_hint)
        return make_set_parser(elem)

    elif get_origin(type_hint) is tuple:
        args = get_args(type_hint)
        if len(args) == 1 and args[0] == ():
            elems = []
        else:
            elems = [make_parser_from_type_hint(arg) for arg in args]
        return make_tuple_parser(elems)

    elif get_origin(type_hint) is dict:
        key_hint, value_hint = get_args(type_hint)
        key = make_parser_from_type_hint(key_hint)
        value = make_parser_from_type_hint(value_hint)
        return make_dict_parser(key, value)

    elif get_origin(type_hint) is Union:
        options = [make_parser_from_type_hint(arg) for arg in get_args(type_hint)]
        return make_union_parser(options)

    else:
        raise ValueError(f"No parser for type hint: {type_hint!r}")

def has_type(value, type_hint):
    if type_hint is None:
        type_hint = type(None)

    if type_hint in (type(None), bool, int, float, complex, str, bytes):
        return type(value) is type_hint

    elif isinstance(type_hint, type) and issubclass(type_hint, enum.Enum):
        return type(value) is type_hint

    elif isinstance(type_hint, type) and dataclasses.is_dataclass(type_hint):
        return type(value) is type_hint and all(has_type(getattr(value, field.name), field.type)
                                                for field in dataclasses.fields(type_hint))

    elif get_origin(type_hint) is list:
        elem_hint, = get_args(type_hint)
        return type(value) is list and all(has_type(subvalue, elem_hint) for subvalue in value)

    elif get_origin(type_hint) is set:
        elem_hint, = get_args(type_hint)
        return type(value) is set and all(has_type(subvalue, elem_hint) for subvalue in value)

    elif get_origin(type_hint) is tuple:
        elems = get_args(type_hint)
        if len(elems) == 1 and elems[0] == ():
            elems = []
        return (
            type(value) is tuple
            and len(value) == len(elems)
            and all(has_type(subvalue, elem_hint) for subvalue, elem_hint in zip(value, elems))
        )

    elif get_origin(type_hint) is dict:
        key_hint, value_hint = get_args(type_hint)
        return (
            type(value) is dict
            and all(has_type(key, key_hint) and has_type(subvalue, value_hint) for key, subvalue in value.items())
        )

    elif get_origin(type_hint) is Union:
        options = get_args(type_hint)
        return any(has_type(value, option) for option in options)

    else:
        raise ValueError(f"Unknown type hint: {type_hint!r}")

def get_used_custom_types(value):
    if value is None:
        return {*()}

    elif type(value) in (bool, int, float, complex, str, bytes):
        return {*()}

    elif type(value) in (list, tuple, set):
        return {typ for subvalue in value for typ in get_used_custom_types(subvalue)}

    elif type(value) is dict:
        return {typ for entry in value.items() for subvalue in entry for typ in get_used_custom_types(subvalue)}

    elif isinstance(value, enum.Enum):
        return {type(value)}

    elif dataclasses.is_dataclass(value):
        return {type(value)} | {
            typ
            for field in dataclasses.fields(value)
            for typ in get_used_custom_types(getattr(value, field.name))
        }

    else:
        raise TypeError(f"Cannot configure type {type(value)}")

def format_value(value):
    if value is None:
        return "None"

    elif type(value) in (bool, int, float):
        return repr(value)

    elif type(value) is complex:
        repr_value = repr(value)
        # remove parentheses
        if repr_value.startswith("(") and repr_value.endswith(")"):
            repr_value = repr_value[1:-1]
        return repr_value

    elif type(value) is bytes:
        # make sure it uses double quotation
        return 'b"' + repr(value + b'"')[2:-2].replace('"', r'\"').replace(r"\'", "'") + '"'

    elif type(value) is str:
        # make sure it uses double quotation
        return '"' + repr(value + '"')[1:-2].replace('"', r'\"').replace(r"\'", "'") + '"'

    elif type(value) is list:
        return "[%s]" % ", ".join(format_value(subvalue) for subvalue in value)

    elif type(value) is tuple:
        if len(value) == 1:
            return "(%s,)" % format_value(value[0])
        return "(%s)" % ", ".join(format_value(subvalue) for subvalue in value)

    elif type(value) is set:
        if not value:
            return "set()"
        return "{%s}" % ", ".join(format_value(subvalue) for subvalue in value)

    elif type(value) is dict:
        return "{%s}" % ", ".join(
            format_value(key) + ":" + format_value(subvalue)
            for key, subvalue in value.items()
        )

    elif isinstance(value, enum.Enum):
        cls = type(value)
        validate_identifier(cls.__name__)
        validate_identifier(value.name)
        return f"{cls.__name__}.{value.name}"

    elif dataclasses.is_dataclass(value):
        cls = type(value)
        fields = dataclasses.fields(cls)
        validate_identifier(cls.__name__)
        for field in fields:
            validate_identifier(field.name)
        return f"{cls.__name__}(%s)" % ", ".join(
            field.name + "=" + format_value(getattr(value, field.name)) for field in fields
        )

    else:
        raise TypeError(f"Cannot format value of type {type(value)}")