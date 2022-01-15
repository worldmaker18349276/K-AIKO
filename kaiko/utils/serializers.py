import contextlib
import keyword
import ast
import enum
import pathlib
import dataclasses
import typing
from typing import Dict, List, Set, Tuple, Union
from . import parsec as pc


# literals

def validate_identifier(name):
    if not isinstance(name, str) or not str.isidentifier(name) or keyword.iskeyword(name):
        raise ValueError(f"Invalid identifier {name!r}")

class ParseSuggestion(pc.ParseFailure):
    def __init__(self, expected, suggestions):
        self.expected = expected
        self.suggestions = suggestions

@pc.parsec
def suggest(parser, suggestions):
    try:
        return (yield parser)
    except pc.ParseFailure as failure:
        raise ParseSuggestion(failure.expected, suggestions) from failure

@contextlib.contextmanager
def failure_choice(failure):
    try:
        yield
    except pc.ParseFailure as failure2:
        raise pc.ParseChoiceFailure([failure, failure2]) from failure2

def _make_literal_parser(expr, desc, suggestions):
    return suggest(pc.regex(expr).map(ast.literal_eval).desc(desc), suggestions)

def make_none_parser(suggestions=[None]):
    return _make_literal_parser(r"None", "None", [format_value(sugg) for sugg in set(suggestions)])

def make_bool_parser(suggestions=[False]):
    return _make_literal_parser(r"False|True", "bool", [format_value(sugg) for sugg in set(suggestions)])

def make_int_parser(suggestions=[0]):
    return _make_literal_parser(
        r"[-+]?(0|[1-9][0-9]*)(?![0-9\.\+eEjJ])", "int",
        [format_value(sugg) for sugg in set(suggestions)],
    )

def make_float_parser(suggestions=[0.0]):
    return _make_literal_parser(
        r"[-+]?([0-9]+\.[0-9]+(e[-+]?[0-9]+)?|[0-9]+[eE][-+]?[0-9]+)(?![0-9\+jJ])",
        "float",
        [format_value(sugg) for sugg in set(suggestions)],
    )

def make_complex_parser(suggestions=[0j]):
    return _make_literal_parser(
        r"[-+]?({0}[-+])?{0}[jJ]".format(
            r"(0|[1-9][0-9]*|[0-9]+\.[0-9]+(e[-+]?[0-9]+)?|[0-9]+e[-+]?[0-9]+)"
        ),
        "complex",
        [format_value(sugg) for sugg in set(suggestions)],
    )

def make_bytes_parser(suggestions=[b""]):
    return _make_literal_parser(
        r'b"('
        r'(?![\r\n\\"])[\x01-\x7f]'
        r'|\\[0-7]{1,3}'
        r'|\\x[0-9a-fA-F]{2}'
        r'|\\u[0-9a-fA-F]{4}'
        r'|\\U[0-9a-fA-F]{8}'
        r'|\\(?![xuUN])[\x01-\x7f]'
        r')*"',
        "bytes",
        [format_value(sugg) for sugg in set(suggestions)],
    )

def make_str_parser(suggestions=[""]):
    return _make_literal_parser(
        r'"('
        r'[^\r\n\\"\x00]'
        r'|\\[0-7]{1,3}'
        r'|\\x[0-9a-fA-F]{2}'
        r'|\\u[0-9a-fA-F]{4}'
        r'|\\U[0-9a-fA-F]{8}'
        r'|\\(?![xuUN\x00]).'
        r')*"',
        "str",
        [format_value(sugg) for sugg in set(suggestions)],
    )

def make_path_parser(suggestions=[pathlib.Path(".")]):
    return (
        pc.string("Path(")
        >> make_str_parser([str(sugg) for sugg in suggestions]).map(pathlib.Path)
        << pc.string(")")
    )


# composite

@pc.parsec
def make_list_parser(elem):
    opening = suggest(pc.regex(r"\[\s*").desc("left bracket"), ["["])
    comma = suggest(pc.regex(r"\s*,\s*").desc("comma"), [","])
    closing = suggest(pc.regex(r"\s*\]").desc("right bracket"), ["]"])
    yield opening
    results = []
    try:
        while True:
            results.append((yield elem))
            yield comma
    except pc.ParseFailure as failure:
        with failure_choice(failure):
            yield closing
        return results

@pc.parsec
def make_set_parser(elem):
    opening = suggest(pc.regex(r"set\(\s*").desc("'set('"), ["set("])
    closing = suggest(pc.regex(r"\s*\)").desc("right parenthesis"), [")"])
    content = make_list_parser(elem)
    yield opening
    res = yield content
    yield closing
    return set(res)

@pc.parsec
def make_dict_parser(key, value):
    opening = suggest(pc.regex(r"\{\s*").desc("left brace"), ["{"])
    colon = suggest(pc.regex(r"\s*:\s*").desc("colon"), [":"])
    comma = suggest(pc.regex(r"\s*,\s*").desc("comma"), [","])
    closing = suggest(pc.regex(r"\s*\}").desc("right brace"), ["}"])
    item = colon.join((key, value))

    yield opening
    results = {}
    try:
        while True:
            k, v = yield item
            results[k] = v
            yield comma
    except pc.ParseFailure as failure:
        with failure_choice(failure):
            yield closing
        return results

@pc.parsec
def make_tuple_parser(elems):
    opening = suggest(pc.regex(r"\(\s*").desc("left parenthesis"), ["("])
    comma = suggest(pc.regex(r"\s*,\s*").desc("comma"), [","])
    closing = suggest(pc.regex(r"\s*\)").desc("right parenthesis"), [")"])

    if len(elems) == 0:
        yield opening
        yield closing
        return ()

    elif len(elems) == 1:
        yield opening
        res = yield elems[0]
        yield comma
        yield closing
        return (res,)

    else:
        yield opening
        results = []
        is_first = True
        for elem in elems:
            if not is_first:
                yield comma
            is_first = False
            results.append((yield elem))
        yield comma.optional()
        yield closing
        return tuple(results)

@pc.parsec
def make_dataclass_parser(cls, fields):
    opening = suggest(
        pc.regex(fr"{cls.__name__}\s*\(\s*").desc(f"'{cls.__name__}('"),
        [f"{cls.__name__}("],
    )
    comma = suggest(pc.regex(r"\s*,\s*").desc("comma"), [","])
    closing = suggest(pc.regex(r"\s*\)").desc("right parenthesis"), [")"])

    yield opening
    results = {}
    is_first = True
    for key, field in fields.items():
        if not is_first:
            yield comma
        is_first = False
        yield suggest(pc.regex(fr"{key}\s*=\s*").desc(f"{key}="), [f"{key}="])
        value = yield field
        results[key] = value
    yield comma.optional()
    yield closing
    return cls(**results)

def make_union_parser(options):
    if len(options) == 0:
        raise ValueError("empty union")
    elif len(options) == 1:
        return options[0]
    else:
        return pc.choice(*options)

@pc.parsec
def make_enum_parser(cls):
    yield suggest(pc.string(f"{cls.__name__}."), [f"{cls.__name__}."])
    option = yield pc.choice(*[suggest(pc.string(option.name), [option.name]) for option in cls])
    return getattr(cls, option)


def get_args(type_hint):
    if hasattr(typing, 'get_args'):
        return typing.get_args(type_hint)
    else:
        return getattr(type_hint, '__args__', ())

def get_origin(type_hint):
    if hasattr(typing, 'get_origin'):
        return typing.get_origin(type_hint)
    else:
        origin = getattr(type_hint, '__origin__', None)
        if origin is None:
            origin = None
        elif origin == List:
            origin = list
        elif origin == Tuple:
            origin = tuple
        elif origin == Set:
            origin = set
        elif origin == Dict:
            origin = dict
        elif origin == Union:
            origin = Union
        else:
            raise ValueError
        return origin

def get_base(type_hint):
    if type_hint is None:
        type_hint = type(None)

    if type_hint is type(None):
        return type(None)

    elif type_hint is bool:
        return bool

    elif type_hint is int:
        return int

    elif type_hint is float:
        return float

    elif type_hint is complex:
        return complex

    elif type_hint is str:
        return str

    elif type_hint is bytes:
        return bytes

    elif isinstance(type_hint, type) and issubclass(type_hint, pathlib.Path):
        return pathlib.Path

    elif isinstance(type_hint, type) and issubclass(type_hint, enum.Enum):
        return type_hint

    elif isinstance(type_hint, type) and dataclasses.is_dataclass(type_hint):
        return type_hint

    elif get_origin(type_hint) is list:
        return list

    elif get_origin(type_hint) is set:
        return set

    elif get_origin(type_hint) is tuple:
        return tuple

    elif get_origin(type_hint) is dict:
        return dict

    elif get_origin(type_hint) is Union:
        return Union

    else:
        raise ValueError

def get_sub(type_hint):
    if type_hint is None:
        type_hint = type(None)

    if type_hint in (type(None), bool, int, float, complex, str, bytes):
        return ()

    elif isinstance(type_hint, type) and issubclass(type_hint, pathlib.Path):
        return ()

    elif isinstance(type_hint, type) and issubclass(type_hint, enum.Enum):
        return ()

    elif isinstance(type_hint, type) and dataclasses.is_dataclass(type_hint):
        return tuple(field.type for field in dataclasses.fields(type_hint))

    elif get_origin(type_hint) in (list, set, dict):
        return get_args(type_hint)

    elif get_origin(type_hint) is tuple:
        args = get_args(type_hint)
        if len(args) == 1 and args[0] == ():
            return ()
        else:
            return args

    elif get_origin(type_hint) is Union:
        return get_args(type_hint)

    else:
        raise ValueError

def get_types(type_hint):
    bases = set()
    base = get_base(type_hint)
    if base is not Union:
        bases.add(base)
    for sub in get_sub(type_hint):
        bases |= get_types(sub)
    return bases

def make_parser_from_type_hint(type_hint, suggestions=[]):
    """Make Parser from type hint.

    Parameters
    ----------
    type_hint : type or type hint
        The type to parse.
    suggestions : list, optional
        The suggestions value, which should be instances of `type_hint`.

    Returns
    -------
    Parsec
        The parser of the given type.
    """
    if type_hint is None:
        type_hint = type(None)

    if type_hint is type(None):
        return make_none_parser(suggestions) if suggestions else make_none_parser()

    elif type_hint is bool:
        return make_bool_parser(suggestions) if suggestions else make_bool_parser()

    elif type_hint is int:
        return make_int_parser(suggestions) if suggestions else make_int_parser()

    elif type_hint is float:
        return make_float_parser(suggestions) if suggestions else make_float_parser()

    elif type_hint is complex:
        return make_complex_parser(suggestions) if suggestions else make_complex_parser()

    elif type_hint is str:
        return make_str_parser(suggestions) if suggestions else make_str_parser()

    elif type_hint is bytes:
        return make_bytes_parser(suggestions) if suggestions else make_bytes_parser()

    elif isinstance(type_hint, type) and issubclass(type_hint, pathlib.Path):
        return make_path_parser(suggestions) if suggestions else make_path_parser()

    elif isinstance(type_hint, type) and issubclass(type_hint, enum.Enum):
        validate_identifier(type_hint.__name__)
        for option in type_hint:
            validate_identifier(option.name)
        return make_enum_parser(type_hint)

    elif isinstance(type_hint, type) and dataclasses.is_dataclass(type_hint):
        validate_identifier(type_hint.__name__)
        fields = {}
        for field in dataclasses.fields(type_hint):
            validate_identifier(field.name)
            subsuggestions = [getattr(sugg, field.name) for sugg in suggestions]
            if field.default is not dataclasses.MISSING:
                subsuggestions.append(field.default)
            elif field.default_factory is not dataclasses.MISSING:
                subsuggestions.append(field.default_factory())
            fields[field.name] = make_parser_from_type_hint(field.type, subsuggestions)
        return make_dataclass_parser(type_hint, fields)

    elif get_origin(type_hint) is list:
        elem_hint, = get_args(type_hint)
        elem = make_parser_from_type_hint(elem_hint, [elem for sugg in suggestions for elem in sugg])
        return make_list_parser(elem)

    elif get_origin(type_hint) is set:
        elem_hint, = get_args(type_hint)
        elem = make_parser_from_type_hint(elem_hint, [elem for sugg in suggestions for elem in sugg])
        return make_set_parser(elem)

    elif get_origin(type_hint) is tuple:
        args = get_args(type_hint)
        if len(args) == 1 and args[0] == ():
            elems = []
        else:
            elems = [
                make_parser_from_type_hint(arg, [sugg[i] for sugg in suggestions])
                for i, arg in enumerate(args)
            ]
        return make_tuple_parser(elems)

    elif get_origin(type_hint) is dict:
        key_hint, value_hint = get_args(type_hint)
        key = make_parser_from_type_hint(key_hint, [key for sugg in suggestions for key in sugg.keys()])
        value = make_parser_from_type_hint(value_hint, [value for sugg in suggestions for value in sugg.values()])
        return make_dict_parser(key, value)

    elif get_origin(type_hint) is Union:
        type_hints = get_args(type_hint)
        bases = set(get_base(type_hint) for type_hint in type_hints)
        assert Union not in bases
        if len(bases) != len(type_hints):
            raise TypeError("Unable to construct parsers for unions of the same base type")
        options = [
            make_parser_from_type_hint(type_hint, [sugg for sugg in suggestions if isinstance(sugg, get_base(type_hint))])
            for type_hint in type_hints
        ]
        return make_union_parser(options)

    else:
        raise ValueError(f"No parser for type hint: {type_hint!r}")

def get_suggestions(failure):
    suggestions = []
    if isinstance(failure, pc.ParseChoiceFailure):
        for subfailure in failure.failures:
            suggestions.extend(get_suggestions(subfailure))
    elif isinstance(failure, ParseSuggestion):
        suggestions.extend(failure.suggestions)
    return suggestions

def has_type(value, type_hint):
    if type_hint is None:
        type_hint = type(None)

    if type_hint in (type(None), bool, int, float, complex, str, bytes):
        return type(value) is type_hint

    elif isinstance(type_hint, type) and issubclass(type_hint, pathlib.Path):
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
    if type(value) in (type(None), bool, int, float, complex, str, bytes):
        return {*()}

    elif type(value) in (list, tuple, set):
        return {typ for subvalue in value for typ in get_used_custom_types(subvalue)}

    elif type(value) is dict:
        return {typ for entry in value.items() for subvalue in entry for typ in get_used_custom_types(subvalue)}

    elif isinstance(value, pathlib.Path):
        return {pathlib.Path}

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
        return "set([%s])" % ", ".join(format_value(subvalue) for subvalue in value)

    elif type(value) is dict:
        return "{%s}" % ", ".join(
            format_value(key) + ":" + format_value(subvalue)
            for key, subvalue in value.items()
        )

    elif isinstance(value, pathlib.Path):
        return "Path(%s)" % format_value(str(value))

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
