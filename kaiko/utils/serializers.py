import keyword
import ast
import enum
import pathlib
import dataclasses
import typing
from typing import Any, Callable, Dict, List, Set, Tuple, Union
from . import parsec as pc


# literals


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


def get_suggestions(failure):
    suggestions = []
    if isinstance(failure, pc.ParseChoiceFailure):
        for subfailure in failure.failures:
            suggestions.extend(get_suggestions(subfailure))
    elif isinstance(failure, ParseSuggestion):
        suggestions.extend(failure.suggestions)
    return suggestions


@dataclasses.dataclass
class Serializer:
    parser: pc.Parsec
    formatter: Callable[[Any], str]
    validator: Callable[[Any], Set[type]]

    def suggest(self, suggestions):
        # assert all(bool(self.validator(sugg)) for sugg in suggestions)
        suggestions_str = [self.formatter(sugg) for sugg in set(suggestions)]
        suggested_parser = suggest(self.parser, suggestions_str)
        return Serializer(suggested_parser, self.formatter, self.validator)

    @staticmethod
    def make_literal(typ, expr, formatter):
        validator = lambda value: {typ} if type(value) is typ else set()
        parser = pc.regex(expr).map(ast.literal_eval).desc(typ.__name__)
        return Serializer(parser, formatter, validator)


def make_none_serializer(suggestions=[]):
    return Serializer.make_literal(type(None), r"None", repr).suggest(
        suggestions or [None]
    )


def make_bool_serializer(suggestions=[]):
    suggestions = list(suggestions)
    if False not in suggestions:
        suggestions.append(False)
    if True not in suggestions:
        suggestions.append(True)
    return Serializer.make_literal(bool, r"False|True", repr).suggest(suggestions)


def make_int_serializer(suggestions=[]):
    expr = r"[-+]?(0|[1-9][0-9]*)(?![0-9\.\+eEjJ])"
    return Serializer.make_literal(int, expr, repr).suggest(suggestions or [0])


def make_float_serializer(suggestions=[]):
    expr = r"[-+]?([0-9]+\.[0-9]+(e[-+]?[0-9]+)?|[0-9]+[eE][-+]?[0-9]+)(?![0-9\+jJ])"
    return Serializer.make_literal(float, expr, repr).suggest(suggestions or [0.0])


def make_complex_serializer(suggestions=[]):
    expr = r"[-+]?({0}[-+])?{0}[jJ]".format(
        r"(0|[1-9][0-9]*|[0-9]+\.[0-9]+(e[-+]?[0-9]+)?|[0-9]+e[-+]?[0-9]+)"
    )

    def format_complex(value):
        repr_value = repr(value)
        # remove parentheses
        if repr_value.startswith("(") and repr_value.endswith(")"):
            repr_value = repr_value[1:-1]
        return repr_value

    return Serializer.make_literal(complex, expr, format_complex).suggest(
        suggestions or [0j]
    )


def make_bytes_serializer(suggestions=[]):
    expr = (
        r'b"('
        r'(?![\r\n\\"])[\x01-\x7f]'
        r"|\\[0-7]{1,3}"
        r"|\\x[0-9a-fA-F]{2}"
        r"|\\u[0-9a-fA-F]{4}"
        r"|\\U[0-9a-fA-F]{8}"
        r"|\\(?![xuUN])[\x01-\x7f]"
        r')*"'
    )

    def format_bytes(value):
        # make sure it uses double quotation
        return (
            'b"'
            + repr(value + b'"')[2:-2].replace('"', r"\"").replace(r"\'", "'")
            + '"'
        )

    return Serializer.make_literal(bytes, expr, format_bytes).suggest(
        suggestions or [b""]
    )


def make_str_serializer(suggestions=[]):
    expr = (
        r'"('
        r'[^\r\n\\"\x00]'
        r"|\\[0-7]{1,3}"
        r"|\\x[0-9a-fA-F]{2}"
        r"|\\u[0-9a-fA-F]{4}"
        r"|\\U[0-9a-fA-F]{8}"
        r"|\\(?![xuUN\x00])."
        r')*"'
    )

    def format_str(value):
        # make sure it uses double quotation
        return (
            '"' + repr(value + '"')[1:-2].replace('"', r"\"").replace(r"\'", "'") + '"'
        )

    return Serializer.make_literal(str, expr, format_str).suggest(suggestions or [""])


# composite


def IIFE(func):
    return func()


def make_list_serializer(elem):
    elem_parser = elem.parser
    elem_formatter = elem.formatter
    elem_validator = elem.validator

    @IIFE
    @pc.parsec
    def parser():
        opening = suggest(pc.regex(r"\[\s*").desc("left bracket"), ["["])
        comma = suggest(pc.regex(r"\s*,\s*").desc("comma"), [","])
        closing = suggest(pc.regex(r"\s*\]").desc("right bracket"), ["]"])
        yield opening
        results = []
        try:
            while True:
                results.append((yield elem_parser))
                yield comma
        except pc.ParseFailure as failure:
            with failure.retry():
                yield closing
            return results

    def validator(value):
        if type(value) is not list:
            return set()
        types = {list}
        for elem in value:
            elem_types = elem_validator(elem)
            if not elem_types:
                return set()
            types |= elem_types
        return types

    formatter = lambda value: "[%s]" % ", ".join(elem_formatter(elem) for elem in value)

    return Serializer(parser, formatter, validator)


def make_set_serializer(elem):
    elem_parser = elem.parser
    elem_formatter = elem.formatter
    elem_validator = elem.validator

    @IIFE
    @pc.parsec
    def parser():
        opening = suggest(pc.regex(r"set\(\[\s*").desc("'set(['"), ["set(["])
        closing = suggest(pc.regex(r"\s*\]\)").desc("'])'"), ["])"])
        comma = suggest(pc.regex(r"\s*,\s*").desc("comma"), [","])
        yield opening
        results = set()
        try:
            while True:
                results.add((yield elem_parser))
                yield comma
        except pc.ParseFailure as failure:
            with failure.retry():
                yield closing
            return results

    def validator(value):
        if type(value) is not set:
            return set()
        types = {set}
        for elem in value:
            elem_types = elem_validator(elem)
            if not elem_types:
                return set()
            types |= elem_types
        return types

    formatter = lambda value: "set([%s])" % ", ".join(
        elem_formatter(elem) for elem in value
    )

    return Serializer(parser, formatter, validator)


def make_dict_serializer(key, value):
    key_parser = key.parser
    key_formatter = key.formatter
    key_validator = key.validator
    value_parser = value.parser
    value_formatter = value.formatter
    value_validator = value.validator

    @IIFE
    @pc.parsec
    def parser():
        opening = suggest(pc.regex(r"\{\s*").desc("left brace"), ["{"])
        colon = suggest(pc.regex(r"\s*:\s*").desc("colon"), [":"])
        comma = suggest(pc.regex(r"\s*,\s*").desc("comma"), [","])
        closing = suggest(pc.regex(r"\s*\}").desc("right brace"), ["}"])
        item = colon.join((key_parser, value_parser))

        yield opening
        results = {}
        try:
            while True:
                k, v = yield item
                results[k] = v
                yield comma
        except pc.ParseFailure as failure:
            with failure.retry():
                yield closing
            return results

    def validator(value):
        if type(value) is not dict:
            return set()
        types = {dict}
        for k, v in value.items():
            k_types = key_validator(k)
            if not k_types:
                return set()
            types |= k_types
            v_types = value_validator(v)
            if not v_types:
                return set()
            types |= v_types
        return types

    def formatter(value):
        return "{%s}" % ", ".join(
            key_formatter(k) + ":" + value_formatter(v) for k, v in value.items()
        )

    return Serializer(parser, formatter, validator)


def make_tuple_serializer(elems):
    elem_parsers = [elem.parser for elem in elems]
    elem_formatters = [elem.formatter for elem in elems]
    elem_validators = [elem.validator for elem in elems]

    @IIFE
    @pc.parsec
    def parser():
        opening = suggest(pc.regex(r"\(\s*").desc("left parenthesis"), ["("])
        comma = suggest(pc.regex(r"\s*,\s*").desc("comma"), [","])
        closing = suggest(pc.regex(r"\s*\)").desc("right parenthesis"), [")"])

        if len(elem_parsers) == 0:
            yield opening
            yield closing
            return ()

        elif len(elem_parsers) == 1:
            yield opening
            res = yield elem_parsers[0]
            yield comma
            yield closing
            return (res,)

        else:
            yield opening
            results = []
            is_first = True
            for elem_parser in elem_parsers:
                if not is_first:
                    yield comma
                is_first = False
                results.append((yield elem_parser))
            yield comma.optional()
            yield closing
            return tuple(results)

    def validator(value):
        if type(value) is not tuple or len(value) != len(elem_validators):
            return set()
        types = {tuple}
        for elem, elem_validator in zip(value, elem_validators):
            elem_types = elem_validator(elem)
            if not elem_types:
                return set()
            types |= elem_types
        return types

    def formatter(value):
        if len(elem_formatters) == 1:
            return "(%s,)" % elem_formatters[0].formatter(value)
        else:
            return "(%s)" % ", ".join(
                elem_formatter(elem)
                for elem, elem_formatter in zip(value, elem_formatters)
            )

    return Serializer(parser, formatter, validator)


def make_union_serializer(options):
    if len(options) == 0:
        raise ValueError("empty union")
    elif len(options) == 1:
        return list(options.values())[0]

    option_validators = [option.validator for option in options.values()]
    option_parsers = [option.parser for option in options.values()]
    option_formatters = [(base, option.formatter) for base, option in options.items()]

    def validator(value):
        for option_validator in option_validators:
            res = option_validator(value)
            if res:
                return res
        return set()

    def formatter(value):
        for base, option_formatter in option_formatters:
            if isinstance(value, base):
                return option_formatter(value)
        raise TypeError

    parser = pc.choice(*option_parsers)
    return Serializer(parser, formatter, validator)


# custom


def make_enum_serializer(cls):
    @IIFE
    @pc.parsec
    def parser():
        yield suggest(pc.string(f"{cls.__name__}."), [f"{cls.__name__}."])
        option = yield pc.choice(
            *[suggest(pc.string(option.name), [option.name]) for option in cls]
        )
        return getattr(cls, option)

    validator = lambda value: {cls} if type(value) is cls else set()
    formatter = lambda value: f"{cls.__name__}.{value.name}"
    return Serializer(parser, formatter, validator)


def make_path_serializer(suggestions=[pathlib.Path(".")]):
    str_serializer = make_str_serializer([str(sugg) for sugg in suggestions])
    str_parser = str_serializer.parser
    str_formatter = str_serializer.formatter

    @IIFE
    @pc.parsec
    def parser():
        opening = suggest(pc.regex(r"Path\(\s*").desc("'Path('"), ["Path("])
        closing = suggest(pc.regex(r"\s*\)").desc("')'"), [")"])
        yield opening
        path = yield str_parser
        yield closing
        return pathlib.Path(path)

    validator = (
        lambda value: {pathlib.Path} if isinstance(value, pathlib.Path) else set()
    )
    formatter = lambda value: "Path(%s)" % str_formatter(str(value))
    return Serializer(parser, formatter, validator)


def make_dataclass_serializer(cls, fields):
    field_parsers = {key: serializer.parser for key, serializer in fields.items()}
    field_validators = {key: serializer.validator for key, serializer in fields.items()}
    field_formatters = {key: serializer.formatter for key, serializer in fields.items()}

    @IIFE
    @pc.parsec
    def parser():
        opening = suggest(
            pc.regex(fr"{cls.__name__}\s*\(\s*").desc(f"'{cls.__name__}('"),
            [f"{cls.__name__}("],
        )
        comma = suggest(pc.regex(r"\s*,\s*").desc("comma"), [","])
        closing = suggest(pc.regex(r"\s*\)").desc("right parenthesis"), [")"])

        yield opening
        results = {}
        is_first = True
        for key, field_parser in field_parsers.items():
            if not is_first:
                yield comma
            is_first = False
            yield suggest(pc.regex(fr"{key}\s*=\s*").desc(f"{key}="), [f"{key}="])
            value = yield field_parser
            results[key] = value
        yield comma.optional()
        yield closing
        return cls(**results)

    def validator(value):
        if type(value) is not cls:
            return set()
        types = {cls}
        for key, field_validator in field_validators.items():
            field_types = field_validator(getattr(value, key))
            if not field_types:
                return set()
            types |= field_types
        return types

    def formatter(value):
        return f"{cls.__name__}(%s)" % ", ".join(
            key + "=" + field_formatter(getattr(value, key))
            for key, field_formatter in field_formatters.items()
        )

    return Serializer(parser, formatter, validator)


def get_args(type_hint):
    if hasattr(typing, "get_args"):
        return typing.get_args(type_hint)
    else:
        return getattr(type_hint, "__args__", ())


def get_origin(type_hint):
    if hasattr(typing, "get_origin"):
        return typing.get_origin(type_hint)
    else:
        origin = getattr(type_hint, "__origin__", None)
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


def validate_identifier(name):
    if (
        not isinstance(name, str)
        or not str.isidentifier(name)
        or keyword.iskeyword(name)
    ):
        raise ValueError(f"Invalid identifier {name!r}")


def make_serializer_from_type_hint(type_hint, suggestions=[]):
    """Make Serializer from type hint.

    Parameters
    ----------
    type_hint : type or generic
        The type to parse.
    suggestions : list, optional
        The suggested values, which should be instances of `type_hint`.

    Returns
    -------
    Serializer
        The serializer of the given type.
    """
    if type_hint is None:
        type_hint = type(None)

    if type_hint is type(None):
        return make_none_serializer(suggestions)

    elif type_hint is bool:
        return make_bool_serializer(suggestions)

    elif type_hint is int:
        return make_int_serializer(suggestions)

    elif type_hint is float:
        return make_float_serializer(suggestions)

    elif type_hint is complex:
        return make_complex_serializer(suggestions)

    elif type_hint is str:
        return make_str_serializer(suggestions)

    elif type_hint is bytes:
        return make_bytes_serializer(suggestions)

    elif isinstance(type_hint, type) and issubclass(type_hint, enum.Enum):
        validate_identifier(type_hint.__name__)
        for option in type_hint:
            validate_identifier(option.name)
        return make_enum_serializer(type_hint)

    elif isinstance(type_hint, type) and issubclass(type_hint, pathlib.Path):
        return make_path_serializer(suggestions)

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
            fields[field.name] = make_serializer_from_type_hint(
                field.type, subsuggestions
            )
        return make_dataclass_serializer(type_hint, fields)

    elif get_origin(type_hint) is list:
        (elem_hint,) = get_args(type_hint)
        elem = make_serializer_from_type_hint(
            elem_hint, [elem for sugg in suggestions for elem in sugg]
        )
        return make_list_serializer(elem)

    elif get_origin(type_hint) is set:
        (elem_hint,) = get_args(type_hint)
        elem = make_serializer_from_type_hint(
            elem_hint, [elem for sugg in suggestions for elem in sugg]
        )
        return make_set_serializer(elem)

    elif get_origin(type_hint) is tuple:
        args = get_args(type_hint)
        if len(args) == 1 and args[0] == ():
            elems = []
        else:
            elems = [
                make_serializer_from_type_hint(arg, [sugg[i] for sugg in suggestions])
                for i, arg in enumerate(args)
            ]
        return make_tuple_serializer(elems)

    elif get_origin(type_hint) is dict:
        key_hint, value_hint = get_args(type_hint)
        key = make_serializer_from_type_hint(
            key_hint, [k for sugg in suggestions for k in sugg.keys()]
        )
        value = make_serializer_from_type_hint(
            value_hint, [v for sugg in suggestions for v in sugg.values()]
        )
        return make_dict_serializer(key, value)

    elif get_origin(type_hint) is Union:
        type_hints = get_args(type_hint)
        bases = {get_base(type_hint): type_hint for type_hint in type_hints}
        assert Union not in bases
        if len(bases) != len(type_hints):
            raise TypeError(
                "Unable to construct parsers for unions of the same base type"
            )
        options = {
            base: make_serializer_from_type_hint(
                type_hint, [sugg for sugg in suggestions if isinstance(sugg, base)]
            )
            for base, type_hint in bases.items()
        }
        return make_union_serializer(options)

    else:
        raise ValueError(f"No serializer for type {type_hint!r}")
