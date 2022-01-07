"""
A configuration system using biparser.
The format of configuration file is a sub-language of python.
"""

import itertools
import keyword
import re
import ast
import enum
import dataclasses
import typing
from typing import Dict, List, Set, Tuple, Union
from collections import OrderedDict
from inspect import cleandoc
from pathlib import Path
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
        return set()

    elif type(value) in (bool, int, float, complex, str, bytes):
        return set()

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


@pc.parsec
def make_field_parser(config_type):
    """Parser for fields of configuration.

    It parse a series of field names of the configuration::

        subfieldname1.subfieldname2.fieldname3

    """
    current_fields = []
    current_type = config_type

    while isinstance(current_type, ConfigurableMeta):
        current_field = yield parse_identifiers(*current_type.__configurable_fields__.keys())
        current_type = current_type.__configurable_fields__[current_field]
        if isinstance(current_type, ConfigurableMeta):
            yield pc.tokens(["."])
        current_fields.append(current_field)

    return tuple(current_fields)

def get_field_tree(config_type):
    fields = {}
    for field_name, field_type in config_type.__configurable_fields__.items():
        if isinstance(field_type, ConfigurableMeta):
            fields[field_name + "."] = get_field_tree(field_type)
        else:
            fields[field_name] = lambda token: tuple(token.split("."))
    return fields

@pc.parsec
def make_configuration_parser(config_type, config_name):
    """Parser for Configurable.

    It parses a configuration like a python script::

        from some.where import SomeSettings

        settings = SomeSettings()
        settings.fieldname1 = 1
        settings.fieldname2 = 'asd'
        settings.subfieldname.fieldname3 = 3.14

    """

    vindent = pc.regex(r"(#[^\n]*|[ ]*)(\n|$)").many()
    equal = pc.regex(r"[ ]*=[ ]*").desc("'='")
    nl = pc.regex(r"[ ]*(\n|$)").desc(r"'\n'")
    field = make_field_parser(config_type)
    end = pc.eof().optional()
    identifier = (
        pc.regex(r"[a-zA-Z_][a-zA-Z0-9_]*")
            .reject(lambda name: None if not keyword.iskeyword(name) else "not keyword")
            .desc("identifier")
    )

    field_hints = config_type.__field_hints__

    # parse header
    imports = (
        pc.regex(r"from[ ]+").desc("'from '")
        >> identifier.sep_by(pc.tokens(["."])).map(".".join)
        << pc.regex(r"[ ]+import[ ]+").desc("' import '")
    ) + identifier.sep_by(pc.regex(r"[ ]*,[ ]*").desc("','"))
    init = (
        parse_identifiers(config_name)
        >> equal
        >> parse_identifiers(config_type.__name__)
        >> pc.tokens(["()"])
    )
    header = (imports << nl << vindent).many_till(init << nl << vindent)

    yield vindent
    imported = yield header
    imported_modules = {name: module for module, names in imported for name in names}

    # start building config
    config = config_type()
    if (config_type.__name__, config_type.__module__) not in imported_modules.items():
        raise pc.ParseFailure(
            f"import statement for module {config_type.__module__}.{config_type.__name__} at the beginning of the file"
        )
    
    while True:
        if (yield end):
            return config

        yield vindent

        # parse field name
        yield parse_identifiers(config_name)
        yield pc.tokens(["."])
        field_key = yield field

        # parse field value
        yield equal
        field_value = yield make_parser_from_type_hint(field_hints[field_key][0])
        for typ in get_used_custom_types(field_value):
            if (typ.__name__, typ.__module__) not in imported_modules.items():
                raise pc.ParseFailure(
                    f"import statement for module {typ.__module__}.{typ.__name__} at the beginning of the file"
                )
        config.set(field_key, field_value)

        yield nl

class SubConfigurable:
    def __init__(self, cls):
        self.cls = cls

def subconfig(cls):
    return SubConfigurable(cls)

class ConfigurableMeta(type):
    def __init__(self, name, supers, attrs):
        super().__init__(name, supers, attrs)

        if not hasattr(self, '__configurable_excludes__'):
            self.__configurable_excludes__ = []
        annotations = typing.get_type_hints(self)

        fields = OrderedDict(annotations)
        for name in dir(self):
            if isinstance(getattr(self, name), SubConfigurable):
                fields[name] = getattr(self, name).cls
        for name in self.__configurable_excludes__:
            if name in fields:
                del fields[name]

        fields_doc = ConfigurableMeta._parse_fields_doc(self.__doc__)
        field_hints = ConfigurableMeta._make_field_hints(fields, fields_doc)

        self.__configurable_fields__ = fields
        self.__configurable_fields_doc__ = fields_doc
        self.__field_hints__ = field_hints

    def __configurable_init__(self, instance):
        for field_name, field_type in self.__configurable_fields__.items():
            if isinstance(field_type, ConfigurableMeta):
                instance.__dict__[field_name] = field_type()

    def __call__(self, *args, **kwargs):
        instance = self.__new__(self, *args, **kwargs)
        self.__configurable_init__(instance)
        self.__init__(instance, *args, **kwargs)
        return instance

    @staticmethod
    def _parse_fields_doc(doc):
        # r"""
        # Fields
        # ------
        # field1 : type1
        #     This is the field-level description for field 1.
        # field2 : type2 or sth...
        #     This is the field 2.
        #     And bla bla bla.
        # """

        res = {}

        if doc is None:
            return res

        doc = cleandoc(doc)

        m = re.search(r"Fields\n------\n", doc)
        if not m:
            return res
        doc = doc[m.end(0):]

        while True:
            m = re.match(r"([0-9a-zA-Z_]+) : [^\n]+\n+((?:[ ]+[^\n]*(?:\n+|$))*)", doc)
            if not m:
                return res
            res[m.group(1)] = cleandoc(m.group(2)).strip()
            doc = doc[m.end(0):]

    @staticmethod
    def _make_field_hints(fields, fields_doc):
        """Make hints for configurable fields of this configuration.

        Parameters
        ----------
        fields : dict
            Dictionary of configurable fields.
        fields_doc : dict
            Docstring of configurable fields.

        Returns
        -------
        field_hints : dict
            A dictionary which maps a series of field names to its field type.
            If it has item `(('a', 'b', 'c'), (float, "floating point number"))`,
            then this configuration should have the field `config.a.b.c` with
            type `float`.
        """
        field_hints = {}
        for field_name, field_type in fields.items():
            if not isinstance(field_type, ConfigurableMeta):
                field_doc = fields_doc.get(field_name, None)
                field_hints[(field_name,)] = (field_type, field_doc)
            else:
                for subfield_names, subfield_hint in field_type.__field_hints__.items():
                    field_hints[(field_name, *subfield_names)] = subfield_hint
        return field_hints

    def get_field_type(self, field):
        if field not in self.__field_hints__:
            raise ValueError("No such field: " + ".".join(field))
        annotation, _ = self.__field_hints__[field]
        return annotation

    def get_field_doc(self, field):
        if field not in self.__field_hints__:
            raise ValueError("No such field: " + ".".join(field))
        _, doc = self.__field_hints__[field]
        return doc

class Configurable(metaclass=ConfigurableMeta):
    """The super class for configuration.

    With this type, the configuration can be easily defined::

        class SomeSettings(Configurable):
            field1: int = 123
            field2: str = 'abc'

            @subconfig
            class subsettings(Configurable):
                field3: bool = True
                field4: float = 3.14

        settings = SomeSettings()

        print(settings.field1)  # 123
        print(settings.subsettings.field3)  # True

        settings.field1 = 456
        settings.subsettings.field3 = False

        print(settings.field1)  # 456
        print(settings.subsettings.field3)  # False

    The field that is annotated or is Configurable type object will be
    assigned as a field of this configuration.  One can define an
    exclusion list `__configurable_excludes__` to exclude them.  The
    Configurable type object in this class will become sub-configuration,
    and will be created before initializing object.  The others fields
    will become the field of this configuration, which is initially absent.
    So in the above example, the field access at the beginning is the
    fallback value of the static field in the class.
    """
    def set(self, fields, value):
        """Set a field of the configuration to the given value.

        Parameters
        ----------
        fields : list of str
            The series of field names.
        value : any
            The value to set.

        Raises
        ------
        ValueError
            If there is no such field.
        """
        if len(fields) == 0:
            raise ValueError("empty field")

        parent, curr = None, self
        field = fields[0]

        for i, field in enumerate(fields):
            if not isinstance(curr, Configurable):
                raise ValueError("not configurable field: " + repr(fields[:i]))

            if field not in curr.__configurable_fields__:
                raise ValueError("no such field: " + repr(fields[:i+1]))

            parent, curr = curr, curr.__dict__.get(field, None)

        else:
            setattr(parent, field, value)

    def unset(self, fields):
        """Unset a field of the configuration.

        Parameters
        ----------
        fields : list of str
            The series of field names.

        Raises
        ------
        ValueError
            If there is no such field.
        """
        if len(fields) == 0:
            raise ValueError("empty field")

        parent, curr = None, self
        field = fields[0]

        for i, field in enumerate(fields):
            if not isinstance(curr, Configurable):
                raise ValueError("not configurable field: " + repr(fields[:i]))

            if field not in curr.__configurable_fields__:
                raise ValueError("no such field: " + repr(fields[:i+1]))

            parent, curr = curr, curr.__dict__.get(field, None)

        else:
            if field in parent.__dict__:
                delattr(parent, field)

    def get(self, fields):
        """Get a field of the configuration.

        Parameters
        ----------
        fields : list of str
            The series of field names.

        Returns
        -------
        value : any
            The value of the field.

        Raises
        ------
        ValueError
            If there is no such field.
        """
        if len(fields) == 0:
            raise ValueError("empty field")

        parent, curr = None, self
        field = fields[0]

        for i, field in enumerate(fields):
            if not isinstance(curr, Configurable):
                raise ValueError("not configurable field: " + repr(fields[:i]))

            if field not in curr.__configurable_fields__:
                raise ValueError("no such field: " + repr(fields[:i+1]))

            parent, curr = curr, curr.__dict__.get(field, None)

        else:
            return getattr(parent, field)

    def has(self, fields):
        """Check if a field of the configuration has a value.

        Parameters
        ----------
        fields : list of str
            The series of field names.

        Returns
        -------
        res : bool
            True if this field has a value.

        Raises
        ------
        ValueError
            If there is no such field.
        """
        if len(fields) == 0:
            raise ValueError("empty field")

        parent, curr = None, self
        field = fields[0]

        for i, field in enumerate(fields):
            if not isinstance(curr, Configurable):
                return False

            if field not in curr.__configurable_fields__:
                return False

            parent, curr = curr, curr.__dict__.get(field, None)

        else:
            return field in parent.__dict__

    def get_default(self, fields):
        """Get default value of a field of the configuration.

        Parameters
        ----------
        fields : list of str
            The series of field names.

        Returns
        -------
        value : any
            The value of the field.

        Raises
        ------
        ValueError
            If there is no such field.
        """
        if len(fields) == 0:
            raise ValueError("empty field")

        parent, curr = None, self
        field = fields[0]

        for i, field in enumerate(fields):
            if not isinstance(curr, Configurable):
                raise ValueError("not configurable field: " + repr(fields[:i]))

            if field not in curr.__configurable_fields__:
                raise ValueError("no such field: " + repr(fields[:i+1]))

            parent, curr = curr, curr.__dict__.get(field, None)

        else:
            return getattr(type(parent), field)

    def has_default(self, fields):
        """Check if a field of the configuration has a default value.

        Parameters
        ----------
        fields : list of str
            The series of field names.

        Returns
        -------
        res : bool
            True if this field has a default value.

        Raises
        ------
        ValueError
            If there is no such field.
        """
        if len(fields) == 0:
            raise ValueError("empty field")

        parent, curr = None, self
        field = fields[0]

        for i, field in enumerate(fields):
            if not isinstance(curr, Configurable):
                raise ValueError("not configurable field: " + repr(fields[:i]))

            if field not in curr.__configurable_fields__:
                raise ValueError("no such field: " + repr(fields[:i+1]))

            parent, curr = curr, curr.__dict__.get(field, None)

        else:
            return hasattr(type(parent), field)

    @classmethod
    def parse(cls, text, name="settings"):
        parser = make_configuration_parser(cls, name)
        return parser.parse(text)

    def format(self, name="settings"):
        cls = type(self)
        res = []
        res.append(f"{name} = {cls.__name__}()\n")
        custom_types = set()

        for key, (type_hint, _) in cls.__field_hints__.items():
            if not self.has(key):
                continue
            value = self.get(key)
            if not has_type(value, type_hint):
                raise TypeError(f"Invalid type {value!r}, expecting {type_hint}")
            custom_types |= get_used_custom_types(value)
            field = ".".join(key)
            res.append(f"{name}.{field} = {format_value(value)}\n")

        imports = []

        cls = type(self)
        for submodule in cls.__module__.split("."):
            validate_identifier(submodule)
        validate_identifier(cls.__name__)
        imports.append(f"from {cls.__module__} import {cls.__name__}\n")
        
        custom_types_list = sorted(list(custom_types), key=lambda typ: typ.__module__)
        for module, types in itertools.groupby(custom_types_list, key=lambda typ: typ.__module__):
            names = [typ.__name__ for typ in types]
            for submodule in module.split("."):
                validate_identifier(submodule)
            for name in names:
                validate_identifier(name)
            imports.append(f"from {module} import " + ", ".join(names) + "\n")

        return "".join(imports) + "\n" + "".join(res)

    @classmethod
    def read(cls, path, name="settings"):
        """Read configuration from a file.

        Parameters
        ----------
        path : str or Path
            The path of file.
        name : str, optional
            The name of the resulting configuration.

        Returns
        -------
        config

        Raises
        ------
        ValueError
            If there is no such file.
        DecodeError
            If decoding fails.
        """
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise ValueError(f"No such file: {path!s}")

        # text = open(path, 'r').read()
        # locals = {}
        # exec(text, globals(), locals)
        # return locals[self.name]

        text = open(path, 'r').read()
        res = cls.parse(text, name=name)
        return res

    def write(self, path, name="settings"):
        """Write this configuration to a file.

        Parameters
        ----------
        path : str or Path
            The path of file.
        name : str, optional
            The name of the resulting configuration.

        Raises
        ------
        biparsers.EncodeError
            If encoding fails.
        """
        if isinstance(path, str):
            path = Path(path)

        text = self.format(name)
        open(path, 'w').write(text)

