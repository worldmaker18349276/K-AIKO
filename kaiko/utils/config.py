"""
A configuration system using biparser.
The format of configuration file is a sub-language of python.
"""

import itertools
import keyword
import re
import typing
from collections import OrderedDict
from inspect import cleandoc
from pathlib import Path
from . import parsec as pc
from . import serializers as sz


@pc.parsec
def make_field_parser(config_type):
    """Parser for fields of configuration.

    It parse a series of field names of the configuration::

        subfieldname1.subfieldname2.fieldname3

    """
    current_fields = []
    current_type = config_type

    while isinstance(current_type, ConfigurableMeta):
        field_names = list(current_type.__configurable_fields__.keys())
        for field_name in field_names:
            sz.validate_identifier(field_name)
        current_field = yield pc.tokens(field_names)
        current_type = current_type.__configurable_fields__[current_field]
        if isinstance(current_type, ConfigurableMeta):
            yield pc.string(".")
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
        >> identifier.sep_by(pc.string(".")).map(".".join)
        << pc.regex(r"[ ]+import[ ]+").desc("' import '")
    ) + identifier.sep_by(pc.regex(r"[ ]*,[ ]*").desc("','"))
    sz.validate_identifier(config_name)
    sz.validate_identifier(config_type.__name__)
    init = (
        pc.string(config_name)
        >> equal
        >> pc.string(config_type.__name__ + "()")
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
        sz.validate_identifier(config_name)
        yield pc.string(config_name + ".")
        field_key = yield field

        # parse field value
        yield equal
        field_value = yield sz.make_parser_from_type_hint(field_hints[field_key][0])
        for typ in sz.get_used_custom_types(field_value):
            if (typ.__name__, typ.__module__) not in imported_modules.items():
                raise pc.ParseFailure(
                    f"import statement for module {typ.__module__}.{typ.__name__} at the beginning of the file"
                )
        set(config, field_key, field_value)

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

def set(config, fields, value):
    """Set a field of the configuration to the given value.

    Parameters
    ----------
    config : Configurable
        The configuration to manipulate.
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

    parent, curr = None, config
    field = fields[0]

    for i, field in enumerate(fields):
        if not isinstance(curr, Configurable):
            raise ValueError("not configurable field: " + repr(fields[:i]))

        if field not in curr.__configurable_fields__:
            raise ValueError("no such field: " + repr(fields[:i+1]))

        parent, curr = curr, curr.__dict__.get(field, None)

    else:
        setattr(parent, field, value)

def unset(config, fields):
    """Unset a field of the configuration.

    Parameters
    ----------
    config : Configurable
        The configuration to manipulate.
    fields : list of str
        The series of field names.

    Raises
    ------
    ValueError
        If there is no such field.
    """
    if len(fields) == 0:
        raise ValueError("empty field")

    parent, curr = None, config
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

def get(config, fields):
    """Get a field of the configuration.

    Parameters
    ----------
    config : Configurable
        The configuration to manipulate.
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

    parent, curr = None, config
    field = fields[0]

    for i, field in enumerate(fields):
        if not isinstance(curr, Configurable):
            raise ValueError("not configurable field: " + repr(fields[:i]))

        if field not in curr.__configurable_fields__:
            raise ValueError("no such field: " + repr(fields[:i+1]))

        parent, curr = curr, curr.__dict__.get(field, None)

    else:
        return getattr(parent, field)

def has(config, fields):
    """Check if a field of the configuration has a value.

    Parameters
    ----------
    config : Configurable
        The configuration to manipulate.
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

    parent, curr = None, config
    field = fields[0]

    for i, field in enumerate(fields):
        if not isinstance(curr, Configurable):
            return False

        if field not in curr.__configurable_fields__:
            return False

        parent, curr = curr, curr.__dict__.get(field, None)

    else:
        return field in parent.__dict__

def get_default(config, fields):
    """Get default value of a field of the configuration.

    Parameters
    ----------
    config : Configurable
        The configuration to manipulate.
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

    parent, curr = None, config
    field = fields[0]

    for i, field in enumerate(fields):
        if not isinstance(curr, Configurable):
            raise ValueError("not configurable field: " + repr(fields[:i]))

        if field not in curr.__configurable_fields__:
            raise ValueError("no such field: " + repr(fields[:i+1]))

        parent, curr = curr, curr.__dict__.get(field, None)

    else:
        return getattr(type(parent), field)

def has_default(config, fields):
    """Check if a field of the configuration has a default value.

    Parameters
    ----------
    config : Configurable
        The configuration to manipulate.
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

    parent, curr = None, config
    field = fields[0]

    for i, field in enumerate(fields):
        if not isinstance(curr, Configurable):
            raise ValueError("not configurable field: " + repr(fields[:i]))

        if field not in curr.__configurable_fields__:
            raise ValueError("no such field: " + repr(fields[:i+1]))

        parent, curr = curr, curr.__dict__.get(field, None)

    else:
        return hasattr(type(parent), field)

def parse(cls, text, name="settings"):
    parser = make_configuration_parser(cls, name)
    return parser.parse(text)

def format(cls, config, name="settings"):
    res = []
    res.append(f"{name} = {cls.__name__}()\n")
    custom_types = {*()}

    for key, (type_hint, _) in cls.__field_hints__.items():
        if not has(config, key):
            continue
        value = get(config, key)
        if not sz.has_type(value, type_hint):
            raise TypeError(f"Invalid type {value!r}, expecting {type_hint}")
        custom_types |= sz.get_used_custom_types(value)
        field = ".".join(key)
        res.append(f"{name}.{field} = {sz.format_value(value)}\n")

    imports = []

    for submodule in cls.__module__.split("."):
        sz.validate_identifier(submodule)
    sz.validate_identifier(cls.__name__)
    imports.append(f"from {cls.__module__} import {cls.__name__}\n")
    
    custom_types_list = sorted(list(custom_types), key=lambda typ: typ.__module__)
    for module, types in itertools.groupby(custom_types_list, key=lambda typ: typ.__module__):
        names = [typ.__name__ for typ in types]
        for submodule in module.split("."):
            sz.validate_identifier(submodule)
        for name in names:
            sz.validate_identifier(name)
        imports.append(f"from {module} import " + ", ".join(names) + "\n")

    return "".join(imports) + "\n" + "".join(res)

def read(cls, path, name="settings"):
    """Read configuration from a file.

    Parameters
    ----------
    cls : ConfigurableMeta
        The type of configuration.
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
    res = parse(cls, text, name=name)
    return res

def write(cls, config, path, name="settings"):
    """Write this configuration to a file.

    Parameters
    ----------
    cls : ConfigurableMeta
        The type of configuration.
    config : Configurable
        The configuration to manipulate.
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

    text = format(cls, config, name)
    open(path, 'w').write(text)

