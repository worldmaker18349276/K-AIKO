"""
A configuration system using biparser.
The format of configuration file is a sub-language of python.
"""

import itertools
import keyword
import re
import typing
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
    identifier = (
        pc.regex(r"[a-zA-Z_][a-zA-Z0-9_]*")
        .reject(lambda name: None if not keyword.iskeyword(name) else "not keyword")
        .desc("identifier")
    )

    field = []
    current = config_type.__configurable_fields__
    is_first = True
    while isinstance(current, dict):
        if not is_first:
            yield pc.string(".")
        is_first = False
        name = yield identifier.reject(
            lambda name: None
            if name in current
            else " or ".join(map(repr, current.keys()))
        )
        current = current[name]
        field.append(name)

    return tuple(field)


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

    vindent = pc.regex(r"((#[^\n]*|[ ]*)(\n|$))*")
    equal = pc.regex(r"[ ]*=[ ]*").desc("'='")
    nl = pc.regex(r"[ ]*(\n|$)").desc(r"'\n'")
    field = make_field_parser(config_type)
    end = pc.eof().optional()
    identifier = (
        pc.regex(r"[a-zA-Z_][a-zA-Z0-9_]*")
        .reject(lambda name: None if not keyword.iskeyword(name) else "not keyword")
        .desc("identifier")
    )

    # parse header
    imports = (
        pc.regex(r"from[ ]+").desc("'from '")
        >> identifier.sep_by(pc.string(".")).map(".".join)
        << pc.regex(r"[ ]+import[ ]+").desc("' import '")
    ) + identifier.sep_by(pc.regex(r"[ ]*,[ ]*").desc("','"))
    sz.validate_identifier(config_name)
    sz.validate_identifier(config_type.__name__)
    init = pc.string(config_name) >> equal >> pc.string(config_type.__name__ + "()")
    header = (imports << nl << vindent).many_till(init << nl << vindent)

    def require(typ):
        if typ.__module__ == "builtins":
            return
        if (typ.__name__, typ.__module__) not in imported_modules.items():
            raise pc.ParseFailure(
                f"import statement for module {typ.__module__}.{typ.__name__} at the beginning of the file"
            )

    yield vindent
    imported = yield header
    imported_modules = {name: module for module, names in imported for name in names}

    # start building config
    config = config_type()
    require(config_type)

    while True:
        if (yield end):
            return config

        yield vindent

        # parse field name
        sz.validate_identifier(config_name)
        yield pc.string(config_name + ".")
        field_key = yield field
        yield equal

        # parse field value
        serializer = config_type.get_field_serializer(field_key)
        field_value = yield serializer.parser
        for typ in serializer.validator(field_value):
            require(typ)
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

        self.__configurable_fields__ = self.make_configurable_fields()

    def make_configurable_fields(self):
        excludes = getattr(self, "__configurable_excludes__", [])
        fields_doc = ConfigurableMeta._parse_fields_doc(self.__doc__)

        configurable_fields = {}

        for name, typ in typing.get_type_hints(self).items():
            if name not in excludes:
                if name not in configurable_fields:
                    configurable_fields[name] = (typ, fields_doc.get(name, None))

        for name in dir(self):
            if isinstance(getattr(self, name), SubConfigurable):
                if name not in configurable_fields:
                    configurable_fields[name] = getattr(
                        self, name
                    ).cls.__configurable_fields__

        return configurable_fields

    def __configurable_init__(self, instance):
        for field_name in dir(self):
            if isinstance(getattr(self, field_name), SubConfigurable):
                instance.__dict__[field_name] = getattr(self, field_name).cls()

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
        doc = doc[m.end(0) :]

        while True:
            m = re.match(r"([0-9a-zA-Z_]+) : [^\n]+\n+((?:[ ]+[^\n]*(?:\n+|$))*)", doc)
            if not m:
                return res
            res[m.group(1)] = cleandoc(m.group(2)).strip()
            doc = doc[m.end(0) :]

    def iter_all_fields(self):
        def it(current):
            for name, value in current.items():
                if isinstance(value, dict):
                    for field in it(value):
                        yield (name, *field)
                else:
                    yield (name,)

        yield from it(self.__configurable_fields__)

    def get_field_hint(self, field):
        current = self.__configurable_fields__

        if len(field) == 0:
            raise ValueError("Empty field")

        for name in field:
            if not isinstance(current, dict) or name not in current:
                raise ValueError(f"No such field: {name}")
            current = current[name]

        if isinstance(current, dict):
            raise ValueError(f"No such field: {'.'.join(field)}")
        return current

    def get_field_type(self, field):
        return self.get_field_hint(field)[0]

    def get_field_doc(self, field):
        return self.get_field_hint(field)[1]

    def get_field_serializer(self, field):
        field_type_hint, _ = self.get_field_hint(field)
        return sz.make_serializer_from_type_hint(field_type_hint)


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

    The field that is annotated or is Configurable type object will be assigned
    as a field of this configuration. One can define an exclusion list
    `__configurable_excludes__` to exclude them. The Configurable type object in
    this class will become sub-configuration, and will be created before
    initializing object. The others fields will become the field of this
    configuration, which is initially absent. So in the above example, the field
    access at the beginning is the fallback value of the static field in the
    class.
    """

    @classmethod
    def read(cls, path, name="settings"):
        return read(cls, path, name)

    def write(self, path, name="settings"):
        return write(type(self), self, path, name)

    def copy(self):
        return copy(type(self), self)


def set(config, field, value):
    """Set a field of the configuration to the given value.

    Parameters
    ----------
    config : Configurable
        The configuration to manipulate.
    field : list of str
        The series of field names.
    value : any
        The value to set.

    Raises
    ------
    ValueError
        If there is no such field.
    """
    type(config).get_field_hint(field)
    for name in field[:-1]:
        config = config.__dict__.get(name)
    else:
        setattr(config, field[-1], value)


def unset(config, field):
    """Unset a field of the configuration.

    Parameters
    ----------
    config : Configurable
        The configuration to manipulate.
    field : list of str
        The series of field names.

    Raises
    ------
    ValueError
        If there is no such field.
    """
    type(config).get_field_hint(field)
    for name in field[:-1]:
        config = config.__dict__.get(name)
    else:
        delattr(config, field[-1])


def get(config, field):
    """Get a field of the configuration.

    Parameters
    ----------
    config : Configurable
        The configuration to manipulate.
    field : list of str
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
    type(config).get_field_hint(field)
    for name in field[:-1]:
        config = config.__dict__.get(name)
    else:
        return getattr(config, field[-1])


def has(config, field):
    """Check if a field of the configuration has a value.

    Parameters
    ----------
    config : Configurable
        The configuration to manipulate.
    field : list of str
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
    type(config).get_field_hint(field)
    for name in field[:-1]:
        config = config.__dict__.get(name)
    else:
        return field[-1] in config.__dict__


def get_default(config, field):
    """Get default value of a field of the configuration.

    Parameters
    ----------
    config : Configurable
        The configuration to manipulate.
    field : list of str
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
    type(config).get_field_hint(field)
    for name in field[:-1]:
        config = config.__dict__.get(name)
    else:
        return getattr(type(config), field[-1])


def has_default(config, field):
    """Check if a field of the configuration has a default value.

    Parameters
    ----------
    config : Configurable
        The configuration to manipulate.
    field : list of str
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
    type(config).get_field_hint(field)
    for name in field[:-1]:
        config = config.__dict__.get(name)
    else:
        return hasattr(type(config), field[-1])


def parse(cls, text, name="settings"):
    parser = make_configuration_parser(cls, name)
    return parser.parse(text)


def format(cls, config, name="settings"):
    res = []
    res.append(f"{name} = {cls.__name__}()\n")
    used_types = {*()}

    for field in cls.iter_all_fields():
        if not has(config, field):
            continue
        value = get(config, field)
        serializer = cls.get_field_serializer(field)
        value_types = serializer.validator(value)
        if not value_types:
            raise TypeError(f"Invalid value {value!r}")
        used_types |= value_types
        res.append(f"{name}.{'.'.join(field)} = {serializer.formatter(value)}\n")

    imports = []

    for submodule in cls.__module__.split("."):
        sz.validate_identifier(submodule)
    sz.validate_identifier(cls.__name__)
    imports.append(f"from {cls.__module__} import {cls.__name__}\n")

    custom_types_list = sorted(list(used_types), key=lambda typ: typ.__module__)
    for module, types in itertools.groupby(
        custom_types_list, key=lambda typ: typ.__module__
    ):
        names = [typ.__name__ for typ in types]
        if module == "builtins":
            continue
        for submodule in module.split("."):
            sz.validate_identifier(submodule)
        for name in names:
            sz.validate_identifier(name)
        imports.append(f"from {module} import " + ", ".join(names) + "\n")

    return "".join(imports) + "\n" + "".join(res)


def copy(cls, config):
    copied = cls()

    for field in cls.iter_all_fields():
        if not has(config, field):
            continue
        value = get(config, field)
        set(copied, field, value)

    return copied


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

    text = open(path, "r").read()
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
    open(path, "w").write(text)
