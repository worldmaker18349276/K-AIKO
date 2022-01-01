"""
A configuration system using biparser.
The format of configuration file is a sub-language of python.
"""

import re
import typing
from collections import OrderedDict
from inspect import cleandoc
from pathlib import Path
from . import parsec as pc
from . import formattec as fc


@pc.parsec
def make_field_parser(config_type):
    """Parser for fields of configuration.

    It parse a series of field names of the configuration::

        subfieldname1.subfieldname2.fieldname3

    """
    current_fields = []
    current_type = config_type

    while hasattr(current_type, '__configurable_fields__'):
        fields = {}
        for field_name, field_type in current_type.__configurable_fields__.items():
            field_key = field_name
            if hasattr(field_type, '__configurable_fields__'):
                field_key = field_key + "."
            fields[field_key] = (field_name, field_type)

        option = yield pc.Parsec.tokens(list(fields.keys()))
        current_field, current_type = fields[option]
        current_fields.append(current_field)

    return tuple(current_fields)

def get_field_tree(config_type):
    fields = {}
    for field_name, field_type in config_type.__configurable_fields__.items():
        if hasattr(field_type, '__configurable_fields__'):
            fields[field_name + "."] = get_field_tree(field_type)
        else:
            fields[field_name] = lambda token: tuple(token.split("."))
    return fields

@pc.parsec
def make_configuration_parser(config_type, config_name):
    """Parser for Configurable.

    It parses a configuration like a python script::

        settings = SomeSettings()
        settings.fieldname1 = 1
        settings.fieldname2 = 'asd'
        settings.subfieldname.fieldname3 = 3.14

    """

    vindent = pc.Parsec.regex(r"(#[^\n]*|[ ]*)(\n|$)").many()
    equal = pc.Parsec.regex(r"[ ]*=[ ]*")
    nl = pc.Parsec.regex(r"[ ]*(\n|$)")
    field = make_field_parser(config_type)
    end = pc.Parsec.eof().optional()

    field_hints = config_type.__field_hints__

    # parse header
    yield vindent
    yield pc.Parsec.tokens([config_name])
    yield equal
    yield pc.Parsec.tokens([config_type.__name__ + "()"])
    yield nl

    config = config_type()
    
    while True:
        if (yield end):
            return config

        yield vindent

        # parse field name
        yield pc.Parsec.tokens([config_name + "."])
        field_key = yield field

        # parse field value
        yield equal
        field_value = yield pc.from_type_hint(field_hints[field_key][0])
        config.set(field_key, field_value)

        yield nl

def make_configuration_formatter(config_type, config_name):
    def formatter(value, **contexts):
        yield f"{config_name} = {config_type.__name__}()\n"

        for field_key, (field_type, field_doc) in config_type.__field_hints__.items():
            if not value.has(field_key):
                continue
            field_value = value.get(field_key)
            value_formatter = fc.from_type_hint(field_type, multiline=True)

            yield config_name + "."
            yield from ".".join(field_key)
            yield " = "
            yield from value_formatter.func(field_value)
            yield "\n"
    return fc.Formattec(formatter)

class ConfigurableMeta(type):
    def __init__(self, name, supers, attrs):
        super().__init__(name, supers, attrs)

        if not hasattr(self, '__configurable_excludes__'):
            self.__configurable_excludes__ = []
        annotations = typing.get_type_hints(self)

        fields = OrderedDict()
        for name in dir(self):
            if name not in self.__configurable_excludes__:
                if name in annotations:
                    fields[name] = annotations[name]
                elif isinstance(getattr(self, name), ConfigurableMeta):
                    fields[name] = getattr(self, name)

        fields_doc = ConfigurableMeta._parse_fields_doc(self.__doc__)
        field_hints = ConfigurableMeta._make_field_hints(fields, fields_doc)

        self.__configurable_fields__ = fields
        self.__configurable_fields_doc__ = fields_doc
        self.__field_hints__ = field_hints

    def __configurable_init__(self, instance):
        for field_name, field_type in self.__configurable_fields__.items():
            if hasattr(field_type, '__configurable_fields__'):
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

        parser = make_configuration_parser(cls, name)
        text = open(path, 'r').read()
        res = parser.parse(text)
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

        formatter = make_configuration_formatter(type(self), name)
        text = formatter.format(self)
        open(path, 'w').write(text)

