"""
Bidirectional parsers for structured data type.
"""

import re
import ast
import enum
import dataclasses
import typing


class DecodeError(Exception):
    """Decode error.

    Attributes
    ----------
    text : str
        The text to decode.
    index : int
        The index of text where the decoding fails
    expected : list of str
        The expected texts to decode.  The suffix "\000" means eof.
    info : str, optional
        The description for this exception.
    """
    def __init__(self, text, index, expected, info=None):
        self.text = text
        self.index = index
        self.expected = expected
        self.info = info

    def __str__(self):
        if self.index > len(self.text):
            loc = f"<out of bounds index {self.index}>"

        else:
            line = self.text.count("\n", 0, self.index)
            last_ln = self.text.rfind("\n", 0, self.index)
            col = self.index - (last_ln + 1)
            loc = f"{line}:{col}"

        if self.info:
            return f"parse failed at {loc}: {self.info}"
        else:
            return f"parse failed at {loc}, expect: {self.expected!r}"

class EncodeError(Exception):
    """Encode error.

    Attributes
    ----------
    value : any
        The value to encode.
    pos : any
        The position of value where the encoding fails.
    expected : list
        The expected values to encode.
    info : str, optional
        The description for this exception.
    """
    def __init__(self, value, pos, expected, info=None):
        self.value = value
        self.pos = pos
        self.expected = expected
        self.info = info

class Biparser:
    def decode(self, text, index=0, partial=False):
        """Decode a string into a value.

        Parameters
        ----------
        text : str
            The text to decode.
        index : int
            The index of text where the decoding starts with.
        partial : bool, optional
            True for partially decoding.  It will parse the longest possible prefix of a given string.

        Returns
        -------
        value : any
            The decoded value of the given text.
        end : int
            The index of the end of the matched substring, which is `len(text)` for non-partial decoding.

        Raises
        ------
        DecodeError
            If the decoding fails.
        """
        raise NotImplementedError

    def encode(self, value):
        """Encode a value into a string.

        Parameters
        ----------
        value : any
            The value to encode.

        Returns
        --------
        text : str
            The encoded text of the given value.

        Raises
        ------
        EncodeError
            If the encoding fails.
        """
        raise NotImplementedError

def eof(text, index, optional=False):
    """decode 'end of file'.

    Parameters
    ----------
    text : str
        The text to decode.
    index : int
        The index of text where the decoding starts with.
    optional : bool, optional
        True for optionally decoding.

    Returns
    -------
    value : bool
        True for eof.
    end : int
        The index of the end of the matched substring.

    Raises
    ------
    DecodeError
        If the decoding fails.
    """
    if index == len(text):
        return True, index
    else:
        if optional:
            return False, index
        raise DecodeError(text, index, ["\000"])

def startswith(prefixes, text, index, optional=False, partial=True):
    """match a list of strings.

    Parameters
    ----------
    prefixes : list of str
        The list of strings to match.
    text : str
        The text to decode.
    index : int
        The index of text where the decoding starts with.
    optional : bool, optional
        True for optionally decoding.
    partial : bool, optional
        True for partially decoding.

    Returns
    -------
    value : str
        The matched string.
    end : int
        The index of the end of the matched substring.

    Raises
    ------
    DecodeError
        If the decoding fails.
    """
    if not prefixes:
        if optional:
            return "", index
        raise DecodeError(text, index, [])
    regex = re.compile("|".join(re.escape(prefix) for prefix in sorted(prefixes, reverse=True)))
    m = regex.match(text, index)
    if not m:
        if optional:
            return "", index
        raise DecodeError(text, index, [prefix + ("" if partial else "\000") for prefix in prefixes])
    if not partial and m.end() != len(text):
        raise DecodeError(text, index, [prefix + ("" if partial else "\000") for prefix in prefixes])
    return m.group(), m.end()

def match(regex, expected, text, index, optional=False, partial=True):
    """match a regular expression.

    Parameters
    ----------
    regex : regular expression object
        The regular expression to match.
    expected : list of str
        The expected strings to match.
    text : str
        The text to decode.
    index : int
        The index of text where the decoding starts with.
    optional : bool, optional
        True for optionally decoding.
    partial : bool, optional
        True for partially decoding.

    Returns
    -------
    value : str
        The match object.
    end : int
        The index of the end of the matched substring.

    Raises
    ------
    DecodeError
        If the decoding fails.
    """
    m = re.compile(regex).match(text, index)
    if not m:
        if optional:
            return m, index
        raise DecodeError(text, index, [ex + ("" if partial else "\000") for ex in expected])
    if not partial and m.end() != len(text):
        raise DecodeError(text, index, [ex + ("" if partial else "\000") for ex in expected])
    return m, m.end()


class LiteralBiparser(Biparser):
    """Biparser for Python literal.

    Attributes
    ----------
    regex : regular expression object
        The regular expression to parse literal.
    expected : list of str
        The expected strings to match.
    type : type
        The type of literal.
    """
    def encode(self, value):
        if not isinstance(value, self.type):
            raise EncodeError(value, "", self.type)
        return repr(value)

    def decode(self, text, index=0, partial=False):
        res, index = match(self.regex, self.expected, text, index, partial=partial)
        return ast.literal_eval(res.group()), index

class NoneBiparser(LiteralBiparser):
    regex = "None"
    expected = ["None"]
    type = type(None)

class BoolBiparser(LiteralBiparser):
    regex = "False|True"
    expected = ["False", "True"]
    type = bool

class IntBiparser(LiteralBiparser):
    regex = r"[-+]?(0|[1-9][0-9]*)(?![0-9\.\+eEjJ])"
    expected = ["0"]
    type = int

class FloatBiparser(LiteralBiparser):
    regex = r"[-+]?([0-9]+\.[0-9]+(e[-+]?[0-9]+)?|[0-9]+[eE][-+]?[0-9]+)(?![0-9\+jJ])"
    expected = ["0.0"]
    type = float

class ComplexBiparser(LiteralBiparser):
    regex = r"[-+]?({0}[-+])?{0}[jJ]".format(r"(0|[1-9][0-9]*|[0-9]+\.[0-9]+(e[-+]?[0-9]+)?|[0-9]+e[-+]?[0-9]+)")
    expected = ["0j"]
    type = complex

    def encode(self, value):
        if not isinstance(value, self.type):
            raise EncodeError(value, "", self.type)
        repr_value = repr(value)
        if repr_value.startswith("(") and repr_value.endswith(")"):
            repr_value = repr_value[1:-1]
        return repr_value

class StrBiparser(LiteralBiparser):
    regex = (r'"('
             r'[^\r\n\\"\x00]'
             r'|\\[0-7]{1,3}'
             r'|\\x[0-9a-fA-F]{2}'
             r'|\\u[0-9a-fA-F]{4}'
             r'|\\U[0-9a-fA-F]{8}'
             r'|\\(?![xuUN\x00]).'
             r')*"')
    expected = ['""']
    type = str

    def encode(self, value):
        # make sure it uses double quotation
        return '"' + repr(value + '"')[1:-2].replace('"', r'\"').replace(r"\'", "'") + '"'

class BytesBiparser(LiteralBiparser):
    regex = (r'b"('
             r'(?![\r\n\\"])[\x01-\x7f]'
             r'|\\[0-7]{1,3}'
             r'|\\x[0-9a-fA-F]{2}'
             r'|\\u[0-9a-fA-F]{4}'
             r'|\\U[0-9a-fA-F]{8}'
             r'|\\(?![xuUN])[\x01-\x7f]'
             r')*"')
    expected = ['b""']
    type = bytes

    def encode(self, value):
        # make sure it uses double quotation
        return 'b"' + repr(value + b'"')[2:-2].replace(b'"', rb'\"').replace(rb"\'", b"'") + '"'

class SStrBiparser(LiteralBiparser):
    regex = (r"'("
             r"[^\r\n\\']"
             r"|\\[0-7]{1,3}"
             r"|\\x[0-9a-fA-F]{2}"
             r"|\\u[0-9a-fA-F]{4}"
             r"|\\U[0-9a-fA-F]{8}"
             r"|\\(?![xuUN])."
             r")*'")
    expected = ["''"]
    type = str

    def encode(self, value):
        # make sure it uses single quotation
        return repr(value + '"')[:-2] + "'"


class ListBiparser(Biparser):
    start = r"\[\s*"
    delimiter = r"\s*,\s*"
    end = r"\s*\]"

    def __init__(self, elem_biparser, multiline=False):
        self.elem_biparser = elem_biparser
        self.multiline = multiline

    def decode(self, text, index=0, partial=False):
        res = []

        _, index = match(self.start, ["["], text, index, partial=True)

        while True:
            m, index = match(self.end, ["]"], text, index, optional=True, partial=partial)
            if m: return res, index

            value, index = self.elem_biparser.decode(text, index, partial=True)
            res.append(value)

            m, index = match(f"({self.delimiter})?{self.end}", ["]"], text, index, optional=True, partial=partial)
            if m: return res, index

            _, index = match(self.delimiter, [","], text, index, partial=True)

    def encode(self, value):
        if not isinstance(value, list):
            raise EncodeError(value, "", list)

        elems_strs = []

        for i, elem in enumerate(value):
            try:
                elem_str = self.elem_biparser.encode(elem)
            except EncodeError as e:
                raise EncodeError(value, f"[{i}]{e.pos}", e.expected)
            elems_strs.append(elem_str)

        if not self.multiline:
            return "[" + ", ".join(elems_strs) + "]"
        elif not elems_strs:
            return "[]"
        else:
            return "[\n    " + ",\n".join(elems_strs).replace("\n", "\n    ") + "\n]"

class SetBiparser(Biparser):
    empty = "set\(\)"
    start = r"\{\s*"
    delimiter = r"\s*,\s*"
    end = r"\s*\}"

    def __init__(self, elem_biparser, multiline=False):
        self.elem_biparser = elem_biparser
        self.multiline = multiline

    def decode(self, text, index=0, partial=False):
        res = set()

        m, index = match(self.empty, ["set()"], text, index, optional=True, partial=partial)
        if m: return res, index

        _, index = match(self.start, ["{"], text, index, partial=True)

        while True:
            value, index = self.elem_biparser.decode(text, index, partial=True)
            res.add(value)

            m, index = match(f"({self.delimiter})?{self.end}", ["}"], text, index, optional=True, partial=partial)
            if m: return res, index

            _, index = match(self.delimiter, [","], text, index, partial=True)

    def encode(self, value):
        if not isinstance(value, set):
            raise EncodeError(value, "", set)

        if not value:
            return "set()"

        elems_strs = []

        for i, elem in enumerate(value):
            try:
                elem_str = self.elem_biparser.encode(elem)
            except EncodeError as e:
                raise EncodeError(value, f"[{i}]{e.pos}", e.expected)
            elems_strs.append(elem_str)

        if not self.multiline:
            return "{" + ", ".join(elems_strs) + "}"
        elif not elems_strs:
            return "set()"
        else:
            return "{\n    " + ",\n".join(elems_strs).replace("\n", "\n    ") + "\n}"

class DictBiparser(Biparser):
    start = r"\{\s*"
    colon = r"\s*:\s*"
    delimiter = r"\s*,\s*"
    end = r"\s*\}"

    def __init__(self, key_biparser, value_biparser, multiline=False):
        self.key_biparser = key_biparser
        self.value_biparser = value_biparser
        self.multiline = multiline

    def decode(self, text, index=0, partial=False):
        res = dict()

        _, index = match(self.start, ["{"], text, index, partial=True)

        while True:
            m, index = match(self.end, ["}"], text, index, optional=True, partial=partial)
            if m: return res, index

            key, index = self.key_biparser.decode(text, index, partial=True)
            _, index = match(self.colon, [":"], text, index, partial=True)
            value, index = self.value_biparser.decode(text, index, partial=True)
            res[key] = value

            m, index = match(f"({self.delimiter})?{self.end}", ["}"], text, index, optional=True, partial=partial)
            if m: return res, index

            _, index = match(self.delimiter, [","], text, index, partial=True)

    def encode(self, value):
        if not isinstance(value, dict):
            raise EncodeError(value, "", dict)

        items_str = []

        for i, (key, value) in enumerate(value.items()):
            try:
                key_str = self.key_biparser.encode(key)
            except EncodeError as e:
                raise EncodeError(value, f".keys()[{i}]{e.pos}", e.expected)

            try:
                value_str = self.value_biparser.encode(value)
            except EncodeError as e:
                raise EncodeError(value, f"[{key_str}]{e.pos}", e.expected)

            items_str.append(key_str + ": " + value_str)

        if not self.multiline:
            return "{" + ", ".join(items_str) + "}"
        elif not items_str:
            return "{}"
        else:
            return "{\n    " + ",\n".join(items_str).replace("\n", "\n    ") + "\n}"

class TupleBiparser(Biparser):
    start = r"\(\s*"
    delimiter = r"\s*,\s*"
    end = r"\s*\)"

    def __init__(self, elems_biparsers, multiline=False):
        self.elems_biparsers = elems_biparsers
        self.multiline = multiline

    def decode(self, text, index=0, partial=False):
        res = []

        _, index = match(self.start, ["("], text, index, partial=True)

        length = len(self.elems_biparsers)
        if length > 0:
            for n, elem_biparser in enumerate(self.elems_biparsers):
                value, index = elem_biparser.decode(text, index, partial=True)
                res.append(value)

                _, index = match(self.delimiter, [","], text, index, optional=(n == length-1 > 0), partial=True)

        _, index = match(self.end, [")"], text, index, partial=partial)
        return tuple(res), index

    def encode(self, value):
        if not isinstance(value, tuple):
            raise EncodeError(value, "", tuple)

        elems_str = []

        for i, (elem, biparser) in enumerate(zip(value, self.elems_biparsers)):
            try:
                elem_str = biparser.encode(elem)
            except EncodeError as e:
                raise EncodeError(value, f"[{i}]{e.pos}", e.expected)
            elems_str.append(elem_str)

        length = len(self.elems_biparsers)
        if length == 0:
            return "()"
        elif length == 1:
            return "(" + elems_str[0] + ",)"
        elif not self.multiline:
            return "(" + ", ".join(elems_str) + ")"
        else:
            return "(\n    " + ",\n".join(elems_str).replace("\n", "\n    ") + "\n)"

class DataclassBiparser(Biparser):
    start = r"\(\s*"
    keyequal = r"\s*{}\s*=\s*"
    delimiter = r"\s*,\s*"
    end = r"\s*\)"

    def __init__(self, clz, fields_biparsers, multiline=False):
        self.clz = clz
        self.fields_biparsers = fields_biparsers
        self.multiline = multiline

    def decode(self, text, index=0, partial=False):
        res = dict()

        _, index = startswith([self.clz.__name__ + "("], text, index, partial=True)

        length = len(self.fields_biparsers)
        if length > 0:
            for i, (name, biparser) in enumerate(self.fields_biparsers.items()):
                _, index = match(self.keyequal.format(re.escape(name)), [name + "="], text, index, partial=True)
                value, index = biparser.decode(text, index, partial=True)
                res[name] = value

                _, index = match(self.delimiter, [","], text, index, optional=(i==length-1), partial=True)

        _, index = match(self.end, [")"], text, index, partial=partial)
        return self.clz(**res), index

    def encode(self, value):
        if not isinstance(value, self.clz):
            raise EncodeError(value, "", self.clz)

        fields_str = []

        for i, (name, biparser) in enumerate(self.fields_biparsers.items()):
            try:
                value_str = biparser.encode(getattr(value, name))
            except EncodeError as e:
                raise EncodeError(value, f"[{name}]{e.pos}", e.expected)

            fields_str.append(name + "=" + value_str)

        if not self.multiline:
            return self.clz.__name__ + "(" + ", ".join(fields_str) + ")"
        elif not fields_str:
            return self.clz.__name__ + "()"
        else:
            return self.clz.__name__ + "(\n    " + ",\n".join(fields_str).replace("\n", "\n    ") + "\n)"


class UnionBiparser(Biparser):
    def __init__(self, options_biparsers):
        self.options_biparsers = options_biparsers

    def decode(self, text, index=0, partial=False):
        expected = []
        final_index = index
        for option_biparser in self.options_biparsers:
            try:
                return option_biparser.decode(text, index, partial=partial)
            except DecodeError as e:
                if e.index > final_index:
                    expected = list(e.expected)
                elif e.index == final_index:
                    expected.extend(e.expected)

        raise DecodeError(text, final_index, expected)

    def encode(self, value):
        for biparser in self.options_biparsers:
            try:
                return biparser.encode(value)
            except EncodeError:
                pass

        raise EncodeError(value, "", [])

class EnumBiparser(Biparser):
    nameperiod = r"{}\."

    def __init__(self, enum_class):
        self.enum_class = enum_class
        self.options = sorted(list(enum_class), key=lambda e:e.name, reverse=True)

    def decode(self, text, index=0, partial=False):
        name = self.enum_class.__name__
        _, index = match(self.nameperiod.format(re.escape(name)), [name + "."], text, index, partial=True)

        option, index = startswith([option.name for option in self.options], text, index, partial=partial)
        option = getattr(self.enum_class, option)
        return option, index

    def encode(self, value):
        if not isinstance(value, self.enum_class):
            raise EncodeError(value, "", self.enum_class)
        return self.enum_class.__name__ + "." + value.name


def from_type_hint(type_hint, multiline=False):
    """Make Biparser from type hint.

    Parameters
    ----------
    type_hint : type or type hint
        the type to parse.
    multiline : bool, optional

    Returns
    -------
    biparser : Biparser
        the biparser of the given type.
    """
    if type_hint is None:
        type_hint = type(None)

    if type_hint == type(None):
        return NoneBiparser()

    elif type_hint == bool:
        return BoolBiparser()

    elif type_hint == int:
        return IntBiparser()

    elif type_hint == float:
        return FloatBiparser()

    elif type_hint == complex:
        return ComplexBiparser()

    elif type_hint == str:
        return StrBiparser()

    elif type_hint == bytes:
        return BytesBiparser()

    elif isinstance(type_hint, type) and issubclass(type_hint, enum.Enum):
        return EnumBiparser(type_hint)

    elif isinstance(type_hint, type) and dataclasses.is_dataclass(type_hint):
        fields = {field.name : from_type_hint(field.type, multiline)
                  for field in type_hint.__dataclass_fields__.values()}
        return DataclassBiparser(type_hint, fields, multiline)

    elif getattr(type_hint, '__origin__', None) == typing.List:
        elem = from_type_hint(type_hint.__args__[0], multiline)
        return ListBiparser(elem, multiline)

    elif getattr(type_hint, '__origin__', None) == typing.Set:
        elem = from_type_hint(type_hint.__args__[0], multiline)
        return SetBiparser(elem, multiline)

    elif getattr(type_hint, '__origin__', None) == typing.Tuple:
        if len(type_hint.__args__) == 1 and type_hint.__args__[0] == ():
            elems = []
        else:
            elems = [from_type_hint(arg, multiline) for arg in type_hint.__args__]
        return TupleBiparser(elems, multiline)

    elif getattr(type_hint, '__origin__', None) == typing.Dict:
        key = from_type_hint(type_hint.__args__[0], False)
        value = from_type_hint(type_hint.__args__[1], multiline)
        return DictBiparser(key, value, multiline)

    elif getattr(type_hint, '__origin__', None) == typing.Union:
        options = [from_type_hint(arg, multiline) for arg in type_hint.__args__]
        return UnionBiparser(options)

    else:
        raise ValueError("No parser for type hint: " + repr(type_hint))
