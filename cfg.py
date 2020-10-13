import re
import typing
import dataclasses


sp = r"( [ \t\f]   |  \\ (\r\n | \r(?!\n) | \n | $) )"
nl = r"( (\# [^\r\n$]*)? (\r\n | \r(?!\n) | \n | $) )"
# sp = r"[ ]"
# nl = r"[\n$]"

def make_basic_parser_full():
    none_parser = "(None)"
    bool_parser = "(False | True)"

    _int_parser = r"""
        ( 0      ( _? 0           )*
        | [1-9]  ( _? [0-9]       )*
        | 0 [Bb] ( _? [01]        )+
        | 0 [Oo] ( _? [0-7]       )+
        | 0 [Xx] ( _? [0-9A-Fa-f] )+
        )
        """
    int_parser = fr"""
        ( [-+]? {_int_parser} )
        """
    dec_parser = r"( [-+]? ( 0 (_? 0)* | [1-9] (_? [0-9])* ) )"
    bin_parser = r"( [-+]? 0 [Bb] (_? [01])+ )"
    oct_parser = r"( [-+]? 0 [Oo] (_? [0-7])+ )"
    hex_parser = r"( [-+]? 0 [Xx] (_? [0-9A-Fa-f])+ )"

    _digit = r"[0-9] (_? [0-9])*"
    _float_parser = fr"""
        ( {_digit} \. {_digit}
        |          \. {_digit}
        | {_digit} \.
        | {_digit} (?=[Ee])
        )
        ( [Ee] [-+]? {_digit} )?
        """
    float_parser = fr"""
        ( [-+]? {_float_parser} )
        """

    _complex_parser = fr"""
        ( {_digit} | {_float_parser} ) [Jj]
        """
    complex_parser = fr"""
        ( [-+]? {_complex_parser}
        | \( ( {dec_parser} | {float_parser} ) [-+] {_complex_parser} \)
        )
        """

    str_parser = r"""
        ( [UuRr]? ( '    ( (?! [\\\r\n] | '    ) . | \\. | \\\r\n )* '
                  | "    ( (?! [\\\r\n] | "    ) . | \\. | \\\r\n )* "
                  | \''' ( (?!  \\      | \''' ) . | \\.          )* \'''
                  | \""" ( (?!  \\      | \""" ) . | \\.          )* \"""
                  )
        )
        """
    bytes_parser = str_parser.replace(".", r"[\x00-\x7f]").replace("[UuRr]?", "( [Rr]?[Bb] | [Bb][Rr] )")

    msp = fr" ( {sp}* ( {nl} {sp}* )* ) "
    list_parser = fr"""
        ( \[ {msp} \]
        | \[ {msp}
            ( {{0}} {msp} , {msp} )*
              {{0}} {msp}
            (, {msp})?
          \]
        )
        """
    set_parser = fr"""
        ( set {sp}* \( {msp} \)
        | \x7b # Left Curly Bracket
            {msp}
            ( {{0}} {msp} , {msp} )*
              {{0}} {msp}
            (, {msp})?
          \x7d # Right Curly Bracket
        )
        """
    dict_parser = fr"""
        ( \x7b {msp} \x7d
        | \x7b # Left Curly Bracket
            {msp}
            ( {{0}} {msp} : {msp} {{1}} {msp} , {msp} )*
              {{0}} {msp} : {msp} {{1}} {msp}
            (, {msp})?
          \x7d # Right Curly Bracket
        )
        """

    class TupleParser:
        def format(self, *args):
            length = len(args)

            if length == 0:
                return fr"( \( {msp} \) )"
            elif length == 1:
                return fr"( \( {msp} {args[0]} {msp} , {msp} \) )"
            else:
                list = fr" {msp} , {msp} ".join(args)
                return fr"( \( {msp} {list} {msp} (, {msp})? \) )"

    class UnionParser:
        def format(self, *args):
            return "( " + " | ".join(args) + " )"

    def dataclass_parser(name, **kwargs):
        if len(kwargs) == 0:
            return fr"( {name} {sp}* \( {msp} \) )"

        else:
            args = [fr"{key} {msp} = {msp} {value}" for key, value in kwargs.items()]
            list = fr" {msp} , {msp} ".join(args)
            return fr"( {name} {sp}* \( {msp} {list} {msp} ( , {msp} )? \) )"

    atomic_types = dict(NoneType=none_parser, bool=bool_parser,
                        int=int_parser, float=float_parser, complex=complex_parser,
                        str=str_parser, bytes=bytes_parser)
    composite_types = dict(List=list_parser, Set=set_parser, Dict=dict_parser,
                           Tuple=TupleParser(), Union=UnionParser())

    return atomic_types, composite_types, dataclass_parser

def make_basic_parser_simple():
    # only parse the form of repr()
    # ban `\t, \r, \f, \v, \\\n`, only use SPACE and RETURN
    # no comment

    none_parser = "(None)"
    bool_parser = "(False | True)"
    int_parser = r"( [-+]? ( 0 | [1-9][0-9]* ) )"
    float_parser = r"( [-+]? ( [0-9]+\.[0-9]+ ( e[-+]?[0-9]+ )? | [0-9]+e[-+]?[0-9]+ ) )"
    _number_parser = r"( 0 | [1-9][0-9]* | [0-9]+\.[0-9]+ ( e[-+]?[0-9]+ )? | [0-9]+e[-+]?[0-9]+ )"
    complex_parser = fr"( [-+]? {_number_parser} j | \( [-+]? {_number_parser} [-+] {_number_parser} j \) )"

    str_parser = r"""( ' ( [^\\\n'] | \\. )* ' | " ( [^\\\n"] | \\. )* " )"""
    bytes_parser = r"""( b' ( (?!\\\n') [\x00-\x7f] | \\[\x00-\x7f] )* ' | b" ( (?!\\\n") [\x00-\x7f] | \\[\x00-\x7f] )* " )"""
    list_parser = r"( \[\] | \[ ({0} , [ ])* {0} \] )"
    set_parser = r"( set\(\) | \x7b ({0} , [ ])* {0} \x7d )"
    dict_parser = r"( \x7b\x7d | \x7b ({0} : [ ] {1} , [ ])* {0} : [ ] {1} \x7d )"

    class TupleParser:
        def format(self, *args):
            length = len(args)

            if length == 0:
                return fr"( \(\) )"
            elif length == 1:
                return fr"( \( {args[0]} , \) )"
            else:
                list = fr" , [ ] ".join(args)
                return fr"( \( {list} \) )"

    class UnionParser:
        def format(self, *args):
            return "( " + " | ".join(args) + " )"

    def dataclass_parser(name, **kwargs):
        if len(kwargs) == 0:
            return fr"( {name}\(\) )"

        else:
            args = [fr"{key} = {value}" for key, value in kwargs.items()]
            list = fr" , [ ]".join(args)
            return fr"( {name}\( {list} \) )"

    atomic_types = dict(NoneType=none_parser, bool=bool_parser,
                        int=int_parser, float=float_parser,
                        str=str_parser, bytes=bytes_parser)
    composite_types = dict(List=list_parser, Set=set_parser, Dict=dict_parser,
                           Tuple=TupleParser(), Union=UnionParser())

    return atomic_types, composite_types, dataclass_parser

atomic_types, composite_types, dataclass_parser = make_basic_parser_full()


def parser(clz):
    if clz is None:
        clz = type(None)

    if not isinstance(clz, type):
        raise ValueError(repr(clz) + " is not a type")

    name = clz.__name__

    if name in atomic_types:
        return atomic_types[name]

    elif name in composite_types:
        args = [parser(arg) for arg in clz.__args__]
        return composite_types[name].format(*args)

    elif dataclasses.is_dataclass(clz):
        fields = {field.name : parser(field.type) for field in dataclasses.fields(clz)}
        return dataclass_parser(name, **fields)

    else:
        raise ValueError("Unable to unrepr type " + clz.__name__)

def unrepr(clz, repr_str, strict=True):
    if strict and not re.fullmatch(parser(clz), repr_str, re.X|re.A|re.M|re.S):
        raise SyntaxError
    return eval(repr_str)


def configurable(arg=None, excludes=[]):
    def configurable_decorator(clz):
        fields = typing.get_type_hints(clz)
        for field in excludes:
            del fields[field]
        clz.__configurable_fields__ = fields
        return clz

    if arg is None:
        return configurable_decorator
    else:
        return configurable_decorator(arg)

def get_configurable_fields(clz):
    field_hints = {}
    for field_name, field_type in clz.__configurable_fields__.items():
        if hasattr(field_type, "__configurable_fields__"):
            subfield_hints = get_configurable_fields(field_type)
            for subfield_names, subfield_type in subfield_hints.items():
                field_hints[(field_name, *subfield_names)] = subfield_type
        else:
            field_hints[(field_name,)] = field_type
    return field_hints

def config_read(file, strict=True, **targets):
    if isinstance(file, str):
        file = open(file, "r")
    config_str = file.read()

    if strict:
        field_parsers = []
        for target_name, target in targets.items():
            field_hints = get_configurable_fields(target)
            for names, hint in field_hints.items():
                field_ref = fr" {sp}* \. {sp}* ".join(map(re.escape, [target_name, *names]))
                value_parser = parser(hint)
                field_parsers.append(fr"{field_ref} {sp}* = {sp}* {value_parser}")

        fields_parser = "( " + " | ".join(field_parsers) + " )"
        regex = fr"({sp}* {nl})* ( {fields_parser} ({sp}* {nl})+ )*"

        if not re.fullmatch(regex, config_str, re.X|re.A|re.M|re.S):
            raise SyntaxError

    exec(config_str, globals(), **targets)

def config_write(file, **targets):
    if isinstance(file, str):
        file = open(file, "w")

    for target_name, target in targets.items():
        field_hints = get_configurable_fields(target)

        for names, hint in field_hints.items():
            value = target
            for name in names:
                value = getattr(value, name)

            value_repr = hint.__repr__(value)
            field_ref = ".".join([target_name, *names])
            print(field_ref + " = " + value_repr, file=file)

