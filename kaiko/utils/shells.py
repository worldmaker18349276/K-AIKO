from enum import Enum
import re
from typing import Optional, List
import dataclasses
from . import commands as cmd


class SHLEXER_STATE(Enum):
    SPACED = " "
    PLAIN = "*"
    BACKSLASHED = "\\"
    QUOTED = "'"


@dataclasses.dataclass(frozen=True)
class ShToken:
    r"""The token of shell-like grammar.

    Attributes
    ----------
    string : str
        The tokenized string.
    type : cmd.TOKEN_TYPE or None
        The type of token.
    mask : slice
        The position of this token.
    quotes : list of int
        The indices of all backslashes and quotation marks used for escaping.
        The token is equal to
        `''.join(raw[i] for i in range(*slice.indices(len(raw))) if i not in quotes)`.
    """
    string: str
    type: Optional[cmd.TOKEN_TYPE]
    mask: slice
    quotes: List[int]


def tokenize(raw):
    r"""Tokenizer for shell-like grammar.

    The delimiter is just whitespace, and the token is defined as::

        <nonspace-character> ::= /[^ \\\']/
        <backslashed-character> ::= "\" /./
        <quoted-string> ::= "'" /[^']*/ "'"
        <token> ::= ( <nonspace-character> | <backslashed-character> | <quoted-string> )*

    The backslashes and quotation marks used for escaping will be deleted after
    being interpreted as a string. The input string should be printable, so it
    doesn't contain tab, newline, backspace, etc. In this grammar, the token of
    an empty string can be expressed as `''`.

    Parameters
    ----------
    raw : str or list of str
        The string to tokenize, which should be printable.

    Yields
    ------
    token: ShToken
        the parsed token.

    Returns
    -------
    state : SHLEXER_STATE
        The final state of parsing.
    """
    SPACE = " "
    BACKSLASH = "\\"
    QUOTE = "'"

    length = len(raw)
    raw = enumerate(raw)

    while True:
        try:
            index, char = next(raw)
        except StopIteration:
            return SHLEXER_STATE.SPACED

        # guard space
        if char == SPACE:
            continue

        # parse token
        start = index
        token = []
        quotes = []
        while True:
            if char == SPACE:
                # end parsing token
                yield ShToken("".join(token), None, slice(start, index), quotes)
                break

            elif char == BACKSLASH:
                # escape the next character
                quotes.append(index)

                try:
                    index, char = next(raw)
                except StopIteration:
                    yield ShToken("".join(token), None, slice(start, length), quotes)
                    return SHLEXER_STATE.BACKSLASHED

                token.append(char)

            elif char == QUOTE:
                # escape the following characters until the next quotation mark
                quotes.append(index)

                while True:
                    try:
                        index, char = next(raw)
                    except StopIteration:
                        yield ShToken("".join(token), None, slice(start, length), quotes)
                        return SHLEXER_STATE.QUOTED

                    if char == QUOTE:
                        quotes.append(index)
                        break
                    else:
                        token.append(char)

            else:
                # otherwise, as it is
                token.append(char)

            try:
                index, char = next(raw)
            except StopIteration:
                yield ShToken("".join(token), None, slice(start, length), quotes)
                return SHLEXER_STATE.PLAIN


def quoting(compreply, state=SHLEXER_STATE.SPACED):
    r"""Escape a given string so that it can be inserted into an untokenized string.

    The strategy to escape insert string only depends on the state of insert
    position.

    Parameters
    ----------
    compreply : str
        The string to insert. The suffix `'\000'` indicate closing the token.
        But inserting `'\000'` after backslash results in `''`, since it is
        impossible to close it.
    state : SHLEXER_STATE
        The state of insert position.

    Returns
    -------
    raw : str
        The escaped string which can be inserted into untokenized string
        directly.
    """
    partial = not compreply.endswith("\000")
    if not partial:
        compreply = compreply[:-1]

    if state == SHLEXER_STATE.PLAIN:
        raw = re.sub(r"([ \\'])", r"\\\1", compreply)

    elif state == SHLEXER_STATE.BACKSLASHED:
        if compreply == "":
            # cannot close backslash without deleting it
            return ""
        raw = compreply[0] + re.sub(r"([ \\'])", r"\\\1", compreply[1:])

    elif state == SHLEXER_STATE.QUOTED:
        if partial:
            raw = compreply.replace("'", r"'\''")
        elif compreply == "":
            raw = "'"
        else:
            raw = compreply[:-1].replace("'", r"'\''") + (
                r"'\'" if compreply[-1] == "'" else compreply[-1] + "'"
            )

    elif state == SHLEXER_STATE.SPACED:
        if compreply != "" and " " not in compreply:
            # use backslash if there is no whitespace
            raw = re.sub(r"([ \\'])", r"\\\1", compreply)
        elif compreply == "":
            raw = "''"
        else:
            raw = (
                "'"
                + compreply[:-1].replace("'", r"'\''")
                + (r"'\'" if compreply[-1] == "'" else compreply[-1] + "'")
            )

    else:
        assert False

    return raw if partial else raw + " "


class SemanticAnalyzer:
    r"""Sematic analyzer for beatshell.

    Attributes
    ----------
    parser : commands.RootCommandParser
        The root command parser for beatshell.
    tokens : list of ShToken
        The parsed tokens.
    lex_state : SHLEXER_STATE
        The shlexer state.
    group : str or None
        The group name of parsed command.
    result : object or cmd.CommandParseError or cmd.CommandUnfinishError
        The command object or the error.
    length : int
        The parsed length of tokens.
    """

    def __init__(self, parser):
        self.parser = parser
        self.tokens = []
        self.lex_state = SHLEXER_STATE.SPACED
        self.group = None
        self.result = None
        self.length = 0

    def update_parser(self, parser):
        self.parser = parser

    def parse(self, buffer):
        tokenizer = tokenize(buffer)

        tokens = []
        while True:
            try:
                token = next(tokenizer)
            except StopIteration as e:
                self.lex_state = e.value
                break

            tokens.append(token)

        types, result = self.parser.parse_command(token.string for token in tokens)
        self.result = result
        self.length = len(types)

        types.extend([None] * (len(tokens) - len(types)))
        self.tokens = [
            dataclasses.replace(token, type=type)
            for token, type in zip(tokens, types)
        ]
        self.group = self.parser.get_group(self.tokens[0].string) if self.tokens else None

    def get_all_groups(self):
        return self.parser.get_all_groups()

    def desc(self, length):
        parents = [token.string for token in self.tokens[:length]]
        return self.parser.desc_command(parents)

    def info(self, length):
        parents = [token.string for token in self.tokens[:length-1]]
        target = self.tokens[length-1].string
        return self.parser.info_command(parents, target)

    def suggest(self, length, target):
        parents = [token.string for token in self.tokens[:length]]
        return self.parser.suggest_command(parents, target)

    def find_token(self, pos):
        for index, token in enumerate(self.tokens):
            if token.mask.start <= pos <= token.mask.stop:
                return index, token
        else:
            return None, None

    def find_token_before(self, pos):
        for index, token in enumerate(reversed(self.tokens)):
            if token.mask.start <= pos:
                return len(self.tokens)-1-index, token
        else:
            return None, None

    def find_token_after(self, pos):
        for index, token in enumerate(self.tokens):
            if pos <= token.mask.stop:
                return index, token
        else:
            return None, None


