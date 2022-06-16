from enum import Enum
import re
from typing import Optional, List
import dataclasses
from ..utils import commands as cmd


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


class CachedParser:
    def __init__(self, origin):
        self.origin = origin
        self.token = None
        self.result = None

    def parse(self, token):
        if self.token == token:
            return self.result
        type, next_parser = self.origin.parse(token)
        self.token = token
        self.result = type, CachedParser(next_parser)
        return self.result


class Editor:
    r"""Text editor with sematic parser.

    Attributes
    ----------
    buffer : list of str
        The buffer of current input.
    pos : int
        The caret position of input.

    parser : commands.RootCommandParser
        The root command parser.
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

    def __init__(self, parser, buffer):
        self.update_parser(parser)
        self.init(buffer)

    def init(self, buffer):
        self.buffer = buffer
        self.pos = len(self.buffer)
        self.tokens = []
        self.lex_state = SHLEXER_STATE.SPACED
        self.group = None
        self.result = None
        self.length = 0

    def update_parser(self, parser):
        self.parser = parser
        self._cached_parser = CachedParser(self.parser)

    # text operations

    def replace(self, selection, text):
        if not all(ch.isprintable() for ch in text):
            raise ValueError("invalid text to insert: " + repr("".join(text)))

        start, stop, _ = selection.indices(len(self.buffer))
        self.buffer[selection] = text
        selection = slice(start, start + len(text))
        if start <= self.pos < stop:
            self.pos = selection.stop
        elif stop <= self.pos:
            self.pos += selection.stop - stop
        return selection

    def insert(self, text):
        text = list(text)

        if len(text) == 0:
            return False

        while len(text) > 0 and text[0] == "\b":
            del text[0]
            del self.buffer[self.pos - 1]
            self.pos = self.pos - 1

        self.replace(slice(self.pos, self.pos), text)
        return True

    def backspace(self):
        if self.pos == 0:
            return False
        self.replace(slice(self.pos-1, self.pos), "")
        return True

    def delete(self):
        if self.pos >= len(self.buffer):
            return False
        self.replace(slice(self.pos, self.pos+1), "")
        return True

    def delete_all(self):
        if not self.buffer:
            return False
        self.replace(slice(None, None), "")
        return True

    def move_to(self, pos):
        pos = (
            min(max(0, pos), len(self.buffer)) if pos is not None else len(self.buffer)
        )

        if self.pos == pos:
            return False

        self.pos = pos
        return True

    def to_word_start(self):
        for match in re.finditer(r"\w+|\W+", "".join(self.buffer)):
            if match.end() >= self.pos:
                return slice(match.start(), self.pos)
        else:
            return slice(0, self.pos)

    def to_word_end(self):
        for match in re.finditer(r"\w+|\W+", "".join(self.buffer)):
            if match.end() > self.pos:
                return slice(self.pos, match.end())
        else:
            return slice(self.pos, len(self.buffer))

    # sematic operations

    def get_all_groups(self):
        return self.parser.get_all_groups()

    def parse(self):
        tokenizer = tokenize(self.buffer)

        tokens = []
        while True:
            try:
                token = next(tokenizer)
            except StopIteration as e:
                self.lex_state = e.value
                break

            tokens.append(token)

        # parse tokens
        cached_parser = self._cached_parser
        types = []
        result = None

        for token in tokens:
            try:
                type, cached_parser = cached_parser.parse(token.string)
            except cmd.CommandParseError as err:
                result = err
                break
            types.append(type)
        else:
            try:
                res = cached_parser.origin.finish()
            except cmd.CommandUnfinishError as err:
                result = err
            else:
                result = res

        self.result = result
        self.length = len(types)

        types.extend([None] * (len(tokens) - len(types)))
        self.tokens = [
            dataclasses.replace(token, type=type)
            for token, type in zip(tokens, types)
        ]
        self.group = self.parser.get_group(self.tokens[0].string) if self.tokens else None

    def desc(self, length):
        cached_parser = self._cached_parser
        for token in self.tokens[:length]:
            try:
                _, cached_parser = cached_parser.parse(token.string)
            except cmd.CommandParseError:
                return None
        return cached_parser.origin.desc()

    def info(self, length):
        assert length >= 1
        cached_parser = self._cached_parser
        for token in self.tokens[:length-1]:
            try:
                _, cached_parser = cached_parser.parse(token.string)
            except cmd.CommandParseError:
                return None
        return cached_parser.origin.info(self.tokens[length-1].string)

    def suggest(self, length, target):
        cached_parser = self._cached_parser
        for token in self.tokens[:length]:
            try:
                _, cached_parser = cached_parser.parse(token.string)
            except cmd.CommandParseError:
                return []
        return cached_parser.origin.suggest(target)

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

