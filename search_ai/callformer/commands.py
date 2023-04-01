from abc import ABC, abstractmethod
from typing import Tuple, Union, Type, Optional, Generator
from typing import Literal
import calendar

import numpy as np
import torch
from torch import Tensor

from .tokenizer import Tokenizer


ArgTypes = Union[Type["Date"], Type["Number"]]
ArgsTypes = Tuple[ArgTypes, ...]


def start_wrapper(f):
    def wrapper(self: Union["Argument", "ArgumentList"]) -> Union[tuple[int], tuple[()]]:
        start, toks = self.start_sequence()
        if start:
            return toks
        return f(self)
    return wrapper


class ObjectBase(ABC):
    sequence: list[int]
    tokenizer: Tokenizer = Tokenizer()
    start: str

    def __init__(self):
        self.sequence = []
        self.complete = False

    def start_sequence(self) -> Union[tuple[Literal[True], tuple[int,...]],
                                      tuple[Literal[False], tuple[()]]]:
        if len(self.sequence) == 0:
            return True, (self.tokenizer.vocab_lookup[self.start],)
        return False, ()
    
    @abstractmethod
    @start_wrapper
    def next_tokens(self) -> Union[tuple[int], tuple[()]]:
        ...

    def valid_tokens(self) -> Generator:
        while not self.complete:
            toks = self.next_tokens() 
            yield toks
        

class Argument(ObjectBase):
    sequence: list[int]
    complete: bool
    const_length: Optional[int] = None
    terminator: str = '"'
    start: str = '"'

    def add_token(self, token_index: int) -> None:
        assert not self.complete, "Attempting to add a token to a completed sequence"
        assert token_index in self.next_tokens(), ("Atempting to add an invalid"
                                                   f" token: {token_index} ('{self.tokenizer.vocab[token_index]}')")
        self.sequence.append(token_index)
        if len(self.sequence) > 1 and \
            self.tokenizer.vocab[token_index] == self.terminator:
            self.complete = True


class ArgumentList(ObjectBase):
    # All arguments listed in `args` are required
    sequence: list[int]
    args: tuple[Argument]
    complete: bool = False
    generating: Optional[int] = None
    terminator: str = ')'
    start: str = '('

    def __init__(self, args) -> None:
        self.args = args
        super().__init__()

    def start_sequence(self) -> Union[tuple[Literal[True], tuple[int,...]],
                                      tuple[Literal[False], tuple[()]]]:
        if len(self.sequence) == 0:
            self.generating = 0
            self.sequence.append(self.tokenizer.vocab_lookup[self.start])
        return False, ()


    def add_token(self, token_index: int) -> None:
        assert not self.complete, "Attempting to add a token to a completed sequence"
        assert self.generating is not None, "No argument is currently being generated"
        if self.generating < len(self.args):
            self.args[self.generating].add_token(token_index)
        self.sequence.append(token_index)
        if len(self.sequence) > 1 and \
            self.tokenizer.vocab[token_index] == self.terminator:
            self.complete = True

    @start_wrapper
    def next_tokens(self) -> Union[tuple[int], tuple]:
        assert self.generating is not None
        toks = self.args[self.generating].next_tokens()
        if len(toks) == 0:
            self.generating += 1
            if self.generating == len(self.args):
                
                return (self.tokenizer.vocab_lookup[self.terminator],)
            return self.tokenizer.vocab_lookup[","],
        return toks
    

class Command(ABC):
    token: str
    tokenizer: Tokenizer = Tokenizer()
    index: int
    sequence: list[int]
    complete: bool
    takes_arguments: bool = False
    args_list: ArgumentList

    def __init__(self) -> None:
        self.complete = False
        self.index = self.tokenizer.vocab_lookup[self.token]
        self.sequence = [self.index]

    @abstractmethod
    def next_tokens(self) -> Union[tuple[int, ...], tuple[()]]:
        ...

    def add_token(self, token_index: int) -> None:
        self.args_list.add_token(token_index)
        self.sequence.append(token_index)


class Number(Argument):

    def next_tokens(self):
        
        if len(self.sequence) == 0:
            return (self.tokenizer.vocab_lookup['"'],)
        toks = [
            self.tokenizer.vocab_lookup[f"{i}"] for i in range(10)
            ]
        if len(self.sequence) > 1:
            toks += ['"', '.']
        elif len(self.sequence) == 1:
            toks += ['-']
        return tuple(toks)
        

class Date(Argument):

    def next_tokens(self):

        if len(self.sequence) == 0:
            return (self.tokenizer.vocab_lookup['"'],)
        if len(self.sequence) == 1:
            return (self.tokenizer.vocab_lookup["2"],)
        if len(self.sequence) >= 2 and len(self.sequence) <= 4:
            return tuple([
                self.tokenizer.vocab_lookup[f"{i}"] for i in range(10)
                ])
        if len(self.sequence) == 5:
            return (self.tokenizer.vocab_lookup["-"],)
        if len(self.sequence) == 6:
            return (self.tokenizer.vocab_lookup["0"],
                    self.tokenizer.vocab_lookup["1"],)
        if len(self.sequence) == 7:
            if self.sequence[6] == self.tokenizer.vocab_lookup["0"]:
                return tuple([
                    self.tokenizer.vocab_lookup[f"{i}"] for i in range(1,10)
                    ])
            if self.sequence[6] == self.tokenizer.vocab_lookup["1"]:
                return tuple([
                    self.tokenizer.vocab_lookup[f"{i}"] for i in range(3)
                    ])
        if len(self.sequence) == 8:
            return (self.tokenizer.vocab_lookup["-"],)
        if len(self.sequence) in (9,10):
            year = int("".join([
                self.tokenizer.vocab[i] for i in self.sequence[1:5]
                ]))
            month = int("".join([
                self.tokenizer.vocab[i] for i in self.sequence[6:8]
                ]))
            num_days = calendar.monthrange(year, month)[1]
            if len(self.sequence) == 9:
                toks = (self.tokenizer.vocab_lookup["0"],
                        self.tokenizer.vocab_lookup["1"],
                        self.tokenizer.vocab_lookup["2"],)
                if num_days >= 29:
                    toks += (self.tokenizer.vocab_lookup["3"],)
                return toks
            if len(self.sequence) == 10:
                if self.sequence[9] == self.tokenizer.vocab_lookup["0"]:
                    return tuple([
                        self.tokenizer.vocab_lookup[f"{i}"] for i in range(1,10)
                        ])
                if self.sequence[9] == self.tokenizer.vocab_lookup["1"]:
                    return tuple([
                        self.tokenizer.vocab_lookup[f"{i}"] for i in range(10)
                        ])
                if self.sequence[9] == self.tokenizer.vocab_lookup["2"]:
                    if num_days == 28:
                        return tuple([
                            self.tokenizer.vocab_lookup[f"{i}"] for i in range(9)
                            ])
                    return tuple([
                        self.tokenizer.vocab_lookup[f"{i}"] for i in range(10)
                        ])
                if self.sequence[9] == self.tokenizer.vocab_lookup["3"]:
                    return tuple([
                        self.tokenizer.vocab_lookup[f"{i}"] for i in range(2)
                        ])
        if len(self.sequence) == 11:
            return (self.tokenizer.vocab_lookup['"'],)
        return ()


class SearchNotesCommand1(Command):
    """
    This is the most basic variant of the "searchnotes" command.
    It takes a single argument: a date to start the search at.
    """
    token: str = "<|searchnotes1|>"
    takes_arguments: bool = True

    def __init__(self) -> None:
        super().__init__()
        self.args_list = ArgumentList((Date(),))
        

    def next_tokens(self):
        assert not self.complete

        if len(self.sequence) == 0:
            return (self.index,)
        if self.takes_arguments and self.args_list is not None:
            if len(self.sequence) == 1:
                return (self.tokenizer.vocab_lookup["("],)
            if len(self.sequence) > 1:
                toks = self.args_list.next_tokens()
                if len(toks) == 0:
                    return (self.tokenizer.vocab_lookup[")"],)
        
    def valid_tokens(self) -> Generator:
        assert self.takes_arguments, ("This command does not take arguments"
                                      " and has no tokens to generate")
        for toks in self.args_list.valid_tokens():
            yield toks
