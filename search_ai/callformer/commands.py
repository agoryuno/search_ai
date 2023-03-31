from abc import ABC, abstractmethod
from typing import Tuple, Union, Type, Optional
import calendar

import numpy as np
import torch
from torch import Tensor

from tokenizer import Tokenizer


ArgTypes = Union[Type["Date"], Type["Number"]]
ArgsTypes = Tuple[ArgTypes, ...]


class Argument(ABC):
    start_token: Tuple[str, ...]
    end_token: Tuple[str, ...]
    sequence: list[int] = []
    tokenizer: Tokenizer

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def add_token(self, token_index: int) -> None:
        self.sequence.append(token_index)

    @abstractmethod
    def next_tokens(self) -> list[int]:
        ...


class Date(Argument):
    start_token = ('"',)
    end_token = ('"',)

    def next_tokens(self) -> Tuple[int, ...]:
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
                    self.tokenizer.vocab_lookup[f"{i}"] for i in range(10)
                    ])
            if self.sequence[6] == self.tokenizer.vocab_lookup["1"]:
                return tuple([
                    self.tokenizer.vocab_lookup[f"{i}"] for i in range(1,3)
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
                    if 
                    return tuple([
                        self.tokenizer.vocab_lookup[f"{i}"] for i in range(1,10)
                        ])
                if self.sequence[9] == self.tokenizer.vocab_lookup["3"]:
                    return tuple([
                        self.tokenizer.vocab_lookup[f"{i}"] for i in range(1,2)
                        ])


class Command(ABC):
    token: str
    complete: bool = False
    takes_arguments: bool = False
    argument_types: ArgsTypes = ()

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.index = tokenizer.vocab_lookup[self.token]
        self.sequence = [self.index]
        self.tokenizer = tokenizer

        if self.takes_arguments:
            assert self.argument_types is not None


    @abstractmethod
    def filter_logits(self, logits: Tensor) -> Tensor:
        ...

    def add_token(self, token_index: int) -> None:
        self.sequence.append(token_index)


class StartCommand(Command):
    token = "<|start|>"
    takes_arguments = True
    argument_types = (str, str, int)

    def filter_logits(self, logits: Tensor) -> Tensor:
        """
        Parameters
        ----------
        logits : Tensor
            The logits for  a single batch, 
            shape (n_ctx, vocab_size)

        Returns
            A mask tensor of shape (n_ctx, vocab_size)
            with valid token positions filled with 1 and
            invalid positions filled with -inf
        """
        # ensure the command is not complete yet
        assert not self.complete

        mask = torch.empty_like(logits).fill_(-np.inf)
        if self.takes_arguments:
            if len(self.sequence) == 1:
                # the start of a command with arguments
                # the only valid next token is '('
                mask[:, self.tokenizer.vocab_lookup["("]] = 1
                return mask
            if len(self.sequence) == 2:
                # make sure that the last token was '('
                assert self.sequence[-1] == self.tokenizer.vocab_lookup["("]
                # the next token must be the start of a valid argument
                # if there's an `int` or a `float` together with a `str` 
                # in the argument types then the next token 
                # can be either a digit or a double quote
                if int in self.argument_types or float in self.argument_types:
                    pass




if __name__ == "__main__":
    from tokenizer import Tokenizer
    #s = StartCommand(Tokenizer())
    #s2 = StartCommand(Tokenizer())
    
    import calendar
    d = '"2010-01-01"'
    print (calendar.monthrange(int(d[1:5]), int(d[6:8]))[1])