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
    sequence: list[int] = []
    tokenizer: Tokenizer
    complete: bool = False
    const_length: Optional[int] = None
    optional: bool

    def __init__(self, tokenizer, optional=False) -> None:
        self.tokenizer = tokenizer
        self.optional = optional

    def add_token(self, token_index: int) -> None:
        assert not self.complete, "Attempting to add a token to a completed argument"
        assert token_index in self.next_tokens(), ("Atempting to add an invalid"
                                                   f" token: {token_index}")
        self.sequence.append(token_index)
        if self.const_length is not None:
            if len(self.sequence) == self.const_length:
                self.complete = True

    @abstractmethod
    def next_tokens(self) -> tuple[int]:
        ...


class ArgumentList(ABC):
    # All arguments listed in `args` are required
    args: tuple[Argument]
    tokenizer: Tokenizer
    complete: bool = False
    generating: Optional[int] = None

    def __init__(self, args) -> None:
        self.args = args
        self.tokenizer = self.args[0].tokenizer

    def add_token(self, token_index: int) -> None:
        assert self.generating is not None, "No argument is currently being generated"
        self.args[self.generating].add_token(token_index)

    def next_tokens(self) -> Union[tuple[int], tuple]:
        if self.generating is None:
            toks = self.args[0].next_tokens()
            self.generating = 0
        toks = self.args[self.generating].next_tokens()
        if len(toks) == 0:
            if self.generating == len(self.args) - 1:
                self.complete = True
                return ()
            self.generating += 1
            return self.tokenizer.vocab_lookup[","],
        return toks
    

class Number(Argument):

    def __init__(self, tokenizer, dtype: Union[int, float], *args, **kwargs) -> None:
        super().__init__(tokenizer, *args, **kwargs)
        self.dtype = dtype

    def next_tokens(self) -> tuple[int, ...]:
        tokens = [self.tokenizer.vocab_lookup[f"{i}"] for i in range(10)]
        if len(self.sequence) > 0 and self.dtype == float:
            tokens.append(self.tokenizer.vocab_lookup["."])
        return tuple(tokens)
        

class Date(Argument):
    optional = True
    const_length = 12

    def next_tokens(self) -> tuple[int, ...]:
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


class Command(ABC):
    token: str
    tokenizer: Tokenizer
    index: int
    sequence: list[int]
    complete: bool = False
    takes_arguments: bool = False
    #argument_types: ArgsTypes = ()
    args_list: Optional[ArgumentList] = None
    

    def __init__(self, tokenizer) -> None:
        self.index = tokenizer.vocab_lookup[self.token]
        self.sequence = [self.index]
        self.tokenizer = tokenizer

        #if self.takes_arguments:
        #    assert self.argument_types is not None


    @abstractmethod
    def filter_logits(self, logits: Tensor) -> Tensor:
        ...

    @abstractmethod
    def next_tokens(self) -> Union[tuple[int, ...], tuple[()]]:
        ...

    def add_token(self, token_index: int) -> None:
        self.sequence.append(token_index)


class StartCommand(Command):
    token = "<|start|>"
    takes_arguments = True
    tokenizer = Tokenizer()
    args_list = ArgumentList(Date(tokenizer))

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
    d = Date(Tokenizer())

    d.add_token(d.tokenizer.vocab_lookup['"'])
    print ([d.tokenizer.vocab[i] for i in d.next_tokens()])
    d.add_token(d.tokenizer.vocab_lookup['2'])
    print ([d.tokenizer.vocab[i] for i in d.next_tokens()])
    d.add_token(d.tokenizer.vocab_lookup['0'])
    print ([d.tokenizer.vocab[i] for i in d.next_tokens()])
    d.add_token(d.tokenizer.vocab_lookup['4'])
    print ([d.tokenizer.vocab[i] for i in d.next_tokens()])
    d.add_token(d.tokenizer.vocab_lookup['1'])
    print ([d.tokenizer.vocab[i] for i in d.next_tokens()])
    d.add_token(d.tokenizer.vocab_lookup['-'])
    print ([d.tokenizer.vocab[i] for i in d.next_tokens()])
    d.add_token(d.tokenizer.vocab_lookup['1'])
    print ([d.tokenizer.vocab[i] for i in d.next_tokens()])
    d.add_token(d.tokenizer.vocab_lookup['0'])
    print ([d.tokenizer.vocab[i] for i in d.next_tokens()])
    d.add_token(d.tokenizer.vocab_lookup['-'])
    print ([d.tokenizer.vocab[i] for i in d.next_tokens()])
    d.add_token(d.tokenizer.vocab_lookup['2'])
    print ([d.tokenizer.vocab[i] for i in d.next_tokens()])
    d.add_token(d.tokenizer.vocab_lookup['6'])
    print ([d.tokenizer.vocab[i] for i in d.next_tokens()])
    d.add_token(d.tokenizer.vocab_lookup['"'])
    print ([d.tokenizer.vocab[i] for i in d.next_tokens()])