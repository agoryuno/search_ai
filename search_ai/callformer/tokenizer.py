import torch
import numpy as np


_vocab = [
    "<|pad|>",
    "<|start|>",
    "<|end|>",
    "<|searchnotes1|>",
    "<|searchnotes2|>",
    "<|searchemails|>",
    '<|searchcontacts|>',
    '<searchmessengers|>',
    '<|summarize|>',
    '<|limit|>',
    '<|curr_date|>',
    '(',
    ')',
    " ",
    "-",
    '"',
    "'",
    ",",
]
_vocab += [str(i) for i in range(10)]
API_VOCAB = {i: token for i, token in enumerate(_vocab)}


class Tokenizer:
    vocab: dict[int, str]
    vocab_lookup: dict[str, int]
    vocab_size: int
    max_token_len: int
    sot_sequence: tuple[int]
    eot: int
    sot: int
    pad: int

    def __init__(
                 self, 
                 vocabulary: dict[int, str] = API_VOCAB,
                 ):
        
        self.vocab_lookup = {v: k for k, v in vocabulary.items()}
        self.vocab = vocabulary
        self.vocab_size: int = len(list(self.vocab.keys()))
        self.max_token_len: int = max([len(token) for token in self.vocab_lookup])
        self.sot_sequence = (self.vocab_lookup["<|start|>"],)
        self.eot = self.vocab_lookup["<|end|>"]
        self.sot = self.vocab_lookup["<|start|>"]
        self.pad = self.vocab_lookup["<|pad|>"]

    def encode(self, text: str) -> torch.Tensor:
        tokens = []
        curr_seq = ""
        for char in text:
            curr_seq += char
            if curr_seq in self.vocab_lookup:
                tokens.append(self.vocab_lookup[curr_seq])
                curr_seq = ""
                continue
            if len(curr_seq) > self.max_token_len:
                raise ValueError(f"Unknown token encountered: {curr_seq}")
            
            
        assert curr_seq == "", f"Unknown token encountered: {curr_seq}"
        return torch.tensor(np.array(tokens), dtype=torch.long)
    
    def decode(self, tokens: torch.Tensor) -> list[str]:
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0) # type: ignore
        
        texts = []
        for i in range(tokens.shape[0]):
            text: str = ""
            for token in tokens[i]:
                text += self.vocab[token.item()] # type: ignore
            texts.append(text)

        return texts