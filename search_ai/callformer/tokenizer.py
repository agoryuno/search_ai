from typing import Optional

import torch
import numpy as np


_vocab = [
    "<|pad|>",
    "<|start|>",
    "<|end|>",
    "<|searchnotes|>",
    "<|searchemails|>",
    '<|searchcontacts|>',
    '<|searchwhatsapp|>',
    '<|searchtelegram|>',
    '<|summarize|>',
    '(',
    ')',
    " ",
    "-",
    '"',
    "'",
]
_vocab += [str(i) for i in range(10)]
API_VOCAB = {i: token for i, token in enumerate(_vocab)}


class Tokenizer:
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

    def encode(self, text: str) -> torch.Tensor:
        tokens = []
        curr_seq = ""
        for char in text.split():
            if curr_seq in self.vocab:
                tokens.append(self.vocab[curr_seq])
                curr_seq = ""
                continue
            if len(curr_seq) > self.max_token_len:
                raise ValueError(f"Unknown token encountered: {curr_seq}")
            curr_seq += char
        assert curr_seq == "", f"Unknown token encountered: {curr_seq}"
        return torch.LongTensor(np.array(tokens))
    
    def decode(self, tokens: torch.Tensor) -> list[str]:
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        
        texts = []
        for i in range(tokens.shape[0]):
            text: str = ""
            for token in tokens[i]:
                text += self.vocab[token]
            texts.append(text)

        return texts