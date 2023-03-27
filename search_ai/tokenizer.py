from typing import Optional

import torch
import numpy as np


API_VOCAB = [
    "<|pad|>",
    "<|start|>",
    "<|end|>",
    "<|searchnotes|>",
    "<|searchemails|>",
    '<|searchcontacts|>',
    '<|searchwhatsapp|>',
    '<|searchtelegram|>',
    '(',
    ')',
    " ",
    "-",
    '"',
    "'",
]
API_VOCAB += [str(i) for i in range(10)]
API_VOCAB = {token: i for i, token in enumerate(API_VOCAB)}


class Tokenizer:
    def __init__(self, vocabulary: Optional[dict] = API_VOCAB):
        self.vocab = vocabulary
        self.vocab_size = len(vocabulary)
        self.max_token_len = max([len(token) for token in vocabulary])

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
    
    def decode(self, tokens: torch.Tensor) -> str:
        return " ".join([self.vocab[t] for t in tokens])