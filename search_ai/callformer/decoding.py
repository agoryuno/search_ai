from abc import ABC
from dataclasses import dataclass, field, replace
from typing import Tuple, Sequence, List, Optional, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical
import torch.nn.functional as F

if TYPE_CHECKING:
    from .transformer import CallFormer

from .tokenizer import Tokenizer


class Inference(ABC):
    def logits(self, tokens: Tensor, embedding: Tensor) -> Tensor:
        """Perform a forward pass on the decoder and return per-token logits"""
        raise NotImplementedError

    def rearrange_kv_cache(self, source_indices) -> None:
        """Update the key-value cache according to the updated beams"""
        raise NotImplementedError

    def cleanup_caching(self) -> None:
        """Clean up any resources or hooks after decoding is finished"""
        pass


class PyTorchInference(Inference):
    def __init__(self, model: "CallFormer", initial_token_length: int):
        self.model: "CallFormer" = model
        self.initial_token_length = initial_token_length
        self.kv_cache = {}
        self.hooks = []

    def logits(self, tokens: Tensor, embedding: Tensor) -> Tensor:
        if not self.kv_cache:
            self.kv_cache, self.hooks = self.model.install_kv_cache_hooks()

        if tokens.shape[-1] > self.initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]

        return self.model.decoder(tokens, embedding, kv_cache=self.kv_cache)

    def cleanup_caching(self):
        for hook in self.hooks:
            hook.remove()

        self.kv_cache = {}
        self.hooks = []

    def rearrange_kv_cache(self, source_indices):
        for module, tensor in self.kv_cache.items():
            # update the key/value cache to contain the selected sequences
            self.kv_cache[module] = tensor[source_indices].detach()


class SequenceRanker(ABC):
    def rank(
        self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]
    ) -> List[int]:
        """
        Given a list of groups of samples and their cumulative log probabilities,
        return the indices of the samples in each group to select as the final result
        """
        raise NotImplementedError


class MaximumLikelihoodRanker(SequenceRanker):
    """
    Select the sample with the highest log probabilities, penalized using either
    a simple length normalization or Google NMT paper's length penalty
    """

    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty

    def rank(self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    # from the Google NMT paper
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        # get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]


class TokenDecoder:
    """ From https://github.com/openai/whisper/blob/main/whisper/decoding.py """

    def reset(self):
        """Initialize any stateful variables for decoding a new sequence"""

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        """Specify how to select the next token, based on the current trace and logits

        Parameters
        ----------
        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        sum_logprobs : Tensor, shape = (n_batch)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Tensor, shape = (n_batch, current_sequence_length + 1)
            the tokens, appended with the selected next token

        completed : bool
            True if all sequences has reached the end of text

        """
        raise NotImplementedError

    def finalize(
        self, tokens: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Sequence[Sequence[Tensor]], List[List[float]]]:
        """Finalize search and return the final candidate sequences

        Parameters
        ----------
        tokens : Tensor, shape = (n_audio, n_group, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence

        sum_logprobs : Tensor, shape = (n_audio, n_group)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Sequence[Sequence[Tensor]], length = n_audio
            sequence of Tensors containing candidate token sequences, for each audio input

        sum_logprobs : List[List[float]], length = n_audio
            sequence of cumulative log probabilities corresponding to the above

        """
        raise NotImplementedError


class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if self.temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            next_tokens = Categorical(logits=logits / self.temperature).sample()

        logprobs = F.log_softmax(logits.float(), dim=-1)
        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)

        completed = (tokens[:, -1] == self.eot).all()
        return tokens, completed

    def finalize(self, tokens: Tensor, sum_logprobs: Tensor):
        # make sure each sequence has at least one EOT token at the end
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        return tokens, sum_logprobs.tolist()
    

@dataclass(frozen=True)
class DecodingOptions:
    temperature: float = 0.0

    # This is the maximum number of tokens to be sampled
    sample_len: Optional[int] = None

    tokenizer: Tokenizer = field(default_factory=Tokenizer)

    length_penalty: Optional[float] = None

    fp16: bool = False


@dataclass(frozen=True)
class DecodingResult:
    tokens: List[int] = field(default_factory=list)
    avg_logprob: float = np.nan
    temperature: float = np.nan


class DecodingTask:
    inference: Inference
    sequence_ranker: SequenceRanker
    decoder: TokenDecoder

    def __init__(self, model: "CallFormer", options: DecodingOptions):
        self.model = model

        self.options: DecodingOptions = options

        self.tokenizer: Tokenizer = options.tokenizer 

        self.n_ctx: int = model.dims.n_ctx
        self.sample_len: int = options.sample_len or self.n_ctx // 2

        self.sot_sequence: tuple[int] = self.tokenizer.sot_sequence

        self.initial_tokens: tuple[int] = self.sot_sequence

        self.inference = PyTorchInference(self.model, 
                                          len(self.initial_tokens) )

        self.sequence_ranker = MaximumLikelihoodRanker(length_penalty=self.options.length_penalty)

        self.decoder = GreedyDecoder(temperature=self.options.temperature, eot=self.tokenizer.eot)

    def _main_loop(self,
                   embedding: Tensor,
                   tokens: Tensor,
                   ) -> Tuple[Tensor, Tensor]:
        assert tokens.shape[0] == embedding.shape[0]

        n_batch = tokens.shape[0]
        sum_logprobs: Tensor = torch.zeros(n_batch, device=embedding.device)

        try:
            for i in range(self.sample_len):
                logits = self.inference.logits(tokens, embedding)

                logits = logits[:, -1]

                tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)

                if completed or tokens.shape[-1] > self.n_ctx:
                    break

        finally:
            self.inference.cleanup_caching()

        return tokens, sum_logprobs


    @torch.no_grad()
    def run(
            self, 
            embedding: Tensor,
            initial_tokens: Optional[tuple] = None) -> Tuple[Tensor, Tensor]:
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_batches: int = embedding.shape[0]

        initial_tokens = initial_tokens if initial_tokens is not None else self.initial_tokens

        tokens: Tensor = torch.tensor([initial_tokens] * n_batches, device=embedding.device)
        tokens, sum_logprobs = self._main_loop(embedding, tokens)
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        return tokens, sum_logprobs



@torch.no_grad()
def decode_function(
    model: "CallFormer",
    embedding: Tensor,
    options: DecodingOptions = DecodingOptions(),
    **kwargs,
):

    assert embedding.ndim in (2, 3)
    embedding = embedding.to(model.device)

    if embedding.ndim == 2:
        embedding = embedding.unsqueeze(0)

    if kwargs:
        options = replace(options, **kwargs)

    result = DecodingTask(model, options).run(embedding)

    return result