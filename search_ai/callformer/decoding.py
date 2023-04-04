from abc import ABC
from dataclasses import dataclass, field, replace
from typing import Tuple, Sequence, List, Optional, TYPE_CHECKING
from typing import Type, Callable
from datetime import datetime

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical
import torch.nn.functional as F

if TYPE_CHECKING:
    from .transformer import CallFormer

from .commands import CommandsList, copy_commands_list
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
            print ("creating kv_cache")
            self.kv_cache, self.hooks = self.model.install_kv_cache_hooks()

        if tokens.shape[-1] > self.initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]

        res = self.model.decoder(tokens, embedding, kv_cache=self.kv_cache)
        return res.detach()

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
        self, 
        tokens: Tensor, 
        logits: Tensor, 
        sum_logprobs: Tensor,
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
        self, 
        tokens: Tensor, 
        batches: Optional[list[list[tuple[CommandsList, float]]]] = None,
    ) -> Sequence[Sequence[Tensor]]:
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
    

class BeamSearchDecoder(TokenDecoder):
    def __init__(
                  self,
                  beam_size: int,
                  eot: int,
                  inference: Inference,
                  patience: Optional[float] = None,
                  max_length: Optional[int] = None,
                  temperature: Optional[float] = None,
                ):
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates: int = round (beam_size * self.patience)
        self.finished_sequences: dict[tuple, float] = {}
        self.token_generator = CommandsList()
        self.max_length = max_length or 50

        assert (
            self.max_candidates > 0
        ), f"Invalid beam size ({beam_size}) or patience ({patience})"

    def reset(self):
        self.finished_sequences = {}

    def extend_sequence(
                        self, 
                        sequence: list[int], 
                        logit: Tensor, 
                        sum_logprob: Tensor) -> tuple[Tensor,
                                                      Tensor,
                                                      dict[tuple, float]]:
        
        assert sequence[-1] != self.eot, "Sequence already ended"

        cmd_list = CommandsList()
        for token in sequence:
            next_tokens = cmd_list.next_tokens()
            if len(next_tokens) == 0:
                break
            assert token in next_tokens, f"{token} not in {next_tokens}"
            cmd_list.add_token(token)
        
        next_tokens = cmd_list.next_tokens()

        mask = torch.empty_like(logit).fill_(-np.inf)
        mask[list(next_tokens)] = 0.0
        choice_logits = logit + mask
        choice_probs = F.softmax(choice_logits, dim=-1)

        new_sequences, finished = {}, {}

        logprob = F.log_softmax(logit, dim=-1)
        for p, t in zip(*choice_probs.topk(self.beam_size, dim=-1)):
            prob, token = p.item(), t.item()
            if prob > 0.:
                new_sequences[tuple(sequence + [token])] = (sum_logprob + logprob[token]).item()
                if token == self.eot:
                    finished[tuple(sequence + [token])] = new_sequences[tuple(sequence + [token])]

        sequences = sorted([(k,v) for k,v in new_sequences.items()], 
                           key=lambda x: x[1],
                           reverse=True)

        new_tokens = torch.tensor([list(s[0]) for s in sequences[:self.beam_size]
                                   if not finished.get(s[0], False)])
        new_sumlogprobs = torch.tensor([s[1] for s in sequences[:self.beam_size]
                                        if not finished.get(s[0], False)])
        return new_tokens, new_sumlogprobs, finished
    
    def make_beam(self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor):
        beam_sequences = {}
        assert tokens.shape[0] == self.beam_size

        assigned = {}
        for i in range(tokens.shape[0]):
            sequences, slogprobs, finished = self.extend_sequence(
                                        tokens[i].tolist(), 
                                        logits[i], 
                                        sum_logprobs[i])
            self.finished_sequences.update(finished)
            for j in range(sequences.shape[0]):
                if not assigned.get(tuple(sequences[j].tolist()), False) and \
                    not finished.get(tuple(sequences[j].tolist()), False):
                    beam_sequences[i] = (sequences[j], slogprobs[j])
                    assigned[tuple(sequences[j].tolist())] = True
                    break
            if not beam_sequences.get(i, False):
                beam_sequences[i] = (sequences[0], slogprobs[0])
                assigned[tuple(sequences[0].tolist())] = True

        new_tokens = torch.empty((tokens.shape[0], tokens.shape[1] + 1), dtype=torch.long)
        new_sumlogprobs = torch.empty_like(sum_logprobs)
        for i,t in beam_sequences.items():
            new_tokens[i] = t[0]
            new_sumlogprobs[i] = t[1]
        return new_tokens, new_sumlogprobs

    def update(
               self,
               tokens: Tensor,
               logits: Tensor,
               sum_logprobs: Tensor,
                ) -> Tuple[Tensor, Tensor, bool]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_batches = tokens.shape[0] // self.beam_size

        batch_sequences, batch_sum_logprobs = [], []

        new_tokens = torch.empty((tokens.shape[0], tokens.shape[1]+1), dtype=torch.long)
        new_sumlogprobs = torch.empty_like(sum_logprobs)

        for i in range(0, n_batches, self.beam_size):
            beam_tokens, beam_sum_logprobs = self.make_beam(tokens[:self.beam_size], 
                       logits[:self.beam_size], 
                       sum_logprobs[:self.beam_size])
            new_tokens[i:i+self.beam_size] = beam_tokens
            new_sumlogprobs[i:i+self.beam_size] = beam_sum_logprobs

        complete = False
        if len(list(self.finished_sequences.keys())) == self.max_candidates:
            complete = True
        
        return new_tokens, new_sumlogprobs, complete

    def finalize(self, tokens: Tensor, sum_logprobs: Tensor):
        sequence = sorted([(k,v) for k,v in self.finished_sequences.items()],
               key=lambda x: x[1],)[0]
        self.reset()
        return sequence


def decoder_options_factory(**kwargs) -> Callable:
    return lambda: dict(**kwargs)


def curr_date_factory() -> str:
    return datetime.now().strftime("%Y-%m-%d")


@dataclass(frozen=True)
class DecodingOptions:
    # This is the maximum number of tokens to be sampled
    sample_len: Optional[int] = None

    tokenizer: Tokenizer = field(default_factory=Tokenizer)

    length_penalty: Optional[float] = None

    fp16: bool = False

    initial_tokens_length: Optional[int] = None

    decoder: Type[TokenDecoder] = GreedyDecoder

    sot_sequence: tuple[int] = Tokenizer().sot_sequence

    decoder_options: dict = field(default_factory=decoder_options_factory(
                temperature=0.0, 
                eot=Tokenizer().eot,
                ))
    
    curr_date: str = field(default_factory=curr_date_factory)

    n_groups: int = 2


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

        self.initial_tokens = self._get_initial_tokens()
        self.initial_tokens_length = len(self.initial_tokens)

        self.inference = PyTorchInference(self.model, 
                                          self.initial_tokens_length )

        self.sequence_ranker = MaximumLikelihoodRanker(length_penalty=self.options.length_penalty)

        if self.options.decoder == BeamSearchDecoder:
            self.options.decoder_options['inference'] = self.inference
            self.options.decoder_options['beam_size'] = options.n_groups
        self.decoder = self.options.decoder(**self.options.decoder_options)
        self.n_groups = options.n_groups

    def _get_initial_tokens(self) -> tuple[int]:
        date_str = self.options.curr_date or datetime.now().strftime("%Y-%m-%d")
        cd_token = self.tokenizer.vocab_lookup['<|curr_date|>']
        tokens = (self.sot_sequence + (cd_token,) +
                    tuple(self.tokenizer.encode(f'("{date_str}")').tolist())
                  )
        return tokens

    def _main_loop(self,
                   embedding: Tensor,
                   tokens: Tensor,
                   ) -> tuple[Tensor, Tensor]:
        assert tokens.shape[0] == embedding.shape[0]

        n_batch = tokens.shape[0]
        sum_logprobs: Tensor = torch.zeros(n_batch, device=embedding.device)
        
        try:
            for i in range(self.sample_len):
                logits = self.inference.logits(tokens, embedding)
                logits = logits[:, -1]

                tokens, sum_logprobs, completed = self.decoder.update(tokens, logits, sum_logprobs)

                if completed:
                    break

        finally:
            self.inference.cleanup_caching()
        
        return tokens, sum_logprobs


    @torch.no_grad()
    def run(
            self, 
            embedding: Tensor,
            ) -> Tensor:
        self.decoder.reset()
        n_batches: int = embedding.shape[0]

        tokens: Tensor = torch.tensor([self.initial_tokens] * n_batches, device=embedding.device)
        tokens = tokens.repeat_interleave(self.n_groups, dim=0)
        embedding = embedding.repeat_interleave(self.n_groups, dim=0)
        
        tokens, sum_logprobs = self._main_loop(embedding, tokens)
        tokens = self.decoder.finalize(tokens, sum_logprobs)
        return tokens



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
    
    with torch.no_grad():
        result = DecodingTask(model, options).run(embedding)

    return result