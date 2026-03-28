from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl4co.models.common.constructive.autoregressive import (
    AutoregressiveDecoder,
    AutoregressiveEncoder,
    AutoregressivePolicy,
)
from rl4co.models.nn.mlp import MLP
from rl4co.models.common.constructive.base import NoEncoder

from rl4co.utils.pylogger import get_pylogger
from rl4co.utils.decoding import DecodingStrategy, process_logits
from rl4co.utils.ops import gather_by_index

from hgnn import HetGNNEncoder
from decoder import L2DDecoder

log = get_pylogger(__name__)


class L2DPolicy(AutoregressivePolicy):
    def __init__(
        self,
        encoder: Optional[AutoregressiveEncoder] = None,
        decoder: Optional[AutoregressiveDecoder] = None,
        embed_dim: int = 64,
        num_encoder_layers: int = 2,
        env_name: str = "fjspt",
        scaling_factor: int = 1000,
        normalization: str = "batch",
        init_embedding: Optional[nn.Module] = None,
        stepwise_encoding: bool = False,
        tanh_clipping: float = 10,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "multistart_sampling",
        **constructive_policy_kw,
    ):
        if len(constructive_policy_kw) > 0:
            log.warning(f"Unused kwargs: {constructive_policy_kw}")

        if encoder is None:
            encoder = HetGNNEncoder(
                env_name=env_name,
                embed_dim=embed_dim,
                num_layers=num_encoder_layers,
                normalization="batch",
                init_embedding=init_embedding,
                scaling_factor=scaling_factor,
            )

        # The decoder generates logits given the current td and heatmap
        if decoder is None:
            decoder = L2DDecoder(
                env_name=env_name,
                embed_dim=embed_dim,
                actor_hidden_dim=embed_dim,
                num_encoder_layers=num_encoder_layers,
                init_embedding=init_embedding,
                stepwise=stepwise_encoding,
                scaling_factor=scaling_factor,
                normalization=normalization,
            )

        # Pass to constructive policy
        super(L2DPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            tanh_clipping=tanh_clipping,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **constructive_policy_kw,
        )


class L2DPolicy4PPO(L2DPolicy):
    def __init__(
        self,
        encoder=None,
        decoder=None,
        critic=None,
        embed_dim: int = 64,
        num_encoder_layers: int = 2,
        env_name: str = "fjspt",
        scaling_factor: int = 1000,
        init_embedding=None,
        tanh_clipping: float = 10,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "multistart_sampling",
        **constructive_policy_kw,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            env_name=env_name,
            scaling_factor=scaling_factor,
            init_embedding=init_embedding,
            stepwise_encoding=True,
            tanh_clipping=tanh_clipping,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **constructive_policy_kw,
        )

        if critic is None:
            critic = MLP(3 * embed_dim, 1, num_neurons=[embed_dim] * 2)

        self.critic = critic

    def evaluate(self, td):
        # Encoder: get encoder output and initial embeddings from initial state
        hidden, _ = self.decoder.feature_extractor(td)
        # pool the embeddings for the critic
        h_tuple = (hidden,) if isinstance(hidden, torch.Tensor) else hidden
        pooled = tuple(map(lambda x: x.mean(dim=-2), h_tuple))
        # potentially cat multiple embeddings (pooled ops and machines)
        h_pooled = torch.cat(pooled, dim=-1)
        # pred value via the value head
        value_pred = self.critic(h_pooled)
        # pre decoder / actor hook
        td, _, hidden = self.decoder.actor.pre_decoder_hook(
            td, None, hidden, num_starts=0
        )
        logits, mask = self.decoder.actor(td, *hidden)
        # get logprobs and entropy over logp distribution
        logprobs = process_logits(logits, mask, tanh_clipping=self.tanh_clipping)
        action_logprobs = gather_by_index(logprobs, td["action"], dim=1)
        dist_entropys = Categorical(logprobs.exp()).entropy()

        return action_logprobs, value_pred, dist_entropys

    def act(self, td, phase: str = "train"):
        logits, mask = self.decoder(td, hidden=None, num_starts=0)  # !!!!!!!!!!!!
        logprobs = process_logits(logits, mask, tanh_clipping=self.tanh_clipping)

        # DRL-S, sampling actions following \pi
        if phase == "train":
            action_indexes = DecodingStrategy.sampling(logprobs)
            td["logprobs"] = gather_by_index(logprobs, action_indexes, dim=1)

        # DRL-G, greedily picking actions with the maximum probability
        else:
            action_indexes = DecodingStrategy.greedy(logprobs)

        # memories.states.append(copy.deepcopy(state))
        td["action"] = action_indexes

        return td

    @torch.no_grad()
    def generate(self, td, env=None, phase: str = "train", **kwargs) -> dict:
        assert phase != "train", "dont use generate() in training mode"
        with torch.no_grad():
            out = super().__call__(td, env, phase=phase, **kwargs)
        return out
