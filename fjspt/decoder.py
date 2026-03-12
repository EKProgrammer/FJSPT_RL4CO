import abc

from typing import Any, Tuple

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.models.common.constructive.autoregressive import AutoregressiveDecoder
from rl4co.models.nn.graph.hgnn import HetGNNEncoder
from rl4co.models.nn.mlp import MLP
from rl4co.utils.ops import batchify, gather_by_index


class L2DActor(nn.Module, metaclass=abc.ABCMeta):
    """Base decoder model for actor in L2D. The actor is responsible for generating the logits for the action
    similar to the decoder in autoregressive models. Since the decoder in L2D can have the additional purpose
    of extracting features (i.e. encoding the environment in ever iteration), we need an additional actor class.
    This function serves as template for such actor classes in L2D
    """

    @abc.abstractmethod
    def forward(
        self, td: TensorDict, hidden: Any = None, num_starts: int = 0
    ) -> Tuple[Tensor, Tensor]:
        """Obtain logits for current action to the next ones

        Args:
            td: TensorDict containing the input data
            hidden: Hidden state from the encoder. Can be any type
            num_starts: Number of starts for multistart decoding

        Returns:
            Tuple containing the logits and the action mask
        """
        raise NotImplementedError("Implement me in subclass!")

    def pre_decoder_hook(
        self, td: TensorDict, env=None, hidden: Any = None, num_starts: int = 0
    ) -> Tuple[TensorDict, Any]:
        """By default, we only require the input for the actor to be a tuple
        (in JSSP we only have operation embeddings but in FJSP we have operation
        and machine embeddings. By expecting a tuple we can generalize things.)

        Args:
            td: TensorDict containing the input data
            hidden: Hidden state from the encoder
            num_starts: Number of starts for multistart decoding

        Returns:
            Tuple containing the updated hidden state(s) and the input TensorDict
        """
        # Для FJSP hidden = (op_embeddings, machine_embeddings)

        hidden = (hidden,) if not isinstance(hidden, tuple) else hidden
        # Для FJSP эта строка ничего не делает

        if num_starts > 1:
            # NOTE: when using pomo, we need this
            # multistart: из одного состояния запустить несколько решений
            hidden = tuple(map(lambda x: batchify(x, num_starts), hidden))
            # пример
            # было:
            # op_embeddings.shape = (32, 30, 128)
            # если num_starts = 8, то
            # станет: (32*8, 30, 128)
        return td, env, hidden


class FJSPTActor(L2DActor):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        hidden_layers: int = 2,
        check_nan: bool = True,
    ) -> None:
        super().__init__()

        self.mlp = MLP(
            # MLP принимает job + machine + truck embeddings, поэтому:
            input_dim=3 * embed_dim,
            output_dim=1,
            num_neurons=[hidden_dim] * hidden_layers,
            hidden_act="ReLU",
            out_act="Identity",
            input_norm="None",
            output_norm="None",
        )
        self.dummy = nn.Parameter(torch.rand(3 * embed_dim))
        # Dummy параметр: Создается обучаемый вектор
        # shape = (3 * embed_dim)
        # Он используется для действия "ничего не делать" (noop).
        # Позже он станет (bs, 1, 3*emb)
        self.check_nan = check_nan
        # Если True, будет проверка: нет ли NaN в logits

    def forward(self, td, ops_emb, ma_emb, tr_emb):
        # ops_emb	(bs, n_ops, emb)
        # ma_emb	(bs, n_ma, emb)
        # tr_emb    (bs, n_tr, emb)

        bs = ops_emb.size(0)
        n_ma = ma_emb.size(1)
        n_tr = tr_emb.size(1)

        # td["next_op"].shape = (bs, n_jobs)
        # td["next_op"] содержит индекс следующей операции для каждого job
        # ops_emb.shape = (bs, n_ops, emb)
        # gather берет embedding нужной операции для каждого job.
        job_emb = gather_by_index(ops_emb, td["next_op"], squeeze=False)
        # job_emb.shape = (bs, n_jobs, emb)
        n_jobs = job_emb.size(1)

        job_emb = job_emb.unsqueeze(2).unsqueeze(3)
        job_emb = job_emb.expand(-1, -1, n_ma, n_tr, -1)
        # job_emb.shape = (bs, n_jobs, n_ma, n_tr, emb)

        ma_emb = ma_emb.unsqueeze(1).unsqueeze(3)
        ma_emb = ma_emb.expand(-1, n_jobs, -1, n_tr, -1)
        # ma_emb.shape = (bs, n_jobs, n_ma, n_tr, emb)

        tr_emb = tr_emb.unsqueeze(1).unsqueeze(2)
        tr_emb = tr_emb.expand(-1, n_jobs, n_ma, -1, -1)
        # tr_emb.shape = (bs, n_jobs, n_ma, n_tr, emb)

        h_actions = torch.cat((job_emb, ma_emb, tr_emb), dim=-1)
        # h_actions.shape = (bs, n_jobs, n_ma, n_tr, 3*emb)
        h_actions = h_actions.flatten(1, 3)
        # h_actions.shape = (bs, n_jobs * n_ma * n_tr, 3*emb)

        no_ops = self.dummy[None, None].expand(bs, 1, -1)
        # no_ops.shape = (bs, 1, 3*emb)
        h_actions_w_noop = torch.cat((no_ops, h_actions), 1)
        # h_actions_w_noop.shape = (bs, n_jobs * n_ma * n_tr + 1, 3*emb)

        logits = self.mlp(h_actions_w_noop).squeeze(-1)
        # logits.shape = (bs, n_jobs * n_ma * n_tr + 1)

        if self.check_nan:
            assert not torch.isnan(logits).any(), "Logits contain NaNs"
        # (bs, n_jobs * n_ma * n_tr + 1)
        mask = td["action_mask"]
        return logits, mask


class L2DDecoder(AutoregressiveDecoder):
    # feature extractor + actor
    def __init__(
        self,
        feature_extractor: nn.Module = None,
        actor: nn.Module = None,
        init_embedding: nn.Module = None,
        embed_dim: int = 128,
        actor_hidden_dim: int = 128,
        actor_hidden_layers: int = 2,
        num_encoder_layers: int = 3,
        normalization: str = "batch",
        stepwise: bool = False,
        scaling_factor: int = 1000,
    ):
        super(L2DDecoder, self).__init__()

        self.env_name = "fjspt"

        if feature_extractor is None and stepwise:
            feature_extractor = HetGNNEncoder(
                env_name=self.env_name,
                embed_dim=embed_dim,
                num_layers=num_encoder_layers,
                normalization=normalization,
                init_embedding=init_embedding,
                scaling_factor=scaling_factor,
            )

        self.feature_extractor = feature_extractor

        if actor is None:
            actor = FJSPTActor(
                embed_dim=embed_dim,
                hidden_dim=actor_hidden_dim,
                hidden_layers=actor_hidden_layers,
            )

        self.actor = actor

    def forward(self, td, hidden, num_starts):
        if hidden is None:
            # NOTE in case we have multiple starts, td is batchified
            # (through decoding strategy pre decoding hook). Thus the
            # embeddings from feature_extractor have the correct shape
            num_starts = 0
            hidden, _ = self.feature_extractor(td)
            # hidden = (ops_emb, ma_emb)
            # ops_emb.shape = (bs, n_jobs * n_ops, emb_dim), ma_emb.shape = (bs, n_ma, emb_dim)

        td, _, hidden = self.actor.pre_decoder_hook(td, None, hidden, num_starts)

        logits, mask = self.actor(td, *hidden)

        return logits, mask
