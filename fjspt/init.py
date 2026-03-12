import torch
import torch.nn as nn

from tensordict.tensordict import TensorDict

from rl4co.models.nn.ops import PositionalEncoding


def env_init_embedding(env_name: str, config: dict) -> nn.Module:
    """Get environment initial embedding. The init embedding is used to initialize the
    general embedding of the problem nodes without any solution information.
    Consists of a linear layer that projects the node features to the embedding space.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "fjspt": FJSPTInitEmbedding,
    }

    if env_name not in embedding_registry:
        raise ValueError(
            f"Unknown environment name '{env_name}'. Available init embeddings: {embedding_registry.keys()}"
        )

    return embedding_registry[env_name](**config)


class JSSPInitEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim,
        linear_bias: bool = True,
        scaling_factor: int = 1000,
        num_op_feats=5,
    ):
        super(JSSPInitEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.scaling_factor = scaling_factor
        self.init_ops_embed = nn.Linear(num_op_feats, embed_dim, linear_bias)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=0.0)

    def _op_features(self, td):
        proc_times = td["proc_times"]
        mean_durations = proc_times.sum(1) / (proc_times.gt(0).sum(1) + 1e-9)
        feats = [
            mean_durations / self.scaling_factor,
            # td["lbs"] / self.scaling_factor,
            td["is_ready"],
            td["num_eligible"],
            td["ops_job_map"],
            td["op_scheduled"],
        ]
        return torch.stack(feats, dim=-1)

    def _init_ops_embed(self, td: TensorDict):
        ops_feat = self._op_features(td)
        ops_emb = self.init_ops_embed(ops_feat)
        ops_emb = self.pos_encoder(ops_emb, td["ops_sequence_order"])

        # zero out padded and finished ops
        mask = td["pad_mask"]  # NOTE dont mask scheduled - leads to instable training
        ops_emb[mask.unsqueeze(-1).expand_as(ops_emb)] = 0
        return ops_emb

    def forward(self, td):
        return self._init_ops_embed(td)


class FJSPTInitEmbedding(JSSPInitEmbedding):
    def __init__(self, embed_dim, linear_bias=False, scaling_factor: int = 100):
        super().__init__(embed_dim, linear_bias, scaling_factor)
        self.init_ma_embed = nn.Linear(1, self.embed_dim, bias=linear_bias)
        self.init_truck_embed = nn.Linear(1, self.embed_dim, bias=linear_bias)
        self.proc_edge_embed = nn.Linear(1, self.embed_dim, bias=linear_bias)
        self.truck_edge_embed = nn.Linear(1, self.embed_dim, bias=linear_bias)

    def forward(self, td: TensorDict):
        ops_emb = self._init_ops_embed(td)
        ma_emb = self._init_machine_embed(td)
        truck_emb = self._init_truck_embed(td)
        machine_edge_emb = self._init_proc_edge_embed(td)
        truck_edge_emb = self._init_truck_edge_embed(td)
        ma_ma_edge_emb = self._init_ma_ma_edge_embed(td)
        # get edges between operations and machines
        # (bs, ops, ma)
        ops_ma_edges = td["ops_ma_adj"].transpose(1, 2)
        return ops_emb, ma_emb, truck_emb, machine_edge_emb, truck_edge_emb, ma_ma_edge_emb, ops_ma_edges

    def _init_proc_edge_embed(self, td: TensorDict):
        proc_times = td["proc_times"].transpose(1, 2) / self.scaling_factor
        edge_embed = self.proc_edge_embed(proc_times.unsqueeze(-1))
        return edge_embed

    def _init_machine_embed(self, td: TensorDict):
        busy_for = (td["machine_busy_until"] - td["time"].unsqueeze(1)) / self.scaling_factor
        ma_embeddings = self.init_ma_embed(busy_for.unsqueeze(2))
        return ma_embeddings

    def _init_ma_ma_edge_embed(self, td):
        trucks_times = td["trucks_times"] / self.scaling_factor
        ma_ma_edge_emb = self.truck_edge_embed(trucks_times.unsqueeze(-1))
        # trucks_times   (bs, n_mas + 1, n_mas + 1)
        # unsqueeze      (bs, n_mas + 1, n_mas + 1, 1)
        # ma_ma_edge_emb  (bs, n_mas + 1, n_mas + 1, emb_dim)
        return ma_ma_edge_emb

    def _init_truck_edge_embed(self, td):
        truck_loc = td["trucks_location"]  # (bs, n_trucks)
        trucks_times = td["trucks_times"]  # (bs, n_mas+1, n_mas+1)
        truck_loc_exp = truck_loc.unsqueeze(-1).expand(-1, -1, trucks_times.size(-1))
        truck_to_all = torch.gather(
            trucks_times,
            1,
            truck_loc_exp,
        )  # (bs, n_trucks, n_mas+1)
        truck_to_machine = truck_to_all[:, :, 1:]  # (bs, n_trucks, n_mas) без LU
        truck_edge_emb = self.truck_edge_embed(
            (truck_to_machine / self.scaling_factor).unsqueeze(-1)
        )  # (bs, n_trucks, n_mas, emb_dim)
        return truck_edge_emb

    def _init_truck_embed(self, td):
        busy_for = (td["truck_busy_until"] - td["time"].unsqueeze(1)) / self.scaling_factor
        truck_emb = self.init_truck_embed(busy_for.unsqueeze(-1))
        # busy_for   (bs, n_trucks)
        # unsqueeze  (bs, n_trucks, 1)
        # truck_emb  (bs, n_trucks, emb_dim)
        return truck_emb
