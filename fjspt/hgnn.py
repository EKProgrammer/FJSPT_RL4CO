import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import einsum
from torch import Tensor

from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.ops import TransformerFFN


class HetGNNLayer(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        # Используется для вычисления attention к самому узлу.
        self.self_attn = nn.Parameter(torch.rand(size=(embed_dim, 1), dtype=torch.float))
        # Используется для оценки узлов другого типа.
        self.cross_attn = nn.Parameter(torch.rand(size=(embed_dim, 1), dtype=torch.float))
        # Используется для edge features.
        self.edge_attn = nn.Parameter(torch.rand(size=(embed_dim, 1), dtype=torch.float))
        self.activation = nn.ReLU()
        self.scale = 1 / math.sqrt(embed_dim)

    def forward(
        self, self_emb: Tensor, other_emb: Tensor, edge_emb: Tensor, edges: Tensor
    ):
        # self_emb   (bs, n_rows, emb_dim)          узлы типа A
        # other_emb  (bs, n_cols, emb_dim)          узлы типа B
        # edge_emb   (bs, n_rows, n_cols, emb_dim)  признаки ребер
        # edges      (bs, n_rows, n_cols)           adjacency mask

        # объект  смысл
        # rows    machines
        # cols    operations

        bs, n_rows, _ = self_emb.shape

        # concat operation embeddings and o-m edge features (proc times)
        # Calculate attention coefficients
        er = einsum(self_emb, self.self_attn, "b m e, e one -> b m") * self.scale
        # er[b,m] (bs, n_rows) = self_emb[b,m,e] * self_attn[e]
        ec = einsum(other_emb, self.cross_attn, "b o e, e one -> b o") * self.scale
        # ec (bs, n_cols)
        ee = einsum(edge_emb, self.edge_attn, "b m o e, e one -> b m o") * self.scale
        # ee (bs, n_rows, n_cols)

        # компонент	 что означает
        # er         важность machine
        # ec         важность operation
        # ee	     важность edge

        # element wise multiplication similar to broadcast column logits over rows with masking
        ec_expanded = einsum(edges, ec, "b m o, b o -> b m o")
        # ec_expanded[i,j,k] = edges[i,j,k] * ec[i,k]
        # ec_expanded.shape = (bs, n_rows, n_cols)
        # element wise multiplication similar to broadcast row logits over cols with masking
        er_expanded = einsum(edges, er, "b m o, b m -> b m o")
        # er_expanded[i,j,k] = edges[i,j,k] * er[i,k]
        # er_expanded.shape = (bs, n_rows, n_cols)

        # adding the projections of different node types and edges together (equivalent to first concat and then project)
        # (bs, n_rows, n_cols)
        cross_logits = self.activation(ec_expanded + ee + er_expanded)
        # Cross attention logits
        # attention(i,j) =
        # ReLU(
        #  machine_score(i)
        #  + operation_score(j)
        #  + edge_score(i,j)
        # )

        # (bs, n_rows, 1)
        self_logits = self.activation(er + er).unsqueeze(-1)
        # Self attention logits
        # self_logits.shape = (bs, n_rows, 1)

        # (bs, n_ma, n_ops + 1)
        mask = torch.cat(
            (
                edges == 1,
                torch.full(
                    size=(bs, n_rows, 1),
                    dtype=torch.bool,
                    fill_value=True,
                    device=edges.device,
                ),
            ),
            dim=-1,
        )

        # (bs, n_ma, n_ops + 1)
        all_logits = torch.cat((cross_logits, self_logits), dim=-1)
        all_logits[~mask] = -torch.inf
        attn_scores = F.softmax(all_logits, dim=-1)
        # attn_scores.shape = (bs, n_ma)
        # То есть для каждой машины:
        # Σ attention = 1

        # (bs, n_ma, n_ops)
        cross_attn_scores = attn_scores[..., :-1]
        # (bs, n_ma, 1)
        self_attn_scores = attn_scores[..., -1].unsqueeze(-1)

        # augment column embeddings with edge features, (bs, r, c, e)
        other_emb_aug = edge_emb + other_emb.unsqueeze(-3)
        # embedding операции обогащается информацией о ребре

        cross_emb = einsum(cross_attn_scores, other_emb_aug, "b m o, b m o e -> b m e")
        # Σ_o attention(m,o) * embedding(o)
        # cross_emb : (bs, n_ma, e)

        self_emb = self_emb * self_attn_scores
        # (bs, n_ma, emb_dim)
        hidden = cross_emb + self_emb

        # Полная формула слоя
        # Для каждой машины i:
        # h_i^{new} = ∑_j β_{ij} (h_j + e_{ij}) + α_{ii} h_i
        # где
        # символ	значение
        # h_i       embedding машины
        # h_j       embedding операции
        # e_{ij}    edge feature
        # α         self_attn_scores
        # β         cross_attn_scores
        return hidden


class HetGNNBlock(nn.Module):
    def __init__(self, embed_dim, normalization="batch"):
        super().__init__()

        # machines ← operations
        self.ma_from_ops = HetGNNLayer(embed_dim)

        # operations ← machines
        self.ops_from_ma = HetGNNLayer(embed_dim)

        # machines ← machines (transport graph)
        self.ma_from_ma = HetGNNLayer(embed_dim)

        # trucks ← machines
        self.truck_from_ma = HetGNNLayer(embed_dim)

        self.ffn_ma = TransformerFFN(embed_dim, embed_dim * 2, normalization)
        self.ffn_ops = TransformerFFN(embed_dim, embed_dim * 2, normalization)
        self.ffn_truck = TransformerFFN(embed_dim, embed_dim * 2, normalization)

    def forward(
        self,
        ops_emb,
        ma_emb,
        truck_emb,
        machine_edge_emb,
        truck_edge_emb,
        ops_ma_edges,
        available_trucks,
    ):
        # machines ← operations
        ma_msg_ops = self.ma_from_ops(
            ma_emb,
            ops_emb,
            machine_edge_emb,
            ops_ma_edges,
        )

        # machines ← machines (transport)
        ma_ma_edges = torch.ones(
            ma_emb.size(0),
            ma_emb.size(1),
            ma_emb.size(1),
            device=ma_emb.device,
            dtype=torch.bool,
        )
        diag = torch.eye(ma_emb.size(1), device=ma_emb.device).bool()
        ma_ma_edges[:, diag] = False
        ma_msg_ma = self.ma_from_ma(
            ma_emb,
            ma_emb,
            truck_edge_emb[:, : ma_emb.size(1), : ma_emb.size(1)],
            ma_ma_edges,
        )
        ma_hidden = ma_msg_ops + ma_msg_ma
        ma_hidden = self.ffn_ma(ma_hidden, ma_emb)

        # operations ← machines
        ops_msg = self.ops_from_ma(
            ops_emb,
            ma_emb,
            machine_edge_emb.transpose(1, 2),
            ops_ma_edges.transpose(1, 2),
        )
        ops_hidden = self.ffn_ops(ops_msg, ops_emb)

        # trucks ← machines
        truck_ma_edges = torch.ones(
            truck_emb.size(0),
            truck_emb.size(1),
            ma_emb.size(1),
            device=truck_emb.device,
            dtype=torch.bool,
        )
        truck_ma_edges = truck_ma_edges & available_trucks.unsqueeze(-1)
        truck_msg = self.truck_from_ma(
            truck_emb,
            ma_emb,
            truck_edge_emb[:, :truck_emb.size(1), :ma_emb.size(1)],
            truck_ma_edges,
        )
        truck_hidden = self.ffn_truck(truck_msg, truck_emb)

        return ops_hidden, ma_hidden, truck_hidden


class HetGNNEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 2,
        normalization: str = "batch",
        init_embedding=None,
        env_name: str = "fjspt",
        **init_embedding_kwargs,
    ) -> None:
        super().__init__()

        if init_embedding is None:
            init_embedding_kwargs["embed_dim"] = embed_dim
            init_embedding = env_init_embedding(env_name, init_embedding_kwargs)

        self.init_embedding = init_embedding

        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [HetGNNBlock(embed_dim, normalization) for _ in range(self.num_layers)]
        )

    def forward(self, td):
        (
            ops_emb,
            ma_emb,
            truck_emb,
            machine_edge_emb,
            truck_edge_emb,
            ops_ma_edges,
            available_trucks
        ) = self.init_embedding(td)

        for layer in self.layers:
            ops_emb, ma_emb, truck_emb = layer(
                ops_emb,
                ma_emb,
                truck_emb,
                machine_edge_emb,
                truck_edge_emb,
                ops_ma_edges,
                available_trucks,
            )

        return (ops_emb, ma_emb, truck_emb), None
