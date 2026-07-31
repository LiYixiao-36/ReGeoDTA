"""ReGeoDTA model definition."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

EMBED_DIM = 128
MAX_RELATIVE_DISTANCE = 100.0
NUM_DISTANCE_BUCKETS = 27


class ProteinCNN(nn.Module):
    """Extract local protein-sequence features with three 1D convolutions."""

    def __init__(self, num_filters: int, kernel_sizes: Sequence[int]) -> None:
        super().__init__()
        if len(kernel_sizes) != 3:
            raise ValueError("kernel_sizes must contain exactly three values.")

        self.CNN_main = nn.Sequential(
            nn.Conv1d(EMBED_DIM, num_filters, kernel_sizes[0], padding=(kernel_sizes[0] - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters * 2, kernel_sizes[1], padding=(kernel_sizes[1] - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(num_filters * 2, EMBED_DIM, kernel_sizes[2], padding=(kernel_sizes[2] - 1) // 2),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.CNN_main(x)


class RelativePositionAttention(nn.Module):
    """Multi-head attention with learnable distance and direction biases."""

    def __init__(self, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        if EMBED_DIM % num_heads != 0:
            raise ValueError(f"EMBED_DIM ({EMBED_DIM}) must be divisible by num_heads ({num_heads}).")

        self.num_heads = num_heads
        self.head_dim = EMBED_DIM // num_heads
        self.scale = self.head_dim ** -0.5

        self.x_down_conv = nn.Conv1d(EMBED_DIM, EMBED_DIM, kernel_size=8, stride=4, padding=2)
        self.q_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.k_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.v_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.out_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(EMBED_DIM)

        self.relative_distance_bucket_table = nn.Parameter(torch.zeros(NUM_DISTANCE_BUCKETS, num_heads))
        distance_boundaries = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            12.0, 14.0, 16.0, 18.0, 20.0,
            23.0, 26.0, 29.0, 32.0, 35.0,
            40.0, 45.0, 50.0,
            60.0, 80.0, 100.0,
        ]
        self.register_buffer(
            "distance_boundaries", torch.tensor(distance_boundaries, dtype=torch.float32), persistent=False
        )

        self.xyz_mlp = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, num_heads))
        self.dir_scale = nn.Parameter(torch.tensor(0.0))
        self._init_relative_bias()

    def _init_relative_bias(self) -> None:
        initial_bias = self.relative_distance_bucket_table.new_tensor([
            0.00,
            0.60, 0.75, 0.85, 0.90, 0.88, 0.85, 0.80, 0.75, 0.70, 0.65,
            0.60, 0.55, 0.50, 0.45, 0.35,
            0.30, 0.27, 0.24, 0.21, 0.18,
            0.15, 0.12, 0.10,
            0.08, 0.06, 0.04,
        ])
        if initial_bias.numel() != NUM_DISTANCE_BUCKETS:
            raise ValueError(
                f"Expected {NUM_DISTANCE_BUCKETS} distance biases, got {initial_bias.numel()}."
            )

        with torch.no_grad():
            self.relative_distance_bucket_table.copy_(initial_bias.unsqueeze(-1).expand(-1, self.num_heads))

    def _distance_to_bucket(self, distance_matrix: torch.Tensor) -> torch.Tensor:
        boundaries = self.distance_boundaries.to(dtype=distance_matrix.dtype)
        valid_mask = (distance_matrix > 1e-6) & (distance_matrix <= MAX_RELATIVE_DISTANCE)
        bucket_indices = torch.bucketize(distance_matrix.contiguous(), boundaries) + 1
        return torch.where(valid_mask, bucket_indices, torch.zeros_like(bucket_indices)).long()

    def forward(
        self,
        x: torch.Tensor,
        rel_pos_matrix: torch.Tensor,
        padding_mask_k: torch.Tensor | None = None,
        padding_mask_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        distance_matrix = rel_pos_matrix[..., 0]
        xyz_bias = self.xyz_mlp(rel_pos_matrix[..., 1:4]).permute(0, 3, 1, 2)

        x = self.norm(x)
        x_kv = self.x_down_conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        seq_len_down = x_kv.size(1)

        query = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(x_kv).reshape(batch_size, seq_len_down, self.num_heads, self.head_dim)
        value = self.v_proj(x_kv).reshape(batch_size, seq_len_down, self.num_heads, self.head_dim)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        content_scores = torch.matmul(query, key) * self.scale
        distance_mask = (
            (distance_matrix > 1e-6) & (distance_matrix <= MAX_RELATIVE_DISTANCE)
        ).unsqueeze(1)
        content_scores = content_scores.masked_fill(~distance_mask, float("-inf"))

        bucket_indices = self._distance_to_bucket(distance_matrix)
        distance_bias = self.relative_distance_bucket_table[bucket_indices].permute(0, 3, 1, 2)
        distance_bias = distance_bias.masked_fill(~distance_mask, 0.0)
        attention_scores = content_scores + distance_bias + xyz_bias * self.dir_scale

        if padding_mask_k is not None:
            attention_scores = attention_scores.masked_fill(
                ~padding_mask_k.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        if padding_mask_q is not None:
            attention_scores = attention_scores.masked_fill(
                ~padding_mask_q.unsqueeze(1).unsqueeze(3), float("-inf")
            )

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        context = torch.matmul(attention_weights, value)
        context = context.permute(0, 2, 1, 3).reshape(batch_size, seq_len, EMBED_DIM)
        return self.dropout(self.out_proj(context))


class ProteinEncoder(nn.Module):
    """Transformer-style protein encoder block."""

    def __init__(
        self, num_heads: int = 8, ffn_hidden: int | None = None, dropout: float = 0.1, bias: bool = True
    ) -> None:
        super().__init__()
        self.pos_attention = RelativePositionAttention(num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.norm2 = nn.LayerNorm(EMBED_DIM)

        ffn_hidden = ffn_hidden or EMBED_DIM * 4
        self.ffn = nn.Sequential(
            nn.Linear(EMBED_DIM, ffn_hidden, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, EMBED_DIM, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, position: torch.Tensor, key_mask: torch.Tensor, query_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.norm1(x + self.pos_attention(x, position, key_mask, query_mask))
        return self.norm2(x + self.ffn(x))


class DrugFeatureEncoder(nn.Module):
    """Encode categorical and numerical atom features with a gated projection."""

    def __init__(self) -> None:
        super().__init__()
        self.drug_embed = nn.Embedding(12, EMBED_DIM, padding_idx=0)
        self.small_embed = nn.Embedding(3, 32)
        self.linear_in_drug = nn.Linear(9, 96)
        self.gate_network = nn.Sequential(
            nn.Linear(EMBED_DIM + 32 + 96, 256), nn.ReLU(), nn.Linear(256, 512)
        )
        self.norm_features = nn.LayerNorm(256)

    def forward(self, drug_node: torch.Tensor) -> torch.Tensor:
        node1 = self.drug_embed(drug_node[:, 0].long())
        node2 = self.small_embed((drug_node[:, -1] + 1).long())
        node3 = self.linear_in_drug(drug_node[:, 1:10])
        feature_values, feature_gates = self.gate_network(torch.cat([node1, node2, node3], dim=1)).chunk(2, dim=1)
        return self.norm_features(feature_values * torch.sigmoid(feature_gates))


class AffinityRegressor(nn.Module):
    """Predict binding affinity from drug and protein representations."""

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.reg = nn.Sequential(
            nn.Linear(EMBED_DIM * 2, 1024),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
        )
        self.out = nn.Linear(512, 1)
        nn.init.constant_(self.out.bias, 5)

    def forward(self, drug: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
        return self.out(self.reg(torch.cat((drug, protein), dim=-1)))


class DrugGraphNetwork(nn.Module):
    """Dual-branch graph encoder using GCN and GATv2-GCN paths."""

    def __init__(self, graph_mid_dim: int = 96, dropout: float = 0.1) -> None:
        super().__init__()
        self.drug_emb = DrugFeatureEncoder()
        self.relu = nn.ReLU()

        self.gcn1_1 = GCNConv(256, graph_mid_dim)
        self.gcn1_2 = GCNConv(graph_mid_dim, graph_mid_dim * 2)
        self.gcn1_3 = GCNConv(graph_mid_dim * 2, graph_mid_dim * 4)
        self.gat1 = GATv2Conv(256, graph_mid_dim, heads=10)
        self.gcn2_1 = GCNConv(graph_mid_dim * 10, graph_mid_dim * 10)

        self.dropout = nn.Dropout(dropout)
        pooled_dim = graph_mid_dim * 10 * 2 + graph_mid_dim * 4 * 2
        self.fc_g1 = nn.Linear(pooled_dim, 1500)
        self.fc_g2 = nn.Linear(1500, EMBED_DIM)

    def forward(self, drug) -> torch.Tensor:
        node_emb = self.drug_emb(drug.x)
        edge_index = drug.edge_index.long()
        batch = drug.batch

        y1 = self.relu(self.gcn1_1(node_emb, edge_index))
        y1 = self.relu(self.gcn1_2(y1, edge_index))
        y1 = self.relu(self.gcn1_3(y1, edge_index))
        y1 = torch.cat([gmp(y1, batch), gap(y1, batch)], dim=1)

        y2 = self.relu(self.gat1(node_emb, edge_index))
        y2 = self.relu(self.gcn2_1(y2, edge_index))
        y2 = torch.cat([gmp(y2, batch), gap(y2, batch)], dim=1)

        graph_embedding = self.relu(self.fc_g1(torch.cat((y1, y2), dim=1)))
        return self.fc_g2(self.dropout(graph_embedding))


class ReGeoDTA(nn.Module):
    """Drug-target affinity prediction model."""

    def __init__(
        self,
        protein_kernel: Sequence[int] = (5, 9, 13),
        head_num: int = 8,
        dropout_rate: float = 0.2,
        graph_mid_dim: int = 96,
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv1d(1024, EMBED_DIM, kernel_size=1)
        self.encoder = ProteinEncoder(num_heads=head_num, dropout=dropout_rate)
        self.protein_CNNs = ProteinCNN(num_filters=36, kernel_sizes=protein_kernel)
        self.gate_net = nn.Sequential(nn.Linear(EMBED_DIM, EMBED_DIM), nn.Sigmoid())
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.drug_network = DrugGraphNetwork(graph_mid_dim=graph_mid_dim, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        self.reg = AffinityRegressor(dropout_rate)

    def forward(
        self,
        protein: torch.Tensor,
        drug,
        position: torch.Tensor,
        key_mask: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        protein_embedding = self.conv_in(protein.permute(0, 2, 1))
        geometry_features = self.encoder(
            protein_embedding.permute(0, 2, 1), position, key_mask, query_mask
        ).permute(0, 2, 1)

        cnn_features = self.protein_CNNs(protein_embedding)
        protein_gate = self.gate_net(cnn_features.permute(0, 2, 1)).permute(0, 2, 1)
        fused_features = cnn_features + protein_gate * geometry_features

        protein_representation = self.norm2(self.max_pool(fused_features).squeeze(-1))
        drug_representation = self.norm1(self.drug_network(drug))
        return self.reg(drug_representation, protein_representation).squeeze(-1)


# Compatibility aliases for full-model checkpoints created by earlier code versions.
CNN = ProteinCNN
Encoder = ProteinEncoder
process_drug_emb = DrugFeatureEncoder
net_reg = AffinityRegressor
drug_graph_network = DrugGraphNetwork
RPGMFDTA = ReGeoDTA
