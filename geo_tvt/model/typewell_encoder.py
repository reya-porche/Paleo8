"""
geo_tvt/model/typewell_encoder.py
Encodes the typewell GR signature into a fixed-length embedding
that captures local geological "memory" for the target area.

The typewell is the reference well used to define the target stratigraphic window.
Its GR log + formation labels encode what the geology should look like.
This encoder gives the TVT predictor access to that geological context.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class TypewellEncoder(nn.Module):
    """
    1D CNN + self-attention encoder for typewell GR logs.
    
    Input:  [batch, seq_len, n_features]   (GR + formation encoding)
    Output: [batch, embed_dim]             (geological context vector)
    
    Architecture:
      Conv1d(filters) → LayerNorm → Self-Attention → Global Pool → Linear
    """

    def __init__(
        self,
        n_input_features: int = 4,  # GR, depth_norm, formation_enc, lith_enc
        embed_dim: int = 64,
        n_filters: int = 128,
        kernel_size: int = 7,
        n_heads: int = 4,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(n_input_features, n_filters, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(n_filters, n_filters, 5, padding=2)
        self.norm1 = nn.LayerNorm(n_filters)
        self.norm2 = nn.LayerNorm(n_filters)

        self.attention = nn.MultiheadAttention(
            embed_dim=n_filters, num_heads=n_heads, batch_first=True
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.project = nn.Sequential(
            nn.Linear(n_filters, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, features] → [batch, features, seq]
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        # → [batch, seq, filters]
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x_attn, _ = self.attention(x, x, x)
        x = self.norm2(x + x_attn)
        # Global pool → [batch, filters, 1] → [batch, filters]
        x = self.pool(x.permute(0, 2, 1)).squeeze(-1)
        return self.project(x)


def encode_typewell_df(df, gr_col="GR", depth_col="MD",
                       formation_col=None, max_len=200) -> np.ndarray:
    """
    Prepare typewell dataframe into a model-ready tensor.
    
    Args:
        df: DataFrame with typewell log data
        gr_col: column name for gamma ray
        depth_col: column name for measured depth
        formation_col: optional column for formation label encoding
        max_len: max sequence length (pad/truncate)
    
    Returns: np.ndarray of shape [max_len, n_features]
    """
    seq = np.zeros((max_len, 4), dtype=np.float32)

    # GR: normalize to [0, 1] with 99th percentile clipping
    gr = df[gr_col].fillna(df[gr_col].median()).values
    gr_p99 = np.percentile(gr, 99)
    gr_norm = np.clip(gr / max(gr_p99, 1.0), 0.0, 1.0)

    # Depth: normalize to [0, 1]
    depth = df[depth_col].values
    depth_norm = (depth - depth.min()) / max(depth.max() - depth.min(), 1.0)

    # Formation encoding (ordinal if available)
    if formation_col and formation_col in df.columns:
        formations = df[formation_col].fillna("unknown")
        unique_fms = list(formations.unique())
        fm_enc = formations.map({f: i / max(len(unique_fms), 1) for i, f in enumerate(unique_fms)}).values
    else:
        fm_enc = np.zeros(len(gr))

    n = min(len(gr), max_len)
    seq[:n, 0] = gr_norm[:n]
    seq[:n, 1] = depth_norm[:n]
    seq[:n, 2] = fm_enc[:n]
    seq[:n, 3] = 1.0  # validity mask (1 = real data, 0 = padding)

    return seq
