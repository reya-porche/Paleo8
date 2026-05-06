"""
geo_tvt/representation/cross_well_repr.py
Cross-well representation learning — the "geological foundation model" layer
for the anonymized-coordinate case.

The idea:
  Instead of external paleogeographic data, learn what formations look like
  from the wells themselves. Wells drilled through similar geology will have
  similar GR patterns, formation sequences, and TVT behavior.

This module:
  1. Encodes each well's GR log into a fixed-length embedding
  2. Finds geologically similar wells via embedding similarity
  3. Uses similar-well TVT history as a prior for prediction
  4. Learns formation-type prototypes (GR signatures → geological class)

Architecture: contrastive learning
  Wells in the same formation type should have similar embeddings.
  Wells in different formation types should have different embeddings.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_DIR


# ─── GR Sequence Encoder ─────────────────────────────────────────────────────

class GRSequenceEncoder(nn.Module):
    """
    Encodes variable-length GR log sequences into fixed 128-dim embeddings.

    Uses:
      Positional 1D CNN (captures local texture)
      → Bidirectional GRU (captures sequence order)
      → Attention pooling (focus on diagnostic depth intervals)
      → L2-normalized embedding

    The embedding captures:
      - GR baseline level (shale-rich vs clean)
      - Cyclicity (rhythmic vs massive)
      - Sharp vs gradational boundaries
      - Coarsening-up vs fining-up patterns
    """

    def __init__(self, embed_dim: int = 128, n_filters: int = 64, gru_hidden: int = 64):
        super().__init__()
        self.embed_dim = embed_dim

        # Local feature extraction
        self.conv1 = nn.Conv1d(1,          n_filters, kernel_size=7,  padding=3)
        self.conv2 = nn.Conv1d(n_filters,  n_filters, kernel_size=15, padding=7)
        self.conv3 = nn.Conv1d(n_filters,  n_filters, kernel_size=31, padding=15)
        self.norm  = nn.BatchNorm1d(n_filters)

        # Sequence context
        self.gru = nn.GRU(
            input_size=n_filters, hidden_size=gru_hidden,
            num_layers=2, batch_first=True, bidirectional=True,
        )

        # Attention pooling
        self.attn_w = nn.Linear(gru_hidden * 2, 1)

        # Projection to embedding space
        self.proj = nn.Sequential(
            nn.Linear(gru_hidden * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:    [B, T] — normalized GR sequence
        mask: [B, T] — 1 for valid positions, 0 for padding

        Returns: [B, embed_dim] L2-normalized embedding
        """
        # [B, T] → [B, 1, T]
        x = x.unsqueeze(1)

        # Multi-scale CNN
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        feat = self.norm(c1 + c3)  # residual

        # [B, F, T] → [B, T, F]
        feat = feat.permute(0, 2, 1)

        # GRU
        gru_out, _ = self.gru(feat)   # [B, T, 2*H]

        # Attention pooling
        attn_logits = self.attn_w(gru_out).squeeze(-1)  # [B, T]
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_logits, dim=-1)  # [B, T]
        pooled = (gru_out * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, 2*H]

        # Project and normalize
        emb = self.proj(pooled)
        return F.normalize(emb, p=2, dim=-1)


# ─── Contrastive Loss ────────────────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) contrastive loss.
    Standard loss for geological similarity learning.

    Positive pairs: different depth windows from the same well (same geology)
    Negative pairs: windows from different wells (different geology)
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """z_i, z_j: [B, D] — pairs of embeddings"""
        B = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)  # [2B, D]

        sim = torch.mm(z, z.T) / self.temperature  # [2B, 2B]
        sim.fill_diagonal_(float("-inf"))  # remove self-similarity

        # Positive pairs: (i, i+B) and (i+B, i)
        targets = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B,     device=z.device),
        ])

        return F.cross_entropy(sim, targets)


# ─── Formation Prototype Learning ────────────────────────────────────────────

class FormationPrototypes(nn.Module):
    """
    Learns representative GR signature prototypes for each formation type.
    Enables classifying formation from GR signature alone.

    After training:
      - Each formation has a prototype embedding
      - Similarity to prototype → formation probability
      - Used as geological classification features
    """

    def __init__(self, n_formations: int, embed_dim: int = 128):
        super().__init__()
        self.prototypes = nn.Parameter(
            F.normalize(torch.randn(n_formations, embed_dim), p=2, dim=-1)
        )
        self.n_formations = n_formations

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B, D] — well embeddings
        Returns:    [B, n_formations] — formation probabilities
        """
        # Cosine similarity to each prototype
        sims = torch.mm(embeddings, self.prototypes.T)  # [B, n_formations]
        return torch.softmax(sims, dim=-1)


# ─── Well Similarity Retrieval ────────────────────────────────────────────────

class WellEmbeddingIndex:
    """
    Fast nearest-neighbor well retrieval using precomputed embeddings.
    Finds the K most geologically similar wells to a query well.

    Used for:
      - Transferring TVT patterns from similar wells
      - Augmenting training data
      - Flagging outlier wells
    """

    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None
        self.well_ids:   Optional[list]        = None
        self.metadata:   Optional[pd.DataFrame] = None

    def build(
        self,
        encoder: GRSequenceEncoder,
        wells_df: pd.DataFrame,
        gr_col: str    = "GR",
        well_id_col: str = "well_id",
        device: str    = "cpu",
        max_len: int   = 500,
    ) -> "WellEmbeddingIndex":
        """Encode all wells and build the index."""
        encoder.eval()
        self.well_ids = []
        embs = []

        for well_id, grp in wells_df.groupby(well_id_col):
            gr = grp[gr_col].fillna(grp[gr_col].median()).values.astype(np.float32)
            # Normalize and truncate/pad
            from alignment.typewell_matcher import normalize_gr
            gr_norm = normalize_gr(gr)[:max_len]
            if len(gr_norm) < max_len:
                gr_norm = np.pad(gr_norm, (0, max_len - len(gr_norm)))

            x = torch.tensor(gr_norm, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                emb = encoder(x).squeeze(0).cpu().numpy()

            self.well_ids.append(well_id)
            embs.append(emb)

        self.embeddings = np.array(embs)
        print(f"[repr] Built embedding index: {len(self.well_ids)} wells, dim={self.embeddings.shape[1]}")
        return self

    def query(
        self,
        query_gr: np.ndarray,
        encoder: GRSequenceEncoder,
        k: int = 5,
        device: str = "cpu",
        max_len: int = 500,
    ) -> list[tuple]:
        """
        Find K most similar wells to a query GR log.
        Returns list of (well_id, similarity_score).
        """
        from alignment.typewell_matcher import normalize_gr
        gr_norm = normalize_gr(query_gr)[:max_len]
        if len(gr_norm) < max_len:
            gr_norm = np.pad(gr_norm, (0, max_len - len(gr_norm)))

        x = torch.tensor(gr_norm.astype(np.float32), device=device).unsqueeze(0)
        encoder.eval()
        with torch.no_grad():
            q_emb = encoder(x).squeeze(0).cpu().numpy()

        # Cosine similarity
        sims = self.embeddings @ q_emb  # [N]
        top_k = np.argsort(sims)[::-1][:k]

        return [(self.well_ids[i], float(sims[i])) for i in top_k]

    def save(self, path: Optional[Path] = None) -> None:
        path = path or (MODEL_DIR / "well_embedding_index.npz")
        np.savez(path, embeddings=self.embeddings, well_ids=np.array(self.well_ids))
        print(f"[repr] Saved index → {path}")

    def load(self, path: Optional[Path] = None) -> "WellEmbeddingIndex":
        path = path or (MODEL_DIR / "well_embedding_index.npz")
        data = np.load(path, allow_pickle=True)
        self.embeddings = data["embeddings"]
        self.well_ids   = list(data["well_ids"])
        return self


# ─── Training Utilities ───────────────────────────────────────────────────────

def create_contrastive_pairs(
    wells_df: pd.DataFrame,
    gr_col: str = "GR",
    well_id_col: str = "well_id",
    window: int = 100,
    n_pairs: int = 1000,
    max_len: int = 500,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create positive pairs for contrastive training.
    Each positive pair = two non-overlapping windows from the same well.

    Returns: (windows_i [N, max_len], windows_j [N, max_len])
    """
    from alignment.typewell_matcher import normalize_gr
    rng = np.random.RandomState(rng_seed)

    wells = [g for _, g in wells_df.groupby(well_id_col)]
    pairs_i, pairs_j = [], []

    for _ in range(n_pairs):
        # Sample a random well with enough data
        eligible = [w for w in wells if len(w) >= window * 2]
        if not eligible:
            continue
        well = rng.choice(eligible)
        gr   = normalize_gr(well[gr_col].fillna(0).values)

        max_start = len(gr) - window * 2
        start_i   = rng.randint(0, max_start)
        start_j   = rng.randint(start_i + window, min(start_i + window * 3, len(gr) - window))

        wi = np.zeros(max_len)
        wj = np.zeros(max_len)
        wi[:window] = gr[start_i:start_i + window]
        wj[:window] = gr[start_j:start_j + window]

        pairs_i.append(wi)
        pairs_j.append(wj)

    return np.array(pairs_i, dtype=np.float32), np.array(pairs_j, dtype=np.float32)


def train_encoder(
    wells_df: pd.DataFrame,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> GRSequenceEncoder:
    """Train the GR encoder with contrastive loss."""
    encoder = GRSequenceEncoder().to(device)
    loss_fn = NTXentLoss(temperature=0.1)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    print(f"[repr] Creating contrastive pairs...")
    wi, wj = create_contrastive_pairs(wells_df, n_pairs=2000)
    wi_t = torch.tensor(wi, device=device)
    wj_t = torch.tensor(wj, device=device)

    dataset = torch.utils.data.TensorDataset(wi_t, wj_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"[repr] Training encoder: {n_epochs} epochs, {len(dataset)} pairs")
    for epoch in range(n_epochs):
        encoder.train()
        epoch_loss = 0.0
        for batch_i, batch_j in loader:
            z_i = encoder(batch_i)
            z_j = encoder(batch_j)
            loss = loss_fn(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs} — loss: {epoch_loss/len(loader):.4f}")

    # Save
    save_path = MODEL_DIR / "gr_encoder.pt"
    torch.save(encoder.state_dict(), save_path)
    print(f"[repr] Saved encoder → {save_path}")
    return encoder
