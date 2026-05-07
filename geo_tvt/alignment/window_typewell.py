"""
geo_tvt/alignment/window_typewell.py  —  Fast typewell matching.

Original:  O(n_lat × n_tw) pure-Python loop  → minutes per well
New:       single BLAS matmul + stride sampling → milliseconds per well  (100-500×)

Algorithm
---------
1. Build normalized typewell window matrix T  (n_tw_eff × W)  — once per well
2. Build normalized lateral   window matrix L  (n_lat_eff × W) — sampled at lateral_stride
3. All Pearson correlations:  C = L @ T.T   (one BLAS call, float32)
4. Per-row NMS top-K extraction
5. Nearest-neighbour fill for rows skipped by stride

Optional accelerators (auto-detected at import):
  FAISS  — IndexFlatIP search for large typewells (n_tw_eff > 3 000)
  cupy   — GPU matrix multiply when use_gpu=True
"""

import numpy as np
import pandas as pd

try:
    import faiss as _faiss
    _FAISS = True
except ImportError:
    _FAISS = False

try:
    import cupy as _cp
    _CUPY = True
except ImportError:
    _CUPY = False


# ─── Public API ───────────────────────────────────────────────────────────────

def add_typewell_window_features(
    lateral_df:     pd.DataFrame,
    typewell_df:    pd.DataFrame,
    gr_col:         str   = "GR",
    tvt_col:        str   = "TVT_input",
    window_size:    int   = 30,
    top_k:          int   = 5,
    lateral_stride: int   = 10,   # compute every N-th row, NN-fill the rest
    tw_stride:      int   = 2,    # subsample typewell windows
    batch_size:     int   = 1024, # lateral batch size for matmul memory control
    search_radius:  int   = 0,    # 0 = full typewell; >0 = cap search (samples)
    use_gpu:        bool  = False,
) -> pd.DataFrame:
    """
    Add top-K typewell window match features to every row of lateral_df.

    Columns added:
        tw_best_tvt, tw_best_score, tw_second_tvt, tw_weighted_tvt,
        tw_tvt_spread, tw_confidence, tw_tvt_vs_ffill
    """
    out = lateral_df.copy()
    if gr_col not in lateral_df.columns or gr_col not in typewell_df.columns:
        return out

    lat_gr = _clean(lateral_df[gr_col]).astype(np.float32)
    tw_gr  = _clean(typewell_df[gr_col]).astype(np.float32)
    tw_tvt_raw = (
        typewell_df[tvt_col].values.astype(np.float32)
        if tvt_col in typewell_df.columns
        else np.arange(len(tw_gr), dtype=np.float32)
    )
    tvt_filled = (
        pd.Series(tw_tvt_raw)
        .interpolate(limit=5)
        .fillna(float(np.nanmean(tw_tvt_raw)))
        .values.astype(np.float32)
    )

    n_lat, n_tw = len(lat_gr), len(tw_gr)
    half_w = window_size // 2

    # ── Build typewell window matrix (done once) ───────────────────────────────
    tw_centers = np.arange(half_w, n_tw - half_w + 1, max(1, tw_stride))
    if len(tw_centers) == 0:
        return _zero_features(out, lateral_df, tvt_col)

    T         = _extract_windows(tw_gr,  tw_centers, window_size)   # (n_T, W)
    T_normed  = _row_normalize(T)                                    # unit rows
    tvt_at_tw = tvt_filled[np.clip(tw_centers, 0, len(tvt_filled) - 1)]

    # ── Build lateral window matrix (sampled at lateral_stride) ───────────────
    lat_centers = np.arange(0, n_lat, max(1, lateral_stride))
    L        = _extract_windows(lat_gr, lat_centers, window_size)   # (n_L, W)
    L_normed = _row_normalize(L)

    effective = len(lat_centers)
    n_T       = len(tw_centers)
    print(
        f"[typewell_window] {n_lat} lat (sampled {effective}) × {n_tw} tw (sampled {n_T})"
        f"  stride={lateral_stride}  tw_stride={tw_stride}"
        f"{'  FAISS' if _FAISS and n_T > 3000 else ''}"
        f"{'  GPU' if use_gpu and _CUPY else ''}",
        flush=True,
    )

    # ── Compute all correlations ───────────────────────────────────────────────
    if use_gpu and _CUPY:
        C = _gpu_matmul(L_normed, T_normed)
    elif _FAISS and n_T > 3000:
        C = _faiss_scores(L_normed, T_normed, top_k * 4)
    else:
        C = _batched_matmul(L_normed, T_normed, batch_size)  # (n_L, n_T)

    # ── Top-K NMS extraction for each sampled row ──────────────────────────────
    min_sep = max(1, window_size // (2 * max(1, tw_stride)))
    res = _topk_nms(C, tvt_at_tw, top_k, min_sep)               # arrays of length n_L

    # ── Nearest-neighbour fill to all n_lat positions ─────────────────────────
    stride_actual = lat_centers[1] - lat_centers[0] if len(lat_centers) > 1 else 1
    nn_idx = np.clip(
        np.round(np.arange(n_lat) / stride_actual).astype(int),
        0, len(lat_centers) - 1,
    )
    res = {k: v[nn_idx] for k, v in res.items()}

    # ── Write output columns ──────────────────────────────────────────────────
    out["tw_best_tvt"]     = res["best_tvt"]
    out["tw_best_score"]   = res["best_score"]
    out["tw_second_tvt"]   = res["second_tvt"]
    out["tw_weighted_tvt"] = res["weighted_tvt"]
    out["tw_tvt_spread"]   = res["tvt_spread"]
    out["tw_confidence"]   = res["confidence"]

    if tvt_col in lateral_df.columns:
        ffill = lateral_df[tvt_col].ffill().fillna(0).values.astype(np.float32)
        out["tw_tvt_vs_ffill"] = res["best_tvt"] - ffill

    return out


# ─── Window extraction ────────────────────────────────────────────────────────

def _extract_windows(signal: np.ndarray, centers: np.ndarray, window_size: int) -> np.ndarray:
    """
    Fully vectorized window extraction using edge-padded fancy indexing.
    Returns float32 matrix of shape (len(centers), window_size).
    """
    half_w = window_size // 2
    # Pad signal so every center has a full window even at edges
    padded = np.pad(signal, half_w, mode="edge")
    # Centers shift by half_w due to padding
    shifted = centers + half_w
    # Column offsets: 0, 1, ..., window_size-1  →  absolute indices per row
    col_idx = shifted[:, None] + np.arange(window_size)[None, :] - half_w  # (n, W)
    return padded[col_idx].astype(np.float32)


def _row_normalize(mat: np.ndarray) -> np.ndarray:
    """Centre and L2-normalise each row so dot product = Pearson correlation."""
    m = mat - mat.mean(axis=1, keepdims=True)
    norms = np.sqrt((m * m).sum(axis=1, keepdims=True)) + 1e-8
    return (m / norms).astype(np.float32)


# ─── Correlation engines ──────────────────────────────────────────────────────

def _batched_matmul(L: np.ndarray, T: np.ndarray, batch_size: int) -> np.ndarray:
    """L @ T.T in row-batches to bound peak memory."""
    T_T   = T.T.copy()                                    # (W, n_T) — cache-friendly
    n_L   = len(L)
    n_T   = T.shape[0]
    out   = np.empty((n_L, n_T), dtype=np.float32)
    for s in range(0, n_L, batch_size):
        e = min(s + batch_size, n_L)
        out[s:e] = L[s:e] @ T_T                           # single BLAS SGEMM
    return out


def _faiss_scores(L: np.ndarray, T: np.ndarray, k_search: int) -> np.ndarray:
    """
    FAISS IndexFlatIP search: exact inner product, much faster for large n_T.
    Returns a sparse (n_L, n_T) matrix with non-zero only at top k_search hits.
    """
    k_search = min(k_search, len(T))
    index = _faiss.IndexFlatIP(T.shape[1])
    index.add(T)
    scores_top, idx_top = index.search(L, k_search)       # (n_L, k_search) each

    # Expand to dense for consistent downstream API
    n_L, n_T = len(L), len(T)
    out = np.zeros((n_L, n_T), dtype=np.float32)
    rows = np.repeat(np.arange(n_L), k_search)
    cols = idx_top.ravel()
    valid = cols >= 0
    out[rows[valid], cols[valid]] = scores_top.ravel()[valid]
    return out


def _gpu_matmul(L: np.ndarray, T: np.ndarray) -> np.ndarray:
    """cupy GPU matmul, returns numpy array."""
    L_gpu = _cp.asarray(L)
    T_gpu = _cp.asarray(T)
    C_gpu = L_gpu @ T_gpu.T
    return _cp.asnumpy(C_gpu)


# ─── Top-K NMS extraction ─────────────────────────────────────────────────────

def _topk_nms(
    C:        np.ndarray,    # (n_L, n_T)
    tvt_at_T: np.ndarray,    # (n_T,)
    top_k:    int,
    min_sep:  int,
) -> dict:
    """
    For each row extract top-K non-overlapping indices (min_sep apart).
    Returns dict of arrays, each length n_L.
    """
    n_L = len(C)
    best_tvt     = np.empty(n_L, dtype=np.float32)
    best_score   = np.empty(n_L, dtype=np.float32)
    second_tvt   = np.empty(n_L, dtype=np.float32)
    weighted_tvt = np.empty(n_L, dtype=np.float32)
    tvt_spread   = np.empty(n_L, dtype=np.float32)
    confidence   = np.empty(n_L, dtype=np.float32)

    for i in range(n_L):
        row = C[i]

        # Fast pre-selection: only inspect top 5×top_k candidates
        # argpartition is O(n_T), much faster than full sort
        n_cand = min(top_k * 5, len(row))
        cand_idx = np.argpartition(row, -n_cand)[-n_cand:]
        cand_idx = cand_idx[np.argsort(row[cand_idx])[::-1]]   # sort descending

        # NMS: greedy non-maximum suppression by minimum separation
        sel: list[int] = []
        for idx in cand_idx:
            if not any(abs(int(idx) - s) < min_sep for s in sel):
                sel.append(int(idx))
            if len(sel) >= top_k:
                break

        # Pad with last element if fewer than top_k matches
        while len(sel) < top_k:
            sel.append(sel[-1] if sel else 0)

        sel_arr   = np.array(sel[:top_k], dtype=np.int32)
        tvt_vals  = tvt_at_T[sel_arr]
        sc_vals   = row[sel_arr]

        weights = np.clip(sc_vals, 0.0, None) + 1e-9
        w_tvt   = float(np.dot(weights, tvt_vals) / weights.sum())

        best_tvt[i]     = tvt_vals[0]
        best_score[i]   = sc_vals[0]
        second_tvt[i]   = tvt_vals[1] if top_k > 1 else tvt_vals[0]
        weighted_tvt[i] = w_tvt
        tvt_spread[i]   = float(tvt_vals.std())
        confidence[i]   = float(sc_vals[0]) if sc_vals[0] > 0 else 0.0

    return {
        "best_tvt":     best_tvt,
        "best_score":   best_score,
        "second_tvt":   second_tvt,
        "weighted_tvt": weighted_tvt,
        "tvt_spread":   tvt_spread,
        "confidence":   confidence,
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _clean(series: pd.Series) -> np.ndarray:
    return series.interpolate(limit=5).fillna(series.median()).values


def _zero_features(
    out: pd.DataFrame,
    lateral_df: pd.DataFrame,
    tvt_col: str,
) -> pd.DataFrame:
    for col in ("tw_best_tvt", "tw_best_score", "tw_second_tvt",
                "tw_weighted_tvt", "tw_tvt_spread", "tw_confidence",
                "tw_tvt_vs_ffill"):
        out[col] = 0.0
    return out
