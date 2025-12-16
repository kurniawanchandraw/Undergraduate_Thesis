import numpy as np
import torch
import torch.nn.functional as F

from .kernels import build_spatiotemporal_kernel, haversine as _h
from .wls import solve_local_wls
from .model import topk_rows, symmetrize_rows

# -----------------------------------------------------------------------------
# Helper: estimate global bandwidths from OLD graph (stable for OOS)
# -----------------------------------------------------------------------------
def _estimate_bandwidths(coords_blocks_old, times_old, tau_s=1.0, tau_t=1.0):
    all_d = []
    for C in coords_blocks_old:
        n = len(C)
        if n <= 1: 
            continue
        # pairwise upper triangle distances
        for i in range(n):
            lat1,lon1 = C[i]
            for j in range(i+1, n):
                lat2,lon2 = C[j]
                all_d.append(_h(lat1,lon1,lat2,lon2))
    hS = np.median(all_d) if len(all_d) else 100.0
    hS = max(hS / max(tau_s, 1e-6), 1e-6)

    tgrid = np.sort(np.unique(times_old))
    Dt = np.abs(tgrid[:,None] - tgrid[None,:])
    hT = np.median(Dt[Dt>0]) if (Dt>0).any() else 1.0
    hT = max(hT / max(tau_t, 1e-6), 1e-6)
    return hS, hT

# -----------------------------------------------------------------------------
# Mode A — Semi-supervised style: FULL GRAPH forward (train + new) 
#   - Rebuild prior on (OLD ∪ NEW)
#   - Forward the trained GNN once to get learned W on the extended graph
#   - Solve local-WLS and return NEW slice
#   - This mimics how you produced TEST metrics (transductive).
# -----------------------------------------------------------------------------
def predict_new_fullgraph(
    model, X_train, y_train, coords_train, times_train,
    new_df, feature_cols, time_col, lat_col, lon_col,
    tau_s=1.0, tau_t=1.0, knn_k=8, prior_self_weight=1.0,
    wls_kind="ridge", ridge_lambda=5.0, huber_delta=1.0, huber_iters=3,
    graph_topk=None, graph_symmetrize=False, device=None
):
    device = device or (next(model.parameters()).device)

    # ---- Stack OLD + NEW ----
    X_new = new_df[feature_cols].values.astype(np.float32)
    coords_new = new_df[[lat_col, lon_col]].values.astype(np.float32)
    times_new = new_df[time_col].values.astype(float)

    X_comb = np.vstack([X_train, X_new]).astype(np.float32)
    coords_comb = np.vstack([coords_train, coords_new]).astype(np.float32)
    times_comb = np.concatenate([times_train, times_new]).astype(float)

    # Build prior on the combined panel (robust to unbalanced and reordering)
    unique_times = np.sort(np.unique(times_comb))
    coords_blocks = [coords_comb[times_comb == t] for t in unique_times]
    A_prior_ext_np = build_spatiotemporal_kernel(
        coords_blocks=coords_blocks, times=unique_times,
        tau_s=tau_s, tau_t=tau_t, k_neighbors=knn_k,
        prior_self_weight=prior_self_weight, verbose=False
    )

    # ---- Forward once on extended graph ----
    X_comb_t = torch.tensor(X_comb, dtype=torch.float32, device=device)
    A_prior_ext = torch.tensor(A_prior_ext_np, dtype=torch.float32, device=device)
    n_old, n_new = len(X_train), len(X_new)

    with torch.no_grad():
        W_learned, _ = model(X_comb_t, A_prior_ext)
        if graph_topk is not None:
            W_learned = topk_rows(W_learned, graph_topk)
        if graph_symmetrize:
            W_learned = symmetrize_rows(W_learned)
        y_hat = solve_local_wls(
            X_comb_t,
            torch.tensor(np.concatenate([y_train, np.zeros(n_new, np.float32)]), device=device),
            W_learned, kind=wls_kind, ridge=ridge_lambda,
            huber_delta=huber_delta, huber_iters=huber_iters, return_betas=False
        )
    return y_hat[n_old:].cpu().numpy()

# Backward-compatible alias
predict_new = predict_new_fullgraph

# -----------------------------------------------------------------------------
# Mode B — PURE OOS (transductive): freeze OLD rows; NEW rows depend on OLD only
#   - Compute OLD→OLD weights exactly as in training (same prior, same encoder)
#   - Build NEW→OLD via prior distances, then blend with encoder cosine using
#     model.alpha and model.tau (same as training). Optional cross_topk.
#   - NEW→NEW can be set to 0 (default, avoids label leakage) or tiny epsilon.
# -----------------------------------------------------------------------------
def predict_new_oos_transductive(
    model, X_train, y_train, coords_train, times_train,
    new_df, feature_cols, time_col, lat_col, lon_col,
    tau_s=1.0, tau_t=1.0, knn_k=8, prior_self_weight=1.0,
    lambda_blend=0.8,
    wls_kind="ridge", ridge_lambda=5.0, huber_delta=1.0, huber_iters=3,
    graph_topk=None, graph_symmetrize=False, device=None,
    cross_topk=None, new_self_weight=0.0
):
    device = device or (next(model.parameters()).device)

    # ---- Extract ----
    X_new = new_df[feature_cols].values.astype(np.float32)
    coords_new = new_df[[lat_col, lon_col]].values.astype(np.float32)
    times_new = new_df[time_col].values.astype(float)
    n_old, n_new = len(X_train), len(X_new)

    # ---- OLD-only prior & weights (freeze these rows) ----
    unique_times_old = np.sort(np.unique(times_train))
    coords_blocks_old = [coords_train[times_train == t] for t in unique_times_old]
    A_prior_old_np = build_spatiotemporal_kernel(
        coords_blocks=coords_blocks_old, times=unique_times_old,
        tau_s=tau_s, tau_t=tau_t, k_neighbors=knn_k,
        prior_self_weight=prior_self_weight, verbose=False
    )
    A_prior_old = torch.tensor(A_prior_old_np, dtype=torch.float32, device=device)
    X_old_t = torch.tensor(X_train, dtype=torch.float32, device=device)

    with torch.no_grad():
        W_old, H_old = model(X_old_t, A_prior_old)
        if graph_topk is not None:
            W_old = topk_rows(W_old, graph_topk)
        if graph_symmetrize:
            W_old = symmetrize_rows(W_old)

    # ---- Bandwidths from OLD (stable) ----
    hS, hT = _estimate_bandwidths(coords_blocks_old, times_train, tau_s, tau_t)

    # ---- Prior cross NEW→OLD ----
    A_cross = np.zeros((n_new, n_old), dtype=np.float32)
    for i in range(n_new):
        lat_i, lon_i, ti = coords_new[i,0], coords_new[i,1], times_new[i]
        for j in range(n_old):
            lat_j, lon_j, tj = coords_train[j,0], coords_train[j,1], times_train[j]
            d_spa = _h(lat_i, lon_i, lat_j, lon_j)
            d_tmp = abs(ti - tj)
            A_cross[i, j] = np.exp(-0.5 * (d_spa/hS)**2) * np.exp(-0.5 * (d_tmp/hT)**2)

    # sparsify cross block if requested
    if cross_topk is not None and cross_topk > 0:
        k = min(cross_topk, max(1, A_cross.shape[1]-1))
        idx = np.argpartition(-A_cross, kth=k-1, axis=1)[:, :k]
        rows = np.arange(n_new)[:, None]
        mask = np.zeros_like(A_cross)
        mask[rows, idx] = 1.0
        A_cross = A_cross * mask

    # normalize
    A_cross = A_cross / (A_cross.sum(axis=1, keepdims=True) + 1e-12)

    # ---- Blend prior cross with encoder cosine (as in training) ----
    with torch.no_grad():
        H_new = model.encoder(torch.tensor(X_new, dtype=torch.float32, device=device))
        H_old_n = F.normalize(H_old, p=2, dim=1)
        H_new_n = F.normalize(H_new, p=2, dim=1)
        S = H_new_n @ H_old_n.t()            # cosine [-1,1]
        logits = S / model.tau               # temperature scaling
        log_prior_cross = torch.log(torch.tensor(A_cross, device=device) + 1e-12)
        alpha = model.alpha
        log_blend = alpha * log_prior_cross + (1 - alpha) * logits
        W_new2old_gnn = F.softmax(log_blend, dim=1)  # rows sum to 1

        if (lambda_blend is not None) and (0.0 <= lambda_blend <= 1.0):
            W_new2old = lambda_blend * W_new2old_gnn + (1 - lambda_blend) * torch.tensor(A_cross, device=device)
        else:
            W_new2old = W_new2old_gnn

    # ---- NEW→NEW block ----
    if new_self_weight and new_self_weight > 0:
        W_new2new = torch.eye(n_new, device=device) * float(new_self_weight)
    else:
        W_new2new = torch.zeros((n_new, n_new), device=device)

    # ---- Assemble W_full ----
    W_full = torch.zeros((n_old + n_new, n_old + n_new), device=device)
    W_full[:n_old, :n_old] = W_old                    # frozen OLD rows
    W_new_row = torch.cat([W_new2old, W_new2new], 1)  # only NEW rows depend on NEW
    W_new_row = W_new_row / (W_new_row.sum(dim=1, keepdim=True) + 1e-12)
    W_full[n_old:, :] = W_new_row

    # ---- Solve local WLS (only NEW preds returned) ----
    X_comb = np.vstack([X_train, X_new]).astype(np.float32)
    X_comb_t = torch.tensor(X_comb, dtype=torch.float32, device=device)
    y_stub = np.concatenate([y_train, np.zeros(n_new, dtype=np.float32)])
    y_stub_t = torch.tensor(y_stub, dtype=torch.float32, device=device)

    with torch.no_grad():
        y_hat = solve_local_wls(
            X_comb_t, y_stub_t, W_full, kind=wls_kind, ridge=ridge_lambda,
            huber_delta=huber_delta, huber_iters=huber_iters, return_betas=False
        )
    return y_hat[n_old:].cpu().numpy()

# -----------------------------------------------------------------------------
# Mode C — Prior-only OOS (for sanity checks / ablations)
# -----------------------------------------------------------------------------
def predict_new_prior_only(
    X_train, y_train, coords_train, times_train,
    new_df, feature_cols, time_col, lat_col, lon_col,
    tau_s=1.0, tau_t=1.0, knn_k=8,
    wls_kind="ridge", ridge_lambda=5.0, huber_delta=1.0, huber_iters=3,
    cross_topk=None, new_self_weight=0.0, device=None
):
    """OOS using ONLY prior distances (no GNN), with 0 self-weight for NEW."""
    device = device or torch.device("cpu")

    X_new = new_df[feature_cols].values.astype(np.float32)
    coords_new = new_df[[lat_col, lon_col]].values.astype(np.float32)
    times_new = new_df[time_col].values.astype(float)

    n_old, n_new = len(X_train), len(X_new)

    # bandwidths from OLD only
    unique_times_old = np.sort(np.unique(times_train))
    coords_blocks_old = [coords_train[times_train == t] for t in unique_times_old]
    hS, hT = _estimate_bandwidths(coords_blocks_old, times_train, tau_s, tau_t)

    # cross NEW→OLD prior
    A_cross = np.zeros((n_new, n_old), dtype=np.float32)
    for i in range(n_new):
        lat_i, lon_i, ti = coords_new[i,0], coords_new[i,1], times_new[i]
        for j in range(n_old):
            lat_j, lon_j, tj = coords_train[j,0], coords_train[j,1], times_train[j]
            d_spa = _h(lat_i, lon_i, lat_j, lon_j)
            d_tmp = abs(ti - tj)
            A_cross[i, j] = np.exp(-0.5 * (d_spa/hS)**2) * np.exp(-0.5 * (d_tmp/hT)**2)

    if cross_topk is not None and cross_topk > 0:
        k = min(cross_topk, max(1, A_cross.shape[1]-1))
        idx = np.argpartition(-A_cross, kth=k-1, axis=1)[:, :k]
        rows = np.arange(n_new)[:, None]
        mask = np.zeros_like(A_cross)
        mask[rows, idx] = 1.0
        A_cross = A_cross * mask

    A_cross = A_cross / (A_cross.sum(axis=1, keepdims=True) + 1e-12)

    # NEW→NEW self weight (default 0 to avoid leakage)
    if new_self_weight and new_self_weight > 0:
        W_new2new = torch.eye(n_new) * float(new_self_weight)
    else:
        W_new2new = torch.zeros((n_new, n_new))

    W_full = torch.zeros((n_old + n_new, n_old + n_new), dtype=torch.float32, device=device)
    # Old rows identity (not used for NEW predictions)
    W_full[:n_old, :n_old] = torch.eye(n_old, device=device)
    W_full[n_old:, :n_old] = torch.tensor(A_cross, dtype=torch.float32, device=device)
    W_full[n_old:, n_old:] = W_new2new.to(device)
    W_full[n_old:, :] = W_full[n_old:, :] / (W_full[n_old:, :].sum(dim=1, keepdim=True) + 1e-12)

    X_comb = np.vstack([X_train, X_new]).astype(np.float32)
    X_comb_t = torch.tensor(X_comb, dtype=torch.float32, device=device)
    y_stub = np.concatenate([y_train, np.zeros(n_new, dtype=np.float32)])
    y_stub_t = torch.tensor(y_stub, dtype=torch.float32, device=device)

    with torch.no_grad():
        y_hat = solve_local_wls(
            X_comb_t, y_stub_t, W_full, kind=wls_kind, ridge=ridge_lambda, 
            huber_delta=huber_delta, huber_iters=huber_iters, return_betas=False
        )
    return y_hat[n_old:].cpu().numpy()
