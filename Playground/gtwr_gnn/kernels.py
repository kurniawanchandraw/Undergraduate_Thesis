
import math
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def _pairwise_block(C1, C2):
    n1, n2 = len(C1), len(C2)
    D = np.zeros((n1, n2), dtype=np.float64)
    for i in range(n1):
        lat1, lon1 = C1[i]
        for j in range(n2):
            lat2, lon2 = C2[j]
            D[i, j] = haversine(lat1, lon1, lat2, lon2)
    return D

def _check_consistent(coords_blocks, tol=1e-6):
    base = coords_blocks[0]
    for b in coords_blocks[1:]:
        if not np.allclose(b, base, atol=tol):
            return False
    return True

def _sparsify_knn(W, k, self_w=1.0):
    n = W.shape[0]
    if n <= 1:
        return np.eye(n) * self_w
    k_eff = min(k, max(1, n-1))
    W_out = np.zeros_like(W)
    for i in range(n):
        row = W[i].copy()
        row[i] = -np.inf
        idx = np.argpartition(-row, kth=k_eff-1)[:k_eff]
        W_out[i, idx] = W[i, idx]
    np.fill_diagonal(W_out, self_w)
    rs = W_out.sum(axis=1, keepdims=True)
    return W_out / np.where(rs>0, rs, 1.0)

def build_spatiotemporal_kernel(
    coords_blocks, times, tau_s=1.0, tau_t=1.0, k_neighbors=8, prior_self_weight=1.0, verbose=True
):
    times = np.array(times, dtype=float)
    T = len(times)
    Ns = [cb.shape[0] for cb in coords_blocks]
    if verbose:
        print("Building spatio-temporal kernel...")
        print(f"Time periods: {T}, Locations(first): {Ns[0]}")

    # global spatial bandwidth
    all_d = []
    for C in coords_blocks:
        if len(C)>1:
            D = _pairwise_block(C, C)
            all_d.append(D[D>0])
    hS = np.median(np.concatenate(all_d)) if len(all_d) else 1.0
    hS = max(hS / max(tau_s,1e-6), 1e-6)

    # global temporal bandwidth
    Dt = np.abs(times[:,None]-times[None,:])
    hT = np.median(Dt[Dt>0]) if (Dt>0).any() else 1.0
    hT = max(hT / max(tau_t,1e-6), 1e-6)

    consistent = (len(set(Ns))==1) and _check_consistent(coords_blocks)
    if consistent:
        C0 = coords_blocks[0]
        D0 = _pairwise_block(C0, C0)
        K_S = np.exp(-0.5 * (D0/hS)**2)
        np.fill_diagonal(K_S, prior_self_weight)
        K_T = np.exp(-0.5 * (Dt/hT)**2)
        np.fill_diagonal(K_T, 1.0)
        W_full = np.kron(K_T, K_S).astype(np.float64)
    else:
        if verbose:
            print("Using adaptive cross-time kernel...")
        N_total = sum(Ns)
        W_full = np.zeros((N_total, N_total), dtype=np.float64)
        r0 = 0
        for i in range(T):
            C1, n1 = coords_blocks[i], Ns[i]
            c0 = 0
            for j in range(T):
                C2, n2 = coords_blocks[j], Ns[j]
                Dij = _pairwise_block(C1, C2)
                Ks = np.exp(-0.5 * (Dij/hS)**2)
                Kt = math.exp(-0.5 * ((abs(times[i]-times[j]) / hT)**2))
                W_full[r0:r0+n1, c0:c0+n2] = Kt * Ks
                c0 += n2
            r0 += n1

    W_sparse = _sparsify_knn(W_full, k_neighbors, self_w=prior_self_weight)
    if verbose:
        print(f"Kernel construction complete. Sparsity: {np.mean(W_sparse==0):.3f}")
    return W_sparse
