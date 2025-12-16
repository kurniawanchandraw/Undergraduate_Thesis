import numpy as np, torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from .wls import solve_local_wls
from .kernels import build_spatiotemporal_kernel

def train_model(
    model, X_all, y_all, A_prior, train_rows, val_rows=None, test_rows=None,
    epochs=200, lr=1e-3, ridge_lambda=5.0, ent_w=5e-3, smooth_w=1e-3,
    N_per_year=None, times=None, print_every=25, early_stop=True, es_patience=80,
    wls_kind="ridge", huber_delta=1.0, huber_iters=3, graph_topk=None, graph_symmetrize=False, device=None
):
    device = device or (next(model.parameters()).device)
    X_t = torch.tensor(X_all, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_all, dtype=torch.float32, device=device)
    A_t = torch.tensor(A_prior, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val, best_state, patience = float('inf'), None, 0
    hist = []

    T = len(times) if times is not None else None

    for ep in range(1, epochs+1):
        model.train(); opt.zero_grad()

        W, B = model(X_t, A_t)
        if graph_topk is not None:
            from .model import topk_rows, symmetrize_rows
            W = topk_rows(W, graph_topk)
            if graph_symmetrize:
                W = symmetrize_rows(W)

        y_hat, betas = solve_local_wls(
            X_t, y_t, W, kind=wls_kind, ridge=ridge_lambda,
            huber_delta=huber_delta, huber_iters=huber_iters, return_betas=True
        )

        sup = F.mse_loss(y_hat[train_rows], y_t[train_rows])

        Wn = W / (W.sum(dim=1, keepdims=True) + 1e-12)
        ent = -torch.sum(Wn * torch.log(Wn + 1e-12), dim=1).mean()
        ent_loss = -ent_w * ent

        if (T is not None) and (N_per_year is not None):
            smooth = 0.0
            beta_mat = betas.reshape(T, N_per_year, -1)
            for t_idx in range(T):
                bt = beta_mat[t_idx]
                s, e = t_idx*N_per_year, (t_idx+1)*N_per_year
                Wsp = A_t[s:e, s:e]
                diff = bt.unsqueeze(1) - bt.unsqueeze(0)
                smooth = smooth + torch.sum(Wsp.unsqueeze(-1) * diff.pow(2))
            spatial_loss = 1e-3 * smooth
        else:
            spatial_loss = 0.0

        total = sup + ent_loss + spatial_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Eval
        model.eval()
        with torch.no_grad():
            W_eval, _ = model(X_t, A_t)
            if graph_topk is not None:
                from .model import topk_rows, symmetrize_rows
                W_eval = topk_rows(W_eval, graph_topk)
                if graph_symmetrize:
                    W_eval = symmetrize_rows(W_eval)
            y_eval = solve_local_wls(X_t, y_t, W_eval, kind=wls_kind, ridge=ridge_lambda, return_betas=False)
            rmse_tr = np.sqrt(mean_squared_error(y_all[train_rows], y_eval[train_rows].detach().cpu().numpy()))
            rmse_va = np.sqrt(mean_squared_error(y_all[val_rows],   y_eval[val_rows].detach().cpu().numpy())) if val_rows is not None and len(val_rows)>0 else float('inf')
            rmse_te = np.sqrt(mean_squared_error(y_all[test_rows],  y_eval[test_rows].detach().cpu().numpy()))  if test_rows is not None and len(test_rows)>0 else np.nan

        hist.append(dict(epoch=ep, loss=total.item(), rmse_tr=rmse_tr, rmse_va=rmse_va, rmse_te=rmse_te,
                         alpha=float(model.alpha), tau=float(model.tau)))
        if (ep % print_every == 0) or (ep == 1):
            print(f"Epoch {ep:3d} | Loss {total.item():.4f} | RMSE: Train {rmse_tr:.3f} | Val {rmse_va:.3f} | α {float(model.alpha):.3f} | τ {float(model.tau):.3f}")

        score = rmse_va if val_rows is not None and len(val_rows)>0 else rmse_tr
        if score < best_val - 1e-6:
            best_val, best_state, patience = score, {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            patience += 1
            if early_stop and patience >= es_patience:
                print(f"Early stopping at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k,v in best_state.items()})

    # final forward
    model.eval()
    with torch.no_grad():
        W_final, _ = model(X_t, A_t)
        if graph_topk is not None:
            from .model import topk_rows, symmetrize_rows
            W_final = topk_rows(W_final, graph_topk)
            if graph_symmetrize:
                W_final = symmetrize_rows(W_final)
        y_final, betas_final = solve_local_wls(X_t, y_t, W_final, kind=wls_kind, ridge=ridge_lambda, return_betas=True)

    return dict(W=W_final, y_hat=y_final, betas=betas_final, history=hist, best_state=best_state)


# -----------------------------------------------------------------------------
# Fine-tune TRANSDUCTIVE with FUTURE year in graph (labels masked)
#   - Build prior on (2019..future_year) panel
#   - Warm-start from a trained model (recommended)
#   - Optimize only on train_rows (past years); val_rows optional; future_rows masked
#   - Return final predictions including FUTURE year (transductive)
# -----------------------------------------------------------------------------
def finetune_transductive_with_future(
    model, X_all_full, y_all_full, coords_blocks_full, times_full,
    train_rows, val_rows, future_rows,
    lr=1e-4, epochs=150, ridge_lambda=5.0, ent_w=5e-3, smooth_w=1e-3,
    knn_k=8, tau_s=1.0, tau_t=1.0, prior_self_weight=1.0,
    N_per_year=None, print_every=25, patience=40,
    wls_kind="ridge", huber_delta=1.0, huber_iters=3, graph_topk=None, graph_symmetrize=False,
    device=None
):
    device = device or (next(model.parameters()).device)
    # Build prior on full panel
    A_prior_np = build_spatiotemporal_kernel(coords_blocks_full, times_full,
                                             tau_s=tau_s, tau_t=tau_t, k_neighbors=knn_k,
                                             prior_self_weight=prior_self_weight, verbose=False)
    A = torch.tensor(A_prior_np, dtype=torch.float32, device=device)
    X = torch.tensor(X_all_full, dtype=torch.float32, device=device)
    y = torch.tensor(y_all_full, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val, best_state, pat = float('inf'), None, 0
    T = len(times_full)

    mask_train = torch.zeros(X.shape[0], dtype=torch.bool, device=device); mask_train[train_rows] = True
    mask_val   = torch.zeros_like(mask_train);                                  mask_val[val_rows]   = True
    # mask_future exists only for metric display, it's not used in loss
    mask_future= torch.zeros_like(mask_train);                                  mask_future[future_rows] = True

    for ep in range(1, epochs+1):
        model.train(); opt.zero_grad()
        W, B = model(X, A)
        if graph_topk is not None:
            from .model import topk_rows, symmetrize_rows
            W = topk_rows(W, graph_topk)
            if graph_symmetrize:
                W = symmetrize_rows(W)

        y_hat, betas = solve_local_wls(X, y, W, kind=wls_kind, ridge=ridge_lambda,
                                       huber_delta=huber_delta, huber_iters=huber_iters, return_betas=True)
        sup = F.mse_loss(y_hat[mask_train], y[mask_train])

        Wn = W / (W.sum(dim=1, keepdims=True) + 1e-12)
        ent = -torch.sum(Wn * torch.log(Wn + 1e-12), dim=1).mean()
        ent_loss = -ent_w * ent

        smooth = 0.0
        if (N_per_year is not None) and (T is not None):
            beta_mat = betas.reshape(T, N_per_year, -1)
            for t_idx in range(T):
                bt = beta_mat[t_idx]
                s, e = t_idx*N_per_year, (t_idx+1)*N_per_year
                Wsp = A[s:e, s:e]
                diff = bt.unsqueeze(1) - bt.unsqueeze(0)
                smooth = smooth + torch.sum(Wsp.unsqueeze(-1) * diff.pow(2))
        spatial_loss = smooth_w * smooth

        total = sup + ent_loss + spatial_loss
        total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        # eval
        model.eval()
        with torch.no_grad():
            W_eval, _ = model(X, A)
            if graph_topk is not None:
                from .model import topk_rows, symmetrize_rows
                W_eval = topk_rows(W_eval, graph_topk)
                if graph_symmetrize:
                    W_eval = symmetrize_rows(W_eval)
            y_eval = solve_local_wls(X, y, W_eval, kind=wls_kind, ridge=ridge_lambda, return_betas=False)
            import numpy as np
            from sklearn.metrics import mean_squared_error
            rmse_tr = np.sqrt(mean_squared_error(y_all_full[train_rows], y_eval[train_rows].detach().cpu().numpy()))
            rmse_va = np.sqrt(mean_squared_error(y_all_full[val_rows],   y_eval[val_rows].detach().cpu().numpy()))
            rmse_fu = np.sqrt(mean_squared_error(y_all_full[future_rows],y_eval[future_rows].detach().cpu().numpy()))

        if (ep % print_every == 0) or (ep == 1):
            print(f"Ep {ep:3d} | Loss {total.item():.4f} | RMSE: Train {rmse_tr:.3f} | Val {rmse_va:.3f} | Fut {rmse_fu:.3f} | α={float(model.alpha):.3f}, τ={float(model.tau):.3f}")

        if rmse_va < best_val - 1e-6:
            best_val, best_state, pat = rmse_va, {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= patience:
                print(f"Early stop at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k,v in best_state.items()})

    # final
    model.eval()
    with torch.no_grad():
        W_fin, _ = model(X, A)
        if graph_topk is not None:
            from .model import topk_rows, symmetrize_rows
            W_fin = topk_rows(W_fin, graph_topk)
            if graph_symmetrize:
                W_fin = symmetrize_rows(W_fin)
        y_fin, betas_fin = solve_local_wls(X, y, W_fin, kind=wls_kind, ridge=ridge_lambda, return_betas=True)

    return dict(W=W_fin, y_hat=y_fin, betas=betas_fin, best_state=best_state, A_prior=A, X=X, y=y)
