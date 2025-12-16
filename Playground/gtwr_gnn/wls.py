
import torch
import torch.nn.functional as F

def local_wls_ridge(X, y, W, ridge=5.0, return_betas=True):
    N, p = X.shape
    device = X.device
    I = ridge * torch.eye(p, device=device)
    y_hat = torch.zeros(N, device=device)
    betas = torch.zeros(N, p, device=device) if return_betas else None
    for i in range(N):
        w = W[i]
        ws = torch.sqrt(w + 1e-12)
        Xw = X * ws.unsqueeze(1); yw = y * ws
        XtWX = Xw.t() @ Xw + I
        XtWy = Xw.t() @ yw
        try:
            beta = torch.linalg.solve(XtWX, XtWy)
        except RuntimeError:
            beta = torch.linalg.lstsq(XtWX, XtWy.unsqueeze(1)).solution.squeeze()
        y_hat[i] = X[i] @ beta
        if return_betas: betas[i] = beta
    return (y_hat, betas) if return_betas else y_hat

def local_wls_huber(X, y, W, ridge=5.0, delta=1.0, iters=3, return_betas=True):
    N, p = X.shape
    device = X.device
    I = ridge * torch.eye(p, device=device)
    y_hat = torch.zeros(N, device=device)
    betas = torch.zeros(N, p, device=device) if return_betas else None
    for i in range(N):
        w = W[i].clone()
        beta = None
        for _ in range(iters):
            ws = torch.sqrt(w + 1e-12)
            Xw = X * ws.unsqueeze(1); yw = y * ws
            XtWX = Xw.t() @ Xw + I
            XtWy = Xw.t() @ yw
            try:
                beta = torch.linalg.solve(XtWX, XtWy)
            except RuntimeError:
                beta = torch.linalg.lstsq(XtWX, XtWy.unsqueeze(1)).solution.squeeze()
            r = y - X @ beta
            absr = torch.abs(r) + 1e-12
            # Huber weight update (IRLS style)
            w = W[i] * torch.where(absr <= delta, torch.ones_like(absr), (delta / absr))
        y_hat[i] = X[i] @ beta
        if return_betas: betas[i] = beta
    return (y_hat, betas) if return_betas else y_hat

def solve_local_wls(X, y, W, kind="ridge", ridge=5.0, huber_delta=1.0, huber_iters=3, return_betas=True):
    if kind == "ridge":
        return local_wls_ridge(X, y, W, ridge=ridge, return_betas=return_betas)
    elif kind == "huber":
        return local_wls_huber(X, y, W, ridge=ridge, delta=huber_delta, iters=huber_iters, return_betas=return_betas)
    else:
        raise ValueError(f"Unknown WLS kind: {kind}")
