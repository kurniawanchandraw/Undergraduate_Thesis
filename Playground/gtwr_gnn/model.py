# gtwr_gnn/model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MathematicallyCorrectGNNWeightNet(nn.Module):
    """
    Encoder -> cosine similarity -> temperature scaling (tau)
    -> log-linear blend with prior (alpha) -> row-softmax to get W.

    forward(X, A_prior) returns:
      - W: (N, N) row-stochastic weights matrix
      - H: (N, emb) node embeddings
    """
    def __init__(self, d_in, spa_hid=32, emb=16, tau=1.2, alpha_init=0.30):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, spa_hid),
            nn.ReLU(),
            nn.Linear(spa_hid, emb)
        )
        # learnable temperature (clamped in property)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau))))
        # learnable alpha (sigmoid-constrained)
        self.raw_alpha = nn.Parameter(torch.tensor(math.log(alpha_init/(1.0 - alpha_init))))

        # optional: small init for last layer to keep early training stable
        nn.init.kaiming_uniform_(self.encoder[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.encoder[0].bias)
        nn.init.xavier_uniform_(self.encoder[2].weight)
        nn.init.zeros_(self.encoder[2].bias)

    @property
    def tau(self):
        # keep tau in a sane range
        return torch.exp(self.log_tau).clamp(min=0.1, max=10.0)

    @property
    def alpha(self):
        # alpha in (0,1)
        return torch.sigmoid(self.raw_alpha)

    def forward(self, X, A_prior):
        """
        X: (N, p) float32
        A_prior: (N, N) float32, row-stochastic-ish prior (nonnegative)
        """
        H = self.encoder(X)                 # (N, emb)
        Hn = F.normalize(H, p=2, dim=1)     # cosine sim works best if normalized
        S = Hn @ Hn.t()                     # (N, N), cosine in [-1, 1]
        logits = S / self.tau               # temperature scaling

        # log-opinion-pooling with prior
        log_prior = torch.log(A_prior + 1e-12)
        log_combined = self.alpha * log_prior + (1.0 - self.alpha) * logits

        # row-softmax so each row sums to 1 (row-stochastic W)
        W = F.softmax(log_combined, dim=1)
        return W, H


# ---------- Graph post-processing helpers (optional) ----------

def _row_normalize(W: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-normalize a nonnegative matrix to sum to 1 per row."""
    row_sum = W.sum(dim=1, keepdim=True)
    return W / (row_sum + eps)


def topk_rows(W: torch.Tensor, k: int) -> torch.Tensor:
    """
    Keep only top-k entries per row (by value), zero the rest, then row-normalize.
    Assumes W is nonnegative (e.g., softmax output).
    """
    if k is None:
        return W
    n = W.shape[0]
    k_eff = max(1, min(k, n))  # guard
    # indices of top-k per row
    vals, idx = torch.topk(W, k_eff, dim=1)
    mask = torch.zeros_like(W)
    mask.scatter_(1, idx, 1.0)
    W_pruned = W * mask
    return _row_normalize(W_pruned)


def symmetrize_rows(W: torch.Tensor) -> torch.Tensor:
    """
    Make W symmetric by averaging with its transpose, then row-normalize.
    (Still returns a row-stochastic matrix.)
    """
    W_sym = 0.5 * (W + W.t())
    # negative tiny values can appear from numerical issues; clamp to 0
    W_sym = torch.clamp(W_sym, min=0.0)
    return _row_normalize(W_sym)
