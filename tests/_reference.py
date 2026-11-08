"""An independent implementation of Brant's (1990) test in numpy, used only to
check the package. Binary logits by Newton-Raphson; covariances by Brant's
formulas; adjacent-split contrasts."""

import numpy as np
from scipy.stats import chi2


def _logit(y, Xc, iters=100):
    b = np.zeros(Xc.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Xc @ b)))
        H = Xc.T @ (Xc * (p * (1 - p))[:, None])
        step = np.linalg.solve(H, Xc.T @ (y - p))
        b = b + step
        if np.max(np.abs(step)) < 1e-12:
            break
    p = 1.0 / (1.0 + np.exp(-(Xc @ b)))
    return b, p


def brant_reference(y, X, groups=None):
    """(omnibus chi2, df, p, {name: (chi2, df)}) for outcome y and predictors X
    (no constant). groups maps a name to the column indices tested jointly."""
    y = np.asarray(y)
    X = np.asarray(X, float)
    levels = np.unique(y)
    J1, K = len(levels) - 1, X.shape[1]
    Xc = np.column_stack([np.ones(len(y)), X])
    fits = [_logit((y > levels[m]).astype(float), Xc) for m in range(J1)]
    beta = np.concatenate([b[1:] for b, _ in fits])
    V = np.zeros((J1 * K, J1 * K))
    inv = [np.linalg.inv(Xc.T @ (Xc * (p * (1 - p))[:, None])) for _, p in fits]
    for m in range(J1):
        for l in range(m, J1):
            if l == m:
                blk = inv[m]
            else:
                pm, pl = fits[m][1], fits[l][1]
                blk = inv[m] @ (Xc.T @ (Xc * (pl - pm * pl)[:, None])) @ inv[l]
            V[m * K:(m + 1) * K, l * K:(l + 1) * K] = blk[1:, 1:]
            V[l * K:(l + 1) * K, m * K:(m + 1) * K] = blk[1:, 1:].T
    D = np.kron(np.eye(J1 - 1, J1) - np.eye(J1 - 1, J1, 1), np.eye(K))
    d = D @ beta
    stat = float(d @ np.linalg.solve(D @ V @ D.T, d))
    df = D.shape[0]
    out = {}
    for name, idx in (groups or {f"x{k}": [k] for k in range(K)}).items():
        rows = [m * K + k for k in idx for m in range(J1)]
        Dk = np.kron(np.eye(len(idx)), np.eye(J1 - 1, J1) - np.eye(J1 - 1, J1, 1))
        dk = Dk @ beta[rows]
        out[name] = (float(dk @ np.linalg.solve(Dk @ V[np.ix_(rows, rows)] @ Dk.T, dk)),
                     Dk.shape[0])
    return stat, df, float(chi2.sf(stat, df)), out
