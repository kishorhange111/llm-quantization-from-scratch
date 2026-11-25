"""Weight quantisers. Every function takes ``W: (out, in)`` and returns the *dequantised*
weight (same shape, float), so the effect on a model can be measured with ordinary matmuls.

- ``rtn``: round-to-nearest with an asymmetric min/max scale per (row, group of ``group``
  input columns). This is the baseline.
- ``gptq`` (Frantar et al. 2023): quantise one input column at a time and spread its rounding
  error over the columns not yet quantised, weighted by the inverse Hessian ``H = 2XᵀX/n``
  of the layer's calibration inputs. This minimises ``‖XWᵀ − XQᵀ‖²`` greedily. Two tricks
  from the paper are included: the Cholesky factor of H⁻¹ for numerical stability, and lazy
  batch updates, which apply the error to the rest of the matrix once per block of 128
  columns instead of once per column. ``act_order`` quantises columns in decreasing
  ``diag(H)`` order.
- ``awq`` (Lin et al. 2024): before rounding, scale input channel j by ``s_j = mean|x_j|^α``
  (folded back into the activations: ``y = (x/s)(Q(W·s))ᵀ``). This protects the weights
  that meet large activations. α is chosen by grid search on calibration outputs.
"""

from __future__ import annotations

import torch


def _grid(w: torch.Tensor, bits: int):
    """Asymmetric scale and zero point per row of ``w`` (rows, cols)."""
    lo = w.amin(1, keepdim=True).clamp(max=0)
    hi = w.amax(1, keepdim=True).clamp(min=0)
    scale = (hi - lo).clamp(min=1e-8) / (2**bits - 1)
    zero = torch.round(-lo / scale)
    return scale, zero


def _q(w, scale, zero, bits):
    return (torch.clamp(torch.round(w / scale) + zero, 0, 2**bits - 1) - zero) * scale


def rtn(W: torch.Tensor, bits: int, group: int | None = 128) -> torch.Tensor:
    W = W.float()
    if group is None or group >= W.shape[1]:
        return _q(W, *_grid(W, bits), bits)
    out = torch.empty_like(W)
    for i in range(0, W.shape[1], group):
        blk = W[:, i : i + group]
        out[:, i : i + group] = _q(blk, *_grid(blk, bits), bits)
    return out


def hessian(X: torch.Tensor) -> torch.Tensor:
    """H = 2 XᵀX / n for calibration inputs X (n, in)."""
    X = X.float()
    return 2 * X.t() @ X / X.shape[0]


def gptq(
    W: torch.Tensor,
    H: torch.Tensor,
    bits: int,
    group: int | None = 128,
    block: int = 128,
    damp: float = 0.01,
    act_order: bool = False,
) -> torch.Tensor:
    W, H = W.clone().float(), H.clone().float()
    cols = W.shape[1]
    dead = torch.diag(H) == 0  # inputs that are always zero: their weights don't matter
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W, H = W[:, perm], H[perm][:, perm]
    H += damp * torch.diag(H).mean() * torch.eye(cols, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    scale = zero = None
    if group is None:
        scale, zero = _grid(W, bits)
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        W1, Hi = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        Err = torch.zeros_like(W1)
        for i in range(i2 - i1):
            c = i1 + i
            if group is not None and c % group == 0:  # grid from the *updated* weights
                ahead = torch.cat([W1[:, i:], W[:, i2:]], 1)[:, :group]
                scale, zero = _grid(ahead, bits)
            w, d = W1[:, i], Hi[i, i]
            q = _q(w[:, None], scale, zero, bits)[:, 0]
            Q[:, c] = q
            err = (w - q) / d
            W1[:, i:] -= err[:, None] * Hi[i, i:][None, :]
            Err[:, i] = err
        W[:, i2:] -= Err @ Hinv[i1:i2, i2:]  # lazy batch update of the remaining columns
    if perm is not None:
        Q = Q[:, torch.argsort(perm)]
    return Q


def awq(
    W: torch.Tensor, X: torch.Tensor, bits: int, group: int | None = 128, n_grid: int = 20
) -> tuple[torch.Tensor, float]:
    """Returns the effective dequantised weight ``Q(W·s)/s`` and the chosen α."""
    W, X = W.float(), X.float()
    act = X.abs().mean(0).clamp(min=1e-8)  # (in,)
    ref = X @ W.t()
    best = (float("inf"), None, 0.0)
    for a in torch.linspace(0, 1, n_grid + 1).tolist():
        s = act**a
        s = s / (s.max() * s.min()).sqrt()  # keep scales centred around 1
        Wq = rtn(W * s, bits, group) / s
        err = float((X @ Wq.t() - ref).pow(2).mean())
        if err < best[0]:
            best = (err, Wq, a)
    return best[1], best[2]


def output_error(W: torch.Tensor, Wq: torch.Tensor, X: torch.Tensor) -> float:
    """Relative reconstruction error ‖X(W − Wq)ᵀ‖² / ‖XWᵀ‖² on calibration inputs."""
    X = X.float()
    ref = X @ W.float().t()
    return float((X @ (W.float() - Wq.float()).t()).pow(2).sum() / ref.pow(2).sum())
