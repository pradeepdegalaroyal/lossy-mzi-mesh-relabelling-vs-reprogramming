"""Single-photon and multi-photon figures of merit for a lossy linear-optical map."""
from __future__ import annotations

from itertools import combinations, permutations

import numpy as np


def single_photon(u: np.ndarray, t: np.ndarray) -> dict[str, float]:
    """Loss-inclusive fidelity f_eff = |tr(U^dag T)|^2 / n^2, mean transmission eta, and f_h = f_eff / eta.

    ``f_h`` is the heralded (loss-normalised) fidelity, ``eta_min`` the worst input-port transmission.
    """
    n = u.shape[0]
    f_eff = float(abs(np.trace(u.conj().T @ t)) ** 2 / n**2)
    col = np.sum(np.abs(t) ** 2, axis=0)
    eta = float(col.mean())
    return {"f_eff": f_eff, "eta": eta, "f_h": f_eff / eta, "eta_min": float(col.min())}


def permanent(a: np.ndarray) -> complex:
    """Permanent of a small square matrix by direct expansion (N <= 4)."""
    n = a.shape[0]
    return sum(np.prod([a[i, p[i]] for i in range(n)]) for p in permutations(range(n)))


def multi_photon(u: np.ndarray, t: np.ndarray, n_photons: int, rng: np.random.Generator,
                 n_inputs: int = 6) -> dict[str, float]:
    """Mean over random input subsets of the collision-free N-photon output statistics.

    Photons are ideal and indistinguishable. ``p_ratio`` is the lossy success probability
    divided by the ideal one; ``fid`` is the squared Bhattacharyya overlap of the normalised
    output distributions (1 means the heralded statistics are exact).
    """
    n = u.shape[0]
    outs = list(combinations(range(n), n_photons))
    fids, ratios = [], []
    for _ in range(n_inputs):
        s = list(rng.choice(n, n_photons, replace=False))
        p = np.array([abs(permanent(t[np.ix_(r, s)])) ** 2 for r in outs])
        q = np.array([abs(permanent(u[np.ix_(r, s)])) ** 2 for r in outs])
        fids.append(float(np.sum(np.sqrt(p / p.sum() * q / q.sum())) ** 2))
        ratios.append(float(p.sum() / q.sum()))
    return {"fid": float(np.mean(fids)), "p_ratio": float(np.mean(ratios))}
