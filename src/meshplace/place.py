"""Loss-aware port relabelling: search over input/output permutations of the target unitary.

If sources and detectors are interchangeable, the mesh may realise P_out U P_in instead of U.
The optical result is identical up to relabelling, but the programmed phases (and thus the light's
path through lossy devices) change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import MeshLoss, build, transfer
from .metrics import single_photon


@dataclass
class Placement:
    """Port relabelling and the score it reached under the loss map used for the search."""

    pin: np.ndarray
    pout: np.ndarray
    score: float
    evals: int


def relabel(u: np.ndarray, pin: np.ndarray, pout: np.ndarray) -> np.ndarray:
    """Return U'[i, j] = U[pout[i], pin[j]]."""
    return u[np.ix_(pout, pin)]


def score(u: np.ndarray, topology: str, loss: MeshLoss, pin: np.ndarray, pout: np.ndarray,
          objective: str = "f_eff") -> float:
    """Figure of merit (``f_eff`` or ``f_h``) of the mesh programmed for the relabelled target."""
    up = relabel(u, pin, pout)
    return single_photon(up, transfer(build(up, topology), loss))[objective]


def search(u: np.ndarray, topology: str, loss: MeshLoss, budget: int, rng: np.random.Generator,
           strategy: str = "local", move: str = "both", objective: str = "f_eff") -> Placement:
    """Search for a relabelling with ``budget`` mesh evaluations.

    strategy: 'random' (best of K uniformly random permutations) or 'local' (greedy swap hill climb
    from the identity). move: 'both', 'in' or 'out' selects which side may be permuted.
    """
    n = u.shape[0]
    ident = np.arange(n)
    pin, pout = ident.copy(), ident.copy()
    best = score(u, topology, loss, pin, pout, objective)
    evals = 1
    while evals < budget:
        cin, cout = pin.copy(), pout.copy()
        if strategy == "random":
            if move in ("both", "in"):
                cin = rng.permutation(n)
            if move in ("both", "out"):
                cout = rng.permutation(n)
        else:
            side = rng.choice(["in", "out"]) if move == "both" else move
            i, j = rng.choice(n, 2, replace=False)
            tgt = cin if side == "in" else cout
            tgt[i], tgt[j] = tgt[j], tgt[i]
        s = score(u, topology, loss, cin, cout, objective)
        evals += 1
        if s > best:
            best, pin, pout = s, cin, cout
    return Placement(pin, pout, best, evals)
