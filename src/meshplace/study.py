"""Programming pipelines evaluated against the true loss map."""
from __future__ import annotations

import numpy as np

from .mesh import MeshLoss, build, transfer
from .metrics import multi_photon, single_photon
from .place import relabel, search
from .retune import retune

METHODS = ("default", "place_random", "place_local", "retune", "place_local+retune")


def realise(u: np.ndarray, topology: str, method: str, loss_true: MeshLoss,
            loss_est: MeshLoss | None, budget: int, rng: np.random.Generator,
            steps: int = 300, move: str = "both", objective: str = "f_eff") -> tuple[np.ndarray, np.ndarray]:
    """Program a mesh for ``u`` with ``method`` and return (relabelled target, true lossy transfer matrix).

    The programming step sees only ``loss_est`` (the calibrated map); the returned matrix is
    evaluated on ``loss_true``.
    """
    est = loss_est if loss_est is not None else loss_true
    up = u
    if method.startswith("place"):
        strategy = "random" if method.startswith("place_random") else "local"
        pl = search(u, topology, est, budget, rng, strategy=strategy, move=move, objective=objective)
        up = relabel(u, pl.pin, pl.pout)
    mesh = build(up, topology)
    if method.endswith("retune"):
        mesh = retune(mesh, est, up, steps=steps, objective=objective)
    return up, transfer(mesh, loss_true)


def evaluate(u: np.ndarray, topology: str, method: str, loss_true: MeshLoss,
             loss_est: MeshLoss | None, budget: int, rng: np.random.Generator,
             steps: int = 300, photons: tuple[int, ...] = (), move: str = "both",
             objective: str = "f_eff") -> dict[str, float]:
    """Run one method and collect single-photon (and optionally multi-photon) metrics."""
    up, t = realise(u, topology, method, loss_true, loss_est, budget, rng, steps, move, objective)
    out = single_photon(up, t)
    for k in photons:
        mp = multi_photon(up, t, k, np.random.default_rng(7))
        out[f"fid{k}"], out[f"p_ratio{k}"] = mp["fid"], mp["p_ratio"]
    return out
