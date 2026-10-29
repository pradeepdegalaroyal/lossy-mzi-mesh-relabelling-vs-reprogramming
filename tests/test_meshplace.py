"""Fast correctness tests for the mesh decompositions, loss model, search and re-tuning."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from meshplace.losses import dft_unitary, haar_unitary, sample_loss
from meshplace.mesh import MeshLoss, build, lossy_mzi, mzi, transfer
from meshplace.metrics import multi_photon, permanent, single_photon
from meshplace.place import relabel, score, search
from meshplace.retune import _Layout, _forward, retune


@pytest.mark.parametrize("n", [3, 4, 7, 8])
@pytest.mark.parametrize("topology", ["clements", "reck"])
def test_decomposition_is_exact(n: int, topology: str) -> None:
    u = haar_unitary(n, np.random.default_rng(n))
    mesh = build(u, topology)
    assert np.abs(transfer(mesh) - u).max() < 1e-10
    assert len(mesh.modes) == n * (n - 1) // 2


def test_depths() -> None:
    u = haar_unitary(8, np.random.default_rng(0))
    assert build(u, "clements").depth == 8
    assert build(u, "reck").depth == 2 * 8 - 3


def test_lossless_arm_reproduces_mzi() -> None:
    assert np.abs(lossy_mzi(0.7, 0.4) - mzi(0.7, 0.4)).max() < 1e-12


def test_loss_reduces_transmission_and_is_passive() -> None:
    n = 6
    u = haar_unitary(n, np.random.default_rng(1))
    loss = sample_loss(n, 1.0, 0.8, np.random.default_rng(2))
    t = transfer(build(u, "clements"), loss)
    assert np.linalg.norm(t, 2) <= 1 + 1e-9
    m = single_photon(u, t)
    assert 0 < m["eta"] < 1 and 0 < m["f_h"] <= 1 + 1e-12


def test_search_never_worse_than_identity() -> None:
    n = 6
    u = haar_unitary(n, np.random.default_rng(3))
    loss = sample_loss(n, 1.0, 0.8, np.random.default_rng(4))
    ident = np.arange(n)
    base = score(u, "clements", loss, ident, ident)
    pl = search(u, "clements", loss, 15, np.random.default_rng(0))
    assert pl.score >= base and pl.evals == 15
    up = relabel(u, pl.pin, pl.pout)
    assert abs(abs(np.linalg.det(up)) - 1) < 1e-10


def test_torch_forward_matches_numpy() -> None:
    n = 5
    u = haar_unitary(n, np.random.default_rng(5))
    loss = sample_loss(n, 1.0, 0.8, np.random.default_rng(6))
    mesh = build(u, "reck")
    amps = [loss.amplitudes(m, c) for m, c in zip(mesh.modes, mesh.column)]
    f64 = torch.float64
    t = _forward(_Layout(mesh), torch.tensor(mesh.theta, dtype=f64), torch.tensor(mesh.phi, dtype=f64),
                 torch.tensor(np.angle(mesh.diag), dtype=f64), torch.tensor([a for a, _ in amps], dtype=f64),
                 torch.tensor([b for _, b in amps], dtype=f64)).numpy()
    assert np.abs(t - transfer(mesh, loss)).max() < 1e-10


def test_retune_does_not_decrease_fidelity() -> None:
    n = 4
    u = haar_unitary(n, np.random.default_rng(7))
    loss = sample_loss(n, 2.0, 0.8, np.random.default_rng(8))
    mesh = build(u, "clements")
    before = single_photon(u, transfer(mesh, loss))["f_eff"]
    after = single_photon(u, transfer(retune(mesh, loss, u, steps=30), loss))["f_eff"]
    assert after >= before - 1e-12


def test_permanent_and_lossless_multiphoton() -> None:
    a = np.array([[1, 2], [3, 4]], dtype=complex)
    assert permanent(a) == 1 * 4 + 2 * 3
    u = dft_unitary(5)
    out = multi_photon(u, u, 2, np.random.default_rng(0), n_inputs=3)
    assert abs(out["fid"] - 1) < 1e-12 and abs(out["p_ratio"] - 1) < 1e-12


def test_zero_loss_map_is_identity_operation() -> None:
    n = 4
    u = haar_unitary(n, np.random.default_rng(9))
    zero = MeshLoss(np.zeros((n - 1, 2 * n)), np.zeros((n - 1, 2 * n)))
    m = single_photon(u, transfer(build(u, "clements"), zero))
    assert abs(m["f_eff"] - 1) < 1e-10 and abs(m["eta"] - 1) < 1e-10
