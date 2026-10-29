"""Loss-aware reprogramming: gradient re-tuning of MZI phases on a lossy mesh (PyTorch)."""
from __future__ import annotations

import numpy as np
import torch

from .mesh import Mesh, MeshLoss

torch.set_num_threads(6)


class _Layout:
    """Index tensors that scatter every element into one dense matrix per physical column."""

    def __init__(self, mesh: Mesh) -> None:
        n, depth = mesh.n, mesh.depth
        self.n, self.depth = n, depth
        cols = torch.tensor(mesh.column)
        modes = torch.tensor(mesh.modes)
        self.cols, self.modes = cols, modes
        used = torch.zeros(depth, n, dtype=torch.bool)
        used[cols, modes] = True
        used[cols, modes + 1] = True
        ci, ri = torch.nonzero(~used, as_tuple=True)
        self.free_c, self.free_r = ci, ri


def _forward(layout: _Layout, theta: torch.Tensor, phi: torch.Tensor, diag_phase: torch.Tensor,
             arm: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Differentiable transfer matrix; arm/base are amplitude transmissions of each element."""
    n, d = layout.n, layout.depth
    gamma = torch.pi - 2.0 * theta
    x = arm.to(torch.complex128) * torch.exp(1j * gamma)
    pref = -1j * torch.exp(-0.5j * gamma) * base.to(torch.complex128)
    e = torch.exp(1j * phi)
    m, col = layout.modes, layout.cols
    blocks = torch.zeros(d, n, n, dtype=torch.complex128)
    blocks = blocks.index_put((col, m, m), pref * e * 0.5 * (x - 1))
    blocks = blocks.index_put((col, m, m + 1), -pref * 0.5j * (x + 1))
    blocks = blocks.index_put((col, m + 1, m), pref * e * 0.5j * (x + 1))
    blocks = blocks.index_put((col, m + 1, m + 1), -pref * 0.5 * (1 - x))
    blocks = blocks.index_put((layout.free_c, layout.free_r, layout.free_r),
                              torch.ones(len(layout.free_c), dtype=torch.complex128))
    t = blocks[0]
    for k in range(1, d):
        t = blocks[k] @ t
    return torch.exp(1j * diag_phase)[:, None] * t


def effective_fidelity(t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Loss-inclusive process fidelity |tr(U^dagger T)|^2 / n^2."""
    n = u.shape[0]
    return torch.abs(torch.trace(u.conj().T @ t)) ** 2 / n**2


def heralded_fidelity(t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Loss-normalised fidelity |tr(U^dagger T)|^2 / (n tr(T^dagger T))."""
    n = u.shape[0]
    return torch.abs(torch.trace(u.conj().T @ t)) ** 2 / (n * torch.sum(torch.abs(t) ** 2))


def retune(mesh: Mesh, loss: MeshLoss, u_target: np.ndarray, steps: int = 200,
           lr: float = 0.02, objective: str = "f_eff") -> Mesh:
    """Return a copy of ``mesh`` whose phases maximise ``objective`` ('f_eff' or 'f_h') on the lossy device."""
    fid = effective_fidelity if objective == "f_eff" else heralded_fidelity
    layout = _Layout(mesh)
    amps = [loss.amplitudes(m, c) for m, c in zip(mesh.modes, mesh.column)]
    arm = torch.tensor([a for a, _ in amps], dtype=torch.float64)
    base = torch.tensor([b for _, b in amps], dtype=torch.float64)
    theta = torch.tensor(mesh.theta, dtype=torch.float64, requires_grad=True)
    phi = torch.tensor(mesh.phi, dtype=torch.float64, requires_grad=True)
    dph = torch.tensor(np.angle(mesh.diag), dtype=torch.float64, requires_grad=True)
    u = torch.tensor(u_target, dtype=torch.complex128)
    opt = torch.optim.Adam([theta, phi, dph], lr=lr)
    best, state = -1.0, None
    for _ in range(steps):
        opt.zero_grad()
        f = fid(_forward(layout, theta, phi, dph, arm, base), u)
        if f.item() > best:
            best = f.item()
            state = (theta.detach().clone(), phi.detach().clone(), dph.detach().clone())
        (-f).backward()
        opt.step()
    assert state is not None
    th, ph, dp = state
    return Mesh(mesh.n, mesh.topology, list(mesh.modes), th.tolist(), ph.tolist(),
                np.exp(1j * dp.numpy()), list(mesh.column))
