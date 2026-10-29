"""Rectangular (Clements) and triangular (Reck) MZI-mesh decompositions with a per-device loss model.

Convention: the Mach-Zehnder interferometer (MZI) acting on adjacent modes (m, m+1) is
T(theta, phi) = [[e^{i phi} cos(theta), -sin(theta)], [e^{i phi} sin(theta), cos(theta)]].
A mesh is a light-ordered list of elements followed by a diagonal of output phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Array = np.ndarray


def mzi(theta: float, phi: float) -> Array:
    """Return the 2x2 MZI transfer matrix T(theta, phi)."""
    c, s, e = np.cos(theta), np.sin(theta), np.exp(1j * phi)
    return np.array([[e * c, -s], [e * s, c]], dtype=complex)


@dataclass
class Mesh:
    """A programmed mesh: light-ordered MZIs, output phases and physical grid positions."""

    n: int
    topology: str
    modes: list[int] = field(default_factory=list)
    theta: list[float] = field(default_factory=list)
    phi: list[float] = field(default_factory=list)
    diag: Array = field(default_factory=lambda: np.ones(0, dtype=complex))
    column: list[int] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return max(self.column) + 1 if self.column else 0

    def schedule(self) -> None:
        """Assign each element to its earliest physical column (as-soon-as-possible schedule)."""
        last = np.zeros(self.n, dtype=int)
        self.column = []
        for m in self.modes:
            c = int(max(last[m], last[m + 1]))
            self.column.append(c)
            last[m] = last[m + 1] = c + 1


def _null_right(u: Array, row: int, m: int) -> tuple[float, float]:
    """Choose (theta, phi) so that u[row, m] vanishes after u <- u @ T(m)^-1."""
    a, b = u[row, m], u[row, m + 1]
    if abs(b) < 1e-14:
        theta = np.pi / 2 if abs(a) > 1e-14 else 0.0
    else:
        theta = float(np.arctan(abs(a / b)))
    phi = float(np.angle(a) - np.angle(b)) if abs(a) > 1e-14 and abs(b) > 1e-14 else 0.0
    return theta, phi


def _apply_right_inv(u: Array, m: int, theta: float, phi: float) -> None:
    t_inv = mzi(theta, phi).conj().T
    u[:, m : m + 2] = u[:, m : m + 2] @ t_inv


def _null_left(u: Array, m: int, col: int) -> tuple[float, float]:
    """Choose (theta, phi) so that u[m+1, col] vanishes after u <- T(m) @ u."""
    a, b = u[m, col], u[m + 1, col]
    if abs(a) < 1e-14:
        theta = np.pi / 2 if abs(b) > 1e-14 else 0.0
    else:
        theta = float(np.arctan(abs(b / a)))
    phi = float(np.angle(-b) - np.angle(a)) if abs(a) > 1e-14 and abs(b) > 1e-14 else 0.0
    return theta, phi


def clements(u_target: Array) -> Mesh:
    """Decompose a unitary into the rectangular Clements mesh (depth n)."""
    n = u_target.shape[0]
    u = u_target.astype(complex).copy()
    right: list[tuple[int, float, float]] = []
    left: list[tuple[int, float, float]] = []
    for k in range(n - 1):
        if k % 2 == 0:
            for i in range(k + 1):
                m, row = k - i, n - 1 - i
                th, ph = _null_right(u, row, m)
                _apply_right_inv(u, m, th, ph)
                right.append((m, th, ph))
        else:
            for j in range(k + 1):
                m = n + j - k - 2
                th, ph = _null_left(u, m, j)
                u[m : m + 2, :] = mzi(th, ph) @ u[m : m + 2, :]
                left.append((m, th, ph))
    d = np.diag(u).copy()
    mesh = Mesh(n, "clements")
    for m, th, ph in right:
        mesh.modes.append(m), mesh.theta.append(th), mesh.phi.append(ph)
    # commute the diagonal through the left elements: T^-1 D = D' T'
    new_elems: list[tuple[int, float, float]] = []
    for m, th, ph in reversed(left):
        dm, dn = d[m], d[m + 1]
        ph_new = float(np.angle(-dm / dn))
        d[m], d[m + 1] = -np.exp(-1j * ph) * dn, dn
        new_elems.append((m, th, ph_new))
    for m, th, ph in new_elems:
        mesh.modes.append(m), mesh.theta.append(th), mesh.phi.append(ph)
    mesh.diag = d
    mesh.schedule()
    return mesh


def reck(u_target: Array) -> Mesh:
    """Decompose a unitary into the triangular Reck mesh (depth 2n-3)."""
    n = u_target.shape[0]
    u = u_target.astype(complex).copy()
    mesh = Mesh(n, "reck")
    for row in range(n - 1, 0, -1):
        for m in range(row):
            th, ph = _null_right(u, row, m)
            _apply_right_inv(u, m, th, ph)
            mesh.modes.append(m), mesh.theta.append(th), mesh.phi.append(ph)
    mesh.diag = np.diag(u).copy()
    mesh.schedule()
    return mesh


def build(u_target: Array, topology: str) -> Mesh:
    """Decompose with the named topology ('clements' or 'reck')."""
    if topology == "clements":
        return clements(u_target)
    if topology == "reck":
        return reck(u_target)
    raise ValueError(f"unknown topology {topology!r}")


def lossy_mzi(theta: float, phi: float, arm_amp: float = 1.0, base_amp: float = 1.0) -> Array:
    """MZI with amplitude transmission ``arm_amp`` in its internal phase-shifter arm.

    Built from two 50:50 couplers around the internal arm, so that arm_amp = 1 reproduces
    ``mzi(theta, phi)`` exactly. ``base_amp`` is a phase-independent amplitude loss of the device.
    """
    gamma = np.pi - 2.0 * theta
    x = arm_amp * np.exp(1j * gamma)
    core = 0.5 * np.array([[x - 1, 1j * (x + 1)], [1j * (x + 1), 1 - x]])
    blk = (-1j * np.exp(-1j * gamma / 2)) * core @ np.diag([1.0, -1.0])
    blk = blk @ np.diag([np.exp(1j * phi), 1.0])
    return base_amp * blk


def transfer(mesh: Mesh, loss: "MeshLoss | None" = None) -> Array:
    """Return the n x n transfer matrix of the (possibly lossy) mesh."""
    n = mesh.n
    t = np.eye(n, dtype=complex)
    for idx, m in enumerate(mesh.modes):
        if loss is None:
            blk = mzi(mesh.theta[idx], mesh.phi[idx])
        else:
            a, b = loss.amplitudes(m, mesh.column[idx])
            blk = lossy_mzi(mesh.theta[idx], mesh.phi[idx], a, b)
        t[m : m + 2, :] = blk @ t[m : m + 2, :]
    return mesh.diag[:, None] * t


@dataclass
class MeshLoss:
    """Per-device losses in dB on the (mode pair, physical column) grid."""

    arm_db: Array
    base_db: Array

    def amplitudes(self, m: int, c: int) -> tuple[float, float]:
        return 10 ** (-self.arm_db[m, c] / 20), 10 ** (-self.base_db[m, c] / 20)
