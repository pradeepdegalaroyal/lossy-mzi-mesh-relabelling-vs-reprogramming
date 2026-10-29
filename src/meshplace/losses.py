"""Random loss maps and target unitaries."""
from __future__ import annotations

import numpy as np
from scipy.stats import unitary_group

from .mesh import MeshLoss


def sample_loss(n: int, mean_db: float, sigma: float, rng: np.random.Generator,
                base_db: float = 0.0) -> MeshLoss:
    """Lognormal device-to-device spread of the internal-arm loss (mean ``mean_db``, log-std ``sigma``).

    The grid has n - 1 mode pairs and 2n columns, which covers both the Clements and Reck layouts.
    ``base_db`` is a phase-independent insertion loss that is identical on every device.
    """
    shape = (n - 1, 2 * n)
    arm = mean_db * np.exp(sigma * rng.standard_normal(shape) - 0.5 * sigma**2)
    return MeshLoss(arm, np.full(shape, base_db))


def perturb_loss(loss: MeshLoss, sigma_cal: float, rng: np.random.Generator) -> MeshLoss:
    """Return a mis-calibrated estimate of ``loss`` with multiplicative lognormal error."""
    err = np.exp(sigma_cal * rng.standard_normal(loss.arm_db.shape))
    return MeshLoss(loss.arm_db * err, loss.base_db.copy())


def haar_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random n x n unitary."""
    return unitary_group.rvs(n, random_state=rng)


def dft_unitary(n: int) -> np.ndarray:
    """Discrete Fourier transform (balanced n-port splitter)."""
    return np.fft.fft(np.eye(n)) / np.sqrt(n)
