# Method

## Device model
* **Mesh.** An n-mode mesh of n(n-1)/2 MZIs with output phases. Two layouts are decomposed exactly (error below 1e-10, tested): the
  rectangular Clements mesh (depth n) and the triangular Reck mesh (depth 2n-3). Each MZI T(theta, phi) sits at a (mode pair, physical column) position found by an as-soon-as-possible schedule.
* **Loss.** Each MZI is two 50:50 couplers around an internal phase-shifter arm. The arm has amplitude transmission a. With a = 1 the device equals T(theta, phi) exactly; with a < 1 its transmission depends on the programmed theta. The arm loss in dB is log-normal across devices with mean `arm_db` and spread `sigma` (default 0.8). A phase-independent per-device loss is supported but set to zero.
* **Metrics** for the realised n x n transfer matrix T and target U: loss-inclusive fidelity f_eff = |tr(U†T)|²/n²; mean transmission eta = tr(T†T)/n; heralded fidelity f_h = f_eff/eta (distortion only). For N = 2 and 3 ideal indistinguishable photons: collision-free output distribution from matrix permanents, success ratio to the ideal and squared Bhattacharyya overlap of the normalised distributions, averaged over 6 random input subsets.

## Programming strategies
| Name | What it does |
|---|---|
| default | Decompose U with the layout's standard algorithm, ignoring loss (the standard photonic baseline: Clements, plus Reck as the older layout) |
| place_random | Classical baseline: best of K uniformly random input and output permutations, each scored on the calibrated loss map |
| place_local | Greedy swap hill climb on the permutations from the identity, same budget K of mesh evaluations |
| retune | Adam gradient ascent (PyTorch, 200 steps, learning rate 0.02) on all phases of the lossy forward model, starting from the default phases |
| place_local+retune | Both |

Relabelling assumes interchangeable sources and detectors: the mesh realises U'[i, j] = U[pout[i], pin[j]], which is the same optical map up to port names.
The programming step sees the *calibrated* loss map; results are always evaluated on the *true* map.

## Experiments (seeds 0 to 4; unitary and loss map paired across methods)
1. **Loss sweep:** mean arm loss 0.1 to 2 dB, 8 modes, Haar-random targets, both layouts, all five strategies.
2. **Multi-photon and targets:** 1 dB, Clements, Haar and Fourier targets; strategies also optimised for f_h instead of f_eff, to expose the transmission versus fidelity trade-off.
3. **Scaling:** n = 4 to 16 at 0.5 dB; log-log fit of heralded infidelity and mesh loss in dB against n.
4. **Heat map:** sigma 0 to 1.6 times mean loss 0.25 to 2 dB, relabelling gain.
5. **Ablation:** which side of the mesh may be permuted (input, output, both), random versus hill climb, and search budget K = 10 to 300.
6. **Robustness:** the loss map used for programming is the true map multiplied by log-normal error (sigma_cal 0 to 2).

## Compute
Everything runs on a laptop CPU in under 15 minutes (`python experiments/run_all.py`), with `--quick` as a one-minute smoke run.
