"""Reproduce every result, table and figure: ``python experiments/run_all.py`` (add --quick for a smoke run)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from meshplace.losses import dft_unitary, haar_unitary, perturb_loss, sample_loss  # noqa: E402
from meshplace.study import evaluate  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"


def cfg(quick: bool) -> dict:
    """Experiment sizes: the quick profile is a smoke test, the full profile is what the README reports."""
    if quick:
        return dict(seeds=2, n=6, budget=20, steps=40, arms=[0.5, 2.0], sizes=[4, 6], sigmas=[0.0, 0.8],
                    heat_arms=[0.5, 2.0], ks=[10, 20], cals=[0.0, 1.0], photons=(2,))
    return dict(seeds=5, n=8, budget=100, steps=200, arms=[0.1, 0.25, 0.5, 1.0, 2.0],
                sizes=[4, 6, 8, 10, 12, 16], sigmas=[0.0, 0.4, 0.8, 1.2, 1.6],
                heat_arms=[0.25, 0.5, 1.0, 2.0], ks=[10, 30, 100, 300], cals=[0.0, 0.25, 0.5, 1.0, 2.0],
                photons=(2, 3))


def instance(seed: int, n: int, arm: float, sigma: float, target: str = "haar"):
    """Paired (target unitary, true loss map) instance for a seed."""
    u = haar_unitary(n, np.random.default_rng(1000 + seed)) if target == "haar" else dft_unitary(n)
    return u, sample_loss(n, arm, sigma, np.random.default_rng(2000 + seed))


def run(quick: bool) -> None:
    c = cfg(quick)
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    t0 = time.time()
    S, N, K, ST = c["seeds"], c["n"], c["budget"], c["steps"]
    pre = "quick_" if quick else ""

    # E1: loss sweep, all methods, both topologies, Haar targets
    rows = []
    for arm in c["arms"]:
        for s in range(S):
            u, loss = instance(s, N, arm, 0.8)
            for top in ("clements", "reck"):
                for m in ("default", "place_random", "place_local", "retune", "place_local+retune"):
                    r = evaluate(u, top, m, loss, None, K, np.random.default_rng(s), ST)
                    rows.append(dict(arm_db=arm, seed=s, topology=top, method=m, **r))
    pd.DataFrame(rows).to_csv(RES / f"{pre}loss_sweep.csv", index=False)
    print(f"[E1] loss sweep done {time.time()-t0:.0f}s", flush=True)

    # E2: multi-photon check and DFT target at the reference loss
    rows = []
    ref = 1.0 if not quick else 2.0
    for target in ("haar", "dft"):
        for s in range(S):
            u, loss = instance(s, N, ref, 0.8, target)
            for obj in ("f_eff", "f_h"):
                for m in ("default", "place_local", "retune", "place_local+retune"):
                    if m == "default" and obj == "f_h":
                        continue
                    r = evaluate(u, "clements", m, loss, None, K, np.random.default_rng(s), ST, c["photons"],
                                 objective=obj)
                    rows.append(dict(target=target, seed=s, method=m, objective=obj, **r))
    pd.DataFrame(rows).to_csv(RES / f"{pre}multiphoton.csv", index=False)
    print(f"[E2] multiphoton done {time.time()-t0:.0f}s", flush=True)

    # E3: scaling with mesh size (no retuning)
    rows = []
    for n in c["sizes"]:
        for s in range(S):
            u, loss = instance(s, n, 0.5, 0.8)
            for top in ("clements", "reck"):
                for m in ("default", "place_local"):
                    r = evaluate(u, top, m, loss, None, K, np.random.default_rng(s))
                    rows.append(dict(n=n, seed=s, topology=top, method=m, **r))
    pd.DataFrame(rows).to_csv(RES / f"{pre}scaling.csv", index=False)
    print(f"[E3] scaling done {time.time()-t0:.0f}s", flush=True)

    # E4: heterogeneity x mean loss heat map (placement gain)
    rows = []
    for arm in c["heat_arms"]:
        for sg in c["sigmas"]:
            for s in range(S):
                u, loss = instance(s, N, arm, sg)
                base = evaluate(u, "clements", "default", loss, None, K, np.random.default_rng(s))
                pl = evaluate(u, "clements", "place_local", loss, None, K, np.random.default_rng(s))
                rows.append(dict(arm_db=arm, sigma=sg, seed=s, f_eff_default=base["f_eff"],
                                 f_eff_place=pl["f_eff"], f_h_default=base["f_h"], f_h_place=pl["f_h"]))
    pd.DataFrame(rows).to_csv(RES / f"{pre}heatmap.csv", index=False)
    print(f"[E4] heat map done {time.time()-t0:.0f}s", flush=True)

    # E5: ablation of the placement search (moves, strategy, budget)
    rows = []
    for s in range(S):
        u, loss = instance(s, N, 1.0, 0.8)
        base = evaluate(u, "clements", "default", loss, None, K, np.random.default_rng(s))
        rows.append(dict(seed=s, kind="default", variant="default", budget=0, **base))
        for mv in ("in", "out", "both"):
            for m in ("place_random", "place_local"):
                r = evaluate(u, "clements", m, loss, None, K, np.random.default_rng(s), move=mv)
                rows.append(dict(seed=s, kind="moves", variant=f"{m}/{mv}", budget=K, **r))
        for k in c["ks"]:
            for m in ("place_random", "place_local"):
                r = evaluate(u, "clements", m, loss, None, k, np.random.default_rng(s))
                rows.append(dict(seed=s, kind="budget", variant=m, budget=k, **r))
    pd.DataFrame(rows).to_csv(RES / f"{pre}ablation.csv", index=False)
    print(f"[E5] ablation done {time.time()-t0:.0f}s", flush=True)

    # E6: robustness to a mis-calibrated loss map
    rows = []
    for cal in c["cals"]:
        for s in range(S):
            u, loss = instance(s, N, 1.0, 0.8)
            est = perturb_loss(loss, cal, np.random.default_rng(3000 + s))
            for m in ("default", "place_local", "retune", "place_local+retune"):
                r = evaluate(u, "clements", m, loss, est, K, np.random.default_rng(s), ST)
                rows.append(dict(sigma_cal=cal, seed=s, method=m, **r))
    pd.DataFrame(rows).to_csv(RES / f"{pre}calibration.csv", index=False)
    print(f"[E6] calibration done {time.time()-t0:.0f}s", flush=True)

    meta = dict(quick=quick, seconds=round(time.time() - t0, 1), config=c)
    (RES / f"{pre}run_meta.json").write_text(json.dumps(meta, indent=2))

    from make_figures import make_all
    make_all(quick)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (< 1 min)")
    run(ap.parse_args().quick)
