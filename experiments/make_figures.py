"""Summary tables and figures (black/grey only, 200 dpi) from the CSV files in results/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
DPI = 200
STYLE = {  # method -> (marker, linestyle, grey, label)
    "default": ("o", "-", "0.0", "default programming"),
    "place_random": ("s", ":", "0.55", "relabel: best of K random"),
    "place_local": ("^", "--", "0.25", "relabel: swap hill climb"),
    "retune": ("D", "-.", "0.4", "reprogram phases"),
    "place_local+retune": ("v", "-", "0.15", "relabel + reprogram"),
}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.6,
                     "axes.edgecolor": "0.2", "figure.dpi": 100})


def ms(x: pd.Series) -> tuple[float, float]:
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def paired_gain(df: pd.DataFrame, keys: list[str], metric: str, rel: bool = True) -> pd.DataFrame:
    """Per-seed change of ``metric`` of each method relative to the default programming."""
    base = df[df.method == "default"].set_index(keys + ["seed"])[metric]
    d = df.set_index(keys + ["seed"])
    d = d.assign(base=base.reindex(d.index).values)
    d["gain"] = (d[metric] / d["base"] - 1) * 100 if rel else (d[metric] - d["base"]) * 100
    return d.reset_index()


def fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def fig_loss_sweep(pre: str) -> dict:
    df = pd.read_csv(RES / f"{pre}loss_sweep.csv")
    summ = df.groupby(["topology", "arm_db", "method"]).agg(
        f_eff_mean=("f_eff", "mean"), f_eff_std=("f_eff", "std"), eta_mean=("eta", "mean"),
        eta_std=("eta", "std"), f_h_mean=("f_h", "mean"), f_h_std=("f_h", "std")).reset_index()
    summ.to_csv(RES / f"{pre}loss_sweep_summary.csv", index=False)
    g_eff = paired_gain(df, ["topology", "arm_db"], "f_eff", rel=True)
    g_h = paired_gain(df, ["topology", "arm_db"], "f_h", rel=False)
    fig, ax = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex=True, sharey="row")
    for j, top in enumerate(["clements", "reck"]):
        for i, (g, ylab) in enumerate([(g_eff, "change in f_eff (%)"),
                                       (g_h, "change in f_h (pp)")]):
            a = ax[i, j]
            for m, (mk, ls, col, lab) in STYLE.items():
                if m == "default":
                    continue
                s = g[(g.topology == top) & (g.method == m)].groupby("arm_db")["gain"].agg(["mean", "std"])
                a.errorbar(s.index, s["mean"], yerr=s["std"], marker=mk, ls=ls, color=col, ms=5, lw=1.2,
                           capsize=2, label=lab)
            a.axhline(0, color="k", lw=0.8)
            a.set_xscale("log")
            if j == 0:
                a.set_ylabel(ylab)
            if i == 0:
                a.set_title(f"{top.capitalize()} mesh")
            if i == 1:
                a.set_xlabel("mean internal-arm loss per device (dB)")
    ax[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Software levers on a lossy 8-mode mesh: gains are small and trade fidelity for transmission")
    fig.subplots_adjust(left=0.12)
    fig.tight_layout()
    fig.savefig(FIG / f"{pre}fig1_loss_sweep.png", dpi=DPI)
    plt.close(fig)
    return {}


def fig_scaling(pre: str) -> dict:
    df = pd.read_csv(RES / f"{pre}scaling.csv")
    df["inf_h"] = 1 - df["f_h"]
    df["loss_db"] = -10 * np.log10(df["eta"])
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.8))
    slopes = {}
    for k, (col, ylab, ttl) in enumerate([("inf_h", "heralded infidelity 1 - f_h", "Distortion"),
                                          ("loss_db", "mean mesh loss (dB)", "Transmission loss")]):
        for (top, m), (mk, ls, gr) in {("clements", "default"): ("o", "-", "0.0"),
                                       ("clements", "place_local"): ("^", "--", "0.35"),
                                       ("reck", "default"): ("s", "-", "0.5"),
                                       ("reck", "place_local"): ("v", "--", "0.7")}.items():
            s = df[(df.topology == top) & (df.method == m)].groupby("n")[col].agg(["mean", "std"])
            slope = fit_slope(s.index.values.astype(float), s["mean"].values)
            slopes[f"{ttl}|{top}|{m}"] = slope
            ax[k].errorbar(s.index, s["mean"], yerr=s["std"], marker=mk, ls=ls, color=gr, ms=5, capsize=2,
                           label=f"{top}, {'default' if m == 'default' else 'relabelled'} (slope {slope:.2f})")
        ax[k].set_xscale("log")
        ax[k].set_yscale("log")
        ax[k].set_xlabel("number of modes n")
        ax[k].set_ylabel(ylab)
        ax[k].set_title(ttl)
        ax[k].legend(fontsize=7)
    fig.suptitle("Scaling at 0.5 dB mean arm loss: fitted log-log slopes in legend")
    fig.tight_layout()
    fig.savefig(FIG / f"{pre}fig2_scaling.png", dpi=DPI)
    plt.close(fig)
    return slopes


def fig_heatmap(pre: str) -> None:
    df = pd.read_csv(RES / f"{pre}heatmap.csv")
    df["gain_eff"] = (df.f_eff_place / df.f_eff_default - 1) * 100
    df["d_h"] = (df.f_h_place - df.f_h_default) * 100
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for a, col, ttl in [(ax[0], "gain_eff", "Loss-inclusive fidelity change (%)"),
                        (ax[1], "d_h", "Heralded fidelity change (percentage points)")]:
        p = df.groupby(["arm_db", "sigma"])[col].mean().unstack()
        im = a.imshow(p.values, cmap="Greys_r" if col == "d_h" else "Greys", aspect="auto", origin="lower")
        a.set_xticks(range(len(p.columns)), [f"{v:g}" for v in p.columns])
        a.set_yticks(range(len(p.index)), [f"{v:g}" for v in p.index])
        a.set_xlabel("device-to-device spread of arm loss (log-normal sigma)")
        a.set_ylabel("mean internal-arm loss (dB)")
        a.set_title(ttl, fontsize=9)
        a.grid(False)
        lo, hi = np.nanmin(p.values), np.nanmax(p.values)
        for (r, c), v in np.ndenumerate(p.values):
            shade = (v - lo) / (hi - lo) if hi > lo else 0.0
            shade = 1 - shade if col == "d_h" else shade  # 0 = white cell
            a.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8,
                   color="white" if shade > 0.55 else "black")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("Where does relabelling help? Mean over seeds, 8-mode Clements mesh")
    fig.tight_layout()
    fig.savefig(FIG / f"{pre}fig3_heatmap.png", dpi=DPI)
    plt.close(fig)


def fig_ablation(pre: str) -> None:
    ab = pd.read_csv(RES / f"{pre}ablation.csv")
    base = ab[ab.kind == "default"].set_index("seed")["f_eff"]
    ab["gain"] = (ab["f_eff"] / ab["seed"].map(base) - 1) * 100
    cal = pd.read_csv(RES / f"{pre}calibration.csv")
    cb = cal[cal.method == "default"].set_index(["sigma_cal", "seed"])["f_eff"]
    cal["gain"] = (cal.f_eff.values / cb.reindex(pd.MultiIndex.from_arrays([cal.sigma_cal, cal.seed])).values - 1) * 100
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.7))
    mv = ab[ab.kind == "moves"].groupby("variant")["gain"].agg(["mean", "std"]).sort_index()
    for i, (v, r) in enumerate(mv.iterrows()):
        ax[0].errorbar(r["mean"], i, xerr=r["std"], marker="s" if "random" in v else "^", color="0.15", capsize=3)
    ax[0].set_yticks(range(len(mv)), mv.index)
    ax[0].set_xlabel("change in loss-inclusive fidelity (%)")
    ax[0].set_title("(a) which ports may move")
    for m, (mk, ls, col, lab) in STYLE.items():
        if m in ("place_random", "place_local"):
            s = ab[(ab.kind == "budget") & (ab.variant == m)].groupby("budget")["gain"].agg(["mean", "std"])
            ax[1].errorbar(s.index, s["mean"], yerr=s["std"], marker=mk, ls=ls, color=col, capsize=2, label=lab)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("search budget K (mesh evaluations)")
    ax[1].set_ylabel("change in loss-inclusive fidelity (%)")
    ax[1].set_title("(b) search budget")
    ax[1].legend(fontsize=7)
    for m, (mk, ls, col, lab) in STYLE.items():
        if m == "default":
            continue
        if m in ("place_random",):
            continue
        s = cal[cal.method == m].groupby("sigma_cal")["gain"].agg(["mean", "std"])
        ax[2].errorbar(s.index, s["mean"], yerr=s["std"], marker=mk, ls=ls, color=col, capsize=2, label=lab)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xlabel("loss-map calibration error (log-normal sigma)")
    ax[2].set_ylabel("change in loss-inclusive fidelity (%)")
    ax[2].set_title("(c) mis-calibrated loss map")
    ax[2].legend(fontsize=7)
    fig.suptitle("Ablation and robustness (8-mode Clements mesh, 1 dB mean arm loss, evaluated on the true map)")
    fig.tight_layout()
    fig.savefig(FIG / f"{pre}fig4_ablation_robustness.png", dpi=DPI)
    plt.close(fig)


def fig_multiphoton(pre: str) -> dict:
    df = pd.read_csv(RES / f"{pre}multiphoton.csv")
    cols = [c for c in df.columns if c.startswith("fid")]
    ks = sorted(int(c[3:]) for c in cols)
    df["label"] = df.method + np.where(df.objective == "f_h", " [obj f_h]", "")
    summ = df.groupby(["target", "label"])[["f_eff", "eta", "f_h"] + [f"fid{k}" for k in ks]
                                             + [f"p_ratio{k}" for k in ks]].agg(["mean", "std"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.reset_index().to_csv(RES / f"{pre}multiphoton_summary.csv", index=False)
    base = df[df.method == "default"].set_index(["target", "seed"])
    fig, ax = plt.subplots(1, len(ks) + 1, figsize=(3.7 * (len(ks) + 1), 3.8))
    mk = {"place_local": "^", "retune": "D", "place_local+retune": "v"}
    names = {"place_local": "relabel", "retune": "reprogram", "place_local+retune": "relabel + reprogram"}
    pairs = [("eta", "f_h", "single photon: change in eta (%)", "change in f_h (pp)")] + [
        (f"p_ratio{k}", f"fid{k}", f"{k}-photon: change in success probability (%)",
         f"change in {k}-photon fidelity (pp)") for k in ks]
    for i, (xc, yc, xl, yl) in enumerate(pairs):
        for (m, obj), g in df[df.method != "default"].groupby(["method", "objective"]):
            dx = (g[xc].values / base.loc[list(zip(g.target, g.seed)), xc].values - 1) * 100
            dy = (g[yc].values - base.loc[list(zip(g.target, g.seed)), yc].values) * 100
            ax[i].errorbar(dx.mean(), dy.mean(), xerr=dx.std(ddof=1), yerr=dy.std(ddof=1), marker=mk[m],
                           color="0.1" if obj == "f_eff" else "0.6", mfc="0.1" if obj == "f_eff" else "white",
                           ms=7, capsize=2, ls="none",
                           label=f"{names[m]}, objective {obj}")
        ax[i].axhline(0, color="k", lw=0.8)
        ax[i].axvline(0, color="k", lw=0.8)
        ax[i].set_xlabel(xl)
        ax[i].set_ylabel(yl)
    ax[0].legend(fontsize=6, loc="lower left")
    fig.suptitle("Paired change versus default programming (10 instances: Haar and Fourier targets, 1 dB)")
    fig.tight_layout()
    fig.savefig(FIG / f"{pre}fig5_tradeoff.png", dpi=DPI)
    plt.close(fig)
    return {}


def key_numbers(pre: str) -> dict:
    """Headline numbers quoted in the README (paired changes versus default programming, mean and std)."""
    out: dict = {}
    ls = pd.read_csv(RES / f"{pre}loss_sweep.csv")
    for metric, rel in (("f_eff", True), ("eta", True), ("f_h", False)):
        g = paired_gain(ls, ["topology", "arm_db"], metric, rel=rel)
        for (top, arm, m), x in g[g.method != "default"].groupby(["topology", "arm_db", "method"]):
            out[f"loss_sweep|{metric}|{top}|{arm:g}dB|{m}"] = [round(float(x.gain.mean()), 3), round(float(x.gain.std(ddof=1)), 3)]
    c = ls[(ls.topology == "clements")].pivot_table(index=["arm_db", "seed"], columns="method", values="f_eff")
    for arm, x in c.groupby(level=0):
        d = (x["place_local"] / x["place_random"] - 1) * 100
        out[f"local_vs_random|f_eff_pct|clements|{arm:g}dB"] = [round(float(d.mean()), 3), round(float(d.std(ddof=1)), 3)]
        d = (x["place_local"] / x["retune"] - 1) * 100
        out[f"local_vs_retune|f_eff_pct|clements|{arm:g}dB"] = [round(float(d.mean()), 3), round(float(d.std(ddof=1)), 3)]
    mp = pd.read_csv(RES / f"{pre}multiphoton.csv")
    base = mp[mp.method == "default"].set_index(["target", "seed"])
    for (m, obj, tgt), g in mp[mp.method != "default"].groupby(["method", "objective", "target"]):
        b = base.loc[[(tgt, sd) for sd in g.seed]]
        for col in [c_ for c_ in mp.columns if c_.startswith(("fid", "p_ratio"))] + ["f_eff", "eta", "f_h"]:
            d = g[col].values - b[col].values
            if col.startswith("p_ratio") or col in ("eta", "f_eff"):
                d = (g[col].values / b[col].values - 1) * 100
            out[f"multiphoton|{col}|{tgt}|{m}|{obj}"] = [round(float(d.mean()), 3), round(float(d.std(ddof=1)), 3)]
    cal = pd.read_csv(RES / f"{pre}calibration.csv")
    cb = cal[cal.method == "default"].set_index(["sigma_cal", "seed"])["f_eff"]
    cal["gain"] = (cal.f_eff.values / cb.reindex(pd.MultiIndex.from_arrays([cal.sigma_cal, cal.seed])).values - 1) * 100
    for (sc, m), x in cal[cal.method != "default"].groupby(["sigma_cal", "method"]):
        out[f"calibration|f_eff_pct|{sc:g}|{m}"] = [round(float(x.gain.mean()), 3), round(float(x.gain.std(ddof=1)), 3)]
    return out


def make_all(quick: bool) -> None:
    pre = "quick_" if quick else ""
    FIG.mkdir(exist_ok=True)
    fig_loss_sweep(pre)
    slopes = fig_scaling(pre)
    fig_heatmap(pre)
    fig_ablation(pre)
    fig_multiphoton(pre)
    (RES / f"{pre}scaling_slopes.json").write_text(json.dumps(slopes, indent=2))
    (RES / f"{pre}key_numbers.json").write_text(json.dumps(key_numbers(pre), indent=1))


if __name__ == "__main__":
    make_all(False)
