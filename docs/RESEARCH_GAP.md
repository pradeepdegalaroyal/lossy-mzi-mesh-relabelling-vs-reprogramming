# Research gap: how much of a lossy photonic mesh can software recover?

## State of the art (from the supplied literature)

Programmable meshes of Mach-Zehnder interferometers (MZIs) are the workhorse of linear-optical processing.
Clements et al., 2016 (doi:10.1364/OPTICA.3.001460) replaced the triangular Reck layout by a rectangular one with half the
optical depth and better tolerance to loss; Wang et al., 2020 (doi:10.1038/s41566-019-0532-1) and Pelucchi et al., 2022
(doi:10.1038/s42254-021-00398-z) describe how such circuits now approach 1000 components, and both name loss and manufacturing
infrastructure as the roadblocks. Recent work moves from single devices to system-level questions: Yousif et al., 2027
(doi:10.1007/978-3-032-39708-9_27) maps measured device loss to cloud throughput and tail latency, and Di Bartolo et al., 2026
(doi:10.1038/s41534-026-01236-9) shows that a reconfigurable circuit with two indistinguishable photons computes better than one with
distinguishable photons, which makes multi-photon statistics the quantity that matters downstream. Hybrid platforms
(Elshaari et al., 2020, doi:10.1038/s41566-020-0609-x) and thin-film lithium niobate (Wang et al., 2018, doi:10.1038/s41586-018-0551-y)
keep lowering the loss per device, but the reviews (Nusyura et al., 2026, doi:10.1007/s42484-026-00402-8) still list loss management
as an open bottleneck.

## Related work

| Paper | Year | Role | What it does | What it does NOT cover (judged from the abstract) |
|---|---|---|---|---|
| Clements et al., doi:10.1364/OPTICA.3.001460 | 2016 | landmark | Rectangular mesh, half the depth of Reck, more loss-robust | Loss is treated as a design property of the layout, not as a measured per-device map that the programming can use |
| Wang et al., doi:10.1038/s41566-019-0532-1 | 2020 | landmark | Review of integrated photonic quantum technology up to about 1000 components | No loss-aware programming strategy |
| Elshaari et al., doi:10.1038/s41566-020-0609-x | 2020 | landmark | Hybrid integration to overcome monolithic limits | Does not quantify what remains after calibration and software compensation |
| Wang et al., doi:10.1038/s41586-018-0551-y | 2018 | landmark | Lithium niobate modulators with below 0.5 dB on-chip loss | Device-level; no mesh-level consequence of residual loss spread |
| Zhong et al., doi:10.1126/science.abe8770 | 2020 | landmark | 100-mode ultralow-loss interferometer for Gaussian boson sampling | Relies on hardware loss, not on loss-aware programming |
| Flamini et al., doi:10.1088/1361-6633/aad5b2 | 2019 | review | Review of photonic quantum information processing | No quantified loss budget per software lever |
| Slussarenko and Pryde, doi:10.1063/1.5115814 | 2019 | review | Concise review, notes noise-tolerant approaches | Same |
| Pelucchi et al., doi:10.1038/s42254-021-00398-z | 2022 | review | Roadmap for integrated quantum photonics | Same |
| Nusyura et al., doi:10.1007/s42484-026-00402-8 | 2026 | recent | Review that lists loss management and large-scale integration as challenges | No quantitative study |
| Yousif et al., doi:10.1007/978-3-032-39708-9_27 | 2027 | recent | Device loss to cloud throughput and latency | Takes loss as given; does not ask how programming can lower it |
| Di Bartolo et al., doi:10.1038/s41534-026-01236-9 | 2026 | recent | Two-photon reservoir on a reconfigurable circuit | No study of how mesh loss distorts the multi-photon output |
| Bankwitz et al., doi:10.1038/s44310-026-00129-1 | 2026 | recent | Thin-film lithium niobate modulator with more than 50 dB extinction | Single device; no mesh |

## The gap

All the mesh designs above are compared for *identical, lossless-at-design-time* devices or with loss as one scalar. A fabricated mesh has a
**measured per-device loss map**, and in a thermo-optic or carrier-based MZI the loss sits in the internal phase-shifter arm, so the
transmission of a device depends on the phase it is programmed to. Two software degrees of freedom are then free in principle:
(i) relabelling input and output ports (if sources and detectors are interchangeable, the mesh may realise a permuted target), and
(ii) re-optimising all phases on the lossy device. Nothing in the supplied literature quantifies how much loss-inclusive fidelity either
lever recovers, whether it survives an imperfect loss calibration, or what it does to multi-photon statistics.

**Novelty in one sentence:** a simulation study of Reck and Clements meshes with phase-dependent per-device loss that measures,
against the same lossy hardware, how much a calibrated-loss-aware relabelling and phase re-optimisation recover, how that scales with mesh size, and what it costs in heralded multi-photon fidelity.

## Hypotheses and success metric (set after one debugging instance, before the sweep)

Scale: classical simulation of 4 to 16 modes, up to three photons; no hardware, no photon distinguishability or source error.

* **H1.** At 1 dB mean arm loss and 8 modes, a swap hill climb with a budget of K = 100 mesh evaluations raises the loss-inclusive fidelity
  f_eff of the Clements mesh by at least 1% relative to default programming, and does so more than best-of-K random relabelling at the same K.
* **H2.** Re-optimising phases recovers more than relabelling does.
* **H3.** An improvement in f_eff carries over to the heralded three-photon fidelity.

Metric: relative change of f_eff = |tr(U†T)|² / n² (T is the lossy transfer matrix, U the target), mean ± standard deviation over 5 seeds,
paired with the default programming on the same unitary and loss map. A hypothesis is rejected if the mean paired change is below its
threshold or below its comparator.
