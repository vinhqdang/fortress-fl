"""Figure for the recovery policies (n1) and canary power (n3)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3})
out = "results/unlearning"
d = pd.read_csv(f"{out}/n1_recovery.csv")
settings = [("synthetic", "linear", "Synthetic\nlinear"), ("har", "linear", "HAR\nlinear"),
            ("har", "mlp", "HAR\nMLP"), ("nbaiot", "linear", "N-BaIoT\nlinear"),
            ("nbaiot", "mlp", "N-BaIoT\nMLP")]
pol = [("retrain", "retrain"), ("ladder", "ladder"), ("icr_q1.0", "ICR"),
       ("icr_q0.5", "ICR, $q$=0.5"), ("icr_q0.25", "ICR, $q$=0.25")]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.5, 1]})
w = 0.16
for j, (p, lab) in enumerate(pol):
    ys, ws = [], []
    for ds, mo, _ in settings:
        t = d[(d.ds == ds) & (d.model == mo) & (d.policy == p)]
        ys.append(t.energy_mj.mean() / 1e3)
        ws.append((t.status == "ISSUED").mean())
    ax[0].bar(np.arange(len(settings)) + (j - 2) * w, ys, w, label=lab)
ax[0].set_yscale("log"); ax[0].set_ylabel("mean client energy (J)")
ax[0].set_xticks(range(len(settings))); ax[0].set_xticklabels([s[2] for s in settings], fontsize=6.5)
ax[0].legend(fontsize=5.5, ncol=2)
p3 = pd.read_csv(f"{out}/n3_canary_power.csv")
t = p3[(p3.ds == "nbaiot") & (p3.model == "linear")]
for m, ls in ((6, "-"), (12, "--")):
    g = t[t.m == m].groupby("scale").apply(lambda x: (x.status == "REVOKED").mean(), include_groups=False)
    ax[1].plot(g.index, g.values, ls, marker="o", ms=3, label=f"$m$={m}")
ax[1].set_xscale("log", base=2); ax[1].set_ylim(-0.05, 1.05)
ax[1].set_xlabel("canary feature scale"); ax[1].set_ylabel("audit flags leaky model")
ax[1].set_title("N-BaIoT linear", fontsize=8); ax[1].legend(fontsize=6)
fig.tight_layout(); fig.savefig("figs/new_recovery.png", dpi=220)
