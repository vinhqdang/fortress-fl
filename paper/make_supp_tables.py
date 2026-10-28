"""Write the supplementary tables from the CSV results (no hand-typed numbers)."""
import os
import pandas as pd

R = lambda n: pd.read_csv(f"results/unlearning/{n}.csv")
out = []


def tab(caption, label, header, rows, colspec):
    s = ["\\begin{table}[h]", "\\centering", f"\\caption{{{caption}}}\\label{{{label}}}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
         f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    s += [" & ".join(str(c).replace("_", "\\_") for c in r) + " \\\\" for r in rows]
    s += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    out.append("\n".join(s))


def irn(g):
    c = g.status.value_counts()
    return f"{c.get('ISSUED', 0)}/{c.get('UNDETERMINED', 0) + c.get('UNPOWERED', 0)}/{c.get('REVOKED', 0)}"


# S-I: methods, synthetic (e1)
d = R("e1_methods")
order = ["none", "subtract", "replay", "subtract+5", "subtract+10", "subtract+20", "rollback", "retrain"]
rows = []
for m in order:
    g = d[d.method == m]
    rows.append([m, irn(g), f"{g.acc.mean():.3f}", f"{g.delta_upper.mean():.3f}", f"{g.rounds.mean():.0f}",
                 f"{g.comm_mb.mean():.2f}", f"{g.energy_mj.mean() / 1e3:.2f}"])
tab("Synthetic fleet, one wave of 20 of 60 clients, 10 seeds. I/U/R: certificate issued, undetermined, revoked. $\\Delta_{\\rm up}$: mean upper bound on the residual advantage.",
    "tab:s_methods", ["Operator", "I/U/R", "Acc.", "$\\Delta_{\\rm up}$", "Rounds", "MB", "Energy (J)"], rows, "@{}lcccccc@{}")

# S-II: calibration (e7, r3)
rows = []
e7 = R("e7_calibration")
for m in ("none", "retrain"):
    g = e7[e7.method == m]
    rows.append(["synthetic", "linear", m, len(g), irn(g)])
for ds in ("har", "nbaiot"):
    c = R(f"r3_{ds}_calibration")
    for mo in ("linear", "mlp"):
        for m in ("none", "retrain"):
            g = c[(c.model == mo) & (c.method == m)]
            rows.append([ds.upper() if ds == "har" else "N-BaIoT", mo, m, len(g), irn(g)])
tab("Calibration: the un-unlearned model (should be flagged) and the exactly retrained model (should not).",
    "tab:s_calib", ["Data", "Model", "Model state", "Runs", "I/U/R"], rows, "@{}lllcc@{}")

# S-III: real-data methods (r1)
rows = []
for ds in ("har", "nbaiot"):
    d = R(f"r1_{ds}_methods")
    for mo in ("linear", "mlp"):
        for m in ["none", "subtract", "replay", "subtract+5", "subtract+10", "subtract+20", "retrain"]:
            g = d[(d.model == mo) & (d.method == m)]
            rows.append([ds.upper() if ds == "har" else "N-BaIoT", mo, m, irn(g), f"{g.acc.mean():.3f}",
                         f"{g.rounds.mean():.0f}", f"{g.energy_mj.mean() / 1e3:.2f}"])
tab("Real data, a wave of one third of the clients, 5 seeds.", "tab:s_real",
    ["Data", "Model", "Operator", "I/U/R", "Acc.", "Rounds", "Energy (J)"], rows, "@{}lllcccc@{}")

# S-IV: n1 (recovery policies)
d = R("n1_recovery")
names = {"synthetic": "Synthetic", "har": "HAR", "nbaiot": "N-BaIoT"}
rows = []
for (ds, mo), g0 in d.groupby(["ds", "model"], sort=False):
    for p in ["retrain", "ladder", "icr_q1.0", "icr_q0.5", "icr_q0.25"]:
        g = g0[g0.policy == p]
        rows.append([names[ds], mo, p, irn(g), f"{g.acc.mean():.3f}", f"{g.rounds.mean():.1f}",
                     f"{g.energy_mj.mean() / 1e3:.2f}", f"{g.energy_mj.max() / 1e3:.2f}"])
tab("Recovery policies (5 seeds). ``U'' counts undetermined and unpowered. Last column: worst run.",
    "tab:s_icr", ["Data", "Model", "Policy", "I/U/R", "Acc.", "Rounds", "Mean J", "Max J"], rows, "@{}lllcccrr@{}")

# S-V: canary grid (n3)
d = R("n3_canary_power")
rows = []
for (ds, mo, m, sc), g in d.groupby(["ds", "model", "m", "scale"], sort=False):
    rows.append([names[ds], mo, m, f"{sc:g}", f"{(g.status == 'REVOKED').mean():.1f}", f"{g.delta_upper.mean():.2f}", f"{g.acc.mean():.3f}"])
tab("Canary design grid: share of runs (5 seeds) in which the audit flags the un-unlearned model.",
    "tab:s_power", ["Data", "Model", "$m$", "Scale", "Flagged", "$\\Delta_{\\rm up}$", "Acc."], rows, "@{}lllcccc@{}")

# S-VI: scheduler (n4)
d = R("n4_scheduler")
rows = []
for (r, p), g in d.groupby(["rate_per_day", "policy"], sort=False):
    rows.append([f"{r:g}", p, f"{g.requests.mean():.0f}", f"{g.waves.mean():.1f}", f"{g.mean_latency.mean():.1f}",
                 f"{g.max_latency.mean():.1f}", f"{g.energy_per_request_j.mean():.2f}",
                 f"{g.underpowered_waves.sum() / g.waves.sum():.2f}"])
tab("Wave scheduler simulation, one year, 20 runs, deadline 30 days, power target 120 pairs.", "tab:s_sched",
    ["Req./day", "Policy", "Requests", "Waves", "Mean lat.", "Max lat.", "J/request", "Underpowered"], rows, "@{}llccccrc@{}")

# S-VII: baselines (n5)
d = R("n5_baselines")
rows = []
for (ds, m), g in d.groupby(["ds", "method"], sort=False):
    rows.append([names[ds], m, f"{g.marker_asr.mean():.3f}", f"{g.mia_auc.mean():.3f}", irn(g.rename(columns={'cert_status': 'status'}))])
tab("Baselines on the MLP (5 seeds): marker success rate, loss-threshold membership AUC and our certificate outcome.",
    "tab:s_base", ["Data", "Model state", "Marker ASR", "MIA AUC", "Canary I/U/R"], rows, "@{}llccc@{}")

# S-VIII: provenance (e6)
e = R("e6_provenance").mean(numeric_only=True)
rows = [["records (60 rounds)", f"{e.records:.0f}"], ["ledger size (kB)", f"{e.ledger_kb:.0f}"],
        ["bytes per round", f"{e.ledger_bytes_per_round:.0f}"],
        ["ledger / update traffic", f"{e.ledger_over_update_traffic:.3f}"],
        ["chain verification (ms)", f"{e.chain_verify_ms:.1f}"], ["Merkle proof check ($\\mu$s)", f"{e.proof_us:.1f}"],
        ["proof length (hashes)", f"{e.proof_len:.1f}"], ["tampering detected", f"{e.tamper_detected:.2f}"],
        ["checkpoints (kB)", f"{e.checkpoint_kb:.0f}"], ["stored updates (kB)", f"{e.stored_updates_kb:.0f}"]]
tab("Provenance ledger on the synthetic fleet (3 seeds, 200 tamper trials each).", "tab:s_ledger", ["Quantity", "Value"], rows, "@{}lr@{}")

open("paper/supp_tables.tex", "w").write("\n".join(out))
print("ok", len(out))
