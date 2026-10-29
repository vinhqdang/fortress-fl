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

# S-IV: recovery policies (rv1), with early-stopped retraining and reveal cost
d = R("rv1_energy")
d["tot_j"] = (d.energy_mj + d.reveal_bytes * 8 * 1e-7 * 1e3) / 1e3
names = {"synthetic": "Synthetic", "har": "HAR", "nbaiot": "N-BaIoT"}
pol_names = {"retrain": "retrain (60)", "retrain_early": "retrain early", "ladder": "ladder",
             "icr_q1.0": "ICR", "icr_q0.5": "ICR q=0.5", "icr_q0.25": "ICR q=0.25"}
rows = []
for (ds, mo), g0 in d.groupby(["ds", "model"], sort=False):
    for p_ in pol_names:
        g = g0[g0.policy == p_]
        rows.append([names[ds], mo, pol_names[p_], irn(g), f"{g.acc.mean():.3f}", f"{g.rounds.mean():.1f}",
                     f"{g.tot_j.mean():.2f}", f"{g.tot_j.max():.2f}",
                     "--" if g.worst_dev_mj.isna().all() else f"{g.worst_dev_mj.mean() / 1e3:.3f}"])
tab("Recovery policies (5 seeds). Energy includes the canary reveal. U counts undetermined and unpowered. Last columns: worst run, and worst single device.",
    "tab:s_icr", ["Data", "Model", "Policy", "I/U/R", "Acc.", "Rounds", "Mean J", "Max J", "Dev. J"], rows, "@{}lllcccrrr@{}")

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

# ---------------------------------------------------------------- additional tables
d = R("rv1_energy")
rows = []
for (ds, mo), g0 in d.groupby(["ds", "model"], sort=False):
    for p_ in pol_names:
        g = g0[g0.policy == p_]
        rows.append([names[ds], mo, pol_names[p_], f"{g.bytes.mean() / 1e6:.2f}", f"{g.flops.mean() / 1e9:.2f}",
                     f"{g.reveal_bytes.mean() / 1e6:.2f}", f"{g.stored_updates_mb.mean():.1f}",
                     f"{g.entangle_flops.mean() / 1e9:.3f}"])
tab("Energy accounting (5 seeds): traffic, computation, canary reveal traffic, server storage of stored updates, and entanglement-score computation.",
    "tab:s_acct", ["Data", "Model", "Policy", "MB", "GFLOP", "Reveal MB", "Stored MB", "Entangle GFLOP"], rows, "@{}lllrrrrr@{}")

rows = []
for (ds, mo), g in d.groupby(["ds", "model"], sort=False):
    vals = {}
    for ef in (1e-10, 1e-9, 1e-8):
        for eb in (1e-8, 1e-7, 1e-6):
            m = {q: ((x.flops * ef + 8 * (x.bytes + x.reveal_bytes) * eb).mean()) for q, x in g.groupby("policy")}
            vals.setdefault("rt", []).append(1 - m["icr_q1.0"] / m["retrain"])
            vals.setdefault("ea", []).append(1 - m["icr_q1.0"] / m["retrain_early"])
            vals.setdefault("ea25", []).append(1 - m["icr_q0.25"] / m["retrain_early"])
    rows.append([names[ds], mo] + [f"{min(v) * 100:.0f} to {max(v) * 100:.0f}" for v in (vals["rt"], vals["ea"], vals["ea25"])])
tab("Sensitivity of the energy savings (percent) to $e_{\\rm flop}\\in[0.1,10]$~nJ and $e_{\\rm bit}\\in[10,1000]$~nJ/bit. Negative: ICR costs more.",
    "tab:s_sens", ["Data", "Model", "ICR vs retrain (60)", "ICR vs early", "ICR q=0.25 vs early"], rows, "@{}llccc@{}")

d = R("rv3_forgetting")
rows = []
for (ds, mo), g0 in d.groupby(["ds", "model"], sort=False):
    for m in ["none", "subtract", "subtract+5", "subtract+10", "subtract+20", "icr_q1.0", "icr_q0.25", "retrain_early", "retrain_seed3"]:
        g = g0[g0.method == m]
        rows.append([names[ds], mo, m, irn(g), f"{g.gap.mean():.2f}", f"{g.disagree.mean():.3f}", f"{g.loss_ratio.mean():.2f}", f"{g.acc.mean():.3f}"])
tab("Independent forgetting criteria (5 seeds): parameter distance to a retrained model in units of the retraining noise floor, prediction disagreement with it, loss on the revoked clients' real records relative to it, test accuracy.",
    "tab:s_forget", ["Data", "Model", "Method", "I/U/R", "Gap", "Disagree", "Loss ratio", "Acc."], rows, "@{}lllcrrrr@{}")

d = R("rv2_beacon")
rows = []
for (ds, mo, m), g in d.groupby(["ds", "model", "method"], sort=False):
    rows.append([names[ds], mo, m, len(g), irn(g)])
tab("Calibration with an unpredictable (OS entropy) reveal order, $T=30$.", "tab:s_beacon", ["Data", "Model", "State", "Runs", "I/U/R"], rows, "@{}lllcc@{}")

d = R("rv5_dynamics")
rows = []
for (ds, mo, v), g0 in d.groupby(["ds", "model", "variant"], sort=False):
    rt = g0[g0.policy == "retrain"].energy_mj.mean()
    ic = g0[g0.policy == "icr"]
    rb = g0[g0.policy == "rollback"]
    rows.append([names[ds], mo, v, irn(g0[g0.policy == "retrain"]), irn(ic), f"{ic.rounds.mean():.0f}",
                 f"{100 * ic.energy_mj.mean() / rt:.0f}", f"{rb.ckpt.mean():.0f}", f"{rb.rounds.mean():.0f}"])
tab("Dropout, asynchrony and staggered enrolment (5 seeds): certificate outcomes, ICR rounds, ICR energy as percent of retraining, rollback checkpoint and rollback rounds.",
    "tab:s_dyn", ["Data", "Model", "Variant", "Retrain I/U/R", "ICR I/U/R", "ICR rounds", "ICR \\%", "Ckpt", "Rollback rounds"], rows, "@{}lllccrrrr@{}")

d = R("rv6_sharded")
rows = []
for (ds, mo, w), g in d.groupby(["ds", "model", "wave"], sort=False):
    rows.append([names[ds], mo, w, f"{g.n_wave.mean():.0f}", f"{g.shards_touched.mean():.0f}", f"{g.energy_sharded_mj.mean() / 1e3:.2f}",
                 f"{g.energy_mono_retrain_mj.mean() / 1e3:.2f}", f"{g.acc_sharded.mean():.3f}", f"{g.acc_mono_retrain.mean():.3f}",
                 irn(g.rename(columns={'status_sharded': 'status'}))])
tab("Cohort-sharded training (4 cohorts, 5 seeds): wave inside one cohort or spread.", "tab:s_shard",
    ["Data", "Model", "Wave", "Clients", "Cohorts", "Sharded J", "Mono J", "Acc. sh.", "Acc. mono", "I/U/R"], rows, "@{}lllrrrrrrc@{}")

d = R("rv6b_granular")
rows = []
for (ds, mo, k, m), g in d.groupby(["ds", "model", "kind", "method"], sort=False):
    if k == "class":
        rows.append([names[ds], mo, k, m, f"forget acc {g.forget_acc.mean():.3f}, retain acc {g.retain_acc.mean():.3f}", f"{g.rounds.mean():.0f}", f"{g.ckpt.mean():.0f}"])
    else:
        rows.append([names[ds], mo, k, m, f"gap {g.gap.mean():.2f}, acc {g.acc.mean():.3f}", f"{g.rounds.mean():.0f}", f"{g.ckpt.mean():.0f}"])
tab("Class-level (mirai\\_udp on N-BaIoT) and sample-level (25\\% of the largest client) requests (5 seeds).", "tab:s_gran",
    ["Data", "Model", "Kind", "Method", "Result", "Rounds", "Ckpt"], rows, "@{}lllllrr@{}")

d = R("rv7_attacks")
rows = []
for (ds, ck), g in d.groupby(["ds", "ckind"], sort=False):
    rows.append([names[ds], ck, f"{g.detect_auc.mean():.2f}", f"{g.filter_recall.mean():.2f}",
                 f"{(g.none_honest == 'REVOKED').sum()}/{len(g)}", f"{(g.status_after_filter == 'ISSUED').sum()}/{len(g)}",
                 f"{(g.none_cherry_drop == 'ISSUED').sum()}/{len(g)}", f"{(g.none_cherry_worst == 'ISSUED').sum()}/{len(g)}"])
tab("Attacks on the audit (MLP, 5 seeds): detector AUC, share of canaries filtered, un-unlearned model flagged honestly, certified after filtering, certified after a sign-balanced selective reveal under the accept-partial rule and under the strict rule.",
    "tab:s_att", ["Data", "Canary", "AUC", "Recall", "Flagged", "Filter cert.", "Cherry (accept)", "Cherry (strict)"], rows, "@{}llcccccc@{}")

d = R("rv4_anchor")
rows = [[int(r.interval), f"{r.chain_only_detect:.2f}", f"{r.witness_detect:.2f}", int(r.anchors), int(r.receipt_bytes)]
        for r in d.itertuples() if r.interval > 0]
tab("Rewrite-and-relink attack on the ledger (300 trials per interval, 3 witnesses). Ed25519: 200/200 valid, 200/200 tampered certificates rejected.",
    "tab:s_anchor", ["Anchor every", "Chain only", "With witnesses", "Anchors", "Receipt bytes"], rows, "@{}ccccc@{}")

# confidence intervals for headline rates
from scipy.stats import beta
def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return f"{k}/{n}", f"{100 * k / n:.1f}", f"{100 * lo:.1f}--{100 * hi:.1f}"
rv2 = R("rv2_beacon"); r7 = R("rv7_attacks"); r1 = R("rv1_energy")
items = [("Exact retrains flagged, seeded order", 4, 280), ("Exact retrains flagged, random order", int(((rv2.method == "retrain") & (rv2.status == "REVOKED")).sum()), int((rv2.method == "retrain").sum())),
         ("Early-stopped retrains flagged", int(((r1.policy == "retrain_early") & (r1.status == "REVOKED")).sum()), int((r1.policy == "retrain_early").sum())),
         ("Un-unlearned flagged, synthetic+HAR, seeded", 160, 160),
         ("Un-unlearned flagged, synthetic+HAR MLP, random", int(((rv2.method == "none") & (rv2.status == "REVOKED") & (rv2.ds != "nbaiot")).sum()), int(((rv2.method == "none") & (rv2.ds != "nbaiot")).sum())),
         ("Un-unlearned flagged, N-BaIoT MLP, random", int(((rv2.method == "none") & (rv2.status == "REVOKED") & (rv2.ds == "nbaiot")).sum()), int(((rv2.method == "none") & (rv2.ds == "nbaiot")).sum())),
         ("Selective reveal certified, accept-partial", int((r7.none_cherry_drop == "ISSUED").sum()), len(r7)),
         ("Selective reveal certified, strict", int((r7.none_cherry_worst == "ISSUED").sum()), len(r7)),
         ("Filter attack certified, Gaussian", int(((r7.ckind == "gaussian") & (r7.status_after_filter == "ISSUED")).sum()), int((r7.ckind == "gaussian").sum())),
         ("Filter attack certified, crossover", int(((r7.ckind == "crossover") & (r7.status_after_filter == "ISSUED")).sum()), int((r7.ckind == "crossover").sum()))]
for pol in ("retrain", "retrain_early", "icr_q1.0", "icr_q0.25"):
    for (ds, mo), g in r1.groupby(["ds", "model"], sort=False):
        x = g[g.policy == pol]
        items.append((f"{pol} certified, {names[ds]} {mo}", int((x.status == "ISSUED").sum()), len(x)))
rows = [[a] + list(cp(k, n)) for a, k, n in items]
tab("Clopper--Pearson 95\\% intervals for headline rates.", "tab:s_ci", ["Quantity", "k/n", "\\%", "95\\% CI"], rows, "@{}lrrr@{}")

open("paper/supp_tables.tex", "w").write("\n".join(out))
print("ok", len(out))
