"""
Certified federated unlearning experiments.

Usage:
    python experiments/unlearning_experiments.py [--quick] [--only e1,e2,...]

Each experiment writes a CSV under results/unlearning/ and the figure script
reads those files, so every number in the manuscript can be regenerated.
Reveal order of the audit is drawn from a seed (reproducible); a deployment
would use a public beacon (see the VOUCH verifier).
"""

import argparse
import copy
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fortress_fl.unlearning import (UnlearnableFL, UnlearnRequest, ShardedFL, data,
                                    plant_canaries, certify_model, verify_then_escalate,
                                    sign_certificate, verify_certificate,
                                    verify_merkle_proof)
from fortress_fl.unlearning import verification as V
from fortress_fl.unlearning.provenance import digest_array
from fortress_fl.unlearning.certify import LADDER

BASE = dict(N=60, d=1024, n_real=12, m=6, W=20, alpha=1.0, p=0.6, sigma=0.0, T=60,
            eps=0.2, byz=0, lr=1.0)


def pmap(fn, jobs):
    with Pool(4) as p:
        return p.map(fn, jobs)


def cfgd(**kw):
    c = dict(BASE)
    c.update(kw)
    return c


def build(cfg, seed, poison=0):
    byz = tuple(range(cfg["N"] - cfg["byz"], cfg["N"]))
    clients, test, info = data.build_federation(
        "synthetic", n_clients=cfg["N"], n_per_client=cfg["n_real"], dim=cfg["d"],
        alpha=cfg["alpha"], seed=seed, byz_clients=byz)
    dim = info["dim"]
    man = plant_canaries(clients, cfg["m"], dim, seed=seed)
    # revoked wave: first W non-Byzantine clients
    honest = [c["id"] for c in clients if not c["byzantine"]]
    tg = honest[:cfg["W"]]
    bd = None
    if poison:
        for c in clients:
            if c["id"] in tg[:poison]:
                data.add_backdoor(c, 0.5, seed=seed)
        bd = data.backdoor_test_set(test)
    fl = UnlearnableFL(clients, dim, lr=cfg["lr"], participation=cfg["p"],
                       sigma_dp=cfg["sigma"], seed=seed)
    fl.train(cfg["T"])
    return fl, clients, test, man, tg, bd, dim


def audit_bytes(cfg):
    """Canary reveal: both twins of each pair leave the revoked clients once."""
    return cfg["W"] * cfg["m"] * 2 * (cfg["d"] + 1) * 4


def cert_row(theta, man, tg, cfg, seed, fl_cost, test, name, extra=None):
    c, diffs = certify_model(theta, man, tg, eps=cfg["eps"], alpha=0.05,
                             reveal_source="seeded", seed=seed)
    r = dict(method=name, seed=seed, status=c.status, t_stop=c.t_stop,
             n_pairs=len(diffs), delta_upper=c.delta_upper, log_e_cert=c.log_e_cert,
             log_e_rev=c.log_e_rev, acc=V.accuracy(theta, test),
             comm_mb=fl_cost["bytes"] / 1e6, energy_mj=fl_cost["energy_j"] * 1e3,
             rounds=fl_cost["rounds"])
    r.update({k: v for k, v in cfg.items()})
    if extra:
        r.update(extra)
    return r


# ---------------------------------------------------------------------------
# E1: methods vs certificate, with controls (validity + power)
# ---------------------------------------------------------------------------
def e1_job(args):
    cfg, seed = args
    fl, clients, test, man, tg, _, dim = build(cfg, seed)
    req = UnlearnRequest("client", client=tg[0], clients=tg)
    zero = {"bytes": 0.0, "energy_j": 0.0, "rounds": 0}
    rows = [cert_row(fl.theta, man, tg, cfg, seed, zero, test, "none")]
    for name, m, kw in [("retrain", "retrain", {}), ("rollback", "rollback", {}),
                        ("subtract", "subtract", {}), ("replay", "replay", {}),
                        ("subtract+5", "subtract_recover", {"recover_rounds": 5}),
                        ("subtract+10", "subtract_recover", {"recover_rounds": 10}),
                        ("subtract+20", "subtract_recover", {"recover_rounds": 20})]:
        f = copy.deepcopy(fl)
        out = f.unlearn(req, m, **kw)
        au = f.ledger.audit_exclusion(tg, out["model_hash"])["passed"]
        rows.append(cert_row(out["theta"], man, tg, cfg, seed, out["cost"], test, name,
                             dict(ledger_audit=au, ckpt=out["checkpoint"])))
    return rows


def e1(seeds, out, quick):
    rows = sum(pmap(e1_job, [(cfgd(), s) for s in seeds]), [])
    return save(rows, out, "e1_methods")


# ---------------------------------------------------------------------------
# E2: wave size, batching and the cost of the certificate
# ---------------------------------------------------------------------------
def e2_job(args):
    cfg, seed = args
    fl, clients, test, man, tg, _, dim = build(cfg, seed)
    rows = []
    for name, m, kw in [("retrain", "retrain", {}), ("subtract+20", "subtract_recover",
                                                     {"recover_rounds": 20})]:
        f = copy.deepcopy(fl)
        out = f.unlearn(UnlearnRequest("client", client=tg[0], clients=tg), m, **kw)
        r = cert_row(out["theta"], man, tg, cfg, seed, out["cost"], test, name)
        r["energy_per_client_mj"] = r["energy_mj"] / cfg["W"]
        r["audit_mb"] = audit_bytes(cfg) / 1e6
        rows.append(r)
    # same revocations served one at a time (first three requests measured,
    # per-request cost extrapolated to the whole wave)
    f = copy.deepcopy(fl)
    cost = {"bytes": 0.0, "energy_j": 0.0, "rounds": 0}
    n_meas = min(3, len(tg))
    for c in tg[:n_meas]:
        o = f.unlearn(UnlearnRequest("client", client=c), "retrain")
        for k in cost:
            cost[k] += o["cost"][k] * len(tg) / n_meas
    r = cert_row(out["theta"], man, tg, cfg, seed, cost, test, "retrain_one_by_one")
    r["energy_per_client_mj"] = r["energy_mj"] / cfg["W"]
    r["audit_mb"] = audit_bytes(cfg) / 1e6
    rows.append(r)
    return rows


def e2(seeds, out, quick):
    jobs = [(cfgd(W=w), s) for w in ((10, 20) if quick else (4, 8, 12, 20, 30)) for s in seeds]
    rows = sum(pmap(e2_job, jobs), [])
    return save(rows, out, "e2_wave_size")


# ---------------------------------------------------------------------------
# E7: calibration of the verifier on exactly-unlearned and un-unlearned models
# ---------------------------------------------------------------------------
def e7_job(args):
    cfg, seed = args
    fl, clients, test, man, tg, _, dim = build(cfg, seed)
    zero = {"bytes": 0.0, "energy_j": 0.0, "rounds": 0}
    rows = [cert_row(fl.theta, man, tg, cfg, seed, zero, test, "none")]
    out = fl.unlearn(UnlearnRequest("client", client=tg[0], clients=tg), "retrain",
                     seed_offset=7)
    rows.append(cert_row(out["theta"], man, tg, cfg, seed, out["cost"], test, "retrain"))
    return rows


def e7(seeds, out, quick):
    cfg = cfgd(N=40, W=20, T=30)
    rows = sum(pmap(e7_job, [(cfg, s) for s in seeds]), [])
    return save(rows, out, "e7_calibration")


# ---------------------------------------------------------------------------
# E3: heterogeneity, participation, DP noise, capacity
# ---------------------------------------------------------------------------
def e3_job(args):
    cfg, seed, tag = args
    fl, clients, test, man, tg, _, dim = build(cfg, seed)
    req = UnlearnRequest("client", client=tg[0], clients=tg)
    zero = {"bytes": 0.0, "energy_j": 0.0, "rounds": 0}
    rows = [cert_row(fl.theta, man, tg, cfg, seed, zero, test, "none", dict(sweep=tag))]
    for name, m, kw in [("rollback", "rollback", {}),
                        ("subtract+10", "subtract_recover", {"recover_rounds": 10}),
                        ("subtract+20", "subtract_recover", {"recover_rounds": 20})]:
        f = copy.deepcopy(fl)
        out = f.unlearn(req, m, **kw)
        rows.append(cert_row(out["theta"], man, tg, cfg, seed, out["cost"], test, name,
                             dict(sweep=tag, ckpt=out["checkpoint"])))
    return rows


def e3(seeds, out, quick):
    jobs = []
    for a in (0.1, 1.0, 10.0):
        jobs += [(cfgd(alpha=a), s, "alpha") for s in seeds]
    for p in (0.3, 0.6, 1.0):
        jobs += [(cfgd(p=p), s, "p") for s in seeds]
    for sg in (0.0, 0.02, 0.05):
        jobs += [(cfgd(sigma=sg), s, "sigma") for s in seeds]
    for d in (64, 256, 1024):
        jobs += [(cfgd(d=d), s, "dim") for s in seeds]
    rows = sum(pmap(e3_job, jobs), [])
    return save(rows, out, "e3_sweeps")


# ---------------------------------------------------------------------------
# E4: verify-then-escalate vs fixed policies
# ---------------------------------------------------------------------------
def e4_job(args):
    cfg, seed = args
    fl, clients, test, man, tg, _, dim = build(cfg, seed)
    f = copy.deepcopy(fl)
    val = {"X": test["X"][:300], "y": test["y"][:300]}
    test = {"X": test["X"][300:], "y": test["y"][300:]}
    res = verify_then_escalate(f, tg, man, eps=cfg["eps"], alpha=0.05,
                               reveal_source="seeded", seed=seed, val=val)
    rows = [dict(seed=seed, policy="verify_then_escalate", released=res["released"],
                 status=res["status"], acc=V.accuracy(res["theta"], test),
                 energy_mj=res["cost"]["energy_j"] * 1e3, comm_mb=res["cost"]["bytes"] / 1e6,
                 rounds=res["cost"]["rounds"], n_attempts=len(res["attempts"]))]
    for name, m, kw in [("always_retrain", "retrain", {}), ("always_rollback", "rollback", {}),
                        ("fixed_subtract+5", "subtract_recover", {"recover_rounds": 5})]:
        g = copy.deepcopy(fl)
        out = g.unlearn(UnlearnRequest("client", client=tg[0], clients=tg), m, **kw)
        c, _ = certify_model(out["theta"], man, tg, eps=cfg["eps"], alpha=0.05,
                             reveal_source="seeded", seed=seed)
        rows.append(dict(seed=seed, policy=name, released=name, status=c.status,
                         acc=V.accuracy(out["theta"], test),
                         energy_mj=out["cost"]["energy_j"] * 1e3,
                         comm_mb=out["cost"]["bytes"] / 1e6, rounds=out["cost"]["rounds"],
                         n_attempts=1))
    return rows


def e4(seeds, out, quick):
    rows = sum(pmap(e4_job, [(cfgd(), s) for s in seeds]), [])
    return save(rows, out, "e4_escalation")


# ---------------------------------------------------------------------------
# E5: poisoned and Byzantine clients
# ---------------------------------------------------------------------------
def e5_job(args):
    cfg, seed, npz = args
    fl, clients, test, man, tg, bd, dim = build(cfg, seed, poison=npz)
    poisoned = tg[:npz]
    rows = [dict(seed=seed, n_poison=npz, byz=cfg["byz"], stage="trained", method="none",
                 acc=V.accuracy(fl.theta, test), asr=V.backdoor_asr(fl.theta, bd),
                 energy_mj=0.0)]
    req = UnlearnRequest("client", client=poisoned[0], clients=poisoned)
    for name, m, kw in [("retrain", "retrain", {}),
                        ("subtract+10", "subtract_recover", {"recover_rounds": 10}),
                        ("subtract+20", "subtract_recover", {"recover_rounds": 20})]:
        f = copy.deepcopy(fl)
        out = f.unlearn(req, m, **kw)
        rows.append(dict(seed=seed, n_poison=npz, byz=cfg["byz"], stage="unlearned",
                         method=name, acc=V.accuracy(out["theta"], test),
                         asr=V.backdoor_asr(out["theta"], bd),
                         energy_mj=out["cost"]["energy_j"] * 1e3))
    rest = [c for c in clients if c["id"] not in poisoned]
    o = UnlearnableFL(rest, dim, lr=cfg["lr"], participation=cfg["p"], sigma_dp=cfg["sigma"],
                      seed=seed + 500)
    o.train(cfg["T"])
    rows.append(dict(seed=seed, n_poison=npz, byz=cfg["byz"], stage="control",
                     method="never_poisoned", acc=V.accuracy(o.theta, test),
                     asr=V.backdoor_asr(o.theta, bd), energy_mj=0.0))
    return rows


def e5(seeds, out, quick):
    jobs = [(cfgd(W=10, byz=b), s, n) for b in (0, 12) for n in (3, 8) for s in seeds]
    rows = sum(pmap(e5_job, jobs), [])
    return save(rows, out, "e5_poison")


# ---------------------------------------------------------------------------
# E6: provenance ledger and certificate binding
# ---------------------------------------------------------------------------
def e6(seeds, out, quick):
    rows = []
    for seed in seeds[:3]:
        cfg = cfgd()
        fl, clients, test, man, tg, _, dim = build(cfg, seed)
        led = fl.ledger
        n_round_rec = sum(1 for r in led.records if r["event"] == "round")
        upd = sum(len(r["participants"]) for r in led.records if r["event"] == "round") * dim * 4
        t0 = time.time(); led.verify_chain(); t_chain = time.time() - t0
        rec = next(r for r in led.records if r["event"] == "round" and tg[0] in r["participants"])
        pr = led.inclusion_proof(rec["idx"], tg[0])
        t0 = time.time()
        for _ in range(2000):
            ok = verify_merkle_proof(pr["leaf"], pr["proof"], pr["root"])
        t_proof = (time.time() - t0) / 2000
        rng = np.random.default_rng(seed)
        det, trials = 0, 200
        for _ in range(trials):
            g = copy.deepcopy(fl)
            r = g.ledger.records[int(rng.integers(1, len(g.ledger.records)))]
            fld = rng.choice(["merkle_root", "model_hash", "round", "weights", "flagged"])
            if fld in ("merkle_root", "model_hash"):
                r[fld] = r[fld][:-1] + ("0" if r[fld][-1] != "0" else "1")
            elif fld == "round":
                r["round"] += 1
            elif fld == "weights":
                r["weights"] = {k: v + 0.01 for k, v in r["weights"].items()} or {"x": 1.0}
            else:
                r["flagged"] = r["flagged"] + ["Cxx"]
            det += (not g.ledger.verify_chain())
        res = verify_then_escalate(fl, tg, man, eps=cfg["eps"], alpha=0.05,
                                   reveal_source="seeded", seed=seed)
        rep = fl.ledger.audit_exclusion(tg, res["model_hash"])
        cert = sign_certificate({"wave": tg, "ledger_head": fl.ledger.head(),
                                 "model_hash": res["model_hash"],
                                 "status": res["status"],
                                 "e_log_cert": res["certificate"].log_e_cert}, b"auditor-key")
        ok_cert = verify_certificate(cert, b"auditor-key")
        cert["body"]["status"] = "ISSUED" if cert["body"]["status"] != "ISSUED" else "REVOKED"
        tamper_cert = not verify_certificate(cert, b"auditor-key")
        # false lineage claim: say the wave was excluded but keep training with it
        bad = UnlearnableFL(clients, dim, lr=cfg["lr"], participation=cfg["p"], seed=seed)
        bad.train(20)
        bad.ledger.log_rollback({"kind": "client"}, "fake", 0, digest_array(np.zeros(dim)), tg)
        bad._run_rounds(bad.theta, bad.reputations, 20, 3, bad.ids, "recover", 9)
        neg = bad.ledger.audit_exclusion(tg, bad.ledger.records[-1]["model_hash"])
        st = fl.storage_bytes()
        rows.append(dict(seed=seed, records=len(led.records), ledger_kb=st["ledger"] / 1e3,
                         ledger_bytes_per_round=st["ledger"] / max(1, n_round_rec),
                         ledger_over_update_traffic=st["ledger"] / upd,
                         chain_verify_ms=t_chain * 1e3, proof_us=t_proof * 1e6,
                         proof_len=len(pr["proof"]), proof_ok=ok, tamper_detected=det / trials,
                         audit_passed=rep["passed"], cert_verifies=ok_cert,
                         cert_tamper_detected=tamper_cert,
                         false_lineage_rejected=not neg["passed"],
                         checkpoint_kb=st["checkpoints"] / 1e3,
                         stored_updates_kb=st["updates"] / 1e3))
    return save(rows, out, "e6_provenance")


# ---------------------------------------------------------------------------
def save(rows, out, name):
    os.makedirs(out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, name + ".csv"), index=False)
    return df


def make_figures(out, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3})
    os.makedirs(figdir, exist_ok=True)
    R = lambda n: pd.read_csv(os.path.join(out, n + ".csv"))
    col = {"ISSUED": "#2a9d8f", "REVOKED": "#e63946", "UNDETERMINED": "#999999"}

    # Fig: certificate outcome and cost per method
    df = R("e1_methods")
    order = ["none", "subtract", "replay", "subtract+5", "subtract+10", "subtract+20",
             "rollback", "retrain"]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    cnt = df.groupby(["method", "status"]).size().unstack(fill_value=0).reindex(order).fillna(0)
    n = df.seed.nunique()
    bottom = np.zeros(len(order))
    for s in ("ISSUED", "UNDETERMINED", "REVOKED"):
        if s in cnt:
            ax[0].bar(range(len(order)), cnt[s] / n, bottom=bottom, color=col[s], label=s)
            bottom += (cnt[s] / n).values
    ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels(order, rotation=60, fontsize=6)
    ax[0].set_ylabel("fraction of runs"); ax[0].legend(fontsize=5.5)
    g = df.groupby("method").agg(e=("energy_mj", "mean"), du=("delta_upper", "mean"),
                                 acc=("acc", "mean")).reindex(order)
    ax[1].bar(range(len(order)), g.e, color="#457b9d")
    ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels(order, rotation=60, fontsize=6)
    ax[1].set_ylabel("client energy (mJ)")
    ax[2].bar(range(len(order)), g.du, color="#f4a261")
    ax[2].axhline(df.eps.iloc[0], c="k", ls="--", lw=0.8)
    ax[2].set_xticks(range(len(order))); ax[2].set_xticklabels(order, rotation=60, fontsize=6)
    ax[2].set_ylabel("upper bound on $\\Delta$")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "cert_methods.png"), dpi=220); plt.close(fig)

    # Fig: wave size
    w = R("e2_wave_size")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for m in ("retrain", "retrain_one_by_one", "subtract+20"):
        t = w[w.method == m].groupby("W")["energy_per_client_mj"].agg(["mean", "std"])
        ax[0].errorbar(t.index, t["mean"], yerr=t["std"], marker="o", ms=3, capsize=2, label=m)
    ax[0].set_xlabel("clients revoked in one wave"); ax[0].set_ylabel("energy per revoked client (mJ)")
    ax[0].set_yscale("log"); ax[0].legend(fontsize=6)
    t = w[w.method == "retrain"].groupby("W").apply(
        lambda d: (d.status == "ISSUED").mean(), include_groups=False)
    ax[1].plot(t.index, t.values, marker="o", ms=3)
    ax[1].set_xlabel("clients revoked in one wave"); ax[1].set_ylabel("certificate issue rate")
    ax[1].set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "cert_wave.png"), dpi=220); plt.close(fig)

    # Fig: escalation policy
    e = R("e4_escalation")
    g = e.groupby("policy").agg(en=("energy_mj", "mean"), es=("energy_mj", "std"),
                                iss=("status", lambda s: (s == "ISSUED").mean()),
                                acc=("acc", "mean"))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3))
    ax[0].bar(range(len(g)), g.en, yerr=g.es, capsize=2, color="#457b9d")
    ax[1].bar(range(len(g)), g.iss, color="#2a9d8f")
    for a, lab in zip(ax, ["client energy (mJ)", "certified fraction"]):
        a.set_xticks(range(len(g))); a.set_xticklabels(g.index, rotation=30, fontsize=6)
        a.set_ylabel(lab)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "cert_escalation.png"), dpi=220); plt.close(fig)

    # Fig: poison
    pz = R("e5_poison")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3))
    ordr = ["none", "retrain", "subtract+10", "subtract+20", "never_poisoned"]
    for a, met in zip(ax, ["asr", "acc"]):
        for j, (npz, byz) in enumerate([(3, 0), (8, 0), (3, 12), (8, 12)]):
            t = pz[(pz.n_poison == npz) & (pz.byz == byz)].groupby("method")[met].mean().reindex(ordr)
            a.bar(np.arange(len(ordr)) + 0.2 * j, t.values, 0.2,
                  label=f"{npz} poisoned, {byz} Byz.")
        a.set_xticks(np.arange(len(ordr)) + 0.3); a.set_xticklabels(ordr, rotation=30, fontsize=6)
        a.set_ylabel("backdoor ASR" if met == "asr" else "test accuracy")
    ax[0].legend(fontsize=5)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "cert_poison.png"), dpi=220); plt.close(fig)

    # Fig: sweeps
    s = R("e3_sweeps")
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.2))
    for a, (key, lab) in zip(ax, [("alpha", "Dirichlet $\\alpha$"), ("p", "participation"),
                                  ("sigma", "DP noise $\\sigma$"), ("dim", "feature dim $d$")]):
        t = s[s.sweep == key]
        cname = "d" if key == "dim" else key
        for m in ("none", "subtract+10", "subtract+20", "rollback"):
            q = t[t.method == m].groupby(cname)["delta_upper"].mean()
            a.plot(q.index, q.values, marker="o", ms=3, label=m)
        a.axhline(0.2, c="k", ls="--", lw=0.7)
        a.set_xlabel(lab); a.set_ylabel("upper bound on $\\Delta$")
        if key in ("alpha", "dim"):
            a.set_xscale("log")
    ax[0].legend(fontsize=5)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "cert_sweeps.png"), dpi=220); plt.close(fig)


def summarize(out):
    pd.set_option("display.width", 220, "display.max_columns", 40)
    for n, keys, mets in [
        ("e1_methods", ["method"], ["acc", "delta_upper", "log_e_cert", "log_e_rev", "t_stop",
                                    "rounds", "comm_mb", "energy_mj"]),
        ("e2_wave_size", ["W", "method"], ["energy_per_client_mj", "delta_upper", "t_stop"]),
        ("e4_escalation", ["policy"], ["acc", "energy_mj", "rounds", "n_attempts"]),
        ("e5_poison", ["byz", "n_poison", "method"], ["asr", "acc"]),
        ("e7_calibration", ["method"], ["delta_upper", "t_stop"]),
    ]:
        p = os.path.join(out, n + ".csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            print(f"\n== {n}")
            print(df.groupby(keys, sort=False)[mets].mean().round(3))
            if "status" in df:
                print(df.groupby(keys + ["status"], sort=False).size().unstack(fill_value=0))
    p = os.path.join(out, "e3_sweeps.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        print("\n== e3_sweeps")
        for key in ("alpha", "p", "sigma", "d"):
            t = df[df.sweep == ("dim" if key == "d" else key)]
            print(t.groupby([key, "method"])[["delta_upper", "acc", "energy_mj"]].mean().round(3))
            print(t.groupby([key, "method", "status"]).size().unstack(fill_value=0))
    p = os.path.join(out, "e6_provenance.csv")
    if os.path.exists(p):
        print("\n== e6_provenance")
        print(pd.read_csv(p).mean(numeric_only=True).round(4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results/unlearning")
    ap.add_argument("--figs", default="figs")
    ap.add_argument("--only", default="")
    ap.add_argument("--seeds", type=int, default=0)
    a = ap.parse_args()
    ns = a.seeds or (2 if a.quick else 10)
    seeds = list(range(ns))
    want = lambda n: (not a.only) or n in a.only.split(",")
    t0 = time.time()
    for name, fn, ss in [("e1", e1, seeds), ("e2", e2, seeds[:5]), ("e3", e3, seeds[:5]),
                         ("e4", e4, seeds), ("e5", e5, seeds[:5]), ("e6", e6, seeds),
                         ("e7", e7, list(range(40)))]:
        if want(name):
            fn(ss, a.out, a.quick)
            print(f"{name} done at {time.time() - t0:.0f}s", flush=True)
    if not a.only or a.only == "fig":
        make_figures(a.out, a.figs)
    summarize(a.out)
    print(f"\ndone in {time.time() - t0:.0f}s")
