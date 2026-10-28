"""
Real-data and MLP experiments for certified federated unlearning.

Data: UCI HAR smartphone sensor features (30 subjects = 30 clients), downloaded
on first use into data/har. Models: logistic regression and a one-hidden-layer MLP.

    python experiments/unlearning_real.py [--quick] [--only r1,r2,...]
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import copy
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from fortress_fl.unlearning import (UnlearnableFL, UnlearnRequest, LinearModel, MLPModel,
                                    plant_canaries, certify_model, verify_then_escalate, data)
from fortress_fl.unlearning import verification as V
from unlearning_experiments import pmap, save

HAR_DIR = os.environ.get("HAR_DIR", "data/har")
NB_PATH = os.environ.get("NBAIOT_NPZ", "data/nbaiot/sample.npz")
DS = "har"                     # set from --dataset
BASE = dict(model="linear", n_real=40, m=6, W=10, p=0.6, T=60, lr=1.0, sigma=0.0, eps=0.2,
            byz=0, hidden=32)


def cfgd(**kw):
    c = dict(BASE)
    c.update(kw)
    return c


def build(cfg, seed, poison=0, target=None):
    n_total = 30 if DS == "har" else 45
    byz = tuple(range(n_total - cfg["byz"], n_total))
    if DS == "har":
        clients, test, info = data.har_federation(data.fetch_har(HAR_DIR), n_real=cfg["n_real"],
                                                  seed=seed, byz_clients=byz)
    else:
        clients, test, info = data.nbaiot_federation(NB_PATH, n_real=cfg["n_real"], seed=seed,
                                                     byz_clients=byz)
    dim = info["dim"]
    man = plant_canaries(clients, cfg["m"], dim, seed=seed)
    honest = [c["id"] for c in clients if not c["byzantine"]]
    if target is not None:
        tg = [honest[(target + i) % len(honest)] for i in range(cfg["W"])]
    else:
        tg = honest[:cfg["W"]]
    bd = None
    if poison:
        for c in clients:
            if c["id"] in tg[:poison]:
                data.add_backdoor(c, 0.5, seed=seed)
        bd = data.backdoor_test_set(test)
    model = LinearModel(dim) if cfg["model"] == "linear" else MLPModel(dim, cfg["hidden"])
    fl = UnlearnableFL(clients, dim, model=model, lr=cfg["lr"], participation=cfg["p"],
                       sigma_dp=cfg["sigma"], seed=seed)
    fl.train(cfg["T"])
    return fl, clients, test, man, tg, bd, model


def cert_row(theta, model, man, tg, cfg, seed, cost, test, name, extra=None):
    c, diffs = certify_model(theta, man, tg, eps=cfg["eps"], alpha=0.05,
                             reveal_source="seeded", seed=seed, model=model)
    r = dict(method=name, seed=seed, status=c.status, t_stop=c.t_stop, n_pairs=len(diffs),
             delta_upper=c.delta_upper, log_e_cert=c.log_e_cert, log_e_rev=c.log_e_rev,
             acc=V.accuracy(theta, test, model), comm_mb=cost["bytes"] / 1e6,
             energy_mj=cost["energy_j"] * 1e3, rounds=cost["rounds"])
    r.update(cfg)
    if extra:
        r.update(extra)
    return r


OPS = [("retrain", "retrain", {}), ("subtract", "subtract", {}), ("replay", "replay", {}),
       ("subtract+5", "subtract_recover", {"recover_rounds": 5}),
       ("subtract+10", "subtract_recover", {"recover_rounds": 10}),
       ("subtract+20", "subtract_recover", {"recover_rounds": 20})]


# ---------------------------------------------------------------------------
def r1_job(args):
    cfg, seed = args
    fl, clients, test, man, tg, _, model = build(cfg, seed)
    req = UnlearnRequest("client", client=tg[0], clients=tg)
    zero = {"bytes": 0.0, "energy_j": 0.0, "rounds": 0}
    rows = [cert_row(fl.theta, model, man, tg, cfg, seed, zero, test, "none")]
    for name, m, kw in OPS:
        f = copy.deepcopy(fl)
        out = f.unlearn(req, m, **kw)
        rows.append(cert_row(out["theta"], model, man, tg, cfg, seed, out["cost"], test, name,
                             dict(ledger_audit=f.ledger.audit_exclusion(tg, out["model_hash"])["passed"])))
    return rows


def r1(seeds, out, quick):
    jobs = [(cfgd(model=m), s) for m in ("linear", "mlp") for s in seeds]
    return save(sum(pmap(r1_job, jobs), []), out, f"r1_{DS}_methods")


# ---------------------------------------------------------------------------
def r2_job(args):
    cfg, seed = args
    fl, clients, test, man, tg, _, model = build(cfg, seed)
    val = {"X": test["X"][:150], "y": test["y"][:150]}
    test2 = {"X": test["X"][150:], "y": test["y"][150:]}
    f = copy.deepcopy(fl)
    res = verify_then_escalate(f, tg, man, eps=cfg["eps"], alpha=0.05, reveal_source="seeded",
                               seed=seed, val=val)
    rows = [dict(seed=seed, model=cfg["model"], policy="verify_then_escalate",
                 released=res["released"], status=res["status"],
                 acc=V.accuracy(res["theta"], test2, model), energy_mj=res["cost"]["energy_j"] * 1e3,
                 rounds=res["cost"]["rounds"], n_attempts=len(res["attempts"]))]
    for name, m, kw in [("always_retrain", "retrain", {}),
                        ("fixed_subtract+5", "subtract_recover", {"recover_rounds": 5})]:
        g = copy.deepcopy(fl)
        out = g.unlearn(UnlearnRequest("client", client=tg[0], clients=tg), m, **kw)
        c, _ = certify_model(out["theta"], man, tg, eps=cfg["eps"], alpha=0.05,
                             reveal_source="seeded", seed=seed, model=model)
        rows.append(dict(seed=seed, model=cfg["model"], policy=name, released=name,
                         status=c.status, acc=V.accuracy(out["theta"], test2, model),
                         energy_mj=out["cost"]["energy_j"] * 1e3, rounds=out["cost"]["rounds"],
                         n_attempts=1))
    return rows


def r2(seeds, out, quick):
    jobs = [(cfgd(model=m), s) for m in ("linear", "mlp") for s in seeds]
    return save(sum(pmap(r2_job, jobs), []), out, f"r2_{DS}_escalation")


# ---------------------------------------------------------------------------
def r3_job(args):
    cfg, seed = args
    fl, clients, test, man, tg, _, model = build(cfg, seed)
    zero = {"bytes": 0.0, "energy_j": 0.0, "rounds": 0}
    rows = [cert_row(fl.theta, model, man, tg, cfg, seed, zero, test, "none")]
    o = fl.unlearn(UnlearnRequest("client", client=tg[0], clients=tg), "retrain", seed_offset=7)
    rows.append(cert_row(o["theta"], model, man, tg, cfg, seed, o["cost"], test, "retrain"))
    return rows


def r3(seeds, out, quick):
    jobs = [(cfgd(T=30, model=m), s) for m in ("linear", "mlp") for s in range(20 if quick else 60)]
    return save(sum(pmap(r3_job, jobs), []), out, f"r3_{DS}_calibration")


# ---------------------------------------------------------------------------
def r4_job(args):
    cfg, seed, npz = args
    fl, clients, test, man, tg, bd, model = build(cfg, seed, poison=npz)
    poisoned = tg[:npz]
    rows = [dict(seed=seed, model=cfg["model"], n_poison=npz, stage="trained", method="none",
                 acc=V.accuracy(fl.theta, test, model), asr=V.backdoor_asr(fl.theta, bd, model),
                 energy_mj=0.0)]
    req = UnlearnRequest("client", client=poisoned[0], clients=poisoned)
    for name, m, kw in [("retrain", "retrain", {}),
                        ("subtract+10", "subtract_recover", {"recover_rounds": 10}),
                        ("subtract+20", "subtract_recover", {"recover_rounds": 20})]:
        f = copy.deepcopy(fl)
        o = f.unlearn(req, m, **kw)
        rows.append(dict(seed=seed, model=cfg["model"], n_poison=npz, stage="unlearned",
                         method=name, acc=V.accuracy(o["theta"], test, model),
                         asr=V.backdoor_asr(o["theta"], bd, model),
                         energy_mj=o["cost"]["energy_j"] * 1e3))
    rest = [c for c in clients if c["id"] not in poisoned]
    ctl = UnlearnableFL(rest, fl.model.n_params, model=model, lr=cfg["lr"],
                        participation=cfg["p"], seed=seed + 500)
    ctl.train(cfg["T"])
    rows.append(dict(seed=seed, model=cfg["model"], n_poison=npz, stage="control",
                     method="never_poisoned", acc=V.accuracy(ctl.theta, test, model),
                     asr=V.backdoor_asr(ctl.theta, bd, model), energy_mj=0.0))
    return rows


def r4(seeds, out, quick):
    jobs = [(cfgd(model=m), s, n) for m in ("linear", "mlp") for n in (2, 5) for s in seeds]
    return save(sum(pmap(r4_job, jobs), []), out, f"r4_{DS}_poison")


# ---------------------------------------------------------------------------
# Per-client certificates: how many canary pairs per client are needed?
# ---------------------------------------------------------------------------
def r5_job(args):
    cfg, seed = args
    fl, clients, test, man, tg, _, model = build(cfg, seed, target=seed * 3)
    zero = {"bytes": 0.0, "energy_j": 0.0, "rounds": 0}
    rows = []
    n_real = np.mean([len(c["y"]) - c["n_canary"] for c in clients])
    extra = dict(canary_share=cfg["m"] / (cfg["m"] + n_real))
    for eps in (0.2, 0.3):
        cc = dict(cfg); cc["eps"] = eps
        rows.append(cert_row(fl.theta, model, man, tg, cc, seed, zero, test, "none", extra))
    req = UnlearnRequest("client", client=tg[0], clients=tg)
    for name, m, kw in [("retrain", "retrain", {}),
                        ("subtract+20", "subtract_recover", {"recover_rounds": 20})]:
        f = copy.deepcopy(fl)
        o = f.unlearn(req, m, **kw)
        for eps in (0.2, 0.3):
            cc = dict(cfg); cc["eps"] = eps
            rows.append(cert_row(o["theta"], model, man, tg, cc, seed, o["cost"], test, name,
                                 extra))
    return rows


def r5(seeds, out, quick):
    jobs = []
    for m in (10, 40, 100):
        jobs += [(cfgd(model="linear", W=1, m=m), s) for s in seeds]
    for m in (10, 20, 40, 60, 100):
        jobs += [(cfgd(model="mlp", W=1, m=m), s) for s in seeds]
    for m in (60, 100):       # wider MLP: more capacity for the canaries
        jobs += [(cfgd(model="mlp", hidden=128, W=1, m=m), s) for s in seeds]
    return save(sum(pmap(r5_job, jobs), []), out, f"r5_{DS}_per_client")


# ---------------------------------------------------------------------------
def figures(out, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3})
    R = lambda n: pd.read_csv(os.path.join(out, n + ".csv"))
    col = {"ISSUED": "#2a9d8f", "REVOKED": "#e63946", "UNDETERMINED": "#999999"}
    d = R(f"r1_{DS}_methods")
    order = ["none", "subtract", "replay", "subtract+5", "subtract+10", "subtract+20", "retrain"]
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.4))
    for j, mod in enumerate(("linear", "mlp")):
        t = d[d.model == mod]
        n = t.seed.nunique()
        cnt = t.groupby(["method", "status"]).size().unstack(fill_value=0).reindex(order).fillna(0)
        bottom = np.zeros(len(order))
        for s in ("ISSUED", "UNDETERMINED", "REVOKED"):
            if s in cnt:
                ax[2 * j].bar(range(len(order)), cnt[s] / n, bottom=bottom, color=col[s], label=s)
                bottom += (cnt[s] / n).values
        ax[2 * j].set_title(f"{DS.upper()}, {mod}", fontsize=8)
        ax[2 * j].set_ylabel("fraction of runs")
        g = t.groupby("method").energy_mj.mean().reindex(order)
        ax[2 * j + 1].bar(range(len(order)), g.values / 1e3, color="#457b9d")
        ax[2 * j + 1].set_ylabel("client energy (J)")
        ax[2 * j + 1].set_title(f"{DS.upper()}, {mod}", fontsize=8)
    for a in ax:
        a.set_xticks(range(len(order))); a.set_xticklabels(order, rotation=60, fontsize=5.5)
    ax[0].legend(fontsize=5)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, f"{DS}_methods.png"), dpi=220); plt.close(fig)

    p = R(f"r5_{DS}_per_client")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3))
    for a, mod in zip(ax, ("mlp", "linear")):
        t0 = p[p.model == mod]
        for eps, ls in ((0.2, "-"), (0.3, "--")):
            t = t0[(t0.eps == eps) & (t0.method == "retrain")]
            r = t.groupby("m").apply(lambda x: (x.status == "ISSUED").mean(), include_groups=False)
            a.plot(r.index, r.values, ls, marker="o", ms=3, color="#2a9d8f",
                   label=f"retrained certified, $\\epsilon$={eps}")
            t = t0[(t0.eps == eps) & (t0.method == "none")]
            r = t.groupby("m").apply(lambda x: (x.status == "REVOKED").mean(), include_groups=False)
            a.plot(r.index, r.values, ls, marker="s", ms=3, color="#e63946",
                   label=f"no unlearning flagged, $\\epsilon$={eps}")
        a.set_title(f"{DS.upper()}, {mod}, one client", fontsize=8)
        a.set_xlabel("canary pairs per client $m$"); a.set_ylabel("fraction of runs")
        a.set_ylim(-0.05, 1.05)
    ax[0].legend(fontsize=5)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, f"{DS}_per_client.png"), dpi=220); plt.close(fig)


def summarize(out):
    pd.set_option("display.width", 220, "display.max_columns", 40)
    for n, keys, mets in [
        (f"r1_{DS}_methods", ["model", "method"], ["acc", "delta_upper", "t_stop", "rounds", "energy_mj"]),
        (f"r2_{DS}_escalation", ["model", "policy"], ["acc", "energy_mj", "rounds", "n_attempts"]),
        (f"r3_{DS}_calibration", ["model", "method"], ["delta_upper", "t_stop", "acc"]),
        (f"r4_{DS}_poison", ["model", "n_poison", "method"], ["asr", "acc", "energy_mj"]),
        (f"r5_{DS}_per_client", ["model", "hidden", "m", "eps", "method"], ["delta_upper", "acc", "canary_share"]),
    ]:
        p = os.path.join(out, n + ".csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            print(f"\n== {n}")
            print(df.groupby(keys, sort=False)[mets].mean().round(3))
            if "status" in df:
                print(df.groupby(keys + ["status"], sort=False).size().unstack(fill_value=0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results/unlearning")
    ap.add_argument("--figs", default="figs")
    ap.add_argument("--only", default="")
    ap.add_argument("--dataset", default="har", choices=["har", "nbaiot"])
    a = ap.parse_args()
    DS = a.dataset
    if DS == "nbaiot":
        BASE["W"] = 15
    seeds = list(range(2 if a.quick else 5))
    want = lambda n: (not a.only) or n in a.only.split(",")
    t0 = time.time()
    if DS == "har":
        data.fetch_har(HAR_DIR)
    for name, fn, ss in [("r1", r1, seeds), ("r2", r2, seeds), ("r3", r3, seeds),
                         ("r4", r4, seeds), ("r5", r5, seeds)]:
        if want(name):
            fn(ss, a.out, a.quick)
            print(f"{name} done at {time.time() - t0:.0f}s", flush=True)
    if not a.only or a.only == "fig":
        figures(a.out, a.figs)
    summarize(a.out)
    print(f"\ndone in {time.time() - t0:.0f}s")
