"""
Experiments for the recovery, audit-design and scheduling algorithms.

    python experiments/unlearning_new.py [--quick] [--only n1,n2,...]

n1  incremental certified recovery (ICR) and entanglement-targeted recovery
n2  paired real-record twins versus synthetic canaries
n3  power-adaptive canary calibration
n4  deadline-aware wave scheduling (simulation driven by measured wave cost)
n5  baselines: marker-based verification and loss-threshold membership inference
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

import unlearning_real as ur
from unlearning_experiments import pmap, save
from fortress_fl.unlearning import (UnlearnRequest, certify_model, verify_then_escalate,
                                    incremental_certified_recovery, calibrate_canaries,
                                    simulate_scheduler, UnlearnableFL, LinearModel, MLPModel,
                                    plant_canaries, data)
from fortress_fl.unlearning import verification as V

WAVE = {"synthetic": 20, "har": 10, "nbaiot": 15}


def cfg_for(ds, model, **kw):
    c = ur.cfgd(ds=ds, model=model, W=WAVE[ds])
    c.update(kw)
    return c


def split_eval(test):
    return ({"X": test["X"][:150], "y": test["y"][:150]},
            {"X": test["X"][150:], "y": test["y"][150:]})


# ---------------------------------------------------------------------------
# n1: recovery policies
# ---------------------------------------------------------------------------
def n1_job(args):
    cfg, seed = args
    fl, clients, test, man, tg, _, model = ur.build(cfg, seed)
    val, te = split_eval(test)
    rows = []

    def add(policy, status, cost, theta, extra):
        rows.append(dict(seed=seed, ds=cfg["ds"], model=cfg["model"], policy=policy,
                         status=status, acc=V.accuracy(theta, te, model),
                         energy_mj=cost["energy_j"] * 1e3, rounds=cost["rounds"],
                         pre_acc=V.accuracy(fl.theta, te, model), **extra))

    g = copy.deepcopy(fl)
    o = g.unlearn(UnlearnRequest("client", client=tg[0], clients=tg), "retrain")
    c, _ = certify_model(o["theta"], man, tg, eps=cfg["eps"], alpha=0.05,
                         reveal_source="seeded", seed=seed, model=model)
    add("retrain", c.status, o["cost"], o["theta"], dict(blocks=0, rec_clients=len(fl.ids) - len(tg)))
    r = verify_then_escalate(copy.deepcopy(fl), tg, man, eps=cfg["eps"], alpha=0.05,
                             reveal_source="seeded", seed=seed, val=val)
    add("ladder", r["status"], r["cost"], r["theta"], dict(blocks=len(r["attempts"]),
                                                          rec_clients=len(fl.ids) - len(tg)))
    for q in (1.0, 0.5, 0.25):
        r = incremental_certified_recovery(fl, tg, man, eps=cfg["eps"], alpha=0.05, q=q,
                                           val=val, reveal_source="seeded", seed=seed,
                                           commit=False)
        add(f"icr_q{q}", r["status"], r["cost"], r["theta"],
            dict(blocks=len(r["attempts"]), rec_clients=r["n_recovery_clients"]))
    return rows


def n1(seeds, out, quick):
    jobs = [(cfg_for("synthetic", "linear"), s) for s in seeds]
    jobs += [(cfg_for(ds, m), s) for ds in ("har", "nbaiot") for m in ("linear", "mlp")
             for s in seeds]
    return save(sum(pmap(n1_job, jobs), []), "results/unlearning", "n1_recovery")


# ---------------------------------------------------------------------------
# n2: real-record twins versus synthetic canaries
# ---------------------------------------------------------------------------
def n2_job(args):
    cfg, seed = args
    fl, clients, test, man, tg, _, model = ur.build(cfg, seed)
    zero = {"bytes": 0.0, "energy_j": 0.0, "rounds": 0}
    rows = [ur.cert_row(fl.theta, model, man, tg, cfg, seed, zero, test, "none")]
    req = UnlearnRequest("client", client=tg[0], clients=tg)
    for name, m, kw in [("retrain", "retrain", {}), ("subtract", "subtract", {}),
                        ("subtract+5", "subtract_recover", {"recover_rounds": 5}),
                        ("subtract+20", "subtract_recover", {"recover_rounds": 20})]:
        f = copy.deepcopy(fl)
        o = f.unlearn(req, m, **kw)
        rows.append(ur.cert_row(o["theta"], model, man, tg, cfg, seed, o["cost"], test, name))
    return rows


def n2(seeds, out, quick):
    jobs = []
    for ds in ("har", "nbaiot"):
        for m in ("linear", "mlp"):
            for can in ("noise", "real"):
                for s in seeds:
                    jobs.append((cfg_for(ds, m, canary=can), s))
    for can in ("noise", "real"):
        for s in seeds:
            jobs.append((cfg_for("synthetic", "linear", canary=can), s))
    return save(sum(pmap(n2_job, jobs), []), "results/unlearning", "n2_real_twins")


def n2c_job(args):
    cfg, seed = args
    fl, clients, test, man, tg, _, model = ur.build(cfg, seed)
    zero = {"bytes": 0.0, "energy_j": 0.0, "rounds": 0}
    rows = [ur.cert_row(fl.theta, model, man, tg, cfg, seed, zero, test, "none")]
    o = fl.unlearn(UnlearnRequest("client", client=tg[0], clients=tg), "retrain", seed_offset=7)
    rows.append(ur.cert_row(o["theta"], model, man, tg, cfg, seed, o["cost"], test, "retrain"))
    return rows


def n2c(seeds, out, quick):
    n = 20 if quick else 50
    jobs = [(cfg_for(ds, "mlp", canary="real", T=30), s) for ds in ("har", "nbaiot")
            for s in range(n)]
    return save(sum(pmap(n2c_job, jobs), []), "results/unlearning", "n2_real_twins_calibration")


# ---------------------------------------------------------------------------
# n3: power-adaptive canary calibration
# ---------------------------------------------------------------------------
def n3_job(args):
    ds, model, scale, m, seed = args
    cfg = cfg_for(ds, model, scale=scale, m=m)
    fl, clients, test, man, tg, _, mdl = ur.build(cfg, seed)
    c, _ = certify_model(fl.theta, man, tg, eps=0.2, alpha=0.05, reveal_source="seeded",
                         seed=seed, model=mdl)
    return [dict(ds=ds, model=model, scale=scale, m=m, seed=seed, status=c.status,
                 delta_upper=c.delta_upper, log_e_rev=c.log_e_rev,
                 acc=V.accuracy(fl.theta, test, mdl), n_pairs=len(tg) * m)]


def n3(seeds, out, quick):
    jobs = [(ds, mo, sc, m, s) for ds, mo in (("nbaiot", "linear"), ("nbaiot", "mlp"),
                                               ("har", "linear"))
            for sc in (1.0, 2.0, 4.0, 8.0) for m in (6, 12) for s in seeds]
    return save(sum(pmap(n3_job, jobs), []), "results/unlearning", "n3_canary_power")


# ---------------------------------------------------------------------------
# n4: scheduler
# ---------------------------------------------------------------------------
def n4(out, quick, wave_energy_j):
    rows = []
    for rate in (0.2, 1.0, 4.0):
        for need, label in ((1, "immediate"), (120, "batched")):
            for s in range(20):
                r = simulate_scheduler(rate, 365, 6, need, deadline=30, wave_energy_j=wave_energy_j,
                                       seed=s)
                rows.append(dict(rate_per_day=rate, policy=label, seed=s, **r))
    return save(rows, "results/unlearning", "n4_scheduler")


# ---------------------------------------------------------------------------
# n5: baselines, marker-based verification and loss-threshold MIA
# ---------------------------------------------------------------------------
def stamp_markers(clients, wave, frac=0.25, seed=0, n_trig=10, value=3.0):
    """VeriFi-style: each leaving client stamps its own random trigger on a fraction
    of its records and relabels them to the flipped class."""
    rng = np.random.default_rng(seed)
    markers = {}
    for c in clients:
        if c["id"] not in wave:
            continue
        idx = rng.choice(len(c["y"]), max(1, int(frac * len(c["y"]))), replace=False)
        pos = rng.choice(c["X"].shape[1] - 1, n_trig, replace=False)
        c["X"] = c["X"].copy(); c["y"] = c["y"].copy()
        c["X"][np.ix_(idx, pos)] = value
        c["y"][idx] = 1.0 - c["y"][idx]
        markers[c["id"]] = (pos, value)
    return markers


def marker_asr(theta, model, test, markers):
    """Fraction of triggered clean test points classified as the flipped label."""
    X = test["X"].copy()
    pos = np.concatenate([p for p, _ in markers.values()])
    val = next(iter(markers.values()))[1]
    keep = np.arange(len(X))
    Xs = X[keep].copy()
    Xs[:, pos] = val
    p = 1 / (1 + np.exp(-model.logit(theta, Xs)))
    flipped = 1.0 - test["y"][keep]
    return float(np.mean((p > 0.5) == flipped))


def n5_job(args):
    cfg, seed = args
    fl, clients, test, man, tg, _, model = ur.build(cfg, seed)
    # marker baseline needs its own federation with markers on the wave's data
    mk_clients, mk_test, info = (data.har_federation(data.fetch_har(ur.HAR_DIR), n_real=cfg["n_real"], seed=seed)
                                 if cfg["ds"] == "har" else
                                 data.nbaiot_federation(ur.NB_PATH, n_real=cfg["n_real"], seed=seed))
    markers = stamp_markers(mk_clients, set(tg), seed=seed)
    fl_m = UnlearnableFL(mk_clients, info["dim"], model=model, lr=cfg["lr"],
                         participation=cfg["p"], seed=seed)
    fl_m.train(cfg["T"])
    rows = []
    req = UnlearnRequest("client", client=tg[0], clients=tg)
    forget = {"X": np.vstack([next(c for c in clients if c["id"] == t)["X"] for t in tg]),
              "y": np.concatenate([next(c for c in clients if c["id"] == t)["y"] for t in tg])}
    nonmember = {"X": test["X"], "y": test["y"]}
    for name, m, kw in [("none", None, {}), ("retrain", "retrain", {}),
                        ("subtract+20", "subtract_recover", {"recover_rounds": 20})]:
        if m is None:
            th, th_m = fl.theta, fl_m.theta
        else:
            a = copy.deepcopy(fl); th = a.unlearn(req, m, **kw)["theta"]
            b = copy.deepcopy(fl_m); th_m = b.unlearn(req, m, **kw)["theta"]
        n = min(len(forget["y"]), len(nonmember["y"]))
        auc = V._auc(-V.per_sample_loss(th, forget["X"][:n], forget["y"][:n], model),
                     -V.per_sample_loss(th, nonmember["X"][:n], nonmember["y"][:n], model))
        c, _ = certify_model(th, man, tg, eps=cfg["eps"], alpha=0.05, reveal_source="seeded",
                             seed=seed, model=model)
        rows.append(dict(seed=seed, ds=cfg["ds"], model=cfg["model"], method=name,
                         marker_asr=marker_asr(th_m, model, mk_test, markers),
                         mia_auc=auc, cert_status=c.status, delta_upper=c.delta_upper,
                         acc_marker_fed=V.accuracy(th_m, mk_test, model),
                         acc_canary_fed=V.accuracy(th, test, model)))
    return rows


def n5(seeds, out, quick):
    jobs = [(cfg_for(ds, "mlp"), s) for ds in ("har", "nbaiot") for s in seeds]
    return save(sum(pmap(n5_job, jobs), []), "results/unlearning", "n5_baselines")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    seeds = list(range(2 if a.quick else 5))
    want = lambda n: (not a.only) or n in a.only.split(",")
    t0 = time.time()
    if want("n1"): n1(seeds, None, a.quick); print("n1", int(time.time() - t0), flush=True)
    if want("n2"): n2(seeds, None, a.quick); n2c(seeds, None, a.quick); print("n2", int(time.time() - t0), flush=True)
    if want("n3"): n3(seeds, None, a.quick); print("n3", int(time.time() - t0), flush=True)
    if want("n4"):
        try:
            w = pd.read_csv("results/unlearning/n1_recovery.csv")
            e = float(w[(w.policy == "icr_q1.0") & (w.ds == "synthetic")].energy_mj.mean()) / 1e3
        except Exception:
            e = 2.0
        n4(None, a.quick, e); print("n4", int(time.time() - t0), flush=True)
    if want("n5"): n5(seeds, None, a.quick); print("n5", int(time.time() - t0), flush=True)
    print("done", int(time.time() - t0))
