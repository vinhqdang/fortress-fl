"""
Experiments added in response to the review round.

    python experiments/unlearning_review.py [--quick] [--only rv1,rv2,...]

rv1  energy accounting: reveal cost, early-stopped retraining, per-device energy,
     server cost; energy-constant sensitivity is computed from its bytes and flops
rv2  calibration with an unpredictable (OS entropy) reveal order
rv3  independent forgetting criterion (parameter and prediction distance to retraining)
rv4  ledger rewrite attack, witness anchoring and Ed25519 certificates
rv5  dropout, asynchrony, staggered enrolment
rv6  sharded cohorts, class-level and sample-level requests
rv7  canary detectability, operator filtering attack, selective-reveal attack
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
from fortress_fl.unlearning import (UnlearnRequest, UnlearnableFL, ShardedFL, certify_model,
                                    verify_then_escalate, incremental_certified_recovery,
                                    early_stopped_retrain, device_energy_j, participation_counts,
                                    data)
from fortress_fl.unlearning import verification as V
from fortress_fl.unlearning.anchor import (Witness, LedgerAnchor, verify_against_witnesses,
                                           generate_keypair, sign_certificate_ed25519,
                                           verify_certificate_ed25519)
from fortress_fl.unlearning.canary import pair_scores

WAVE = {"synthetic": 20, "har": 10, "nbaiot": 15}
OUT = "results/unlearning"
SETTINGS = [("synthetic", "linear"), ("har", "linear"), ("har", "mlp"),
            ("nbaiot", "linear"), ("nbaiot", "mlp")]


def cfg_for(ds, model, **kw):
    c = ur.cfgd(ds=ds, model=model, W=WAVE[ds])
    c.update(kw)
    return c


def split_eval(test):
    return ({"X": test["X"][:150], "y": test["y"][:150]},
            {"X": test["X"][150:], "y": test["y"][150:]})


def reveal_bytes(man, wave):
    return sum(len(man[c]["pairs"]) * 2 * len(man[c]["pairs"][0]["in"][0]) * 4 for c in wave)


def cert_status(theta, man, tg, cfg, seed, model, source="seeded"):
    c, _ = certify_model(theta, man, tg, eps=cfg["eps"], alpha=0.05, reveal_source=source,
                         seed=seed if source == "seeded" else None, model=model)
    return c.status


# ---------------------------------------------------------------------------
# rv1: energy accounting
# ---------------------------------------------------------------------------
def rv1_job(args):
    cfg, seed = args
    fl, clients, test, man, tg, _, model = ur.build(cfg, seed)
    val, te = split_eval(test)
    rb = reveal_bytes(man, tg)
    n_entries = sum(len(u) for u in fl.updates.values())
    rows = []

    def add(policy, status, theta, cost, counts, extra=None):
        dev = list(device_energy_j(fl, counts).values()) if counts else []
        rows.append(dict(seed=seed, ds=cfg["ds"], model=cfg["model"], policy=policy,
                         status=status, acc=V.accuracy(theta, te, model), rounds=cost["rounds"],
                         bytes=cost["bytes"], flops=cost["flops"], energy_mj=cost["energy_j"] * 1e3,
                         reveal_bytes=rb, worst_dev_mj=1e3 * max(dev) if dev else np.nan,
                         mean_dev_mj=1e3 * float(np.mean(dev)) if dev else np.nan,
                         stored_updates_mb=fl.storage_bytes()["updates"] / 1e6,
                         entangle_flops=3.0 * n_entries * fl.dim if "q" in policy else 0.0,
                         **(extra or {})))

    req = UnlearnRequest("client", client=tg[0], clients=tg)
    g = copy.deepcopy(fl)
    o = g.unlearn(req, "retrain")
    add("retrain", cert_status(o["theta"], man, tg, cfg, seed, model), o["theta"], o["cost"],
        participation_counts(g, 0))
    e = early_stopped_retrain(fl, tg, val)
    add("retrain_early", cert_status(e["theta"], man, tg, cfg, seed, model), e["theta"], e["cost"],
        e["participation"])
    r = verify_then_escalate(copy.deepcopy(fl), tg, man, eps=cfg["eps"], alpha=0.05,
                             reveal_source="seeded", seed=seed, val=val)
    add("ladder", r["status"], r["theta"], r["cost"], None)
    for q in (1.0, 0.5, 0.25):
        r = incremental_certified_recovery(fl, tg, man, eps=cfg["eps"], alpha=0.05, q=q, val=val,
                                           reveal_source="seeded", seed=seed, commit=False)
        add(f"icr_q{q}", r["status"], r["theta"], r["cost"], r["participation"])
    return rows


def rv1(seeds, quick):
    jobs = [(cfg_for(ds, m), s) for ds, m in SETTINGS for s in seeds]
    return save(sum(pmap(rv1_job, jobs), []), OUT, "rv1_energy")


# ---------------------------------------------------------------------------
# rv2: calibration with an unpredictable reveal order
# ---------------------------------------------------------------------------
def rv2_job(args):
    cfg, seed = args
    fl, clients, test, man, tg, _, model = ur.build(cfg, seed)
    rows = [dict(seed=seed, ds=cfg["ds"], model=cfg["model"], method="none",
                 status=cert_status(fl.theta, man, tg, cfg, seed, model, "local"))]
    o = fl.unlearn(UnlearnRequest("client", client=tg[0], clients=tg), "retrain", seed_offset=7)
    rows.append(dict(seed=seed, ds=cfg["ds"], model=cfg["model"], method="retrain",
                     status=cert_status(o["theta"], man, tg, cfg, seed, model, "local")))
    return rows


def rv2(seeds, quick):
    n = 20 if quick else 60
    jobs = [(cfg_for(ds, m, T=30), s) for ds, m in (("synthetic", "linear"), ("har", "mlp"),
                                                    ("nbaiot", "mlp")) for s in range(n)]
    return save(sum(pmap(rv2_job, jobs), []), OUT, "rv2_beacon")


# ---------------------------------------------------------------------------
# rv3: independent forgetting criterion
# ---------------------------------------------------------------------------
def disagree(model, a, b, X):
    return float(np.mean((V.predict_proba(a, X, model) > 0.5) != (V.predict_proba(b, X, model) > 0.5)))


def rv3_job(args):
    cfg, seed = args
    fl, clients, test, man, tg, _, model = ur.build(cfg, seed)
    val, te = split_eval(test)
    req = UnlearnRequest("client", client=tg[0], clients=tg)
    ref1 = copy.deepcopy(fl).unlearn(req, "retrain", seed_offset=1)["theta"]
    ref2 = copy.deepcopy(fl).unlearn(req, "retrain", seed_offset=2)["theta"]
    realX = np.vstack([next(c for c in clients if c["id"] == t)["X"][:-cfg["m"]] for t in tg])
    realy = np.concatenate([next(c for c in clients if c["id"] == t)["y"][:-cfg["m"]] for t in tg])
    lref = V.per_sample_loss(ref1, realX, realy, model).mean()
    rows = []

    def add(name, theta):
        rows.append(dict(seed=seed, ds=cfg["ds"], model=cfg["model"], method=name,
                         status=cert_status(theta, man, tg, cfg, seed, model),
                         gap=V.param_gap(theta, ref1, ref2),
                         disagree=disagree(model, theta, ref1, te["X"]),
                         disagree_floor=disagree(model, ref2, ref1, te["X"]),
                         loss_ratio=float(V.per_sample_loss(theta, realX, realy, model).mean() / (lref + 1e-12)),
                         acc=V.accuracy(theta, te, model)))
    add("none", fl.theta)
    for name, m, kw in [("subtract", "subtract", {}),
                        ("subtract+5", "subtract_recover", {"recover_rounds": 5}),
                        ("subtract+10", "subtract_recover", {"recover_rounds": 10}),
                        ("subtract+20", "subtract_recover", {"recover_rounds": 20}),
                        ("retrain_seed3", "retrain", {"seed_offset": 3})]:
        add(name, copy.deepcopy(fl).unlearn(req, m, **kw)["theta"])
    for q in (1.0, 0.25):
        r = incremental_certified_recovery(fl, tg, man, eps=cfg["eps"], q=q, val=val,
                                           reveal_source="seeded", seed=seed, commit=False)
        add(f"icr_q{q}", r["theta"])
    add("retrain_early", early_stopped_retrain(fl, tg, val)["theta"])
    return rows


def rv3(seeds, quick):
    jobs = [(cfg_for(ds, m), s) for ds, m in SETTINGS for s in seeds]
    return save(sum(pmap(rv3_job, jobs), []), OUT, "rv3_forgetting")


# ---------------------------------------------------------------------------
# rv4: ledger rewrite attack, witnesses, signatures
# ---------------------------------------------------------------------------
def rewrite_and_relink(ledger, pos):
    g = copy.deepcopy(ledger)
    rec = g.records[pos]
    rec["flagged"] = list(rec.get("flagged", [])) + ["Cxx"]
    prev = g.records[pos - 1]["hash"]
    for r in g.records[pos:]:
        r["prev"] = prev
        body = {k: v for k, v in r.items() if k != "hash"}
        r["hash"] = g._record_hash(body)
        prev = r["hash"]
    return g


def rv4(seeds, quick):
    cfg = cfg_for("synthetic", "linear")
    fl, clients, test, man, tg, _, model = ur.build(cfg, 0)
    n = len(fl.ledger.records)
    rng = np.random.default_rng(0)
    rows = []
    for interval in (5, 10, 20, 60):
        det_chain = det_wit = 0
        trials = 300
        for _ in range(trials):
            L = int(rng.integers(20, n + 1))                # ledger length at attack time
            led = copy.deepcopy(fl.ledger)
            led.records = led.records[:L]
            ws = [Witness("A"), Witness("B"), Witness("C")]
            an = LedgerAnchor(ws, interval=interval)
            for idx in range(0, L, interval):
                head = led.records[idx]["hash"]
                sig = an.sk.sign(f"{idx}:{head}".encode())
                for w in ws:
                    w.receive(idx, head, sig, an.pk)
            pos = int(rng.integers(1, L))
            g = rewrite_and_relink(led, pos)
            det_chain += (not g.verify_chain())
            det_wit += (not verify_against_witnesses(g, ws)["passed"])
        rows.append(dict(interval=interval, records=n, anchors=n // interval + 1,
                         chain_only_detect=det_chain / trials, witness_detect=det_wit / trials,
                         last_anchor=-1, receipt_bytes=3 * (n // interval + 1) * 200))
    sk, pk = generate_keypair()
    ok = bad = 0
    for i in range(200):
        c = sign_certificate_ed25519({"wave": tg, "status": "ISSUED", "i": i}, sk)
        ok += verify_certificate_ed25519(c, pk)
        c["body"]["status"] = "REVOKED"
        bad += (not verify_certificate_ed25519(c, pk))
    rows.append(dict(interval=-1, records=n, anchors=0, chain_only_detect=np.nan,
                     witness_detect=np.nan, last_anchor=-1, receipt_bytes=0,
                     sig_valid=ok / 200, sig_tamper_detect=bad / 200))
    return save(rows, OUT, "rv4_anchor")


# ---------------------------------------------------------------------------
# rv5: dropout, asynchrony, staggered enrolment
# ---------------------------------------------------------------------------
VARIANTS = {"base": {}, "dropout0.3": {"dropout": 0.3}, "stale3": {"stale": 3},
            "dropout+stale": {"dropout": 0.3, "stale": 3}, "staggered": {"join": 40}}


def rv5_job(args):
    cfg, seed, vname = args
    fl, clients, test, man, tg, _, model = ur.build(cfg, seed)
    val, te = split_eval(test)
    req = UnlearnRequest("client", client=tg[0], clients=tg)
    rows = []

    def add(policy, status, theta, cost, ck=None):
        rows.append(dict(seed=seed, ds=cfg["ds"], model=cfg["model"], variant=vname, policy=policy,
                         status=status, acc=V.accuracy(theta, te, model), rounds=cost["rounds"],
                         energy_mj=cost["energy_j"] * 1e3, ckpt=ck,
                         k0=fl.ledger.first_participation(tg[0])))
    zero = {"rounds": 0, "energy_j": 0.0}
    add("none", cert_status(fl.theta, man, tg, cfg, seed, model), fl.theta, zero)
    for name, m in (("retrain", "retrain"), ("rollback", "rollback")):
        g = copy.deepcopy(fl)
        o = g.unlearn(req, m)
        add(name, cert_status(o["theta"], man, tg, cfg, seed, model), o["theta"], o["cost"],
            o["checkpoint"])
    r = incremental_certified_recovery(fl, tg, man, eps=cfg["eps"], q=1.0, val=val,
                                       reveal_source="seeded", seed=seed, commit=False)
    add("icr", r["status"], r["theta"], r["cost"])
    return rows


def rv5(seeds, quick):
    jobs = []
    for ds, m in (("synthetic", "linear"), ("har", "linear"), ("har", "mlp")):
        for vname, extra in VARIANTS.items():
            for s in seeds:
                jobs.append((cfg_for(ds, m, **extra), s, vname))
    return save(sum(pmap(rv5_job, jobs), []), OUT, "rv5_dynamics")


# ---------------------------------------------------------------------------
# rv6: sharded cohorts, class-level and sample-level requests
# ---------------------------------------------------------------------------
def rv6_job(args):
    cfg, seed = args
    fl, clients, test, man, tg, _, model = ur.build(cfg, seed)
    val, te = split_eval(test)
    rows = []
    sh0 = ShardedFL(clients, fl.dim, n_shards=4, model=model, lr=cfg["lr"], participation=cfg["p"],
                    seed=seed)
    sh0.train(cfg["T"])
    base_acc = V.accuracy(sh0.theta, te, model)
    mono_acc = V.accuracy(fl.theta, te, model)
    honest = [c["id"] for c in clients if not c["byzantine"]]
    one_shard = [c for c in honest if sh0.owner[c] == 0][:max(3, min(cfg["W"] // 2, 5))]
    for label, wave in (("spread", tg), ("one_shard", one_shard)):
        sh = ShardedFL(clients, fl.dim, n_shards=4, model=model, lr=cfg["lr"],
                       participation=cfg["p"], seed=seed)
        sh.train(cfg["T"])
        o = sh.unlearn_wave(wave)
        g = copy.deepcopy(fl)
        mo = g.unlearn(UnlearnRequest("client", client=wave[0], clients=wave), "retrain")
        rows.append(dict(seed=seed, ds=cfg["ds"], model=cfg["model"], wave=label, n_wave=len(wave),
                         shards_touched=o["shards_touched"], acc_sharded=V.accuracy(o["theta"], te, model),
                         acc_sharded_before=base_acc, acc_mono_before=mono_acc,
                         acc_mono_retrain=V.accuracy(mo["theta"], te, model),
                         energy_sharded_mj=o["cost"]["energy_j"] * 1e3,
                         energy_mono_retrain_mj=mo["cost"]["energy_j"] * 1e3,
                         status_sharded=cert_status(o["theta"], man, wave, cfg, seed, model)))
    return rows


def rv6b_job(args):
    cfg, seed = args
    fl, clients, test, man, tg, _, model = ur.build(cfg, seed)
    rows = []
    zero = {"rounds": 0, "energy_j": 0.0}
    if cfg["ds"] == "nbaiot":
        kinds = list(np.load(ur.NB_PATH)["kinds"])
        cls = kinds.index("mirai_udp")
        req = UnlearnRequest("class", client=None, classes=[cls])
        m = np.isin(test["latent"], [cls])
        fa = lambda th: float(np.mean((V.predict_proba(th, test["X"][m], model) > .5) == test["y"][m]))
        ra = lambda th: V.accuracy(th, {"X": test["X"][~m], "y": test["y"][~m]}, model)
        rows.append(dict(seed=seed, ds=cfg["ds"], model=cfg["model"], kind="class", method="none",
                         forget_acc=fa(fl.theta), retain_acc=ra(fl.theta), rounds=0, energy_mj=0.0, ckpt=0))
        for name in ("retrain", "rollback"):
            g = copy.deepcopy(fl)
            o = g.unlearn(req, name)
            rows.append(dict(seed=seed, ds=cfg["ds"], model=cfg["model"], kind="class", method=name,
                             forget_acc=fa(o["theta"]), retain_acc=ra(o["theta"]),
                             rounds=o["cost"]["rounds"], energy_mj=o["cost"]["energy_j"] * 1e3,
                             ckpt=o["checkpoint"]))
    tid = max(clients, key=lambda c: len(c["y"]))["id"]
    tgt = next(c for c in clients if c["id"] == tid)
    idx = np.random.default_rng(seed).choice(len(tgt["y"]) - cfg["m"], max(1, (len(tgt["y"]) - cfg["m"]) // 4),
                                             replace=False)
    req = UnlearnRequest("sample", client=tid, indices=idx)
    ref2 = copy.deepcopy(fl).unlearn(req, "retrain", seed_offset=2)["theta"]
    ref1 = copy.deepcopy(fl).unlearn(req, "retrain", seed_offset=1)["theta"]
    rows.append(dict(seed=seed, ds=cfg["ds"], model=cfg["model"], kind="sample", method="none",
                     gap=V.param_gap(fl.theta, ref1, ref2), acc=V.accuracy(fl.theta, test, model),
                     rounds=0, energy_mj=0.0, ckpt=0))
    for name in ("retrain", "rollback"):
        g = copy.deepcopy(fl)
        o = g.unlearn(req, name, seed_offset=3)
        rows.append(dict(seed=seed, ds=cfg["ds"], model=cfg["model"], kind="sample", method=name,
                         gap=V.param_gap(o["theta"], ref1, ref2), acc=V.accuracy(o["theta"], test, model),
                         rounds=o["cost"]["rounds"], energy_mj=o["cost"]["energy_j"] * 1e3,
                         ckpt=o["checkpoint"]))
    return rows


def rv6(seeds, quick):
    jobs = [(cfg_for(ds, m), s) for ds, m in (("synthetic", "linear"), ("har", "linear"),
                                              ("har", "mlp")) for s in seeds]
    save(sum(pmap(rv6_job, jobs), []), OUT, "rv6_sharded")
    jobs = [(cfg_for(ds, m), s) for ds, m in (("nbaiot", "linear"), ("nbaiot", "mlp"), ("har", "mlp"))
            for s in seeds]
    return save(sum(pmap(rv6b_job, jobs), []), OUT, "rv6b_granular")


# ---------------------------------------------------------------------------
# rv7: canary detectability, filtering and selective-reveal attacks
# ---------------------------------------------------------------------------
def rv7_job(args):
    from sklearn.ensemble import IsolationForest
    cfg, seed = args
    fl, clients, test, man, tg, _, model = ur.build(cfg, seed)
    X_all = np.vstack([c["X"] for c in clients])
    is_can = np.concatenate([np.r_[np.zeros(len(c["y"]) - c["n_canary"], bool), np.ones(c["n_canary"], bool)]
                             for c in clients])
    det = IsolationForest(n_estimators=100, random_state=seed).fit(X_all)
    score = -det.score_samples(X_all)
    auc = V._auc(score[is_can], score[~is_can])
    rows = []
    # operator filters the most anomalous records (as many as there are canaries) and trains on the rest
    thr = np.sort(score)[::-1][int(is_can.sum()) - 1]
    flagged = score >= thr
    recall = float(np.mean(flagged[is_can]))
    filt, k = [], 0
    for c in clients:
        n = len(c["y"])
        keep = ~flagged[k:k + n]
        k += n
        d = dict(c)
        d["X"], d["y"] = c["X"][keep], c["y"][keep]
        filt.append(d)
    f2 = UnlearnableFL(filt, fl.dim, model=model, lr=cfg["lr"], participation=cfg["p"], seed=seed)
    f2.train(cfg["T"])
    st_filter = cert_status(f2.theta, man, tg, cfg, seed, model)
    # selective reveal: the revoking clients open a sign-balanced subset of their pairs, so
    # that the revealed signs look like fair coins (the attack needs only to hide the leak)
    out_extra = {}
    for tname, theta_t in (("none", fl.theta),
                           ("subtract", copy.deepcopy(fl).unlearn(UnlearnRequest("client", client=tg[0], clients=tg), "subtract")["theta"])):
        dl = {(c, i): pair_scores(theta_t, p, model)["loss"] for c in tg for i, p in enumerate(man[c]["pairs"])}
        neg = [k for k, v in dl.items() if v <= 0]
        pos = [k for k, v in dl.items() if v > 0]
        k = min(len(neg), len(pos))
        chosen = neg[:k] + pos[:k]
        revealed = {c: [i for (cc, i) in chosen if cc == c] for c in tg}
        out_extra[f"{tname}_honest"] = certify_model(theta_t, man, tg, eps=cfg["eps"], alpha=0.05,
                                                     reveal_source="seeded", seed=seed, model=model)[0].status
        out_extra[f"{tname}_cherry_drop"] = certify_model(theta_t, man, tg, eps=cfg["eps"], alpha=0.05,
                                                          reveal_source="seeded", seed=seed, model=model,
                                                          revealed=revealed, missing="drop")[0].status
        out_extra[f"{tname}_cherry_worst"] = certify_model(theta_t, man, tg, eps=cfg["eps"], alpha=0.05,
                                                           reveal_source="seeded", seed=seed, model=model,
                                                           revealed=revealed, missing="worst")[0].status
        out_extra[f"{tname}_revealed_frac"] = len(chosen) / max(1, len(dl))
    st_honest = out_extra["none_honest"]
    rows.append(dict(seed=seed, ds=cfg["ds"], model=cfg["model"], ckind=cfg.get("ckind", "gaussian"),
                     detect_auc=auc, filter_recall=recall, status_honest=st_honest,
                     status_after_filter=st_filter, acc_filtered=V.accuracy(f2.theta, test, model),
                     **out_extra))
    return rows


def rv7(seeds, quick):
    jobs = [(cfg_for(ds, "mlp", ckind=k), s) for ds in ("har", "nbaiot") for k in ("gaussian", "crossover")
            for s in seeds]
    return save(sum(pmap(rv7_job, jobs), []), OUT, "rv7_attacks")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    seeds = list(range(2 if a.quick else 5))
    want = lambda n: (not a.only) or n in a.only.split(",")
    t0 = time.time()
    for name, fn in [("rv1", rv1), ("rv2", rv2), ("rv3", rv3), ("rv4", rv4), ("rv5", rv5),
                     ("rv6", rv6), ("rv7", rv7)]:
        if want(name):
            fn(seeds, a.quick)
            print(name, int(time.time() - t0), flush=True)
    print("done", int(time.time() - t0))
