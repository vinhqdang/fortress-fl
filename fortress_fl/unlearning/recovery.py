"""
Recovery algorithms that use the provenance ledger's stored updates.

* ``incremental_certified_recovery`` (ICR) continues recovery rounds on one model
  and audits it after each block, instead of restarting for every rung of a
  ladder, so a failed attempt costs nothing that the next attempt does not reuse.
* ``select_entangled`` picks the retained clients whose past updates are most
  aligned with those of the revoked clients, for entanglement-targeted recovery.
"""

import copy
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from .certify import certify_model
from .provenance import digest_array
from .system import UnlearnableFL, UnlearnRequest
from .verification import accuracy


def entanglement_scores(fl: UnlearnableFL, wave: Sequence[str]) -> Dict[str, float]:
    """
    Mean cosine similarity between each retained client's stored update and the
    mean stored update of the revoked clients, over rounds where both took part.
    High scores mark clients whose training signal overlaps most with the wave.
    """
    wave = set(wave)
    sums: Dict[str, float] = {}
    cnts: Dict[str, int] = {}
    for t, ups in fl.updates.items():
        w = [ups[c] for c in ups if c in wave]
        if not w:
            continue
        m = np.mean(w, axis=0)
        nm = np.linalg.norm(m)
        if nm < 1e-12:
            continue
        for c, g in ups.items():
            if c in wave:
                continue
            ng = np.linalg.norm(g)
            if ng < 1e-12:
                continue
            sums[c] = sums.get(c, 0.0) + float(g @ m / (ng * nm))
            cnts[c] = cnts.get(c, 0) + 1
    return {c: sums[c] / cnts[c] for c in sums}


def select_entangled(fl: UnlearnableFL, wave: Sequence[str], q: float,
                     min_clients: int = 4) -> List[str]:
    """Top ``q`` fraction of the retained clients by entanglement score."""
    active = [c for c in fl.consent.active_clients() if c not in set(wave)]
    if q >= 1.0:
        return active
    sc = entanglement_scores(fl, wave)
    ranked = sorted(active, key=lambda c: -sc.get(c, -1.0))
    k = max(min_clients, int(np.ceil(q * len(active))))
    return ranked[:k]


def incremental_certified_recovery(fl: UnlearnableFL, wave: Sequence[str], manifests: Dict,
                                   eps: float = 0.2, alpha: float = 0.05, block: int = 5,
                                   max_rounds: int = 60, q: float = 1.0,
                                   val: Optional[Dict] = None, max_drop: float = 0.03,
                                   reveal_source: str = "seeded", seed: Optional[int] = None,
                                   commit: bool = True, require_power: bool = True) -> Dict:
    """
    Subtract the wave's stored contributions, then run recovery rounds in blocks of
    ``block`` and audit after every block (including block 0), releasing the first
    model that is certified and keeps validation accuracy within ``max_drop``.

    Error control: the audit after block k is run at level alpha_k = alpha/(k(k+1)),
    and sum_k alpha_k <= alpha, so by the union bound the probability that a leaky
    model is ever released as certified is at most alpha, whatever the dependence
    between the successive models. ``q < 1`` restricts recovery to the most
    entangled retained clients (see ``select_entangled``).
    """
    f = copy.deepcopy(fl)
    T = f.round
    wave = list(wave)
    req = UnlearnRequest("client", client=wave[0], clients=wave)
    base_acc = accuracy(fl.theta, val, fl.model) if val is not None else None
    pre, _ = certify_model(fl.theta, manifests, wave, eps, alpha, reveal_source=reveal_source,
                           seed=seed, model=fl.model)
    audit_power = pre.status == "REVOKED"

    f.unlearn(req, "subtract")                       # no client cost
    active = select_entangled(f, wave, q)
    theta, reps = f.theta.copy(), dict(f.reputations)
    cost = {"bytes": 0.0, "flops": 0.0, "energy_j": 0.0, "rounds": 0}
    attempts, done, k = [], 0, 1
    while True:
        a_k = alpha / (k * (k + 1))
        cert, _ = certify_model(theta, manifests, wave, eps, a_k, reveal_source=reveal_source,
                                seed=seed, model=f.model)
        acc = accuracy(theta, val, f.model) if val is not None else None
        ok = base_acc is None or acc >= base_acc - max_drop
        attempts.append({"block": k - 1, "rounds": done, "status": cert.status, "val_acc": acc,
                         "utility_ok": ok, "delta_upper": cert.delta_upper})
        if cert.status == "ISSUED" and ok:
            break
        if done >= max_rounds:
            break
        theta, reps, c = f._run_rounds(theta, reps, T + done, block, active, "recover",
                                       f.seed + 1)
        for key in cost:
            cost[key] += c.get(key, 0)
        done += block
        k += 1
    status = attempts[-1]["status"] if attempts[-1]["utility_ok"] else "UTILITY_REJECTED"
    if require_power and not audit_power and status == "ISSUED":
        status = "UNPOWERED"
    part = participation_counts(f, T)
    f.theta, f.reputations, f.round = theta.copy(), dict(reps), T + done
    f.ledger.log_release(digest_array(theta), "icr")
    if commit:
        fl.__dict__.update(f.__dict__)
    return {"status": status, "theta": theta, "cost": cost, "rounds": done,
            "attempts": attempts, "audit_power": audit_power, "targeted": q < 1.0,
            "model_hash": digest_array(theta), "n_recovery_clients": len(active),
            "participation": part}


def participation_counts(f: UnlearnableFL, after_round: int = 0) -> Dict[str, int]:
    """Rounds each client uploaded an update in recovery-phase ledger records."""
    c = Counter()
    for rec in f.ledger.records:
        if rec["event"] == "round" and rec["phase"] == "recover" and rec["round"] > after_round:
            c.update(rec["participants"])
    return dict(c)


def device_energy_j(f: UnlearnableFL, counts: Dict[str, int]) -> Dict[str, float]:
    """Per-device energy of the counted rounds under the system's cost model."""
    out = {}
    for cid, n in counts.items():
        n_rec = len(f.clients[cid]["y"])
        out[cid] = n * f.cost.round_cost([n_rec], f.dim, f.model.flops_per_sample)["energy_j"]
    return out


def early_stopped_retrain(fl: UnlearnableFL, wave: Sequence[str], val: Dict, tol: float = 0.005,
                          block: int = 5, max_rounds: int = 60, seed_offset: int = 1) -> Dict:
    """
    A stronger retraining baseline: restart from the initial model on the retained
    clients and stop as soon as validation accuracy is within ``tol`` of the accuracy
    of the model before unlearning (checked every ``block`` rounds). Retraining for a
    fixed T rounds over-provisions when the task converges early.
    """
    f = copy.deepcopy(fl)
    wave = list(wave)
    for w in wave:
        f.consent.revoke(w, {"kind": "client"}, f.round)
    target = accuracy(fl.theta, val, fl.model) - tol
    theta, reps = f.model.init(f.seed), {c: 0.5 for c in f.ids}
    active = f.consent.active_clients()
    cost = {"bytes": 0.0, "flops": 0.0, "energy_j": 0.0, "rounds": 0}
    done = 0
    while done < max_rounds:
        theta, reps, c = f._run_rounds(theta, reps, done, block, active, "recover",
                                       f.seed + seed_offset)
        for k in cost:
            cost[k] += c.get(k, 0)
        done += block
        if accuracy(theta, val, f.model) >= target:
            break
    f.theta, f.reputations = theta.copy(), dict(reps)
    return {"theta": theta, "cost": cost, "rounds": done, "state": f,
            "participation": participation_counts(f, 0)}
