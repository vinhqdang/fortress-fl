"""
Unlearnable FORTRESS-FL: federated training with per-round checkpoints, a
provenance ledger, consent revocation and several unlearning strategies.

Training reuses the FORTRESS-FL building blocks (Pedersen commitments, spectral
Byzantine detection, reputation weighting, Gaussian noise). Round state is the
pair (model, reputations), because reputations also carry information about past
contributions and must be rolled back together with the model.
"""

import contextlib
import io
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..crypto.pedersen import setup_pedersen_commitment, commit_gradient, verify_commitment
from ..aggregation.spectral_clustering import (
    spectral_clustering_byzantine_detection, compute_similarity_matrix)
from ..aggregation.reputation import update_reputations
from .provenance import ProvenanceLedger, ConsentRegistry, digest_array
from .models import LinearModel


def _sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))),
                    np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def logistic_gradient(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X.T @ (_sigmoid(X @ theta) - y) / len(y)


@dataclass
class CostModel:
    """
    Resource model for an IoT client. The constants are assumptions chosen to be
    representative of low-power devices; they are not measurements. Use
    ``scaled`` to explore sensitivity.
    """
    bytes_per_param: int = 4          # float32 on the wire
    commitment_bytes: int = 64        # 512-bit group element
    e_flop_j: float = 1e-9            # joules per floating point operation
    e_bit_j: float = 1e-7             # joules per transmitted/received bit

    def round_cost(self, n_samples: List[int], dim: int, flops_per_sample: float = None):
        fps = 4.0 * dim if flops_per_sample is None else flops_per_sample
        flops = sum(n * fps for n in n_samples)                # fwd + bwd
        up = len(n_samples) * (dim * self.bytes_per_param + self.commitment_bytes)
        down = len(n_samples) * dim * self.bytes_per_param
        bytes_total = up + down
        energy = flops * self.e_flop_j + 8.0 * bytes_total * self.e_bit_j
        return {"bytes": bytes_total, "flops": flops, "energy_j": energy}


@dataclass
class UnlearnRequest:
    kind: str = "client"              # 'client' | 'sample' | 'class'
    client: Optional[str] = None
    indices: Optional[np.ndarray] = None   # sample indices within client
    classes: Optional[List[int]] = None    # latent classes (class-level)
    clients: Optional[List[str]] = None    # several clients in one wave

    @property
    def targets(self) -> List[str]:
        return list(self.clients) if self.clients else [self.client]

    def describe(self) -> Dict:
        return {"kind": self.kind, "client": self.client,
                "n_indices": None if self.indices is None else int(len(self.indices)),
                "classes": self.classes,
                "clients": self.clients if self.clients else None}


def _empty_cost():
    return {"bytes": 0.0, "flops": 0.0, "energy_j": 0.0, "rounds": 0}


def _add_cost(a, b):
    return {k: a[k] + b.get(k, 0) for k in a}


class UnlearnableFL:
    """Federated trainer with provenance and unlearning support."""

    def __init__(self, clients: List[Dict], dim: int, lr: float = 0.5,
                 participation: float = 1.0, clip: float = 1.0, sigma_dp: float = 0.0,
                 lambda_rep: float = 0.1, penalty: float = 0.2, gate_tau: float = 0.25,
                 comm_params=None, security_param: int = 512, seed: int = 0,
                 cost_model: Optional[CostModel] = None, store_updates: bool = True,
                 model=None, dropout: float = 0.0, max_staleness: int = 0,
                 join_round: Optional[Dict[str, int]] = None):
        self.clients = {c["id"]: c for c in clients}
        self.ids = [c["id"] for c in clients]
        self.model = model if model is not None else LinearModel(dim)
        self.dim = self.model.n_params
        self.lr = lr
        self.participation = participation      # float, or dict client -> probability
        self.join_round = join_round or {}      # client -> first round it may take part in
        self.dropout = dropout                  # selected client fails before uploading
        self.max_staleness = max_staleness      # asynchronous: gradient at a model up to S rounds old
        self.clip = clip
        self.sigma_dp = sigma_dp
        self.lambda_rep = lambda_rep
        self.penalty = penalty
        self.gate_tau = gate_tau
        self.seed = seed
        self.cost = cost_model or CostModel()
        self.store_updates = store_updates
        if comm_params is None:
            with contextlib.redirect_stdout(io.StringIO()):
                comm_params = setup_pedersen_commitment(security_param)
        self.comm_params = comm_params
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.theta = self.model.init(self.seed)
        self.reputations = {c: 0.5 for c in self.ids}
        self.round = 0
        self.ledger = ProvenanceLedger()
        self.consent = ConsentRegistry(self.ids)
        self.checkpoints = {0: (self.theta.copy(), dict(self.reputations))}
        self.updates: Dict[int, Dict[str, np.ndarray]] = {}
        self.round_meta: Dict[int, Dict] = {}
        self.train_cost = _empty_cost()
        self.ledger.log_round(0, "train", [], [], {}, [], digest_array(self.theta))

    # ------------------------------------------------------------------
    def _participants(self, t: int, active: List[str], seed: int) -> List[str]:
        """Independent Bernoulli(p) per (round, client): removing a client does
        not change who else participates."""
        out = []
        for c in active:
            if t < self.join_round.get(c, 0):
                continue
            h = np.random.default_rng([seed, t, zlib.crc32(c.encode())])
            p = self.participation[c] if isinstance(self.participation, dict) else self.participation
            if p >= 1.0 or h.random() < p:
                out.append(c)
        if self.dropout > 0 and len(out) > 3:
            kept = [c for c in out
                    if np.random.default_rng([seed, t, zlib.crc32(c.encode()), 7]).random() >= self.dropout]
            out = kept if len(kept) >= 3 else out[:3]
        if len(out) < 3:   # fall back so clustering is defined
            out = list(active)[:max(3, len(out))]
        return out

    def _local_update(self, cid: str, theta: np.ndarray, data_view=None) -> np.ndarray:
        c = self.clients[cid]
        X, y = (c["X"], c["y"]) if data_view is None else data_view
        if c.get("byzantine"):
            g = self.model.grad(theta, X, y)
            if c.get("attack") == "sign_flip":
                g = -2.0 * g
            elif c.get("attack") == "noise":
                g = np.random.default_rng(self.seed + 7).normal(0, 1.0, self.dim)
        else:
            g = self.model.grad(theta, X, y)
        n = np.linalg.norm(g)
        return g * (self.clip / n) if n > self.clip else g

    def _aggregate(self, ids, grads, theta, reps, t, noise_seed):
        """One TrustChain round with a separation gate on the cluster split."""
        p, q, g_, h_ = self.comm_params
        coms, opens = [], []
        for g in grads:
            com, op = commit_gradient(g, p, q, g_, h_)
            coms.append(com)
            opens.append(op)
        ok = [verify_commitment(g, c, o, p, q, g_, h_)
              for g, c, o in zip(grads, coms, opens)]
        keep = [i for i, v in enumerate(ok) if v]
        flagged = [ids[i] for i, v in enumerate(ok) if not v]

        byz_rel: List[int] = []
        if len(keep) >= 4:
            vg = [grads[i] for i in keep]
            with contextlib.redirect_stdout(io.StringIO()):
                labels, bid = spectral_clustering_byzantine_detection(vg, k=2)
            S = compute_similarity_matrix(vg)
            hon = np.where(labels != bid)[0]
            byz = np.where(labels == bid)[0]
            if len(byz) > 0 and len(hon) > len(byz):
                cross = S[np.ix_(hon, byz)].mean()
                within = S[np.ix_(hon, hon)][np.triu_indices(len(hon), 1)].mean() \
                    if len(hon) > 1 else 1.0
                if within - cross > self.gate_tau:
                    byz_rel = byz.tolist()
        byz_ids = [ids[keep[i]] for i in byz_rel]
        flagged += byz_ids
        hon_idx = [keep[i] for i in range(len(keep)) if i not in byz_rel]
        hon_ids = [ids[i] for i in hon_idx]

        w = np.array([reps[c] for c in hon_ids], dtype=float)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(hon_ids)) / len(hon_ids)
        agg = sum(wi * grads[i] for wi, i in zip(w, hon_idx))
        rng = np.random.default_rng(noise_seed)
        if self.sigma_dp > 0:
            agg = agg + rng.normal(0.0, self.sigma_dp, self.dim)

        new_reps = dict(reps)
        sub_ids = ids
        upd = update_reputations(sub_ids, grads, agg, hon_idx,
                                 [i for i in range(len(ids)) if ids[i] in flagged],
                                 {c: reps[c] for c in sub_ids},
                                 lambda_param=self.lambda_rep, penalty=self.penalty)
        new_reps.update(upd)
        weights = dict(zip(hon_ids, w))
        return agg, new_reps, weights, flagged

    # ------------------------------------------------------------------
    def _run_rounds(self, theta, reps, start, n_rounds, active, phase, seed,
                    data_override=None, record=True):
        cost = _empty_cost()
        hist = [theta]
        for k in range(n_rounds):
            t = start + k + 1
            ids = self._participants(t, active, seed)

            def stale_model(c):
                if self.max_staleness <= 0:
                    return theta
                s_ = int(np.random.default_rng([seed, t, zlib.crc32(c.encode()), 99])
                         .integers(0, self.max_staleness + 1))
                return hist[max(0, len(hist) - 1 - s_)]
            grads = [self._local_update(
                c, stale_model(c), None if data_override is None else data_override.get(c))
                for c in ids]
            agg, reps, weights, flagged = self._aggregate(
                ids, grads, theta, reps, t, noise_seed=seed * 100003 + t)
            theta = theta - self.lr * agg
            hist.append(theta)
            if len(hist) > self.max_staleness + 1:
                hist.pop(0)
            nsamp = [len(self.clients[c]["y"] if data_override is None or c not in data_override
                         else data_override[c][1]) for c in ids]
            cost = _add_cost(cost, self.cost.round_cost(nsamp, self.dim, self.model.flops_per_sample))
            cost["rounds"] += 1
            if record:
                self.ledger.log_round(t, phase, ids, [digest_array(g) for g in grads],
                                      weights, flagged, digest_array(theta))
                self.checkpoints[t] = (theta.copy(), dict(reps))
                if self.store_updates:
                    self.updates[t] = {c: g.copy() for c, g in zip(ids, grads)}
                self.round_meta[t] = {"participants": ids, "weights": weights,
                                      "flagged": flagged}
        return theta, reps, cost

    def train(self, n_rounds: int):
        self.theta, self.reputations, c = self._run_rounds(
            self.theta, self.reputations, self.round, n_rounds,
            self.consent.active_clients(), "train", self.seed)
        self.round += n_rounds
        self.train_cost = _add_cost(self.train_cost, c)
        return self.theta

    # ------------------------------------------------------------------
    def _first_affected_round(self, req: UnlearnRequest) -> int:
        owners = self._owners(req)
        firsts = [self.ledger.first_participation(c) for c in owners]
        firsts = [f for f in firsts if f is not None]
        return min(firsts) if firsts else self.round + 1

    def _owners(self, req: UnlearnRequest) -> List[str]:
        if req.kind == "client":
            return req.targets
        if req.kind == "sample":
            return [req.client]
        owners = []
        for c in self.ids:
            if np.isin(self.clients[c]["latent"], req.classes).any():
                owners.append(c)
        return owners

    def _edited_views(self, req: UnlearnRequest) -> Dict[str, tuple]:
        views = {}
        if req.kind == "sample":
            c = self.clients[req.client]
            m = np.ones(len(c["y"]), bool)
            m[req.indices] = False
            views[req.client] = (c["X"][m], c["y"][m])
        elif req.kind == "class":
            for cid in self._owners(req):
                c = self.clients[cid]
                m = ~np.isin(c["latent"], req.classes)
                if m.sum() == 0:
                    views[cid] = (c["X"][:0], c["y"][:0])
                else:
                    views[cid] = (c["X"][m], c["y"][m])
        return views

    def unlearn(self, req: UnlearnRequest, method: str = "rollback",
                recover_rounds: Optional[int] = None, seed_offset: int = 1) -> Dict:
        """
        Execute an unlearning request against the current trained state.

        methods:
          retrain    restart from round 0 on the retained data
          rollback   restore the last clean checkpoint, re-run remaining rounds
          subtract   remove the stored weighted updates of the client (client-level)
          subtract_recover   subtract, then ``recover_rounds`` fresh rounds
          replay     re-run rounds from the clean checkpoint using stored updates
                     of the remaining clients (no client computation)
        """
        T = self.round
        owners = self._owners(req)
        for o in owners:
            self.consent.revoke(o, req.describe(), T)
        k0 = self._first_affected_round(req)
        ck = max(k0 - 1, 0)
        views = self._edited_views(req)
        if req.kind == "client":
            active = [c for c in self.consent.active_clients()]
        else:
            active = list(self.ids)
        seed = self.seed + seed_offset
        saved = (self.theta.copy(), dict(self.reputations))
        cost = _empty_cost()
        self.ledger.log_rollback(req.describe(), method, ck,
                                 digest_array(self.checkpoints[ck][0]), owners)

        if method == "retrain":
            theta0, reps0, start = self.model.init(self.seed), {c: 0.5 for c in self.ids}, 0
            n = T
        elif method == "rollback":
            theta0, reps0 = self.checkpoints[ck]
            start, n = ck, T - ck
        elif method in ("subtract", "subtract_recover", "replay"):
            if req.kind != "client":
                raise ValueError(f"{method} supports client-level requests only")
            theta0, reps0 = self.checkpoints[ck]
            start, n = ck, T - ck
        else:
            raise ValueError(method)

        if method in ("retrain", "rollback"):
            theta, reps, cost = self._run_rounds(
                theta0, reps0, start, n, active, "recover", seed,
                data_override=views or None)
        elif method in ("subtract", "subtract_recover"):
            theta = saved[0].copy()
            reps = dict(self.reputations)
            for t in range(ck + 1, T + 1):
                for o in owners:
                    w = self.round_meta[t]["weights"].get(o, 0.0)
                    if w > 0 and o in self.updates.get(t, {}):
                        theta = theta + self.lr * w * self.updates[t][o]
            for o in owners:
                reps[o] = 0.0
            self.ledger.log_round(T, "recover", [], [], {}, [], digest_array(theta))
            if method == "subtract_recover" and (recover_rounds or 0) > 0:
                theta, reps, cost = self._run_rounds(
                    theta, reps, T, recover_rounds, active, "recover", seed)
        elif method == "replay":
            theta, reps = theta0.copy(), dict(reps0)
            for t in range(ck + 1, T + 1):
                meta = self.round_meta[t]
                ids = [c for c in meta["participants"] if c not in owners]
                if len(ids) == 0:
                    continue
                w = np.array([meta["weights"].get(c, 0.0) for c in ids])
                if w.sum() == 0:
                    continue
                w = w / w.sum()
                agg = sum(wi * self.updates[t][c] for wi, c in zip(w, ids))
                theta = theta - self.lr * agg
            self.ledger.log_round(T, "recover", [], [], {}, [], digest_array(theta))

        if method in ("subtract", "replay"):
            # derived checkpoints and stored updates still embed the client's
            # influence; drop them and keep only the unlearned head state
            for t in [t for t in self.checkpoints if t > ck]:
                del self.checkpoints[t]
            for t in [t for t in self.updates if t > ck]:
                for o in owners:
                    self.updates[t].pop(o, None)
            self.checkpoints[T] = (theta.copy(), dict(reps))
        self.theta, self.reputations = theta.copy(), dict(reps)
        for o in owners:
            self.consent.mark_fulfilled(o)
        self.ledger.log_release(digest_array(theta), f"unlearn:{method}")
        return {"theta": theta, "reputations": reps, "cost": cost,
                "checkpoint": ck, "method": method,
                "model_hash": digest_array(theta)}

    # ------------------------------------------------------------------
    def storage_bytes(self) -> Dict[str, int]:
        ck = sum(t.nbytes + 8 * len(r) for t, r in self.checkpoints.values())
        up = sum(g.nbytes for r in self.updates.values() for g in r.values())
        return {"checkpoints": ck, "updates": up, "ledger": self.ledger.storage_bytes()}


class ShardedFL:
    """
    Cohort-isolated training: clients are split into S shards, each trained as an
    independent FORTRESS-FL federation, and the served model is the sample-weighted
    average of the shard models. A client-level request retrains one shard only.
    """

    def __init__(self, clients: List[Dict], dim: int, n_shards: int = 4, **kw):
        self.dim = dim
        self.shards: List[UnlearnableFL] = []
        self.owner: Dict[str, int] = {}
        buckets = [[] for _ in range(n_shards)]
        for i, c in enumerate(clients):
            buckets[i % n_shards].append(c)
            self.owner[c["id"]] = i % n_shards
        shared = None
        for b in buckets:
            fl = UnlearnableFL(b, dim, comm_params=shared, **kw)
            shared = fl.comm_params
            self.shards.append(fl)
        self.n_samples = [sum(len(c["y"]) for c in b) for b in buckets]

    @property
    def theta(self) -> np.ndarray:
        w = np.array(self.n_samples, dtype=float)
        w /= w.sum()
        return sum(wi * s.theta for wi, s in zip(w, self.shards))

    def train(self, n_rounds: int) -> np.ndarray:
        for s in self.shards:
            s.train(n_rounds)
        return self.theta

    def unlearn(self, client: str, seed_offset: int = 1) -> Dict:
        i = self.owner[client]
        res = self.shards[i].unlearn(UnlearnRequest("client", client=client),
                                     "retrain", seed_offset=seed_offset)
        self.n_samples[i] -= len(self.shards[i].clients[client]["y"])
        res["theta"] = self.theta
        return res

    def unlearn_wave(self, clients: List[str], seed_offset: int = 1) -> Dict:
        """Retrain every shard that holds a member of the wave (others are untouched)."""
        groups: Dict[int, List[str]] = {}
        for c in clients:
            groups.setdefault(self.owner[c], []).append(c)
        cost = {"bytes": 0.0, "flops": 0.0, "energy_j": 0.0, "rounds": 0}
        for i, grp in groups.items():
            res = self.shards[i].unlearn(UnlearnRequest("client", client=grp[0], clients=grp),
                                         "retrain", seed_offset=seed_offset)
            for k in cost:
                cost[k] += res["cost"].get(k, 0)
            self.n_samples[i] -= sum(len(self.shards[i].clients[c]["y"]) for c in grp)
        return {"theta": self.theta, "cost": cost, "shards_touched": len(groups)}
