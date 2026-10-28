"""Tests for the federated unlearning and provenance layer."""

import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fortress_fl.unlearning import (UnlearnableFL, UnlearnRequest, ShardedFL, data,
                                    plant_canaries, certify_model, sign_certificate,
                                    verify_certificate, verify_merkle_proof)
from fortress_fl.unlearning.provenance import (ProvenanceLedger, ConsentRegistry,
                                               merkle_root, merkle_proof, _leaf)
from fortress_fl.unlearning import verification as V


def small_system(n=8, d=24, seed=0, **kw):
    clients, test, info = data.build_federation("synthetic", n_clients=n, n_per_client=20,
                                                dim=d, alpha=1.0, seed=seed)
    fl = UnlearnableFL(clients, info["dim"], participation=0.8, seed=seed,
                       security_param=256, **kw)
    return fl, clients, test


def test_merkle_proofs():
    leaves = [_leaf(f"C{i}", f"d{i}") for i in range(7)]
    root = merkle_root(leaves)
    for i in range(7):
        assert verify_merkle_proof(leaves[i], merkle_proof(leaves, i), root)
    assert not verify_merkle_proof(leaves[0], merkle_proof(leaves, 1), root)


def test_ledger_detects_tampering():
    fl, _, _ = small_system()
    fl.train(6)
    assert fl.ledger.verify_chain()
    rec = fl.ledger.records[3]
    rec["model_hash"] = rec["model_hash"][:-1] + ("0" if rec["model_hash"][-1] != "0" else "1")
    assert not fl.ledger.verify_chain()


def test_consent_registry():
    reg = ConsentRegistry(["C00", "C01"])
    reg.revoke("C00", {"kind": "client"}, 5)
    assert not reg.is_active("C00") and reg.active_clients() == ["C01"]
    assert len(reg.pending()) == 1
    reg.mark_fulfilled("C00")
    assert reg.pending() == []


def test_methods_run_and_audit():
    fl, clients, test = small_system()
    fl.train(10)
    tg = [clients[0]["id"], clients[1]["id"]]
    req = UnlearnRequest("client", client=tg[0], clients=tg)
    for m, kw in [("retrain", {}), ("rollback", {}), ("subtract", {}), ("replay", {}),
                  ("subtract_recover", {"recover_rounds": 3})]:
        f = copy.deepcopy(fl)
        out = f.unlearn(req, m, **kw)
        assert np.all(np.isfinite(out["theta"]))
        assert f.ledger.audit_exclusion(tg, out["model_hash"])["passed"], m
        assert not any(f.consent.is_active(c) for c in tg)
        assert out["cost"]["rounds"] in (0, 3, 10 - out["checkpoint"], 10)


def test_audit_rejects_false_lineage():
    fl, clients, _ = small_system()
    fl.train(6)
    tg = [clients[0]["id"]]
    fl.ledger.log_rollback({"kind": "client"}, "fake", 0, fl.ledger.records[0]["model_hash"], tg)
    fl._run_rounds(fl.theta, fl.reputations, 6, 2, fl.ids, "recover", 5)
    rep = fl.ledger.audit_exclusion(tg, fl.ledger.records[-1]["model_hash"])
    assert not rep["passed"]


def test_subtract_removes_stored_contribution():
    fl, clients, _ = small_system()
    fl.train(6)
    cid = clients[0]["id"]
    out = fl.unlearn(UnlearnRequest("client", client=cid), "subtract")
    assert all(cid not in u for u in fl.updates.values())


def test_sharded_unlearn_touches_one_shard():
    clients, test, info = data.build_federation("synthetic", n_clients=8, n_per_client=20,
                                                dim=24, seed=1)
    sh = ShardedFL(clients, info["dim"], n_shards=2, participation=1.0, seed=1,
                   security_param=256)
    sh.train(5)
    before = [s.theta.copy() for s in sh.shards]
    cid = clients[0]["id"]
    sh.unlearn(cid)
    changed = [not np.allclose(a, s.theta) for a, s in zip(before, sh.shards)]
    assert changed.count(True) <= 1


def test_certificate_binding():
    cert = sign_certificate({"head": "abc", "status": "ISSUED"}, b"k")
    assert verify_certificate(cert, b"k")
    cert["body"]["status"] = "REVOKED"
    assert not verify_certificate(cert, b"k")


def test_certify_symmetric_scores_are_not_revoked():
    clients, test, info = data.build_federation("synthetic", n_clients=6, n_per_client=10,
                                                dim=16, seed=2)
    man = plant_canaries(clients, 20, info["dim"], seed=2)
    theta = np.zeros(info["dim"])        # a model that never saw any canary
    cert, _ = certify_model(theta, man, [c["id"] for c in clients], eps=0.3,
                            reveal_source="seeded", seed=0)
    assert cert.status != "REVOKED"


def test_mlp_gradient_matches_finite_differences():
    from fortress_fl.unlearning import MLPModel
    rng = np.random.default_rng(0)
    m = MLPModel(6, hidden=4)
    X, y = rng.normal(size=(9, 6)), (rng.random(9) < 0.5).astype(float)
    th = m.init(1) + 0.1 * rng.normal(size=m.n_params)

    def loss(t):
        p = 1 / (1 + np.exp(-m.logit(t, X)))
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    g = m.grad(th, X, y)
    for i in rng.choice(m.n_params, 12, replace=False):
        e = np.zeros(m.n_params); e[i] = 1e-6
        assert abs((loss(th + e) - loss(th - e)) / 2e-6 - g[i]) < 1e-6


def test_mlp_system_unlearns_and_audits():
    from fortress_fl.unlearning import MLPModel
    clients, test, info = data.build_federation("synthetic", n_clients=8, n_per_client=20,
                                                dim=16, seed=3)
    model = MLPModel(info["dim"], hidden=4)
    fl = UnlearnableFL(clients, info["dim"], model=model, participation=0.8, seed=3,
                       security_param=256)
    fl.train(6)
    tg = [clients[0]["id"]]
    out = fl.unlearn(UnlearnRequest("client", client=tg[0]), "subtract_recover",
                     recover_rounds=2)
    assert out["theta"].shape == (model.n_params,)
    assert fl.ledger.audit_exclusion(tg, out["model_hash"])["passed"]


def test_escalation_reports_unpowered_audit():
    from fortress_fl.unlearning import verify_then_escalate
    clients, test, info = data.build_federation("synthetic", n_clients=8, n_per_client=20,
                                                dim=8, seed=4)
    man = plant_canaries(clients, 3, info["dim"], seed=4)
    fl = UnlearnableFL(clients, info["dim"], participation=1.0, seed=4, security_param=256)
    fl.train(4)
    tg = [c["id"] for c in clients[:4]]
    res = verify_then_escalate(fl, tg, man, eps=0.3, reveal_source="seeded", seed=0)
    assert res["audit_power"] is False and res["status"] != "ISSUED"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
