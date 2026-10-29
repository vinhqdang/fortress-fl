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


def test_icr_certifies_exact_and_respects_alpha_split():
    from fortress_fl.unlearning import incremental_certified_recovery
    clients, test, info = data.build_federation("synthetic", n_clients=12, n_per_client=12,
                                                dim=128, seed=5)
    man = plant_canaries(clients, 6, info["dim"], seed=5)
    fl = UnlearnableFL(clients, info["dim"], participation=0.8, seed=5, security_param=256)
    fl.train(20)
    tg = [c["id"] for c in clients[:6]]
    r = incremental_certified_recovery(fl, tg, man, eps=0.3, block=4, max_rounds=8, seed=0,
                                       commit=False)
    assert r["rounds"] <= 8 and len(r["attempts"]) >= 1
    assert r["rounds"] == 4 * (len(r["attempts"]) - 1)
    assert r["cost"]["rounds"] == r["rounds"]


def test_entanglement_selection_size():
    from fortress_fl.unlearning import select_entangled
    fl, clients, _ = small_system()
    fl.train(6)
    tg = [clients[0]["id"], clients[1]["id"]]
    fl.unlearn(UnlearnRequest("client", client=tg[0], clients=tg), "subtract")
    sel = select_entangled(fl, tg, 0.5)
    assert 4 <= len(sel) <= 6 and not set(sel) & set(tg)


def test_real_twins_hold_out_ghosts():
    from fortress_fl.unlearning import plant_real_twins
    clients, _, _ = data.build_federation("synthetic", n_clients=4, n_per_client=20, dim=8, seed=6)
    n0 = [len(c["y"]) for c in clients]
    man = plant_real_twins(clients, 0.5, seed=0)
    for c, n in zip(clients, n0):
        pairs = man[c["id"]]["pairs"]
        assert len(pairs) == int(0.5 * n) // 2
        assert len(c["y"]) == n - len(pairs)
        trained = {tuple(x.round(6)) for x in c["X"]}
        assert all(tuple(p["in"][0].round(6)) in trained for p in pairs)
        assert all(tuple(p["ghost"][0].round(6)) not in trained for p in pairs)


def test_scheduler_batches_until_enough_pairs_or_deadline():
    from fortress_fl.unlearning import WaveScheduler
    s = WaveScheduler(pairs_per_client=6, pairs_needed=24, deadline=10, margin=1)
    for i in range(3):
        s.submit(float(i), f"R{i}")
    assert s.due(3.0) is None
    s.submit(3.0, "R3")
    assert s.due(3.0) == ["R0", "R1", "R2", "R3"]
    s.submit(4.0, "R4")
    assert s.due(12.0) is None and s.due(13.0) == ["R4"]


def test_anchor_detects_relinked_rewrite_and_signatures():
    import copy
    from fortress_fl.unlearning.anchor import (Witness, LedgerAnchor, verify_against_witnesses,
                                               generate_keypair, sign_certificate_ed25519,
                                               verify_certificate_ed25519)
    fl, _, _ = small_system()
    fl.train(10)
    ws = [Witness("A"), Witness("B")]
    an = LedgerAnchor(ws, interval=3)
    an.maybe_publish(fl.ledger, force=True)
    assert verify_against_witnesses(fl.ledger, ws)["passed"]
    g = copy.deepcopy(fl.ledger)
    g.records[2]["flagged"] = ["Cxx"]
    prev = g.records[1]["hash"]
    for r in g.records[2:]:
        r["prev"] = prev
        r["hash"] = g._record_hash({k: v for k, v in r.items() if k != "hash"})
        prev = r["hash"]
    assert g.verify_chain() and not verify_against_witnesses(g, ws)["passed"]
    sk, pk = generate_keypair()
    c = sign_certificate_ed25519({"s": "ISSUED"}, sk)
    assert verify_certificate_ed25519(c, pk)
    c["body"]["s"] = "REVOKED"
    assert not verify_certificate_ed25519(c, pk)


def test_dropout_staleness_and_enrolment():
    clients, test, info = data.build_federation("synthetic", n_clients=10, n_per_client=20, dim=16, seed=7)
    jr = {c["id"]: (0 if i < 5 else 6) for i, c in enumerate(clients)}
    fl = UnlearnableFL(clients, info["dim"], participation=0.9, dropout=0.4, max_staleness=2,
                       join_round=jr, seed=7, security_param=256)
    fl.train(8)
    assert all(np.isfinite(fl.theta))
    late = clients[9]["id"]
    first = fl.ledger.first_participation(late)
    assert first is None or first >= 7


def test_crossover_canaries_and_strict_reveal():
    clients, _, info = data.build_federation("synthetic", n_clients=4, n_per_client=20, dim=12, seed=8)
    man = plant_canaries(clients, 5, info["dim"], seed=8, kind="crossover")
    from fortress_fl.unlearning import cohort_diffs
    th = np.zeros(info["dim"])
    ids = [c["id"] for c in clients[:2]]
    assert len(cohort_diffs(th, man, ids)) == 10
    assert len(cohort_diffs(th, man, ids, revealed={ids[0]: [0], ids[1]: []}, missing="drop")) == 1
    worst = cohort_diffs(th, man, ids, revealed={ids[0]: [0], ids[1]: []}, missing="worst")
    assert len(worst) == 10 and sum(d["loss"] > 1e5 for d in worst) == 9


def test_early_stopped_retrain_and_sharded_wave():
    from fortress_fl.unlearning import early_stopped_retrain
    clients, test, info = data.build_federation("synthetic", n_clients=10, n_per_client=20, dim=16, seed=9)
    fl = UnlearnableFL(clients, info["dim"], participation=0.9, seed=9, security_param=256)
    fl.train(10)
    val = {"X": test["X"][:100], "y": test["y"][:100]}
    r = early_stopped_retrain(fl, [clients[0]["id"], clients[1]["id"]], val, block=2, max_rounds=10)
    assert 2 <= r["rounds"] <= 10 and r["cost"]["rounds"] == r["rounds"]
    sh = ShardedFL(clients, info["dim"], n_shards=2, participation=0.9, seed=9, security_param=256)
    sh.train(4)
    o = sh.unlearn_wave([clients[0]["id"], clients[2]["id"]])
    assert o["shards_touched"] == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
