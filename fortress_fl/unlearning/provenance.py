"""
Update provenance and consent management for federated unlearning.

The ledger is an append-only, hash-chained log. Every training round stores a
Merkle root over the (client, update-digest) leaves, the aggregation weights,
the flagged clients and the digest of the resulting global model. Unlearning
requests are recorded as ``rollback`` events that name the checkpoint the model
was restored to, so an auditor can later check that the model that is served
descends from a state that no revoked client has ever influenced.
"""

import hashlib
import hmac
import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

GENESIS = "0" * 64


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_array(arr: np.ndarray) -> str:
    """Digest of a float vector (stored as float32, the wire format)."""
    return sha256_hex(np.ascontiguousarray(arr, dtype=np.float32).tobytes())


# --------------------------------------------------------------------------
# Merkle tree
# --------------------------------------------------------------------------

def _leaf(client_id: str, update_digest: str) -> str:
    return sha256_hex(b"\x00" + client_id.encode() + b"|" + update_digest.encode())


def _node(left: str, right: str) -> str:
    return sha256_hex(b"\x01" + bytes.fromhex(left) + bytes.fromhex(right))


def merkle_root(leaves: List[str]) -> str:
    if not leaves:
        return GENESIS
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [_node(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def merkle_proof(leaves: List[str], index: int) -> List[Tuple[str, str]]:
    """Authentication path for ``leaves[index]`` as (side, sibling_hash) pairs."""
    proof = []
    level = list(leaves)
    idx = index
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        sibling = idx ^ 1
        proof.append(("L" if sibling < idx else "R", level[sibling]))
        level = [_node(level[i], level[i + 1]) for i in range(0, len(level), 2)]
        idx //= 2
    return proof


def verify_merkle_proof(leaf: str, proof: List[Tuple[str, str]], root: str) -> bool:
    h = leaf
    for side, sibling in proof:
        h = _node(sibling, h) if side == "L" else _node(h, sibling)
    return h == root


# --------------------------------------------------------------------------
# Consent registry
# --------------------------------------------------------------------------

class ConsentRegistry:
    """Tracks consent grants and revocations per client and scope."""

    def __init__(self, client_ids: List[str]):
        self.state: Dict[str, Dict] = {
            c: {"granted": True, "revocations": []} for c in client_ids
        }

    def revoke(self, client_id: str, scope: Dict, round_idx: int) -> Dict:
        entry = {"scope": scope, "round": round_idx, "fulfilled": False}
        self.state[client_id]["revocations"].append(entry)
        if scope.get("kind") == "client":
            self.state[client_id]["granted"] = False
        return entry

    def is_active(self, client_id: str) -> bool:
        return self.state[client_id]["granted"]

    def active_clients(self) -> List[str]:
        return [c for c, s in self.state.items() if s["granted"]]

    def mark_fulfilled(self, client_id: str) -> None:
        for r in self.state[client_id]["revocations"]:
            r["fulfilled"] = True

    def pending(self) -> List[Tuple[str, Dict]]:
        return [(c, r) for c, s in self.state.items()
                for r in s["revocations"] if not r["fulfilled"]]


# --------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------

class ProvenanceLedger:
    """Append-only hash-chained log of training and unlearning events."""

    def __init__(self):
        self.records: List[Dict] = []
        self._leaves: Dict[int, List[str]] = {}

    @staticmethod
    def _record_hash(body: Dict) -> str:
        payload = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return sha256_hex(payload.encode())

    def _append(self, body: Dict) -> Dict:
        body = dict(body)
        body["idx"] = len(self.records)
        body["prev"] = self.records[-1]["hash"] if self.records else GENESIS
        body["hash"] = self._record_hash(body)
        self.records.append(body)
        return body

    # -- logging -----------------------------------------------------------
    def log_round(self, round_idx: int, phase: str, participants: List[str],
                  update_digests: List[str], weights: Dict[str, float],
                  flagged: List[str], model_hash: str) -> Dict:
        leaves = [_leaf(c, d) for c, d in zip(participants, update_digests)]
        self._leaves[len(self.records)] = leaves
        return self._append({
            "event": "round", "phase": phase, "round": round_idx,
            "participants": list(participants),
            "merkle_root": merkle_root(leaves),
            "weights": {k: round(float(v), 8) for k, v in weights.items()},
            "flagged": list(flagged), "model_hash": model_hash,
        })

    def log_rollback(self, request: Dict, method: str, checkpoint_round: int,
                     checkpoint_hash: str, excluded: List[str]) -> Dict:
        return self._append({
            "event": "rollback", "request": request, "method": method,
            "checkpoint_round": checkpoint_round,
            "checkpoint_hash": checkpoint_hash, "excluded": list(excluded),
        })

    def log_release(self, model_hash: str, note: str = "") -> Dict:
        return self._append({"event": "release", "model_hash": model_hash,
                             "note": note})

    # -- queries -----------------------------------------------------------
    def verify_chain(self) -> bool:
        prev = GENESIS
        for i, rec in enumerate(self.records):
            body = {k: v for k, v in rec.items() if k != "hash"}
            if rec["idx"] != i or rec["prev"] != prev:
                return False
            if self._record_hash(body) != rec["hash"]:
                return False
            prev = rec["hash"]
        return True

    def contributions(self, client_id: str, phase: Optional[str] = None) -> List[Dict]:
        out = []
        for rec in self.records:
            if rec["event"] != "round":
                continue
            if phase is not None and rec["phase"] != phase:
                continue
            if client_id in rec["participants"]:
                out.append({"round": rec["round"], "idx": rec["idx"],
                            "weight": rec["weights"].get(client_id, 0.0),
                            "flagged": client_id in rec["flagged"]})
        return out

    def first_participation(self, client_id: str) -> Optional[int]:
        c = self.contributions(client_id)
        return c[0]["round"] if c else None

    def inclusion_proof(self, record_idx: int, client_id: str) -> Optional[Dict]:
        rec = self.records[record_idx]
        if rec["event"] != "round" or client_id not in rec["participants"]:
            return None
        i = rec["participants"].index(client_id)
        leaves = self._leaves[record_idx]
        return {"leaf": leaves[i], "proof": merkle_proof(leaves, i),
                "root": rec["merkle_root"]}

    def storage_bytes(self) -> int:
        return len(json.dumps(self.records, separators=(",", ":")).encode())

    def head(self) -> str:
        return self.records[-1]["hash"] if self.records else GENESIS

    # -- audit -------------------------------------------------------------
    def audit_exclusion(self, client_ids: List[str], final_model_hash: str) -> Dict:
        """
        Check that the released model descends from a checkpoint untouched by
        the revoked clients and that no later round used their updates.

        The check is structural: it certifies lineage, not statistical
        indistinguishability (see ``verification.py`` for the latter).
        """
        report = {"chain_valid": self.verify_chain(), "checks": {}}
        rb_idx = None
        for rec in self.records:
            if rec["event"] == "rollback" and set(client_ids) <= set(rec["excluded"]):
                rb_idx = rec["idx"]
        report["checks"]["rollback_recorded"] = rb_idx is not None
        if rb_idx is None:
            report["passed"] = False
            return report
        rb = self.records[rb_idx]

        # The checkpoint must exist in the log from before the rollback ...
        ck_ok = False
        ck_clean = True
        for rec in self.records[:rb_idx]:
            if rec["event"] == "round" and rec["model_hash"] == rb["checkpoint_hash"] \
                    and rec["round"] == rb["checkpoint_round"]:
                ck_ok = True
        # ... and no revoked client may appear in any round that produced it.
        for rec in self.records[:rb_idx]:
            if rec["event"] == "round" and rec["round"] <= rb["checkpoint_round"] \
                    and rec["phase"] == "train":
                if set(client_ids) & set(rec["participants"]):
                    ck_clean = False
        if rb["checkpoint_round"] == 0:
            ck_ok, ck_clean = True, True  # initial model: no training yet
        report["checks"]["checkpoint_in_log"] = ck_ok
        report["checks"]["checkpoint_clean"] = ck_clean

        # Rounds after the rollback must exclude the revoked clients.
        post_clean = True
        last_hash = rb["checkpoint_hash"]
        for rec in self.records[rb_idx + 1:]:
            if rec["event"] == "round":
                if set(client_ids) & set(rec["participants"]):
                    post_clean = False
                last_hash = rec["model_hash"]
        report["checks"]["post_rollback_excludes"] = post_clean
        report["checks"]["final_hash_matches"] = (last_hash == final_model_hash)
        report["passed"] = (report["chain_valid"] and
                            all(report["checks"].values()))
        return report


def sign_certificate(payload: Dict, key: bytes) -> Dict:
    """HMAC-SHA256 tag over a canonical encoding of the certificate body."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {"body": payload, "tag": hmac.new(key, body, hashlib.sha256).hexdigest(),
            "issued": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def verify_certificate(cert: Dict, key: bytes) -> bool:
    body = json.dumps(cert["body"], sort_keys=True, separators=(",", ":")).encode()
    expected = hmac.new(key, body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, cert["tag"])
