"""
Public-key signatures and external anchoring for the provenance ledger.

A hash chain held by the operator can be rewritten and re-linked wholesale, so on its
own it only detects partial edits. Anchoring publishes the chain head at intervals to
independent witnesses; a verifier then compares the ledger it is shown with what the
witnesses recorded. Certificates are signed with Ed25519 so that third parties verify
with a public key instead of a shared secret.
"""

import json
from typing import Dict, List, Optional, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (Ed25519PrivateKey,
                                                               Ed25519PublicKey)

from .provenance import ProvenanceLedger


def generate_keypair() -> Tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    sk = Ed25519PrivateKey.generate()
    return sk, sk.public_key()


def public_bytes(pk: Ed25519PublicKey) -> bytes:
    return pk.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def _canon(payload: Dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def sign_certificate_ed25519(payload: Dict, sk: Ed25519PrivateKey) -> Dict:
    return {"body": payload, "sig": sk.sign(_canon(payload)).hex()}


def verify_certificate_ed25519(cert: Dict, pk: Ed25519PublicKey) -> bool:
    try:
        pk.verify(bytes.fromhex(cert["sig"]), _canon(cert["body"]))
        return True
    except (InvalidSignature, ValueError):
        return False


class Witness:
    """An independent party that stores signed ledger heads it was sent."""

    def __init__(self, name: str):
        self.name = name
        self.sk, self.pk = generate_keypair()
        self.heads: Dict[int, str] = {}
        self.receipts: List[Dict] = []

    def receive(self, idx: int, head: str, operator_sig: bytes, operator_pk: Ed25519PublicKey) -> Dict:
        operator_pk.verify(operator_sig, f"{idx}:{head}".encode())
        self.heads[idx] = head
        rec = sign_certificate_ed25519({"witness": self.name, "idx": idx, "head": head}, self.sk)
        self.receipts.append(rec)
        return rec


class LedgerAnchor:
    """Operator-side publisher: signs and sends the chain head to every witness."""

    def __init__(self, witnesses: List[Witness], interval: int = 10):
        self.sk, self.pk = generate_keypair()
        self.witnesses = witnesses
        self.interval = interval
        self.last_anchored = -1

    def maybe_publish(self, ledger: ProvenanceLedger, force: bool = False) -> Optional[int]:
        idx = len(ledger.records) - 1
        if idx < 0:
            return None
        if force or idx - self.last_anchored >= self.interval:
            head = ledger.records[idx]["hash"]
            sig = self.sk.sign(f"{idx}:{head}".encode())
            for w in self.witnesses:
                w.receive(idx, head, sig, self.pk)
            self.last_anchored = idx
            return idx
        return None


def verify_against_witnesses(ledger: ProvenanceLedger, witnesses: List[Witness],
                             quorum: int = 1) -> Dict:
    """
    Check the ledger a verifier is shown against the heads the witnesses recorded.
    Rewriting any record at or before the last anchor changes every later hash, so it
    disagrees with at least ``quorum`` witnesses unless the operator also corrupted them.
    """
    ok_chain = ledger.verify_chain()
    bad = 0
    checked = 0
    for w in witnesses:
        agree = True
        for idx, head in w.heads.items():
            checked += 1
            if idx >= len(ledger.records) or ledger.records[idx]["hash"] != head:
                agree = False
        if not agree:
            bad += 1
    honest = len(witnesses) - bad
    return {"chain_valid": ok_chain, "witnesses_agreeing": honest, "witnesses": len(witnesses),
            "passed": ok_chain and honest >= quorum and bad == 0, "heads_checked": checked}
