"""Certification of revocation waves and the verify-then-escalate policy."""

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vouch.verify import VouchConfig, VouchVerifier

from .canary import cohort_diffs
from .system import UnlearnableFL, UnlearnRequest


def certify_model(theta: np.ndarray, manifests: Dict, client_ids: Sequence[str],
                  eps: float = 0.2, alpha: float = 0.05, manifest_digest: str = "",
                  reveal_source: str = "local", seed: Optional[int] = None):
    diffs = cohort_diffs(theta, manifests, client_ids)
    ver = VouchVerifier(["loss", "margin"], VouchConfig(eps=eps, alpha=alpha),
                        manifest_sha256=manifest_digest)
    cert = ver.run(diffs, shuffle_seed=seed, early_stop=True, reveal_source=reveal_source)
    return cert, diffs


LADDER = [("subtract", {}),
          ("subtract_recover", {"recover_rounds": 5}),
          ("subtract_recover", {"recover_rounds": 20}),
          ("rollback", {})]


def _label(m, kw):
    return m if not kw else f"{m}+{kw['recover_rounds']}"


def verify_then_escalate(fl: UnlearnableFL, clients: Sequence[str], manifests: Dict,
                         ladder=LADDER, eps: float = 0.2, alpha: float = 0.05,
                         commit: bool = True, reveal_source: str = "local",
                         seed: Optional[int] = None, val: Optional[Dict] = None,
                         max_drop: float = 0.03) -> Dict:
    """
    Try unlearning methods from cheapest to most expensive on a scratch copy and
    release the first model whose wave certificate is ISSUED. The error budget
    alpha is split evenly over the K rungs, so the probability that any leaky
    model is ever certified is at most alpha (union bound). If no rung issues,
    the most expensive rung is released as UNCERTIFIED, never as certified.

    A rung is also rejected when accuracy on the server's validation set ``val``
    falls more than ``max_drop`` below the model before unlearning: forgetting
    that wrecks the model (plain update subtraction can) is not accepted.
    """
    from .verification import accuracy
    from .provenance import digest_array
    K = len(ladder)
    req = UnlearnRequest("client", client=clients[0], clients=list(clients))
    spent = {"bytes": 0.0, "flops": 0.0, "energy_j": 0.0, "rounds": 0}
    base_acc = accuracy(fl.theta, val) if val is not None else None
    attempts = []
    chosen = None
    for i, (m, kw) in enumerate(ladder):
        f = copy.deepcopy(fl)
        out = f.unlearn(req, m, **kw)
        for k in spent:
            spent[k] += out["cost"].get(k, 0)
        cert, _ = certify_model(out["theta"], manifests, clients, eps, alpha / K,
                                manifest_digest=digest_array(out["theta"])[:0],
                                reveal_source=reveal_source, seed=seed)
        acc = accuracy(out["theta"], val) if val is not None else None
        ok_util = base_acc is None or acc >= base_acc - max_drop
        attempts.append({"method": _label(m, kw), "status": cert.status, "val_acc": acc,
                         "utility_ok": ok_util,
                         "t_stop": cert.t_stop, "delta_upper": cert.delta_upper,
                         "log_e_cert": cert.log_e_cert, "log_e_rev": cert.log_e_rev,
                         "cost": dict(out["cost"])})
        chosen = (f, out, cert, _label(m, kw))
        if cert.status == "ISSUED" and ok_util:
            break
    f, out, cert, name = chosen
    status = cert.status if attempts[-1]["utility_ok"] else "UTILITY_REJECTED"
    if commit:
        fl.__dict__.update(f.__dict__)
    return {"released": name, "status": status, "theta": out["theta"],
            "attempts": attempts, "cost": spent, "certificate": cert,
            "model_hash": out["model_hash"]}
