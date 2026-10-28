"""Power-adaptive canary calibration and deadline-aware wave scheduling."""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .certify import certify_model


def calibrate_canaries(make_pilot: Callable[[float, int], Tuple[object, Dict, List[str]]],
                       scales: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
                       pair_counts: Sequence[int] = (6, 12, 24), eps: float = 0.2,
                       alpha: float = 0.05, seed: int = 0) -> Dict:
    """
    Choose the cheapest canary design for which the audit has power.

    ``make_pilot(scale, m)`` returns (federation trained for a pilot, manifests,
    audited client ids). A design has power when the audit flags the pilot model
    (which still contains every audited client's influence). Designs are tried in
    order of canary records per client (m), then scale, and the first powered one
    is returned; None means no design in the grid has power.
    """
    trials = []
    for m in pair_counts:
        for s in scales:
            fl, man, ids = make_pilot(s, m)
            cert, _ = certify_model(fl.theta, man, ids, eps, alpha, reveal_source="seeded",
                                    seed=seed, model=fl.model)
            trials.append({"m": m, "scale": s, "status": cert.status,
                           "delta_upper": cert.delta_upper, "log_e_rev": cert.log_e_rev})
            if cert.status == "REVOKED":
                return {"chosen": {"m": m, "scale": s}, "trials": trials}
    return {"chosen": None, "trials": trials}


class WaveScheduler:
    """
    Deadline-aware batching of revocation requests. A request waits until the queue
    holds ``pairs_needed`` canary pairs (so the wave certificate has power) or until
    its age reaches ``deadline`` (a legal response period, minus ``margin``), whichever
    comes first. Times are in days.
    """

    def __init__(self, pairs_per_client: int, pairs_needed: int, deadline: float,
                 margin: float = 1.0):
        self.m, self.need = pairs_per_client, pairs_needed
        self.deadline, self.margin = deadline, margin
        self.queue: List[Tuple[float, str]] = []

    def submit(self, t: float, client: str) -> None:
        self.queue.append((t, client))

    def due(self, now: float) -> Optional[List[str]]:
        if not self.queue:
            return None
        full = len(self.queue) * self.m >= self.need
        late = now - self.queue[0][0] >= self.deadline - self.margin
        if full or late:
            wave = [c for _, c in self.queue]
            self.queue = []
            return wave
        return None


def simulate_scheduler(rate_per_day: float, horizon_days: float, pairs_per_client: int,
                       pairs_needed: int, deadline: float, wave_energy_j: float,
                       seed: int = 0, margin: float = 1.0) -> Dict:
    """Poisson arrivals, one decision per day; returns latency and energy statistics."""
    rng = np.random.default_rng(seed)
    sch = WaveScheduler(pairs_per_client, pairs_needed, deadline, margin)
    t, arrivals = 0.0, []
    while True:
        t += rng.exponential(1.0 / rate_per_day)
        if t > horizon_days:
            break
        arrivals.append(t)
    lat, waves, underpowered, n = [], 0, 0, 0
    qtimes: Dict[str, float] = {}
    day, i = 0, 0
    while day <= horizon_days + deadline:
        while i < len(arrivals) and arrivals[i] <= day:
            cid = f"R{i}"
            qtimes[cid] = arrivals[i]
            sch.submit(arrivals[i], cid)
            i += 1
        w = sch.due(float(day))
        if w:
            waves += 1
            n += len(w)
            underpowered += (len(w) * pairs_per_client < pairs_needed)
            lat += [day - qtimes[c] for c in w]
        day += 1
    return {"requests": n, "waves": waves, "mean_latency": float(np.mean(lat)) if lat else 0.0,
            "max_latency": float(np.max(lat)) if lat else 0.0,
            "energy_per_request_j": waves * wave_energy_j / max(1, n),
            "underpowered_waves": underpowered}
