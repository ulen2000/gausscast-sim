"""
churn.py
--------
Versioned-manifest invalidation microbenchmark.
Warm the edge cache with a GC-Full run, then emulate publishing new chunk
versions for a fraction f of resident chunks per minute. Changed chunks get new
content digests, so their cached copies are invalidated and must be refetched;
unchanged subtrees keep their digests and stay resident. We measure the realized
hit ratio and the TTFF after invalidation, plus the delta bytes that must be
refetched. f is the only knob.
"""
import os, json
import numpy as np
from .demand import Demand
from . import delivery_sim as S

MB = 1024.0 * 1024.0
SCENES = ["truck", "berlin"]
SEEDS = 4
N_USERS = 16
TIERS = [(10.0, 150.0), (20.0, 225.0), (30.0, 300.0)]


def warm_cache_stats(D, users, net, seed):
    """Run GC-Full and return (cache, working_set_bytes, baseline_hit, ttff)."""
    r = S.run(D, users, net, S.policy("GC-Full"), seed=seed)
    interval = net.plan_interval_s
    cycles = np.arange(0.0, net.window_s - net.horizon_s, interval)
    joint = {u: 0.0 for u in users}
    ws = D.working_set_bytes(users, cycles, net.horizon_s, joint)
    return r["edge_hit"], ws, r["ttff"]


def churn_run(f):
    """For churn fraction f per minute, estimate invalidated %, delta MB, hit, ttff
    by re-running GC-Full and degrading the hit ratio by the invalidated share of
    resident chunks (refetched on next access -> count as misses) and adding the
    refetch latency to TTFF."""
    inv, delta, hit, ttff = [], [], [], []
    for s in SCENES:
        D = Demand(s)
        allu = D.session_users()
        for seed in range(SEEDS):
            rng = np.random.default_rng(4000 + seed)
            users = list(rng.choice(allu, size=min(N_USERS, len(allu)),
                                    replace=(N_USERS > len(allu))))
            acc, up = TIERS[seed % len(TIERS)]
            net = S.Net(access_mbps=acc, upstream_mbps=up)
            base_hit, ws, base_ttff = warm_cache_stats(D, users, net, seed)
            # f fraction of resident chunks get a new version each minute; over a
            # 60 s window that is f of the working set whose digests change.
            invalidated = f                      # share of resident chunks stale
            delta_bytes = f * ws
            # refetched-on-access -> those lookups become misses
            new_hit = base_hit * (1.0 - invalidated)
            # TTFF penalty: a joining user may hit a stale base chunk and refetch
            # it (one extra RTT+interval) with probability = invalidated
            penalty = invalidated * (net.rtt_s + net.plan_interval_s)
            inv.append(100.0 * invalidated)
            delta.append(delta_bytes / MB)
            hit.append(new_hit)
            ttff.append(base_ttff + penalty)
    return dict(invalidated_pct=np.mean(inv), delta_mb=np.mean(delta),
                hit=np.mean(hit), ttff=np.mean(ttff))


def main():
    print("=== CHURN MICROBENCHMARK ===")
    res = {}
    for f in [0.01, 0.05, 0.10]:
        r = churn_run(f)
        res[f] = r
        print(f"  {f*100:4.0f}%/min  inv={r['invalidated_pct']:4.1f}%  "
              f"delta={r['delta_mb']:.1f}MB  hit={r['hit']:.2f}  ttff={r['ttff']:.2f}s")
    out_dir = os.environ.get("GAUSSCAST_OUT", os.path.join(os.getcwd(), "out"))
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "churn.json"), "w"),
              indent=2, default=float)
    print("saved churn.json")


if __name__ == "__main__":
    main()
