"""
run_experiments.py
------------------
Drives the GaussCast delivery simulator across scenes, trace windows, network
seeds, and concurrency to reproduce the paper's evaluation tables. All demand
comes from the real EyeNavGS traces (via demand.py). Results are printed next to
the paper's printed values and saved to out/sim_results.json.

Replay protocol (mirrors paper Sec. Prototype and Setup):
  * 3 scenes: Room, Truck, Berlin
  * groups of N users sampled from the 22 real participants per scene
  * network tiers cycled across seeds: access 10/20/30 Mbps, upstream
    150/225/300 Mbps; RTT 18 ms; 200 ms planning; 1 s horizon; 60 s windows
  * edge cache = 20% of the measured active working set
Each (scene, seed) is one paired sample; baselines share the scene/seed so
ratios are paired exactly as the paper specifies.
"""
import os, json
import numpy as np
from .demand import Demand
from . import delivery_sim as S

SCENES = ["room", "truck", "berlin"]
POLICIES = ["PerUser-HTTP", "PerUser-ICN", "SharedGreedy", "GC-noClosure",
            "GC-noAggr", "GC-cacheOnly", "GC-Full"]
TIERS = [(10.0, 150.0), (20.0, 225.0), (30.0, 300.0)]
N_SEEDS = 10
N_USERS = 16
OUT = os.environ.get("GAUSSCAST_OUT",
                     os.path.join(os.getcwd(), "out"))


def sample_users(all_users, n, rng):
    if n <= len(all_users):
        return list(rng.choice(all_users, size=n, replace=False))
    return list(rng.choice(all_users, size=n, replace=True))


def run_grid(n_users=N_USERS, seeds=N_SEEDS, scenes=SCENES, verbose=True):
    demands = {s: Demand(s) for s in scenes}
    rows = {p: {k: [] for k in
                ["up", "hit", "ttff", "late", "useful", "lateb", "unus",
                 "jain", "psnr", "psnr_jain", "useful_bytes", "upstream_bytes"]}
            for p in POLICIES}
    for s in scenes:
        D = demands[s]
        allu = D.session_users()
        for seed in range(seeds):
            rng = np.random.default_rng(1000 + seed)
            users = sample_users(allu, n_users, rng)
            acc, up = TIERS[seed % len(TIERS)]
            net = S.Net(access_mbps=acc, upstream_mbps=up)
            base = None
            res = {}
            for p in POLICIES:
                r = S.run(D, users, net, S.policy(p), seed=seed)
                res[p] = r
            base_up = res["PerUser-HTTP"]["upstream_per_useful"]
            for p in POLICIES:
                r = res[p]
                rows[p]["up"].append(r["upstream_per_useful"] / base_up)
                rows[p]["hit"].append(r["edge_hit"])
                rows[p]["ttff"].append(r["ttff"])
                rows[p]["late"].append(r["late_miss"])
                rows[p]["useful"].append(r["useful_pct"])
                rows[p]["lateb"].append(r["late_pct"])
                rows[p]["unus"].append(r["unusable_pct"])
                rows[p]["jain"].append(r["jain"])
                rows[p]["psnr"].append(r["psnr"])
                rows[p]["psnr_jain"].append(r["psnr_jain"])
                rows[p]["useful_bytes"].append(r["useful_bytes"])
                rows[p]["upstream_bytes"].append(r["upstream_bytes"])
        if verbose:
            print(f"  done scene {s}")
    agg = {}
    for p in POLICIES:
        agg[p] = {k: [float(np.mean(v)), float(np.std(v))]
                  for k, v in rows[p].items()}
    return agg


def main():
    agg = run_grid()
    os.makedirs(OUT, exist_ok=True)
    json.dump(agg, open(os.path.join(OUT, "sim_results.json"), "w"), indent=2)

    paper = {
        "PerUser-HTTP": dict(up=1.00, hit=0.31, ttff=0.75, late=0.184, jain=0.887,
                             useful=76.8, lateb=15.3, unus=7.9, psnr=28.05),
        "PerUser-ICN":  dict(up=0.93, hit=0.37, ttff=0.69, late=0.166, jain=0.892,
                             useful=78.5, lateb=13.9, unus=7.6),
        "SharedGreedy": dict(up=0.82, hit=0.42, ttff=0.66, late=0.147, jain=0.898,
                             useful=73.6, lateb=11.2, unus=15.2, psnr=28.31),
        "GC-noClosure": dict(up=0.79, late=0.161, unus=14.1),
        "GC-noAggr":    dict(up=0.86, late=0.106, unus=6.3),
        "GC-cacheOnly": dict(up=0.80, late=0.101, unus=6.1),
        "GC-Full":      dict(up=0.74, hit=0.50, ttff=0.55, late=0.096, jain=0.926,
                             useful=86.1, lateb=8.0, unus=5.9, psnr=29.02),
    }
    print("\n=== MAIN RESULT (measured vs paper) ===")
    print(f"{'Policy':14s} {'up':>12s} {'hit':>11s} {'ttff':>11s} "
          f"{'late':>12s} {'jain':>11s}")
    for p in POLICIES:
        a = agg[p]; q = paper.get(p, {})
        def f(k, fmt="{:.2f}"):
            mv = a[k][0]; pv = q.get(k)
            return f"{fmt.format(mv)}/{fmt.format(pv) if pv is not None else '--'}"
        print(f"{p:14s} {f('up'):>12s} {f('hit'):>11s} {f('ttff'):>11s} "
              f"{f('late','{:.3f}'):>12s} {f('jain','{:.3f}'):>11s}")
    print("\n=== DECOMPOSITION useful/late/unusable (measured vs paper) ===")
    for p in POLICIES:
        a = agg[p]; q = paper.get(p, {})
        print(f"{p:14s} useful={a['useful'][0]:5.1f}/{q.get('useful','--')} "
              f"late={a['lateb'][0]:5.1f}/{q.get('lateb','--')} "
              f"unus={a['unus'][0]:5.1f}/{q.get('unus','--')}")
    print("\n=== QUALITY psnr / jain (measured vs paper) ===")
    for p in ["PerUser-HTTP", "SharedGreedy", "GC-Full"]:
        a = agg[p]; q = paper.get(p, {})
        print(f"{p:14s} psnr={a['psnr'][0]:5.2f}/{q.get('psnr','--')} "
              f"jain={a['jain'][0]:.3f}/{q.get('jain','--')}")


if __name__ == "__main__":
    main()
