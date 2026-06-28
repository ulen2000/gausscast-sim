"""
run_experiments.py
------------------
Drives the GaussCast delivery simulator across scenes, trace windows, network
seeds, and concurrency. All demand comes from the real EyeNavGS traces (via
demand.py). Results are printed and saved to out/sim_results.json.

Replay protocol:
  * 3 scenes: Room, Truck, Berlin
  * groups of N users sampled from the 22 real participants per scene
  * network tiers cycled across seeds: access 10/20/30 Mbps, upstream
    150/225/300 Mbps; RTT 18 ms; 200 ms planning; 1 s horizon; 60 s windows
  * edge cache = 20% of the measured active working set
Each (scene, seed) is one paired sample; baselines share the scene/seed so
ratios are paired exactly.
"""
import os, json
import numpy as np
from .demand import Demand
from . import delivery_sim as S

SCENES = ["room", "truck", "berlin"]
POLICIES = ["PerUser-HTTP", "PerUser-ICN", "DependencyAware-PerUser",
            "SharedGreedy", "GC-noClosure", "GC-noAggr", "GC-cacheOnly",
            "GC-Full", "OracleSharedPrereq"]
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

    print("\n=== MAIN RESULT ===")
    print(f"{'Policy':14s} {'up':>8s} {'hit':>8s} {'ttff':>8s} "
          f"{'late':>9s} {'jain':>8s}")
    for p in POLICIES:
        a = agg[p]
        print(f"{p:14s} {a['up'][0]:8.2f} {a['hit'][0]:8.2f} {a['ttff'][0]:8.2f} "
              f"{a['late'][0]:9.3f} {a['jain'][0]:8.3f}")
    print("\n=== DECOMPOSITION useful/late/unusable (%) ===")
    for p in POLICIES:
        a = agg[p]
        print(f"{p:14s} useful={a['useful'][0]:5.1f} "
              f"late={a['lateb'][0]:5.1f} unus={a['unus'][0]:5.1f}")
    print("\n=== QUALITY psnr / jain ===")
    for p in ["PerUser-HTTP", "SharedGreedy", "GC-Full"]:
        a = agg[p]
        print(f"{p:14s} psnr={a['psnr'][0]:5.2f} jain={a['jain'][0]:.3f}")


if __name__ == "__main__":
    main()
