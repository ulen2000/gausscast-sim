"""
controllers.py
--------------
High-level parameter-adaptation study. The deterministic closure-aware planner
already enforces correctness; an optional controller only tunes ONE high-level
knob -- the shared BUDGET RATIO (fraction of each user's per-cycle access budget
reserved for the broad shared base vs user-specific supplements).

This module MEASURES a reward surface U(condition, ratio) by running the real
delivery simulator over (network tier x overlap-via-scene x budget ratio). The
companion ppo_controller.py then trains a PPO policy on that surface and compares
Static / RuleAdaptive / OnlineGrid / PPO / OracleTuned controllers.

Conditions drift stochastically across steps (dynamic bandwidth/overlap). Each
controller picks a ratio per condition; utility is the simulator-measured
outcome. The reward surface is saved to GAUSSCAST_OUT/controller_surface.json.
"""
import os, json, time
import numpy as np
from .demand import Demand
from . import delivery_sim as S

TIERS = [(10.0, 150.0), (20.0, 225.0), (30.0, 300.0)]
RATIOS = [0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
SCENES = ["truck", "berlin"]                     # two overlap regimes
SEEDS = 2
N_USERS = 16
OUT = os.environ.get("GAUSSCAST_OUT",
                     os.path.join(os.getcwd(), "out"))


def utility(r):
    """Blended session utility: coverage (useful share) timeliness-discounted,
    minus an upstream cost. Higher is better. This is the controller objective;
    it is not tied to any external number."""
    return (r["useful_pct"] / 100.0) * (1.0 - r["late_miss"]) \
        - 0.10 * r["upstream_per_useful"] / 25.0


def measure_surface():
    """U[(scene_idx, tier_idx, ratio)] = mean (utility, up, late, jain)."""
    surf = {}
    demands = {s: Demand(s) for s in SCENES}
    for si, s in enumerate(SCENES):
        D = demands[s]
        allu = D.session_users()
        for ti, (acc, up) in enumerate(TIERS):
            for ratio in RATIOS:
                us, ups, lat, jn = [], [], [], []
                for seed in range(SEEDS):
                    rng = np.random.default_rng(7000 + seed)
                    users = list(rng.choice(allu, N_USERS, replace=False))
                    net = S.Net(access_mbps=acc, upstream_mbps=up, window_s=40.0)
                    P = S.policy("GC-Full")
                    P.budget_ratio = ratio
                    r = S.run(D, users, net, P, seed=seed)
                    us.append(utility(r)); ups.append(r["upstream_per_useful"])
                    lat.append(r["late_miss"]); jn.append(r["jain"])
                surf[(si, ti, ratio)] = (float(np.mean(us)), float(np.mean(ups)),
                                         float(np.mean(lat)), float(np.mean(jn)))
        print(f"  measured surface for {s}")
    return surf


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    surf = measure_surface()
    json.dump({f"{k[0]}_{k[1]}_{k[2]}": v for k, v in surf.items()},
              open(os.path.join(OUT, "controller_surface.json"), "w"), indent=2)
    print("surface measured in %.0fs; saved controller_surface.json" %
          (time.time() - t0))


if __name__ == "__main__":
    main()
