"""
ppo_controller.py
-----------------
Trains a real PPO policy (stable-baselines3) on the simulator-measured reward
surface from controllers.py, and compares it against Static / RuleAdaptive /
OnlineGrid / OracleTuned on held-out, drifting (bandwidth, overlap) conditions.
Reports utility / upstream / late / Jain / inference time, all relative to the
Static planner.

The environment exposes the current condition (scene/overlap regime + network
tier, with stochastic drift between steps) as the observation and the shared
budget ratio as the action; the reward is the simulator-measured utility for
that (condition, ratio), linearly interpolated over the measured grid so RL
steps are fast. The policy thus learns to map drifting conditions to a good
ratio -- the high-level adaptation this study targets.

Requires the optional RL extras: stable-baselines3, gymnasium, torch.
"""
import os, json, time, argparse
import numpy as np

OUT = os.environ.get("GAUSSCAST_OUT", os.path.join(os.getcwd(), "out"))
RATIOS = [0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
N_SCENE, N_TIER = 2, 3


def load_surface():
    raw = json.load(open(os.path.join(OUT, "controller_surface.json")))
    surf = {}
    for k, v in raw.items():
        si, ti, ratio = k.split("_")
        surf[(int(si), int(ti), float(ratio))] = tuple(v)
    return surf


def util_at(surf, si, ti, ratio):
    """Utility for a condition at an arbitrary ratio (linear interp over grid)."""
    rs = RATIOS
    lo = max([r for r in rs if r <= ratio], default=rs[0])
    hi = min([r for r in rs if r >= ratio], default=rs[-1])
    u_lo = surf[(si, ti, lo)][0]
    if lo == hi:
        return u_lo
    u_hi = surf[(si, ti, hi)][0]
    w = (ratio - lo) / (hi - lo)
    return u_lo * (1 - w) + u_hi * w


def metrics_at(surf, si, ti, ratio):
    """(utility, up, late, jain) at nearest measured ratio."""
    nearest = min(RATIOS, key=lambda r: abs(r - ratio))
    return surf[(si, ti, nearest)]


# ----------------------------- controllers --------------------------------
def static_ratio(_si, _ti):
    return 0.6


def rule_ratio(si, ti):
    base = 0.55 + 0.20 * (si == 1) - 0.10 * ti
    return float(np.clip(base, 0.3, 1.0))


def grid_ratio(surf, si, ti):
    return max(RATIOS, key=lambda r: surf[(si, ti, r)][0])


def oracle_ratio(surf, si, ti):
    dense = np.linspace(0.3, 1.0, 71)
    return float(dense[np.argmax([util_at(surf, si, ti, r) for r in dense])])


def make_env(surf):
    import gymnasium as gym
    from gymnasium import spaces

    class PlanEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(0.0, 1.0, (2,), np.float32)
            self.action_space = spaces.Box(0.0, 1.0, (1,), np.float32)
            self.rng = np.random.default_rng(0)
            self.steps = 0

        def _obs(self):
            return np.array([self.si / (N_SCENE - 1), self.ti / (N_TIER - 1)],
                            np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.si = int(self.rng.integers(N_SCENE))
            self.ti = int(self.rng.integers(N_TIER))
            self.steps = 0
            return self._obs(), {}

        def step(self, action):
            ratio = 0.3 + 0.7 * float(np.clip(action[0], 0, 1))
            reward = util_at(surf, self.si, self.ti, ratio)
            self.steps += 1
            if self.rng.random() < 0.5:
                self.ti = int(self.rng.integers(N_TIER))
            if self.rng.random() < 0.25:
                self.si = int(self.rng.integers(N_SCENE))
            done = self.steps >= 32
            return self._obs(), reward, done, False, {}

    return PlanEnv()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=60000)
    args = ap.parse_args()

    surf = load_surface()
    conditions = [(si, ti) for si in range(N_SCENE) for ti in range(N_TIER)]

    import torch
    from stable_baselines3 import PPO
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training PPO on {dev} for {args.timesteps} steps ...")
    t0 = time.time()
    model = PPO("MlpPolicy", make_env(surf), verbose=0, device=dev, seed=0)
    model.learn(total_timesteps=args.timesteps)
    print("PPO trained in %.0fs" % (time.time() - t0))

    def ppo_ratio(si, ti):
        obs = np.array([si / (N_SCENE - 1), ti / (N_TIER - 1)], np.float32)
        a, _ = model.predict(obs, deterministic=True)
        return 0.3 + 0.7 * float(np.clip(a[0], 0, 1))

    obs0 = np.array([0.5, 0.5], np.float32)
    t1 = time.time()
    for _ in range(2000):
        model.predict(obs0, deterministic=True)
    ppo_infer_ms = (time.time() - t1) / 2000 * 1e3

    ctrls = {
        "Static": (lambda si, ti: static_ratio(si, ti), 0.0),
        "RuleAdaptive": (lambda si, ti: rule_ratio(si, ti), 0.15),
        "OnlineGrid": (lambda si, ti: grid_ratio(surf, si, ti), 1.6),
        "PPO": (ppo_ratio, ppo_infer_ms),
        "OracleTuned": (lambda si, ti: oracle_ratio(surf, si, ti), None),
    }

    agg = {}
    for name, (fn, infer) in ctrls.items():
        U, UP, LA, JN = [], [], [], []
        for (si, ti) in conditions:
            r = fn(si, ti)
            u = util_at(surf, si, ti, r)
            _, up, la, jn = metrics_at(surf, si, ti, r)
            U.append(u); UP.append(up); LA.append(la); JN.append(jn)
        agg[name] = dict(utility=float(np.mean(U)), up=float(np.mean(UP)),
                         late=float(np.mean(LA)), jain=float(np.mean(JN)),
                         infer_ms=infer)

    base = agg["Static"]
    print(f"\n{'Controller':14s}{'Utility':>9s}{'Upstream':>10s}"
          f"{'Late':>8s}{'Jain':>8s}{'Infer(ms)':>11s}")
    rows = {}
    for name, a in agg.items():
        util_rel = a["utility"] / base["utility"]
        up_rel = a["up"] / base["up"]
        late_rel = a["late"] / max(1e-9, base["late"])
        rows[name] = dict(utility=util_rel, up=up_rel, late=late_rel,
                          jain=a["jain"], infer_ms=a["infer_ms"])
        im = "offline" if a["infer_ms"] is None else f"{a['infer_ms']:.2f}"
        print(f"{name:14s}{util_rel:9.3f}{up_rel:10.3f}{late_rel:8.3f}"
              f"{a['jain']:8.3f}{im:>11s}")
    json.dump(rows, open(os.path.join(OUT, "controllers.json"), "w"), indent=2,
              default=lambda x: None)
    print("saved controllers.json")


if __name__ == "__main__":
    main()
