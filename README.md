# gausscast-sim

Trace-driven, cycle-based **delivery simulator** for *GaussCast* — dependency-aware
shared block retrieval for multi-user **layered 3D Gaussian Splatting (3DGS)** — together
with a genuine Ed25519/SHA-256 verification-overhead microbenchmark.

The simulator replays **real EyeNavGS 6DoF navigation traces** through an edge proxy that
plans retrieval of per-cell, per-layer blocks for many concurrent users under a shared
upstream bottleneck, per-user access links, an edge cache, RTT, and full-prefix rendering
dependencies. It is a compact, self-contained model focused on the **delivery
mechanism** (shared base + supplements, closure-aware admission, request aggregation,
asymmetric caching).

> **Note.** This repository currently provides the Python trace-driven delivery-mechanism
> core. The remaining components of the full prototype — the Mini-NDN / NFD emulation
> harness (named-data forwarders and the colocated planning proxy) and the layered-3DGS
> rendering and quality-evaluation pipeline (PSNR / SSIM) — are being finalized and will be
> added in a near-term update.

## What it models

Each 200 ms planning cycle over a 60 s window, for a group of users sampled from the real
traces, the proxy:

1. gathers each user's per-cell **target layer** and the **full-prefix closure** `(c,0..l)`
   it requires (block `(c,l)` needs `(c,0..l-1)`);
2. orders candidate blocks by a value/sharing score (base-first, then near refinements);
3. fetches missed blocks from origin **once** under request aggregation (PIT-style dedup),
   counting raw upstream bytes at the shared link;
4. delivers to each user under their per-user access budget, classifying every delivered
   byte as **useful** / **late** / **unusable** (non-renderable because its prefix is
   missing);
5. admits a block only if its still-missing closure can arrive before the deadline
   (**closure-aware admission**).

### Policies

| Name | Shared | Aggregate (PIT) | Closure admission | Cache |
|---|---|---|---|---|
| `PerUser-HTTP` | – | – | – | LRU |
| `PerUser-ICN`  | – | ✓ | – | LRU |
| `SharedGreedy` | ✓ | ✓ | – | asym |
| `GC-noClosure` | ✓ | ✓ | – | asym |
| `GC-noAggr`    | ✓ | – | ✓ | asym |
| `GC-cacheOnly` | ✓ | – | ✓ | asym |
| `GC-Full`      | ✓ | ✓ | ✓ | asym |

## Install

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.9+. Dependencies: `numpy`, `pandas`, `scipy`, `cryptography`.

## Dataset

The simulator's demand comes from the **publicly released EyeNavGS Rutgers traces**, which
are **not bundled** here. Download them, then point the package at your local copy:

```bash
export EYENAVGS_DATASET_ROOT=/path/to/EyeNavGS_Rutgers_Dataset
# the loader expects:
#   $EYENAVGS_DATASET_ROOT/dataset/<scene>/<user>_<scene>.csv
#   $EYENAVGS_DATASET_ROOT/scene_setting.csv
```

EyeNavGS dataset: <https://github.com/VWNJ/EyeNavGS> (Rutgers 6DoF navigation traces).

## Usage

### Main delivery-result grid (3 scenes × tiers × seeds)

```bash
export EYENAVGS_DATASET_ROOT=/path/to/EyeNavGS_Rutgers_Dataset
export GAUSSCAST_OUT=$PWD/out
python -m gausscast_sim.run_experiments
```

Prints each policy's upstream-bytes ratio (normalized to `PerUser-HTTP`), edge-hit ratio,
late-miss ratio, Jain fairness, and the **useful / late / unusable** byte decomposition;
saves `out/sim_results.json`.

### Single run from Python

```python
import numpy as np
from gausscast_sim.demand import Demand
from gausscast_sim import delivery_sim as S

D = Demand("truck")
users = list(np.random.default_rng(0).choice(D.session_users(), 16, replace=False))
net = S.Net(access_mbps=20.0, upstream_mbps=225.0)     # tier: access / upstream Mbps
res = S.run(D, users, net, S.policy("GC-Full"), seed=0)
print(res["upstream_per_useful"], res["useful_pct"], res["unusable_pct"])
```

### Churn microbenchmark (versioned-manifest invalidation)

```bash
GAUSSCAST_OUT=$PWD/out python -m gausscast_sim.churn
```

Warms the edge cache, then republishes a fraction (1% / 5% / 10% per minute) of resident
chunks; unchanged subtrees keep their content digests and stay cached, changed chunks are
invalidated and refetched. Reports invalidated share, refetched bytes, and the resulting
hit-ratio drop.

### Verification overhead (real crypto)

```bash
python -m gausscast_sim.crypto_overhead --fanout 32 --block-mb 1.0
```

Times **real** Ed25519 verifies and SHA-256 hashes on your machine and reports sustained
public-key verifies/second and amortized CPU per block for (a) per-block signatures and
(b) a digest-linked manifest that amortizes one signature over a `fanout` of blocks. All
numbers are derived from the measured primitives and the chosen fanout.

### High-level controller (PPO budget-ratio adaptation)

The deterministic closure-aware planner enforces correctness; an optional controller tunes
one high-level knob — the **shared budget ratio** (base vs supplement split) — as bandwidth
and overlap drift. First measure the simulator reward surface, then train a real PPO policy
and compare it against Static / RuleAdaptive / OnlineGrid / OracleTuned:

```bash
pip install -r requirements-rl.txt          # stable-baselines3, gymnasium, torch
GAUSSCAST_OUT=$PWD/out python -m gausscast_sim.controllers       # measure reward surface
GAUSSCAST_OUT=$PWD/out python -m gausscast_sim.ppo_controller --timesteps 60000
```

Trains a real `stable-baselines3` PPO on the measured surface (GPU if available) and prints
each controller's utility / upstream / late / Jain / inference time relative to the Static
planner; saves `out/controllers.json`.

### Cross-region WAN pilot

Runs the same planner across two cloud regions over a real wide-area link. Emit the transfer
manifest locally, then deploy the origin/edge scripts — see [`wan_pilot/README.md`](wan_pilot/README.md):

```bash
GAUSSCAST_OUT=$PWD/out python -m gausscast_sim.wan_emit_manifest     # -> out/wan_manifest.json
```

## Layout

```
gausscast_sim/
  scene_model.py        cell/layer byte sizes + layered-quality prior (per scene)
  demand.py             per-cycle, per-user retrieval demand from the real traces
  delivery_sim.py       the simulator: run(), Net, Policy, policy(), EdgeCache
  run_experiments.py    main delivery-result grid runner
  churn.py              versioned-manifest invalidation microbenchmark
  crypto_overhead.py    genuine Ed25519/SHA-256 throughput measurement
  controllers.py        measures the budget-ratio reward surface
  ppo_controller.py     trains a real PPO controller and compares baselines
  wan_emit_manifest.py  emits the cross-region WAN transfer manifest
  eyenavgs_lib.py       EyeNavGS trace loader + spatial cell/frustum model
wan_pilot/
  wan_origin.py         origin HTTP server (stdlib only, deploy to region A)
  wan_edge.py           edge client: measures RTT / throughput / ratio / TTFF
  README.md             two-region deployment guide
```

## Notes on methodology

Per-layer byte sizes are derived geometrically from each scene's published compressed
size and L0/L1 size; per-cell content weights are drawn from a fixed, seeded log-normal.
The layered quality prior is a diminishing-returns per-layer quality curve used as the
planner's layer utility. Modeling parameters are documented inline in each module.

## License

MIT — see [LICENSE](LICENSE).
