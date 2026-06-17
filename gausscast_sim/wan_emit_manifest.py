"""
wan_emit_manifest.py
--------------------
Step 1 of the cross-region WAN pilot. Runs the same trace-driven
simulator/planner used for the main results, captures the exact list of
distinct origin pulls for GC-Full and the PerUser-HTTP baseline over the
large-scene regime, and writes a compact transfer manifest:

  GAUSSCAST_OUT/wan_manifest.json = {
    "block_bytes": { "<id>": nbytes, ... },   # synthetic equal-content blocks
    "policies": {
        "GC-Full":      {"pulls": [[cycle, id, nbytes], ...],
                          "upstream_bytes": <sim raw upstream>,
                          "first_base_id": <id>},
        "PerUser-HTTP": { ... }
    }
  }

The edge replay (wan_pilot/wan_edge.py) fetches these blocks over the real WAN
HTTP link to the origin (wan_pilot/wan_origin.py), so the upstream-bytes ratio
matches the simulator (RTT-independent) while TTFF is measured live under real
cross-region RTT. Blocks carry only byte sizes -- the pilot measures upstream
bytes + TTFF, not rendering.
"""
import os, json
import numpy as np
from .demand import Demand
from . import delivery_sim as S

SCENES = ["truck", "berlin"]          # large-scene / high-overlap regime
SEED = 0
N_USERS = 16
TIER = (20.0, 225.0)                   # 16-user default tier (access, upstream)
OUT = os.environ.get("GAUSSCAST_OUT", os.path.join(os.getcwd(), "out"))


def bid(scene, c, l):
    return f"{scene}:{c}:{l}"


def emit():
    block_bytes = {}
    policies = {"GC-Full": {"pulls": [], "upstream_bytes": 0.0, "first_base_id": None},
                "PerUser-HTTP": {"pulls": [], "upstream_bytes": 0.0, "first_base_id": None}}
    cyc_offset = 0
    for scene in SCENES:
        D = Demand(scene)
        rng = np.random.default_rng(7000 + SEED)
        allu = D.session_users()
        users = list(rng.choice(allu, size=min(N_USERS, len(allu)),
                                replace=(N_USERS > len(allu))))
        acc, up = TIER
        net = S.Net(access_mbps=acc, upstream_mbps=up)
        for pname in policies:
            trace = []
            r = S.run(D, users, net, S.policy(pname), seed=SEED, origin_trace=trace)
            policies[pname]["upstream_bytes"] += r["upstream_bytes"]
            for (ci, c, l, nb) in trace:
                ident = bid(scene, c, l)
                block_bytes[ident] = int(round(nb))
                policies[pname]["pulls"].append([cyc_offset + int(ci), ident, int(round(nb))])
                if l == 0 and policies[pname]["first_base_id"] is None:
                    policies[pname]["first_base_id"] = ident
        cyc_offset += 100000
        print(f"  emitted origin trace for {scene}")
    manifest = {"block_bytes": block_bytes, "policies": policies,
                "regime": {"scenes": SCENES, "n_users": N_USERS, "tier": TIER}}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "wan_manifest.json")
    json.dump(manifest, open(path, "w"))
    gc = policies["GC-Full"]["upstream_bytes"]
    pu = policies["PerUser-HTTP"]["upstream_bytes"]
    print(f"distinct blocks: {len(block_bytes)}")
    print(f"GC-Full pulls: {len(policies['GC-Full']['pulls'])}  "
          f"PerUser pulls: {len(policies['PerUser-HTTP']['pulls'])}")
    print(f"sim upstream ratio GC/PerUser = {gc/pu:.3f}")
    print(f"saved {path}")


if __name__ == "__main__":
    emit()
