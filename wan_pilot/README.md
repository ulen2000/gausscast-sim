# Cross-region WAN pilot

Runs the **same delivery planner** as the main results across two public-cloud
regions, so the origin↔edge path crosses a real wide-area link (~40 ms RTT). It
replays the simulator's origin pulls over real HTTP/1.1 and reports the measured
RTT, throughput, upstream-bytes ratio (GC-Full / PerUser-HTTP), and a live TTFF.

Because cache reuse and request aggregation make the upstream **byte savings**
RTT-independent, the pilot transfers the distinct origin working set once over
the real WAN (the bytes that genuinely cross the link), takes the upstream ratio
from the simulator, and measures RTT / throughput / TTFF live on the real path.

## Steps

### 0. Emit the transfer manifest (locally)
```bash
export EYENAVGS_DATASET_ROOT=/path/to/EyeNavGS_Rutgers_Dataset
export GAUSSCAST_OUT=$PWD/out
python -m gausscast_sim.wan_emit_manifest
```
Produces `out/wan_manifest.json` (the distinct origin blocks + per-policy pull
lists + the simulator's upstream ratio).

### 1. Two cloud instances
- **Region A** (e.g. Singapore): origin
- **Region B** (e.g. Guangzhou/Beijing): edge + clients

A small pay-as-you-go instance (1–2 vCPU / 1–2 GB) per region is plenty. Both
scripts use only the Python 3 standard library — no installs needed on the
cloud boxes. Restrict the origin's port to region B's address via the security
group; the origin server is unauthenticated and intended for the pilot only.

### 2. Origin (region A)
Copy `wan_origin.py` and `out/wan_manifest.json` to A:
```bash
python3 wan_origin.py --manifest wan_manifest.json --port 8080
```
Add `--scale N` to divide every block size by `N` for a lighter/faster run;
use the same `--scale` on the edge.

### 3. Edge + clients (region B)
Copy `wan_edge.py` and `out/wan_manifest.json` to B:
```bash
python3 wan_edge.py --origin http://<A_PUBLIC_IP>:8080 --manifest wan_manifest.json
```
Reports the measured RTT, real throughput, upstream-bytes ratio, and live TTFF,
and saves `wan_cloud_pilot.json`.

### 4. Tear down
Stop both instances and remove the origin's security-group rule after the run.
