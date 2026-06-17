"""
wan_edge.py
-----------
EDGE + clients for the cross-region WAN pilot. Deploy on the cloud instance in
REGION B (e.g. Guangzhou/Beijing), pointing at the origin in REGION A so the
origin<->edge path crosses a real wide-area link. It replays the simulator's
origin pulls over real HTTP, measuring:

  * real RTT (median of /ping round-trips),
  * real throughput (distinct upstream bytes / transfer wall-time),
  * upstream-bytes ratio GC-Full / PerUser-HTTP (from the simulator, which is
    RTT-independent), and
  * live TTFF: wall-clock from session start until the first base (layer-0)
    block is fully resident at the edge, measured over the real link.

Usage on the edge instance:
    python3 wan_edge.py --origin http://<ORIGIN_IP>:8080 \
        --manifest wan_manifest.json [--scale 1] [--ttff-base 0.55]

--scale must match wan_origin.py. --ttff-base is the simulator's emulated-RTT
TTFF (s) used as the comparison baseline; the live cloud TTFF is reported
alongside it. Only the Python 3 standard library is used.
"""
import sys, os, json, time, argparse, statistics, urllib.request

OUT = os.environ.get("GAUSSCAST_OUT", ".")


def measure_rtt(origin, n=12):
    rtts = []
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            urllib.request.urlopen(origin + "/ping", timeout=10).read()
        except Exception as e:
            print("  ping failed:", e)
            continue
        rtts.append((time.perf_counter() - t0) * 1e3)
    rtts.sort()
    return statistics.median(rtts) if rtts else float("nan")


def fetch_block(origin, ident):
    t0 = time.perf_counter()
    with urllib.request.urlopen(origin + "/block/" + ident, timeout=120) as resp:
        data = resp.read()
    return len(data), (time.perf_counter() - t0)


def replay(origin, pulls, block_bytes, scale, first_base_id):
    """Fetch each DISTINCT block once over the real link (a block crosses the
    WAN once; the simulator's duplicate pulls are an origin-link accounting
    figure, not extra distinct WAN content). Returns measured bytes, wall-time,
    and live TTFF (time to first-base-block completion)."""
    seen = set()
    distinct_order = []
    for (_ci, ident, _nb) in pulls:
        if ident not in seen:
            seen.add(ident)
            distinct_order.append(ident)
    total_bytes = 0
    t_start = time.perf_counter()
    ttff = None
    for ident in distinct_order:
        nb, _ = fetch_block(origin, ident)
        total_bytes += nb
        if ident == first_base_id and ttff is None:
            ttff = time.perf_counter() - t_start
    wall = time.perf_counter() - t_start
    if ttff is None:                       # first-base not in distinct list edge case
        ttff = wall
    return total_bytes, wall, ttff, len(distinct_order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", required=True, help="http://<origin_ip>:<port>")
    ap.add_argument("--manifest", default="wan_manifest.json")
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--ttff-base", type=float, default=0.55,
                    help="simulator emulated-RTT TTFF (s) for the comparison row")
    args = ap.parse_args()

    m = json.load(open(args.manifest))
    bb = m["block_bytes"]
    pol = m["policies"]

    print("=== cross-region WAN pilot ===")
    rtt = measure_rtt(args.origin)
    print(f"measured origin<->edge RTT = {rtt:.1f} ms")

    res = {"rtt_ms": rtt, "scale": args.scale, "policies": {}}
    for pname in ["GC-Full", "PerUser-HTTP"]:
        tb, wall, ttff, ndist = replay(args.origin, pol[pname]["pulls"], bb,
                                       args.scale, pol[pname]["first_base_id"])
        tput = (tb * 8 / 1e6) / wall if wall > 0 else float("nan")
        res["policies"][pname] = dict(distinct_blocks=ndist, bytes=tb,
                                      wall_s=wall, throughput_mbps=tput,
                                      live_ttff_s=ttff,
                                      sim_upstream_bytes=pol[pname]["upstream_bytes"])
        print(f"  {pname:14s} distinct={ndist:5d}  bytes={tb/1e6:8.1f}MB  "
              f"wall={wall:6.1f}s  tput={tput:6.1f}Mbps  liveTTFF={ttff*1e3:6.1f}ms")

    gc = res["policies"]["GC-Full"]["sim_upstream_bytes"]
    pu = res["policies"]["PerUser-HTTP"]["sim_upstream_bytes"]
    up_ratio = gc / pu
    res["upstream_ratio_gc_over_peruser"] = up_ratio
    print(f"\nupstream-bytes ratio GC/PerUser = {up_ratio:.3f}")
    print(f"TTFF: emulated(sim) {args.ttff_base:.2f}s -> cloud(live first-base) "
          f"{res['policies']['GC-Full']['live_ttff_s']:.3f}s")
    path = os.path.join(OUT, "wan_cloud_pilot.json")
    json.dump(res, open(path, "w"), indent=2)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
