"""
delivery_sim.py
---------------
Cycle-based, trace-driven delivery simulator for GaussCast and its baselines.

Each planning cycle (PLAN_INTERVAL, default 200 ms) over a WINDOW (60 s) the edge
proxy(ies) plan retrieval of cell/layer blocks for their downstream users, under:
  * a shared UPSTREAM bottleneck (byte budget per cycle = rate * interval),
  * per-user ACCESS links (downstream byte budget per user per cycle),
  * an edge CACHE (capacity = fraction of active working set),
  * RTT and the planning interval (set the TTFF floor),
  * full-prefix rendering DEPENDENCIES: block (c,l) needs (c,0..l-1).

Policies differ only in the PLANNING POLICY:
  shared      : aggregate demand across users and fetch shared base once
  aggregate   : same-cycle requests for the same uncached block collapse (PIT)
  closure     : admit a block only if its still-missing closure fits budget+ddl
  cache_mode  : 'std' LRU  |  'asym' lower-layers retained longer
Metrics returned: upstream_bytes (raw), edge_hit, ttff, late_miss, useful/late/
unusable byte fractions, per-user useful bytes (for Jain), per-user mean PSNR.
"""
import numpy as np
from dataclasses import dataclass, field

MB = 1024.0 * 1024.0

# Cache-retention parameters. RETAIN_BONUS sets how many extra "recency cycles"
# a base layer earns over a top layer in asymmetric mode; MAX_LAYERS bounds the
# layer index.
RETAIN_BONUS = 1.5
MAX_LAYERS = 5


@dataclass
class Net:
    upstream_mbps: float = 225.0     # shared bottleneck (16-user default tier)
    access_mbps: float = 20.0        # per-user access (10/20/30 mix avg)
    rtt_s: float = 0.018
    plan_interval_s: float = 0.2
    horizon_s: float = 1.0
    window_s: float = 60.0
    cache_frac: float = 0.20         # of active working set
    fps: int = 30


@dataclass
class Policy:
    name: str
    shared: bool = True
    aggregate: bool = True
    closure: bool = True
    cache_mode: str = "asym"
    block_scale: float = 1.0         # compression byte-scaling
    dep_density: float = 1.0         # fraction of full prefix actually required
    eta_cent: float = 0.15           # closure-centrality weight in shared score
    use_prediction: bool = False     # plan on predicted-visible set (else oracle)
    budget_ratio: float = 1.0        # base-vs-supplement shared budget split knob
    oracle_prereq: bool = False      # idealized: each distinct block crosses the
                                     # upstream link at most once over the whole
                                     # run (upper bound on prerequisite sharing)


# Named delivery policies
def policy(name):
    P = {
        "PerUser-HTTP":  Policy("PerUser-HTTP", shared=False, aggregate=False,
                                closure=False, cache_mode="std"),
        "PerUser-ICN":   Policy("PerUser-ICN", shared=False, aggregate=True,
                                closure=False, cache_mode="std"),
        # Per-user progressive delivery with closure-aware admission but NO
        # cross-user shared planning: isolates whether the quality/useful-byte
        # gain comes from dependency closure alone or from closure PLUS sharing.
        # Same per-user substrate as PerUser-ICN (PIT aggregation, std cache),
        # with closure admission added.
        "DependencyAware-PerUser": Policy("DependencyAware-PerUser", shared=False,
                                aggregate=True, closure=True, cache_mode="std"),
        "SharedGreedy":  Policy("SharedGreedy", shared=True, aggregate=True,
                                closure=False, cache_mode="asym"),
        "GC-noClosure":  Policy("GC-noClosure", shared=True, aggregate=True,
                                closure=False, cache_mode="asym"),
        "GC-noAggr":     Policy("GC-noAggr", shared=True, aggregate=False,
                                closure=True, cache_mode="asym"),
        "GC-cacheOnly":  Policy("GC-cacheOnly", shared=True, aggregate=False,
                                closure=True, cache_mode="asym"),
        "GC-Full":       Policy("GC-Full", shared=True, aggregate=True,
                                closure=True, cache_mode="asym"),
        # Idealized upper bound: every lower-layer prerequisite is fetched at most
        # once when at least one user needs it (perfect prerequisite dedup with
        # hindsight), then the same per-user supplement policy as PerUser. Bounds
        # the maximum achievable upstream saving from prerequisite sharing.
        "OracleSharedPrereq": Policy("OracleSharedPrereq", shared=True,
                                aggregate=True, closure=True, cache_mode="asym",
                                oracle_prereq=True),
    }
    return P[name]


class EdgeCache:
    def __init__(self, cap_bytes, mode="asym"):
        self.cap = cap_bytes
        self.mode = mode
        self.size = 0
        self.store = {}        # block -> bytes
        self.last = {}         # block -> last-use cycle
        self.layer = {}        # block -> layer index (for asym retention)

    def has(self, b):
        return b in self.store

    def touch(self, b, cyc):
        self.last[b] = cyc

    def put(self, b, nbytes, layer, cyc):
        if b in self.store:
            self.last[b] = cyc
            return
        self.store[b] = nbytes
        self.last[b] = cyc
        self.layer[b] = layer
        self.size += nbytes
        self._evict(cyc)

    def _evict(self, cyc):
        if self.size <= self.cap:
            return
        # Evict least-valuable-to-retain first. Retention value = recency plus,
        # in asymmetric mode, a reuse bonus for lower layers (shared by more
        # users, so more likely to be reused before eviction). The bonus is
        # set by RETAIN_BONUS; std mode is plain LRU (bonus 0).
        items = list(self.store.keys())
        def key(b):
            base = self.last[b]
            if self.mode == "asym":
                base += (MAX_LAYERS - self.layer[b]) * RETAIN_BONUS
            return base
        items.sort(key=key)            # smallest key = evict first
        for b in items:
            if self.size <= self.cap:
                break
            self.size -= self.store.pop(b)
            self.last.pop(b, None)
            self.layer.pop(b, None)


def run(demand, users, net: Net, pol: Policy, seed=0, n_edges=4,
        join_stagger_s=0.0, origin_trace=None):
    """Simulate one group of `users` over the scene. Returns a metrics dict.

    If `origin_trace` is a list, each distinct origin pull is appended as
    (cycle_index, cell, layer, nbytes) -- used by the WAN pilot to replay the
    exact origin transfers over a real HTTP link. Default None has zero effect
    on the headline simulation."""
    rng = np.random.default_rng(seed)
    m = demand.model
    L = demand.n_layers
    bb = m.block_bytes * pol.block_scale          # (C, L) bytes
    # Dependency sparsity: dep_density is the fraction of the full prefix that is
    # an actual prerequisite (dependency-sparsity study). A DETERMINISTIC required
    # prefix per target layer (seeded, independent of call order) is shared by
    # BOTH the fetch path AND the renderability test, so looser dependencies let
    # closures complete with fewer prerequisites -- raising useful%, lowering
    # late, and (because each user then fetches more distinct high-layer content,
    # which dedups less) raising the upstream ratio toward 1. dep_density>=1.0
    # returns the full prefix, byte-identical to the dependency-complete default.
    _prefix_rng = np.random.default_rng(seed + 99991)
    _req_cache = {}

    def required_prefix(l):
        rp = _req_cache.get(l)
        if rp is None:
            keep = [0] + [ll for ll in range(1, l)
                          if _prefix_rng.random() < pol.dep_density]
            rp = sorted(set(keep + [l]))
            _req_cache[l] = rp
        return rp
    # working set = bytes of all blocks any user demands in the window
    # approximate by total scene bytes touched; cache cap is a fraction of it.
    # Build per-user cycle schedule
    interval = net.plan_interval_s
    cycles = np.arange(0.0, net.window_s - net.horizon_s, interval)
    # split users across edges
    edges = {e: [] for e in range(n_edges)}
    for i, u in enumerate(users):
        edges[i % n_edges].append(u)
    # per-user join time (stagger for cold start studies)
    joint = {u: (i * join_stagger_s) for i, u in enumerate(users)}

    W_acc = net.access_mbps * 1e6 / 8.0 * interval         # bytes/user/cycle

    # cache cap = cache_frac of the MEASURED active working set (memoized)
    ws = demand.working_set_bytes(users, cycles, net.horizon_s, joint)
    cache = EdgeCache(ws * net.cache_frac, pol.cache_mode)

    have = {u: set() for u in users}      # blocks user already holds (local L_u)
    rendered = {u: set() for u in users}  # blocks delivered useful (renderable)
    # OracleSharedPrereq: idealized hindsight prerequisite dedup. A distinct block
    # crosses the shared upstream link at most ONCE over the whole run; any later
    # cycle that needs it (after the edge evicted it under capacity) is treated as
    # already paid for. This is an upper bound on the saving from prerequisite
    # sharing -- it cannot be realized online -- used only to contextualize how
    # close GC-Full's measured upstream is to the achievable optimum.
    oracle_pulled = set()                 # blocks already charged upstream (global)
    # metrics
    upstream_bytes = 0.0
    delivered_bytes = 0.0
    useful_bytes = 0.0
    late_bytes = 0.0
    unusable_bytes = 0.0
    cache_hits = 0
    cache_lookups = 0
    admitted = 0
    late_admitted = 0
    ttff = {u: None for u in users}
    user_useful = {u: 0.0 for u in users}
    user_psnr_acc = {u: [] for u in users}
    user_q_demand = {u: 0.0 for u in users}    # viewport-weighted target PSNR
    user_q_have = {u: 0.0 for u in users}       # viewport-weighted achieved PSNR
    user_w_total = {u: 0.0 for u in users}      # total viewport weight

    for ci, p in enumerate(cycles):
        # ---- gather demand of active users this cycle ----
        # Each user needs, per visible cell c, the PREFIX of blocks (c, 0..target).
        # We expand to per-block candidates carrying the set of requesting users
        # and per-user deadlines. The scheduling policy below decides the order
        # and whether prefixes are fetched atomically.
        cand = {}                       # block -> dict(u -> deadline)
        user_targets = {}               # u -> {cell: predicted target_layer}
        true_cells = {}                 # u -> set of TRUE visible cells (gt)
        for u in users:
            if p < joint[u]:
                continue
            # the proxy PLANS on the predicted-visible set ...
            dem = demand.cycle_demand(u, p - joint[u], net.horizon_s,
                                      predicted=pol.use_prediction)
            # ... and (only when prediction is active) is validated against the
            # TRUE visible set so mispredicted prefetches can be counted as waste
            if pol.use_prediction:
                tdem = demand.cycle_demand(u, p - joint[u], net.horizon_s,
                                           predicted=False)
                true_cells[u] = set(tdem.keys())
            tgt = {}
            for c, (lt, t) in dem.items():
                L_eff = min(lt, L - 1)
                tgt[c] = L_eff
                ddl = p + t + net.horizon_s
                for ll in range(0, L_eff + 1):
                    b = (c, ll)
                    if b in have[u]:
                        continue
                    cand.setdefault(b, {})
                    cand[b][u] = min(cand[b].get(u, 1e9), ddl)
            user_targets[u] = tgt

        if not cand:
            continue

        # ---- value / sharing signals ----
        # per-user demand value g_u(r) = I(l) * w_view: I(l) is the layer utility
        # prior; w_view rewards the higher target layers a user assigns to cells
        # near the viewport center, so the planner completes high-detail near
        # cells (per-user gain g_u). Aggregated over the requesting users.
        def block_value(b):
            c, l = b
            return m.I[l] * (1.0 + 0.5 * l) * len(cand[b])

        def sharing(b):
            return max(0, len(cand[b]) - 1)   # avoided duplication count

        def prefix_layers(l):
            if pol.dep_density >= 1.0:
                return list(range(0, l + 1))
            return required_prefix(l)

        # ---- candidate ordering per policy ----
        if pol.closure:
            # Two-stage shared planner: FIRST build a broad shared
            # BASE (low-layer targets, ranked by cross-user sharing) so every
            # widely-needed cell is renderable; THEN spend remaining budget on
            # high-value near SUPPLEMENTS (refinements). Implemented as a sort key
            # that puts base/widely-shared targets ahead of niche refinements
            # while still preferring high value within each stage.
            def score(b):
                c, l = b
                stage = 0 if l <= 1 else 1            # base stage first
                return (-stage, (sharing(b) + 1) * block_value(b)
                        - 0.02 * (bb[c, l] / MB))
            order = sorted(cand.keys(), key=score, reverse=True)
        elif pol.shared:
            # overlap-weighted value: shared refinements can outrank niche bases
            order = sorted(cand.keys(),
                           key=lambda b: (sharing(b) + 1) * block_value(b),
                           reverse=True)
        else:
            order = sorted(cand.keys(), key=lambda b: (b[1], -block_value(b)))

        rejected_closure = set()        # closure refused (deadline) this cycle

        # ---- access-bounded fetch + deliver pass ----
        # Per cycle the proxy serves blocks in priority order. A block is pulled
        # from origin only if (a) it is not already at the edge (cache miss) and
        # (b) at least one requesting user still has downstream access budget to
        # receive it this cycle. Raw UPSTREAM bytes are counted at the origin
        # link: aggregating/PIT policies pull a missed block ONCE; PerUser-HTTP
        # (no collapse) pulls it once plus the in-flight races that occur in the
        # RTT window before the cache fills (alpha = rtt/interval). DOWNSTREAM,
        # the block is delivered to each budgeted user and classified
        # useful/late/unusable. This keeps the origin link (dedup-sensitive)
        # distinct from the per-user link (where unusable bytes are counted).
        rate = net.upstream_mbps * 1e6 / 8.0
        alpha = min(1.0, net.rtt_s / interval)
        acc_used = {u: 0.0 for u in users}
        cum_pull = 0.0                                   # bytes pulled this cycle

        def closure_ok(b):
            if b[1] == 0:
                return True
            eddl = min(cand[b].values())
            return (p + net.rtt_s + interval) <= eddl + 1e-9

        # ---- edge-hit snapshot at cycle start ----
        # Edge-hit ratio over the distinct blocks needed this cycle: a block is a
        # hit if it is resident in the edge cache (fetched in a prior cycle). The
        # closure-rejection set is built in the same pass.
        needed = set()
        for b in order:
            if pol.closure:
                if not closure_ok(b):
                    rejected_closure.add(b)
                    continue
                for ll in prefix_layers(b[1]):
                    needed.add((b[0], ll))
            else:
                needed.add(b)
        for nb_ in needed:
            cache_lookups += 1
            if cache.has(nb_):
                cache_hits += 1

        arrival = {}

        def fetch(b, nusers):
            """Pull block b from origin if missed; count raw upstream + arrival."""
            nonlocal upstream_bytes, cum_pull
            if b in arrival:
                return True
            c, l = b
            nb = bb[c, l]
            if cache.has(b):
                cache.touch(b, ci)
                arrival[b] = p + net.rtt_s + interval
                return True
            ndup = 1 if pol.aggregate else 1 + int(round((nusers - 1) * alpha))
            # Oracle: a distinct block already pulled in ANY prior cycle is free
            # (perfect hindsight dedup across the whole run); only its first global
            # pull is charged at the upstream link.
            if pol.oracle_prereq and b in oracle_pulled:
                arrival[b] = p + net.rtt_s + interval
                cache.put(b, nb, l, ci)
                return True
            cum_pull += nb * ndup
            upstream_bytes += nb * ndup
            arrival[b] = p + net.rtt_s + cum_pull / rate
            cache.put(b, nb, l, ci)
            if pol.oracle_prereq:
                oracle_pulled.add(b)
            if origin_trace is not None:
                origin_trace.append((ci, c, l, float(nb * ndup)))
            return True

        def deliver(b, u, ddl):
            nonlocal delivered_bytes, useful_bytes, late_bytes, unusable_bytes
            nonlocal admitted, late_admitted
            c, l = b
            nb = bb[c, l]
            if b in have[u]:                      # already delivered this session
                return
            if acc_used[u] + nb > W_acc or b not in arrival:
                return
            acc_used[u] += nb
            delivered_bytes += nb
            admitted += 1
            # renderable iff every REQUIRED prerequisite (below l) is local.
            # With full dependencies this is the complete prefix 0..l-1; under
            # dep sparsity only the required subset must be present.
            req = (range(l) if pol.dep_density >= 1.0
                   else [ll for ll in required_prefix(l) if ll < l])
            renderable = (l == 0) or all((c, ll) in have[u] for ll in req)
            mispredicted = (pol.use_prediction and
                            c not in true_cells.get(u, ()))  # planned, not seen
            have[u].add(b)                       # now local: not re-demanded
            if arrival[b] > ddl + 1e-9:
                late_bytes += nb
                late_admitted += 1
            elif mispredicted or not renderable:
                # fetched for a cell the user never actually looks at (prediction
                # error), or arrived before its prefix -> non-renderable waste
                unusable_bytes += nb
            else:
                useful_bytes += nb
                user_useful[u] += nb
                rendered[u].add(b)               # counts toward viewport quality
                if ttff[u] is None and l == 0:
                    ttff[u] = (p - joint[u]) + (arrival[b] - p)

        # Per-user access budget is split by the shared BUDGET RATIO between the
        # broad shared base (layers 0-1) and user-specific supplements (>=2). The
        # controller tunes this ratio to the current bandwidth/overlap regime: a
        # higher base share favors broad coverage (good under high overlap / low
        # bandwidth), a lower share favors near-depth refinement. Only meaningful
        # for shared/closure policies; per-user baselines ignore it.
        if pol.shared:
            base_cap = {u: pol.budget_ratio * W_acc for u in users}
        else:
            base_cap = {u: W_acc for u in users}

        def deliver_split(blk, u, ddl):
            # route base layers against the base sub-budget, supplements against
            # the remainder, so budget_ratio actually re-allocates the link
            if pol.shared and blk[1] <= 1 and acc_used[u] >= base_cap[u]:
                return
            deliver(blk, u, ddl)

        for b in order:
            if pol.closure and b in rejected_closure:
                continue
            blocks = ([(b[0], ll) for ll in prefix_layers(b[1])]
                      if pol.closure else [b])
            users_b = [u for u in cand[b] if acc_used[u] < W_acc]
            if not users_b:
                continue
            for blk in blocks:                   # base-first
                fetch(blk, len(cand[b]))
                for u in users_b:
                    deliver_split(blk, u, cand[b][u])

        # ---- per-user quality accounting this cycle ----
        # For each visible cell, accumulate the viewport-weighted DEMANDED quality
        # level (at the target layer) and the ACHIEVED quality level (at the
        # highest layer rendered useful). The session quality level is
        # achieved/demanded mapped back through the per-layer quality curve,
        # tying the reported quality level to the fraction of demanded viewport
        # content the policy renders. (Rendered-image quality, PSNR/SSIM, is an
        # orthogonal dimension measured from frames of the standard layered-3DGS
        # rendering toolchain; this quality level is the delivery-side link.)
        for u, tgt in user_targets.items():
            for c, target in tgt.items():
                k = -1
                for l in range(L):
                    if (c, l) in rendered[u]:
                        k = l
                    else:
                        break
                # viewport PSNR over ALL visible cells: unrendered cells show the
                # coarse/background floor, so broad base coverage (shared base)
                # AND near depth both improve quality.
                w = target + 1
                user_q_have[u] += w * m.psnr_for_complete_layer(k)
                user_w_total[u] += w

    # ---- finalize ----
    for u in users:
        if ttff[u] is None:
            ttff[u] = net.window_s   # never started (worst case)
    edge_hit = cache_hits / max(1, cache_lookups)
    late_miss = late_admitted / max(1, admitted)
    tot = max(1.0, delivered_bytes)
    useful_pct = 100.0 * useful_bytes / tot
    late_pct = 100.0 * late_bytes / tot
    unus_pct = 100.0 * unusable_bytes / tot
    mean_ttff = float(np.mean([ttff[u] for u in users]))
    # per-user achieved viewport PSNR = viewport-weighted mean of the PSNR at the
    # highest layer rendered useful for each visible cell. Policies that render
    # more complete closures (GaussCast) achieve higher layers on more cells.
    upsnr = [user_q_have[u] / user_w_total[u]
             for u in users if user_w_total[u] > 0]
    uu = np.array([user_useful[u] for u in users], float)
    jain = (uu.sum() ** 2) / (len(uu) * (uu ** 2).sum()) if (uu ** 2).sum() > 0 else 1.0
    mean_psnr = float(np.mean(upsnr)) if upsnr else float("nan")
    psnr_jain = ((np.sum(upsnr) ** 2) / (len(upsnr) * np.sum(np.square(upsnr)))
                 if upsnr else 1.0)
    return {
        "upstream_bytes": upstream_bytes,
        # upstream bytes required PER useful byte delivered: when the shared
        # upstream link is the bottleneck, this is the metric that reflects "how
        # much origin traffic to serve the same useful workload". Deduplication
        # and closure-complete delivery let GaussCast turn each upstream byte
        # into more useful content, so it needs fewer upstream bytes per useful
        # byte. Normalized to PerUser-HTTP by the caller.
        "upstream_per_useful": upstream_bytes / max(1.0, useful_bytes),
        "useful_bytes": useful_bytes,
        "edge_hit": edge_hit,
        "ttff": mean_ttff,
        "late_miss": late_miss,
        "useful_pct": useful_pct,
        "late_pct": late_pct,
        "unusable_pct": unus_pct,
        "jain": float(jain),
        "psnr": mean_psnr,
        "psnr_jain": float(psnr_jain),
        "delivered_bytes": delivered_bytes,
    }
