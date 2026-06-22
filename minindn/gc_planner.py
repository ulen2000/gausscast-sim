"""
gc_planner.py
-------------
Closure-aware two-stage retrieval planner and a synthetic multi-user demand
generator for the GaussCast Mini-NDN harness. Pure standard library.

The full evaluation uses real EyeNavGS 6DoF traces through the trace-driven
simulator (``gausscast_sim/``). For a self-contained NFD emulation that runs
without the dataset, this module generates demand with a controllable cross-user
overlap so the harness can exercise the same mechanism the paper measures: many
users sharing lower-layer prerequisites, fetched once upstream and reused.

Demand model (per user):
  * a set of visible cells; with probability ``overlap`` a cell is drawn from a
    shared "common region", otherwise from the user's own region. Shared cells
    are what concurrent users have in common -- the lower-layer overlap the
    shared base exploits.
  * each visible cell gets a target layer: shared/background cells tend to need
    only the base, cells in the user's own focus get higher refinement layers.

Planner stages (Algorithm 1 in the paper, deterministic core):
  1. SHARED BASE: rank candidate blocks by a cross-user sharing-weighted value
     density and admit each block together with its still-missing closure, so a
     widely needed prerequisite is fetched once and referenced by every user
     that needs it.
  2. PER-USER SUPPLEMENTS: spend the remaining budget on each user's
     higher-layer refinements, again admitting the full still-missing closure as
     one unit (a refinement is never scheduled before its prerequisites).

The output is, per user, an ORDERED list of (cell, layer, tag) entries where tag
is "base" (shared-base reference) or "supp" (user-specific supplement), with all
required lower layers present ahead of each refinement. The orchestrator turns
these into NDN segment Interests.
"""

import random

from gc_blocks import SceneBlocks


def make_users(n_users, sb: SceneBlocks, overlap=0.5, focus_layer=None,
               common_cells=None, cells_per_user=4, seed=0):
    """Synthesize per-user demand {cell: target_layer} for ``n_users`` users.

    overlap        fraction of each user's cells drawn from the shared common
                   region (higher overlap -> more shared lower-layer demand)
    common_cells   number of cells in the shared common region
    cells_per_user number of visible cells per user
    focus_layer    highest refinement layer a user requests on its focus cells
                   (defaults to the scene's top layer)
    """
    rng = random.Random(1234 + seed)
    Lmax = sb.n_layers - 1
    focus_layer = Lmax if focus_layer is None else min(focus_layer, Lmax)
    common_cells = common_cells or max(2, sb.n_cells // 3)
    common = list(range(common_cells))
    own_pool = list(range(common_cells, sb.n_cells))
    if not own_pool:                     # tiny scenes: reuse common as own
        own_pool = common

    users = {}
    for u in range(n_users):
        cells = {}
        for _ in range(cells_per_user):
            if rng.random() < overlap and common:
                c = rng.choice(common)
                # shared/background cells: mostly base, occasional mid layer
                lt = 0 if rng.random() < 0.7 else min(1, Lmax)
            else:
                c = rng.choice(own_pool)
                # the user's own focus cells warrant high refinement
                lt = rng.randint(min(1, Lmax), focus_layer)
            cells[c] = max(cells.get(c, 0), lt)
        users[f"u{u}"] = cells
    return users


def _value(sb, cell, layer, n_req):
    """Per-block value: layer utility I(l) times a near-detail boost, scaled by
    how many users request it (aggregated demand)."""
    return sb.I[layer] * (1.0 + 0.5 * layer) * n_req


def plan(users, sb: SceneBlocks, shared=True, closure=True,
         budget_segments=None, budget_ratio=0.6):
    """Run the planner for a group of users. Returns:

      plans[user] = ordered list of (cell, layer, tag)   tag in {base, supp}
      shared_base = set of (cell, layer) chosen as shared base

    With shared=False the planner degrades to independent per-user planning
    (the PerUser baselines): no shared base, each user's prefix is planned on its
    own. With closure=False a block may be admitted without first ensuring its
    full missing prefix is scheduled (the no-closure ablation)."""
    Lmax = sb.n_layers - 1

    # candidate blocks -> set of users requesting (the demand carried by a block)
    cand = {}
    for u, cells in users.items():
        for c, lt in cells.items():
            lt = min(lt, Lmax)
            # full-prefix demand: a target layer implies its whole prefix
            layers = range(lt + 1) if closure else [lt]
            for l in layers:
                cand.setdefault((c, l), set()).add(u)

    # per-user plan, and which blocks are already in each user's plan
    plans = {u: [] for u in users}
    in_plan = {u: set() for u in users}
    shared_base = set()

    if budget_segments is None:
        budget_segments = sum(sb.n_segments(c, l) for (c, l) in cand)
    spent = [0]                          # mutable for closures

    def admit(u, c, l, tag):
        if (c, l) in in_plan[u]:
            return
        in_plan[u].add((c, l))
        plans[u].append((c, l, tag))

    def admit_closure(u, c, l, tag):
        """Admit (c,l) for user u together with its still-missing prefix, base
        layers first, so prerequisites always precede the refinement."""
        seq = sb.closure(c, l) if closure else [(c, l)]
        for (cc, ll) in seq:             # closure() is already base-first
            t = "base" if (cc, ll) in shared_base else tag
            admit(u, cc, ll, t)

    # ---- Stage 1: shared base ------------------------------------------
    if shared:
        base_budget = int(budget_segments * budget_ratio)
        # rank widely shared, low-layer prerequisites first (closure-density:
        # value per upstream segment, favouring blocks many users share)
        def base_key(b):
            c, l = b
            n = len(cand[b])
            dens = (_value(sb, c, l, n) * n) / sb.n_segments(c, l)
            return (l, -dens)            # base layers first, then density
        for b in sorted(cand, key=base_key):
            c, l = b
            if len(cand[b]) < 2:         # not actually shared -> leave to supp
                continue
            seg = sb.n_segments(c, l)
            # upstream cost charged ONCE for the shared base (the whole point)
            new_cost = 0 if b in shared_base else seg
            if spent[0] + new_cost > base_budget:
                continue
            if b not in shared_base:
                shared_base.add(b)
                spent[0] += seg
            # reference it (and its prefix) into every requesting user's plan
            for u in cand[b]:
                admit_closure(u, c, l, "base")

    # ---- Stage 2: per-user supplements ---------------------------------
    # remaining budget spent on each user's highest-value refinements
    for u, cells in users.items():
        # user's own target blocks, highest value first
        targets = sorted(
            ((c, min(lt, Lmax)) for c, lt in cells.items()),
            key=lambda cl: _value(sb, cl[0], cl[1], len(cand.get(cl, [1]))),
            reverse=True)
        for (c, l) in targets:
            # cost = still-missing prefix not already planned for this user and
            # not already in the shared base (which is charged once, upstream)
            seq = sb.closure(c, l) if closure else [(c, l)]
            missing = [(cc, ll) for (cc, ll) in seq
                       if (cc, ll) not in in_plan[u]]
            new_segs = sum(sb.n_segments(cc, ll) for (cc, ll) in missing
                           if (cc, ll) not in shared_base)
            if spent[0] + new_segs > budget_segments:
                continue
            spent[0] += new_segs
            admit_closure(u, c, l, "supp")

    return plans, shared_base


def plan_stats(plans, shared_base, sb: SceneBlocks):
    """Summarize a plan: distinct blocks, distinct segments, and how many
    upstream segments are saved by the shared base vs naive per-user fetch."""
    distinct = set()
    per_user_total = 0
    for u, items in plans.items():
        for (c, l, _t) in items:
            distinct.add((c, l))
            per_user_total += sb.n_segments(c, l)
    distinct_segs = sum(sb.n_segments(c, l) for (c, l) in distinct)
    base_segs = sum(sb.n_segments(c, l) for (c, l) in shared_base)
    return {
        "distinct_blocks": len(distinct),
        "distinct_segments": distinct_segs,
        "per_user_sum_segments": per_user_total,
        "shared_base_blocks": len(shared_base),
        "shared_base_segments": base_segs,
    }


if __name__ == "__main__":
    sb = SceneBlocks("truck", n_cells=8)
    for ov in (0.2, 0.5, 0.8):
        users = make_users(8, sb, overlap=ov, seed=1)
        for shared in (False, True):
            plans, base = plan(users, sb, shared=shared, closure=True)
            st = plan_stats(plans, base, sb)
            tag = "shared" if shared else "peruser"
            print(f"overlap={ov} {tag:7s} "
                  f"distinct_segs={st['distinct_segments']:4d} "
                  f"per_user_sum={st['per_user_sum_segments']:4d} "
                  f"base_blocks={st['shared_base_blocks']}")
