"""
gc_blocks.py
------------
Self-contained cell/layer/chunk retrieval-block model and NDN naming for the
GaussCast Mini-NDN harness. Pure standard library (no numpy, no dataset) so it
runs inside the Mini-NDN node namespaces, whose Python only ships python-ndn.

This mirrors the byte/dependency model of the trace-driven simulator in
``gausscast_sim/scene_model.py`` closely enough for the emulation to exercise the
same delivery mechanism, while staying dependency-free:

  * Per-LAYER total bytes follow a geometric growth s_l = s0 * g^l, solved per
    scene from two constraints: s0 + s1 = (L0/L1 size) and sum_l s_l = comp size.
  * Per-CELL split of each layer's bytes follows a fixed, seeded log-normal
    weight, so cells are heterogeneous but reproducible (matching the simulator's
    intent without importing numpy).
  * DEPENDENCY (full-prefix closure): block (c, l) requires (c, 0..l-1).
  * A retrieval block (c, l) is segmented into SEG_BYTES segments for transfer;
    scheduling and caching remain at block granularity, but Interest aggregation
    in NFD operates at segment granularity (one Data packet per segment), exactly
    as described in the paper.

NDN naming (versioned, digest-linked), per the paper's namespace:

  /gc/scene/<scid>/ver/<v>/cell/<cid>/layer/<l>/chunk/<k>/seg/<s>/<digest>

The trailing <digest> binds the name to its content so a cached copy can be
verified without contacting the publisher. The producer signs segment Data with
a digest (SHA-256) signer; the digest in the name is the SHA-256 of the segment
payload, giving the digest-linked verification the paper describes.
"""

import hashlib
import math
import random

MB = 1024.0 * 1024.0
GB = 1024.0 * MB

# One retrieval block is transferred as fixed-size segments. NFD carries one Data
# packet per segment, so Interest aggregation / content-store reuse happen at
# this granularity (see the paper's retrieval section).
SEG_BYTES = 4096

# The harness moves REAL packets over REAL NFD forwarders, so it works on a
# compact transfer scale rather than the full multi-gigabyte scenes (the
# trace-driven simulator covers full-scale byte accounting). We size blocks so a
# base-layer block is a few segments and cap the segment count per block, keeping
# one run to a few hundred segments while preserving the layer/cell structure,
# dependencies, naming, and digests that the mechanism depends on.
TARGET_BASE_SEGS = 3      # average segments in a layer-0 block
MAX_SEGS = 16             # cap so high layers stay transfer-friendly

# Scene cards, identical to gausscast_sim/scene_model.py SCENE_CARD.
SCENE_CARD = {
    "room":   {"dataset": "Mip-NeRF 360",  "gauss": 1.6e6, "comp": 38 * MB,
               "cells": 96,  "layers": 3, "l01": 6.1 * MB},
    "truck":  {"dataset": "Tanks&Temples", "gauss": 2.4e6, "comp": 142 * MB,
               "cells": 128, "layers": 4, "l01": 19.8 * MB},
    "berlin": {"dataset": "Zip-NeRF",      "gauss": 21e6,  "comp": 1.27 * GB,
               "cells": 384, "layers": 5, "l01": 121 * MB},
}


def solve_layer_sizes(l01, total, n_layers):
    """Solve geometric per-layer totals s_l = s0 * g^l s.t. s0 + s1 = l01 and
    sum_l s_l = total. Returns (list-of-bytes, g). Pure-Python bisection, the
    same method as the simulator's numpy version."""
    ratio = total / l01  # = (sum_{l<L} g^l) / (1 + g)

    def f(g):
        if abs(g - 1.0) < 1e-9:
            geo = n_layers
        else:
            geo = (g ** n_layers - 1.0) / (g - 1.0)
        return geo / (1.0 + g) - ratio

    lo, hi = 1e-3, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    g = 0.5 * (lo + hi)
    s0 = l01 / (1.0 + g)
    sizes = [s0 * g ** l for l in range(n_layers)]
    scale = total / sum(sizes)
    sizes = [s * scale for s in sizes]
    return sizes, g


def layer_quality_prior(n_layers):
    """Marginal per-layer quality gain I(l) in (0,1], a non-increasing
    diminishing-returns curve used as the planner's layer utility. Mirrors
    gausscast_sim/scene_model.layer_quality_prior."""
    inc = [2.6 / (1.6 ** l) for l in range(n_layers)]
    mx = max(inc)
    return [x / mx for x in inc]


class SceneBlocks:
    """Byte sizes, naming, digests, and full-prefix closures for one scene.

    To keep the emulation tractable (Berlin alone is 1.27 GB), the harness works
    on a configurable SUBSET of cells (``n_cells``) and a byte SCALE so that real
    packets cross real NFD forwarders without moving gigabytes. The relative
    layer/cell structure, dependencies, naming, and digests are unchanged, so the
    mechanism being demonstrated -- shared-base reuse under PIT aggregation and
    content-store caching -- is identical to the full-scale simulator."""

    def __init__(self, scene="truck", version="1", n_cells=None,
                 byte_scale=None, seed=12345):
        if scene not in SCENE_CARD:
            raise ValueError(f"unknown scene {scene!r}")
        card = SCENE_CARD[scene]
        self.scene = scene
        self.version = str(version)
        self.n_layers = card["layers"]
        self.full_cells = card["cells"]
        self.n_cells = n_cells or card["cells"]
        self.layer_total, self.geom_g = solve_layer_sizes(
            card["l01"], card["comp"], card["layers"])
        self.I = layer_quality_prior(card["layers"])
        # per-cell weights: fixed, seeded log-normal (stdlib random)
        rng = random.Random(seed)
        w = [math.exp(rng.gauss(0.0, 0.6)) for _ in range(self.n_cells)]
        sw = sum(w)
        self.cell_w = [x / sw for x in w]
        # Choose a byte scale so the AVERAGE layer-0 block is TARGET_BASE_SEGS
        # segments. This keeps a run to a few hundred real segments while the
        # relative layer/cell structure (and thus the sharing opportunity) is
        # preserved. An explicit byte_scale overrides the heuristic.
        if byte_scale is None:
            avg_w = 1.0 / self.n_cells
            raw_base = avg_w * self.layer_total[0]      # avg layer-0 block bytes
            target = TARGET_BASE_SEGS * SEG_BYTES
            byte_scale = target / raw_base
        self.byte_scale = byte_scale

    def block_bytes(self, cell, layer):
        """Transfer bytes of block (cell, layer) after the harness byte scale,
        capped at MAX_SEGS segments so high layers stay transfer-friendly."""
        raw = self.cell_w[cell % self.n_cells] * self.layer_total[layer]
        nb = int(round(raw * self.byte_scale))
        nb = max(SEG_BYTES, nb)
        return min(nb, MAX_SEGS * SEG_BYTES)

    def n_segments(self, cell, layer):
        nb = self.block_bytes(cell, layer)
        return max(1, (nb + SEG_BYTES - 1) // SEG_BYTES)

    def closure(self, cell, layer):
        """Full-prefix dependency closure of (cell, layer): the set of blocks
        (cell, 0..layer) that must all be present for (cell, layer) to render."""
        return [(cell, l) for l in range(layer + 1)]

    # ---- NDN naming -------------------------------------------------------
    def block_prefix(self, cell, layer):
        """Name prefix of a block (without the per-segment suffix)."""
        return (f"/gc/scene/{self.scene}/ver/{self.version}"
                f"/cell/{cell}/layer/{layer}/chunk/1")

    def seg_name(self, cell, layer, seg):
        """Full segment name including the content digest, per the paper's
        digest-linked namespace .../seg/<s>/<digest>."""
        digest = self.seg_digest(cell, layer, seg)
        return f"{self.block_prefix(cell, layer)}/seg/{seg}/{digest}"

    def seg_payload(self, cell, layer, seg):
        """Deterministic segment payload. Content is synthetic (the harness
        measures delivery, not rendering) but reproducible, so the digest in the
        name is a stable function of (scene, ver, cell, layer, seg)."""
        nb = self.block_bytes(cell, layer)
        nseg = self.n_segments(cell, layer)
        this = SEG_BYTES if seg < nseg - 1 else (nb - SEG_BYTES * (nseg - 1))
        this = max(1, this)
        head = (f"{self.scene}|{self.version}|{cell}|{layer}|{seg}|").encode()
        # fill deterministically without storing large buffers
        body = (head * ((this // len(head)) + 1))[:this]
        return body

    def seg_digest(self, cell, layer, seg):
        return hashlib.sha256(self.seg_payload(cell, layer, seg)).hexdigest()


if __name__ == "__main__":
    for s in SCENE_CARD:
        sb = SceneBlocks(s, n_cells=8)
        szs = [round(x / MB, 2) for x in sb.layer_total]
        print(f"{s:7s} layers(MB)={szs} g={sb.geom_g:.2f} I="
              f"{[round(x,3) for x in sb.I]}")
        c, l = 0, min(2, sb.n_layers - 1)
        print(f"        block(0,{l}) bytes={sb.block_bytes(c,l)} "
              f"segs={sb.n_segments(c,l)} closure={sb.closure(c,l)}")
        print(f"        seg name={sb.seg_name(c,l,0)[:90]}...")
