"""
scene_model.py
--------------
Cell / layer / chunk retrieval-block model for the three GaussCast evaluation
scenes, calibrated to their published scene-card values:

  Scene  Dataset        #Gauss  Comp.size  #Cells  #Lyr  L0/L1 size
  Room   Mip-NeRF360    1.6 M   38 MB      96      3     6.1 MB
  Truck  Tanks&Temples  2.4 M   142 MB     128     4     19.8 MB
  Berlin Zip-NeRF       21 M    1.27 GB    384     5     121 MB

Modeling choices:
  * Per-LAYER total bytes follow a geometric growth s_l = s0 * g^l. We solve
    (s0, g) per scene from two constraints: s0+s1 = L0/L1 size, and
    sum_l s_l = compressed size. Higher layers carry more refinement bytes,
    matching layered-3DGS encoders where detail layers dominate size.
  * Per-CELL split of each layer's bytes is proportional to a fixed per-cell
    content weight drawn from a log-normal (seeded), so cells are heterogeneous
    but reproducible. A retrieval block is one (cell, layer) unit; large blocks
    are conceptually segmented into 0.5-2 MB chunks for transfer but scheduled
    and cached at block granularity.
  * DEPENDENCY (full-prefix closure): block (c, l) requires (c, 0..l-1). This is
    the conservative full-prefix dependency the sparsity study varies.
  * Per-LAYER quality prior I(l): marginal quality gain of adding layer l, a
    non-increasing, diminishing-returns curve normalized to (0,1]. Used as the
    planner's layer utility and to relate the set of complete layers a user
    holds to a per-layer quality level. (Rendered-image quality, PSNR/SSIM, is an
    orthogonal dimension measured from frames of the standard layered-3DGS
    rendering toolchain; this prior is the delivery-side link to it.)
"""
import numpy as np

MB = 1024.0 * 1024.0
GB = 1024.0 * MB

SCENE_CARD = {
    "room":   {"dataset": "Mip-NeRF 360",   "gauss": 1.6e6, "comp": 38 * MB,
               "cells": 96,  "layers": 3, "l01": 6.1 * MB},
    "truck":  {"dataset": "Tanks&Temples",  "gauss": 2.4e6, "comp": 142 * MB,
               "cells": 128, "layers": 4, "l01": 19.8 * MB},
    "berlin": {"dataset": "Zip-NeRF",       "gauss": 21e6,  "comp": 1.27 * GB,
               "cells": 384, "layers": 5, "l01": 121 * MB},
}


def solve_layer_sizes(l01, total, n_layers):
    """Solve geometric per-layer totals s_l = s0 * g^l s.t. s0+s1 = l01 and
    sum_l s_l = total. Returns array of length n_layers (bytes)."""
    ratio = total / l01  # = (sum_{l<L} g^l) / (1 + g)
    # Solve f(g) = (g^L - 1)/(g - 1) / (1 + g) - ratio = 0 for g > 0.
    def f(g):
        if abs(g - 1.0) < 1e-9:
            geo = n_layers
        else:
            geo = (g ** n_layers - 1.0) / (g - 1.0)
        return geo / (1.0 + g) - ratio
    lo, hi = 1e-3, 1e3
    # ratio increases with g; bisect
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    g = 0.5 * (lo + hi)
    s0 = l01 / (1.0 + g)
    sizes = np.array([s0 * g ** l for l in range(n_layers)], float)
    # numerical renorm so the total is exact
    sizes *= total / sizes.sum()
    return sizes, g


def layer_quality_prior(n_layers):
    """Marginal PSNR gain per added layer (diminishing returns), and the
    cumulative PSNR a user sees holding layers 0..k. Follows a typical
    layered-3DGS curve: base layer ~24 dB, saturating near ~30 dB, giving the
    per-layer quality levels the layer utility I(r) is derived from."""
    # cumulative PSNR after completing layers 0..k  (k = -1 -> nothing)
    # base (L0/L1) gives a coarse image; each refinement adds diminishing dB.
    base = 24.0
    # diminishing per-layer increments
    inc = np.array([2.6 / (1.6 ** l) for l in range(n_layers)], float)
    cum = base + np.cumsum(inc)            # PSNR after completing layer l
    cum = np.concatenate([[18.0], cum])    # index 0 = nothing delivered yet
    # marginal gain normalized to (0,1] for the planner's I(l)
    marg = np.diff(cum)
    I = marg / marg.max()
    return I, cum


class SceneModel:
    def __init__(self, scene, seed=12345):
        c = SCENE_CARD[scene]
        self.scene = scene
        self.n_cells = c["cells"]
        self.n_layers = c["layers"]
        self.comp = c["comp"]
        self.l01 = c["l01"]
        self.layer_total, self.geom_g = solve_layer_sizes(
            c["l01"], c["comp"], c["layers"])
        self.I, self.psnr_cum = layer_quality_prior(c["layers"])
        rng = np.random.default_rng(seed)
        # per-cell content weight (log-normal), fixed/reproducible
        w = rng.lognormal(mean=0.0, sigma=0.6, size=self.n_cells)
        self.cell_w = w / w.sum()
        # block_bytes[cell, layer]
        self.block_bytes = np.outer(self.cell_w, self.layer_total)  # (C, L)
        assert abs(self.block_bytes.sum() - self.comp) / self.comp < 1e-6

    def closure_bytes(self, cell, upto_layer):
        """Total bytes of (cell, 0..upto_layer)."""
        return self.block_bytes[cell, :upto_layer + 1].sum()

    def psnr_for_complete_layer(self, k):
        """PSNR a cell contributes when its highest complete layer is k
        (k = -1 means base not yet complete)."""
        return self.psnr_cum[k + 1]


if __name__ == "__main__":
    for s in ["room", "truck", "berlin"]:
        m = SceneModel(s)
        szs = m.layer_total / MB
        print(f"{s:7s} layers(MB)={np.round(szs,2)} sum={szs.sum():.1f} "
              f"L0+L1={szs[0]+szs[1]:.1f} g={m.geom_g:.2f} "
              f"I={np.round(m.I,3)} psnr_cum={np.round(m.psnr_cum,2)}")
