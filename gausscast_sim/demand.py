"""
demand.py
---------
Build the per-planning-cycle, per-user retrieval DEMAND for a scene from the
real EyeNavGS traces. This is the input to the delivery simulator.

For each scene we build a PARTITION grid whose cell count approximates the
paper's #Cells (Table tab:scene-card: 96 / 128 / 384). The set of cells ever
visible across all users defines the scene's active cells; the SceneModel's
byte budget is distributed over exactly those cells.

For each user u and planning cycle starting at playback time p (cycles spaced by
PLAN_INTERVAL over a WINDOW), demand over the horizon [p, p+H] is:
  * visible cells V = union of view-frustum partition cells over the horizon;
  * per visible cell c, a TARGET LAYER lt(c) set by proximity: cells the user
    is close to / looking directly at warrant full detail; far visible cells
    warrant only base layers. This produces layered, dependency-bearing demand.
  * useful deadline t_ddl(c): first playback time in the horizon at which c
    becomes visible (earliest-visibility), used by the planner's deadline test.

The constant-velocity + gaze predictor of the paper is what the proxy actually
uses to FORECAST these sets; ground truth comes from the trace. We expose both
so the prediction-horizon study can measure realized prediction error.
"""
import numpy as np
from . import eyenavgs_lib as E
from .scene_model import SceneModel, SCENE_CARD

SAMPLE_FPS = 10.0      # pose sampling for visibility (denser than workload's 5)
VIEW_RANGE = 6.0
MARGIN = 1.0


def build_partition(scene):
    """Grid whose total cell count is closest to the paper's #Cells."""
    target = SCENE_CARD[scene]["cells"]
    bmin, bmax = E.scene_bounds(scene, margin=MARGIN)
    span = bmax - bmin
    vol = float(np.prod(span))
    # choose cell_size so that prod(ceil(span/cs)) ~ target
    cs = (vol / target) ** (1.0 / 3.0)
    # one refinement pass: the set of *visible* cells overshoots the raw grid
    # product because frustums reach across the box, so coarsen slightly to land
    # the active-cell count near the paper's #Cells.
    grid = E.CellGrid(bmin, bmax, cs * 1.18)
    return grid


def proximity_layer(dist, n_layers, view_range=VIEW_RANGE):
    """Target layer by distance: near -> full layers, far -> base only.
    Linear bands across [0, view_range]."""
    frac = np.clip(dist / view_range, 0.0, 1.0)
    # near (frac~0) -> top layer ; far (frac~1) -> layer 0 (base only)
    lt = np.round((1.0 - frac) * (n_layers - 1)).astype(int)
    return np.maximum(lt, 0)  # far cells need only the L0 base


class Demand:
    """Per-cycle demand generator for one scene, sharing one partition grid."""

    def __init__(self, scene, fps=SAMPLE_FPS):
        self.scene = scene
        self.grid = build_partition(scene)
        self.centers, self.ids = self.grid.all_centers()
        self.n_layers = SCENE_CARD[scene]["layers"]
        self.fps = fps
        self.users = {}
        active = set()
        for u in E.ALL_USERS:
            h = E.load_head(scene, u)
            if h is None or len(h) < 2:
                continue
            dur = float(h["t"].iloc[-1])
            hr = E.resample_head(h, fps=fps, t0=0.0, t1=min(dur, 130.0))
            if hr is None or len(hr) < 2:
                continue
            P = hr[["x", "y", "z"]].values
            F = hr[["fx", "fy", "fz"]].values
            FOV = hr["fov_h"].values
            T = hr["t"].values
            self.users[u] = {"P": P, "F": F, "FOV": FOV, "T": T, "dur": dur}
        # determine active cells across all users (sparse sampling)
        for u, d in self.users.items():
            P, F, FOV = d["P"], d["F"], d["FOV"]
            step = max(1, len(P) // 200)
            for i in range(0, len(P), step):
                vc, _ = self._visible(P[i], F[i], FOV[i])
                active.update(vc.tolist())
        self.active = sorted(active)
        self.remap = {c: i for i, c in enumerate(self.active)}
        self.n_cells = len(self.active)
        self._dcache = {}
        self._wscache = {}
        # build the byte/quality model on exactly the active cell count
        self.model = SceneModel(scene)
        self._rescale_model()

    def _rescale_model(self):
        """Rebuild SceneModel.block_bytes over the actual active cell count."""
        m = self.model
        rng = np.random.default_rng(777)
        w = rng.lognormal(0.0, 0.6, size=self.n_cells)
        m.cell_w = w / w.sum()
        m.n_cells = self.n_cells
        m.block_bytes = np.outer(m.cell_w, m.layer_total)

    def _visible(self, pos, fwd, fov):
        d = self.centers - pos[None, :]
        dist = np.sqrt((d * d).sum(axis=1))
        within = dist <= VIEW_RANGE
        if not within.any():
            return np.empty(0, np.int64), np.empty(0)
        dn = d[within] / np.maximum(dist[within, None], 1e-9)
        cosang = dn @ fwd
        half = min(float(fov) * 0.5, np.pi * 0.49)
        vis = cosang >= np.cos(half)
        return self.ids[within][vis], dist[within][vis]

    def working_set_bytes(self, users, cycles, horizon, joint=None):
        """Total bytes of the distinct blocks `users` demand over `cycles`
        (the active working set). Memoized per (users, cycles-signature)."""
        joint = joint or {u: 0.0 for u in users}
        key = (tuple(sorted(users)), round(float(cycles[0]), 3),
               round(float(cycles[-1]), 3), len(cycles), round(horizon, 3),
               tuple(round(joint[u], 3) for u in sorted(users)))
        v = self._wscache.get(key)
        if v is not None:
            return v
        bb = self.model.block_bytes
        blocks = set()
        for u in users:
            for p in cycles:
                if p < joint[u]:
                    continue
                dem = self.cycle_demand(u, p - joint[u], horizon)
                for c, (lt, t) in dem.items():
                    for ll in range(0, min(lt, self.n_layers - 1) + 1):
                        blocks.add((c, ll))
        ws = float(sum(bb[c, l] for (c, l) in blocks)) if blocks else float(bb.sum())
        self._wscache[key] = ws
        return ws

    def cycle_demand(self, u, p, horizon):
        """Cached per-user, per-cycle demand. Returns dict {cell: [target,t]}.
        Computed once per (user, rounded p, horizon) and memoized so repeated
        simulation runs over the same traces are fast."""
        key = (u, round(p, 3), round(horizon, 3))
        c = self._dcache.get(key)
        if c is not None:
            return c
        out = self._compute_cycle_demand(u, p, horizon)
        self._dcache[key] = out
        return out

    def _compute_cycle_demand(self, u, p, horizon):
        d = self.users[u]
        T = d["T"]
        lo = np.searchsorted(T, p, "left")
        hi = np.searchsorted(T, p + horizon, "right")
        if hi <= lo:
            k = min(max(lo, 0), len(T) - 1)
            lo, hi = k, k + 1
        out = {}
        for i in range(lo, hi):
            vc, dist = self._visible(d["P"][i], d["F"][i], d["FOV"][i])
            if len(vc) == 0:
                continue
            lt = proximity_layer(dist, self.n_layers)
            tframe = T[i] - p
            for cid, L, t in zip(vc.tolist(), lt.tolist(), [tframe] * len(vc)):
                lc = self.remap.get(cid)
                if lc is None:
                    continue
                if lc not in out:
                    out[lc] = [L, t]
                else:
                    out[lc][0] = max(out[lc][0], L)
                    out[lc][1] = min(out[lc][1], t)
        return out

    def session_users(self):
        return list(self.users.keys())


if __name__ == "__main__":
    for s in ["room", "truck", "berlin"]:
        D = Demand(s)
        print(f"{s:7s} target_cells={SCENE_CARD[s]['cells']} "
              f"grid_dims={D.grid.dims.tolist()} active_cells={D.n_cells} "
              f"users={len(D.users)} total_MB={D.model.block_bytes.sum()/1024/1024:.1f}")
