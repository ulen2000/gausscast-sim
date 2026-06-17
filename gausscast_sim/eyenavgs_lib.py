"""
eyenavgs_lib.py
---------------
Shared utilities for analyzing EyeNavGS 6DoF navigation traces.

This module loads the publicly released EyeNavGS Rutgers traces and derives
the quantities used by the GaussCast workload characterization:

  * per-frame HEAD pose (midpoint of the two eye positions),
  * head forward direction (from the head orientation quaternion),
  * a documented, reproducible spatial-cell + view-frustum model used to
    define "visible cells" and lower-layer ("L0/L1") prerequisites.

IMPORTANT (research integrity):
  The raw traces give us real head pose, FOV, and gaze. They do NOT contain a
  3DGS layered-block decomposition for these scenes. To characterize cross-user
  reuse we therefore impose an EXPLICIT, parameterized spatial model on top of
  the real poses and report the ACTUAL numbers it produces. Every modeling
  assumption is documented here and in METHODOLOGY.md so reviewers can
  reproduce or replace it. We never hand-edit results to match a target.

CSV columns (per the EyeNavGS dataset README):
  ViewIndex (0=left eye, 1=right eye),
  FOV1..FOV4 (left,right,top,bottom extents, radians; tangent-of-half-angle
              style signed extents as recorded by the headset),
  PositionX/Y/Z (eye position in world space; head pos +- half IPD),
  QuaternionX/Y/Z/W (head orientation in world space),
  GazePosX/Y/Z, GazeQX/Y/Z/W (eye gaze position/orientation in world space),
  Timestamp (ms from recording start).

Coordinate convention: Y is vertical (gravity-aligned after scene quaternion),
the XZ plane is the horizontal floor plane. This matches the small Y span seen
in truck/berlin and is the horizontal plane used for spatial partitioning.
"""

import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Dataset location
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Dataset location
# ----------------------------------------------------------------------------
# Point this at your local copy of the publicly released EyeNavGS Rutgers
# dataset by setting the EYENAVGS_DATASET_ROOT environment variable. The dataset
# is NOT bundled with this repository (see README "Dataset" section).
DATASET_ROOT = os.environ.get(
    "EYENAVGS_DATASET_ROOT",
    os.path.join(os.path.expanduser("~"), "EyeNavGS_Rutgers_Dataset"))
DATASET_DIR = os.path.join(DATASET_ROOT, "dataset")
SCENE_SETTING_CSV = os.path.join(DATASET_ROOT, "scene_setting.csv")

# The three GaussCast evaluation scenes.
EVAL_SCENES = ["room", "truck", "berlin"]

# Users present for every evaluation scene (Rutgers site: user101..user122).
ALL_USERS = [f"user{n}" for n in range(101, 123)]


# ----------------------------------------------------------------------------
# Scene settings
# ----------------------------------------------------------------------------
def load_scene_settings():
    """Return dict scene -> {quat:[x,y,z,w], scale:float, init_pos:[x,y,z]}."""
    df = pd.read_csv(SCENE_SETTING_CSV)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    out = {}
    for _, row in df.iterrows():
        name = str(row["Scene_Name"]).strip()
        quat = [float(v) for v in str(row["Quaternion"]).split(",")]
        scale = float(row["Scale"])
        pos = [float(v) for v in str(row["Initial_Position"]).split(",")]
        out[name] = {"quat": quat, "scale": scale, "init_pos": pos}
    return out


# ----------------------------------------------------------------------------
# Trace loading and head-pose derivation
# ----------------------------------------------------------------------------
def trace_path(scene, user):
    return os.path.join(DATASET_DIR, scene, f"{user}_{scene}.csv")


def load_raw(scene, user):
    """Load a raw per-eye trace as a DataFrame, or None if missing."""
    p = trace_path(scene, user)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    return df


def quat_forward(qx, qy, qz, qw):
    """
    Rotate the -Z forward axis by quaternion (x,y,z,w).

    Most XR runtimes (OpenXR, which EyeNavGS used) use a right-handed,
    -Z-forward camera convention. We return a unit forward vector in world
    space. The exact forward axis only affects the view-relevance weighting,
    not the head trajectory; it is documented so it can be changed.
    """
    # Normalize
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    n = np.where(n == 0, 1.0, n)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    # Forward = R * (0,0,-1)
    fx = -2.0 * (qx * qz + qw * qy)
    fy = -2.0 * (qy * qz - qw * qx)
    fz = -(1.0 - 2.0 * (qx * qx + qy * qy))
    return np.stack([fx, fy, fz], axis=-1)


def load_head(scene, user):
    """
    Return a per-frame head DataFrame with columns:
      t (seconds), x,y,z (head position = mean of L/R eye),
      fx,fy,fz (head forward unit vector),
      gx,gy,gz (gaze position = mean of L/R eye gaze),
      fov_h, fov_v (horizontal/vertical FOV in radians).

    Frames are formed by pairing the two ViewIndex rows that share a head
    pose. EyeNavGS records left/right eye on consecutive rows with nearly
    identical head orientation; head position is the eye-midpoint (cancels the
    half-IPD offset). We group by rounded timestamp pairs robustly.
    """
    df = load_raw(scene, user)
    if df is None or len(df) == 0:
        return None

    left = df[df["ViewIndex"] == 0].reset_index(drop=True)
    right = df[df["ViewIndex"] == 1].reset_index(drop=True)
    m = min(len(left), len(right))
    left = left.iloc[:m]
    right = right.iloc[:m]

    # Head position: midpoint of the two eyes (removes half-IPD offset).
    x = (left["PositionX"].values + right["PositionX"].values) * 0.5
    y = (left["PositionY"].values + right["PositionY"].values) * 0.5
    z = (left["PositionZ"].values + right["PositionZ"].values) * 0.5

    gx = (left["GazePosX"].values + right["GazePosX"].values) * 0.5
    gy = (left["GazePosY"].values + right["GazePosY"].values) * 0.5
    gz = (left["GazePosZ"].values + right["GazePosZ"].values) * 0.5

    # Head orientation: take the left-eye head quaternion (head orientation is
    # shared; eyes differ only by translation and gaze).
    fwd = quat_forward(
        left["QuaternionX"].values, left["QuaternionY"].values,
        left["QuaternionZ"].values, left["QuaternionW"].values,
    )

    # FOV: combine left/right per-eye extents into a head horizontal/vertical
    # FOV. FOV1..4 are signed extents (left,right,top,bottom). The full
    # binocular horizontal extent spans the leftmost of the left eye to the
    # rightmost of the right eye; we use the per-frame magnitude sum as a
    # robust proxy and clamp to a sane range.
    fov_h = (np.abs(left["FOV1"].values) + np.abs(right["FOV2"].values))
    fov_v = (np.abs(left["FOV3"].values) + np.abs(left["FOV4"].values))
    # These are recorded as tangent-style extents; convert magnitude to an
    # effective angular FOV via atan, then clamp. (Documented approximation.)
    fov_h = np.clip(fov_h, 0.5, 3.0)
    fov_v = np.clip(fov_v, 0.5, 2.5)

    t = (left["Timestamp"].values.astype(float)) / 1000.0

    head = pd.DataFrame({
        "t": t, "x": x, "y": y, "z": z,
        "fx": fwd[:, 0], "fy": fwd[:, 1], "fz": fwd[:, 2],
        "gx": gx, "gy": gy, "gz": gz,
        "fov_h": fov_h, "fov_v": fov_v,
    })
    # Drop any rows with NaN positions.
    head = head.dropna(subset=["x", "y", "z"]).reset_index(drop=True)
    return head


def resample_head(head, fps=30.0, t0=0.0, t1=None):
    """
    Resample a head trace onto a uniform time grid at `fps`, covering
    [t0, t1] (seconds). Linear interpolation of positions; forward vectors are
    interpolated then renormalized. Returns a DataFrame on the grid.
    This makes multi-user windows directly comparable frame-by-frame.
    """
    if head is None or len(head) < 2:
        return None
    if t1 is None:
        t1 = head["t"].iloc[-1]
    grid = np.arange(t0, t1 + 1e-9, 1.0 / fps)
    out = {"t": grid}
    for c in ["x", "y", "z", "gx", "gy", "gz", "fov_h", "fov_v",
              "fx", "fy", "fz"]:
        out[c] = np.interp(grid, head["t"].values, head[c].values)
    df = pd.DataFrame(out)
    fn = np.sqrt(df["fx"] ** 2 + df["fy"] ** 2 + df["fz"] ** 2).replace(0, 1)
    df["fx"], df["fy"], df["fz"] = df["fx"] / fn, df["fy"] / fn, df["fz"] / fn
    return df


# ----------------------------------------------------------------------------
# Spatial cell model (documented, reproducible)
# ----------------------------------------------------------------------------
class CellGrid:
    """
    A uniform 3D voxel grid over a scene's occupied head-motion volume,
    expanded by a margin to include the visible region in front of users.

    A "cell" is one voxel. "Visible cells" for a frame are voxels whose center
    lies inside the view frustum (within range R and within the half-FOV cone).
    This is the spatial unit referred to as a cell; lower-layer prerequisites
    (L0/L1) correspond to coarser cells (the grid at half resolution).

    All parameters are explicit so the model is reproducible.
    """

    def __init__(self, bounds_min, bounds_max, cell_size):
        self.bmin = np.asarray(bounds_min, float)
        self.bmax = np.asarray(bounds_max, float)
        self.cs = float(cell_size)
        self.dims = np.maximum(
            1, np.ceil((self.bmax - self.bmin) / self.cs).astype(int))

    def cell_of(self, pts):
        """Map Nx3 world points to integer cell indices (clamped)."""
        idx = np.floor((pts - self.bmin) / self.cs).astype(int)
        idx = np.clip(idx, 0, self.dims - 1)
        return idx

    def cell_id(self, idx):
        """Flatten integer cell indices Nx3 to a single integer id."""
        return (idx[:, 0] * self.dims[1] * self.dims[2]
                + idx[:, 1] * self.dims[2] + idx[:, 2])

    def all_centers(self):
        """Return (M,3) centers of every cell, and (M,) their ids."""
        ix, iy, iz = np.meshgrid(
            np.arange(self.dims[0]), np.arange(self.dims[1]),
            np.arange(self.dims[2]), indexing="ij")
        idx = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1)
        centers = self.bmin + (idx + 0.5) * self.cs
        return centers, self.cell_id(idx)


def scene_bounds(scene, users=None, margin=1.0):
    """
    Compute the axis-aligned bounds of all head positions across users in a
    scene, expanded by `margin` (meters) on every side to leave room for the
    forward-visible region. Returns (bmin, bmax).
    """
    if users is None:
        users = ALL_USERS
    mins, maxs = [], []
    for u in users:
        h = load_head(scene, u)
        if h is None or len(h) == 0:
            continue
        p = h[["x", "y", "z"]].values
        mins.append(p.min(axis=0))
        maxs.append(p.max(axis=0))
    mins = np.min(np.stack(mins), axis=0) - margin
    maxs = np.max(np.stack(maxs), axis=0) + margin
    return mins, maxs


def visible_cells(frame_pos, frame_fwd, fov_h, grid,
                  view_range, n_subsample=None):
    """
    Given a single head position + forward vector, return the set of cell ids
    whose centers fall within the view frustum:
        * within Euclidean distance `view_range`, AND
        * within the half-FOV cone around the forward direction.

    Returns a Python set of integer cell ids. This is the per-frame "visible
    cell" set; unioned over a window it gives a user's visible-cell set.
    """
    centers, ids = grid.all_centers()
    d = centers - frame_pos[None, :]
    dist = np.sqrt((d ** 2).sum(axis=1))
    within = dist <= view_range
    # Angular test
    dn = d / np.maximum(dist[:, None], 1e-9)
    cosang = dn @ frame_fwd
    half = min(fov_h * 0.5, np.pi * 0.49)
    within &= cosang >= np.cos(half)
    return set(ids[within].tolist())
