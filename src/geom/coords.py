

import yaml
import numpy as np
import cv2
import os


def _load_H(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    H = np.array(cfg["homography"], dtype=float)
    world_size = tuple(cfg.get("world_size", [0.0, 0.0]))
    Hinv = np.linalg.inv(H)
    return H, Hinv, world_size


class CourtGeometry:
    """
    Pixel -> world mapping using 4 homographies:
      - front wall: (u=y, v=z) at x=0
      - floor:      (u=y, v=x) at z=0
      - left wall:  (u=x, v=z) at y=0
      - right wall: (u=x, v=z) at y=640
    """

    def __init__(
        self,
        frontwall_yaml: str = "configs/calib.frontwall.yaml",
        floor_yaml: str = "configs/calib.floor.yaml",
        leftwall_yaml: str = "configs/calib.leftwall.yaml",
        rightwall_yaml: str = "configs/calib.rightwall.yaml",
        on_plane_px_tol: float = 3.0,
    ):
        assert os.path.exists(frontwall_yaml), f"Missing {frontwall_yaml}"
        assert os.path.exists(floor_yaml), f"Missing {floor_yaml}"
        assert os.path.exists(leftwall_yaml), f"Missing {leftwall_yaml}"
        assert os.path.exists(rightwall_yaml), f"Missing {rightwall_yaml}"

        self.COURT_W = 640.0
        self.COURT_L = 975.0
        self.OUT_Z = 457.0

        self.Hw, self.Hw_inv, self.wall_size = _load_H(frontwall_yaml)
        self.Hf, self.Hf_inv, self.floor_size = _load_H(floor_yaml)
        self.Hl, self.Hl_inv, self.left_size = _load_H(leftwall_yaml)
        self.Hr, self.Hr_inv, self.right_size = _load_H(rightwall_yaml)

        self.tol = float(on_plane_px_tol)

    # ---- helpers ----

    @staticmethod
    def _p2(arr):
        return np.array(arr, dtype=np.float32).reshape(1, -1, 2)

    @staticmethod
    def _warp(p, H):
        return cv2.perspectiveTransform(p, H).reshape(-1, 2)

    def _reproj_err(self, xy_px, H, Hinv):
        w = self._warp(self._p2([xy_px]), H)[0]
        back = self._warp(self._p2([w]), Hinv)[0]
        return float(np.linalg.norm(np.array(xy_px, dtype=float) - back))

    # ---- plane mappings ----

    def px_to_wall_plane(self, xy_px):
        return self._warp(self._p2([xy_px]), self.Hw)[0]

    def px_to_floor_plane(self, xy_px):
        return self._warp(self._p2([xy_px]), self.Hf)[0]

    def px_to_left_plane(self, xy_px):
        return self._warp(self._p2([xy_px]), self.Hl)[0]

    def px_to_right_plane(self, xy_px):
        return self._warp(self._p2([xy_px]), self.Hr)[0]

    # ---- world coords ----

    def wall_cm(self, xy_px):
        # (u=y, v=z)
        u, v = self.px_to_wall_plane(xy_px)
        return 0.0, float(u), float(v)

    def floor_cm(self, xy_px):
        # (u=y, v=x)
        u, v = self.px_to_floor_plane(xy_px)
        return float(v), float(u), 0.0

    def left_cm(self, xy_px):
        # (u=x, v=z)
        u, v = self.px_to_left_plane(xy_px)
        return float(u), 0.0, float(v)

    def right_cm(self, xy_px):
        # (u=x, v=z)
        u, v = self.px_to_right_plane(xy_px)
        return float(u), self.COURT_W, float(v)

    # ---- main API (Option A: no polygon gating) ----

    def px_to_court_xyz(self, xy_px):
        """
        Try all planes and pick the one with lowest reprojection error.

        plane ∈ {"front_wall","left_wall","right_wall","floor","air"}
        """
        plane_errs = [
            ("front_wall", self._reproj_err(xy_px, self.Hw, self.Hw_inv)),
            ("left_wall",  self._reproj_err(xy_px, self.Hl, self.Hl_inv)),
            ("right_wall", self._reproj_err(xy_px, self.Hr, self.Hr_inv)),
            ("floor",      self._reproj_err(xy_px, self.Hf, self.Hf_inv)),
        ]
        plane, err = min(plane_errs, key=lambda t: t[1])

        if err > self.tol:
            return "air", (float("nan"), float("nan"), float("nan")), err

        if plane == "front_wall":
            return "front_wall", self.wall_cm(xy_px), err
        if plane == "left_wall":
            return "left_wall", self.left_cm(xy_px), err
        if plane == "right_wall":
            return "right_wall", self.right_cm(xy_px), err
        if plane == "floor":
            return "floor", self.floor_cm(xy_px), err

        return "air", (float("nan"), float("nan"), float("nan")), err


def speed_cm_per_s(p_cm_prev, p_cm_now, fps: float):
    p0 = np.array(p_cm_prev, dtype=float)
    p1 = np.array(p_cm_now, dtype=float)
    return float(np.linalg.norm(p1 - p0) * fps)


def classify_plane(cg: CourtGeometry, xy_px):
    plane, (x_cm, y_cm, z_cm), err = cg.px_to_court_xyz(xy_px)
    if plane == "air":
        return "air", None, err
    return plane, (x_cm, y_cm), err
