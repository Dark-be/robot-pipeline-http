"""Minimal ACT-format HDF5 collector.

Receives a ``standard_obs`` dict (from ``Robot.get_standard_obs()``) and
writes it as an ACT-format HDF5 file::

    /observations/qpos            (T, D)   float32
    /observations/cameras/{name}  (T,) vlen uint8   (jpeg)
                                 (T, H, W[, C])     (raw)
    /action                       (T, D)   float32

Usage::

    c = ActHDF5Collector({"save_dir": "./data", "image_format": "jpeg"})
    c.collect(standard_obs)   # from robot.get_standard_obs()
    c.finish(episode_idx=0)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np

from utils.base.data_handler import debug_print

# standard_obs keys (see Robot.get_standard_obs())
KEY_QPOS = "observations/qpos"
KEY_ACTION = "action"
CAMERA_PREFIX = "observations/images/"


class ActHDF5Collector:
    def __init__(self, collector_config: dict):
        self.name = "ActHDF5Collector"
        self._save_dir = Path(collector_config.get("save_dir", "./data"))
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._image_format = collector_config.get("image_format", "jpeg")  # "jpeg" | "raw"
        self._timesteps: list[dict] = []
        debug_print(self.name, f"Initialized with save_dir={self._save_dir}, image_format={self._image_format}", "INFO")

    # -- public API ---------------------------------------------------------
    def set_save_dir(self, path):
        self._save_dir = Path(path)

    @property
    def save_dir(self) -> Path:
        """当前数据保存目录（供 server 上报 data_status）。"""
        return self._save_dir
    def _to_image(self, val):
        is_jpeg = isinstance(val, np.ndarray) and val.ndim == 1 and val.dtype == np.uint8

        if self._image_format == "jpeg":
            if is_jpeg:
                return val.ravel()
            ok, buf = cv2.imencode(".jpg", val)
            if not ok:
                raise ValueError("Failed to encode image as JPEG")
            return buf.ravel()
        else:
            if is_jpeg:
                img = cv2.imdecode(np.frombuffer(val, dtype=np.uint8), cv2.IMREAD_COLOR)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return np.asarray(val, dtype=np.uint8)

    def collect(self, standard_obs: dict):
        step = {
            "qpos": np.asarray(standard_obs[KEY_QPOS], dtype=np.float32),
            "action": np.asarray(standard_obs[KEY_ACTION], dtype=np.float32),
            "images": {},
        }
        for key, val in standard_obs.items():
            if key.startswith(CAMERA_PREFIX) and val is not None:
                cam_name = key[len(CAMERA_PREFIX):]
                step["images"][cam_name] = self._to_image(val)
        self._timesteps.append(step)

    def finish(self, episode_idx: int):
        if not self._timesteps:
            return

        path = self._save_dir / f"episode_{episode_idx}.hdf5"
        num = len(self._timesteps)

        qpos_dim = self._timesteps[0]["qpos"].shape[0]
        action_dim = self._timesteps[0]["action"].shape[0]
        debug_print(self.name, f"Collector finish: episode {episode_idx}, {num} steps, "
              f"qpos_dim={qpos_dim}, action_dim={action_dim}")

        with h5py.File(path, "w") as f:
            obs = f.create_group("observations")

            obs.create_dataset(
                "qpos",
                data=np.stack([t["qpos"] for t in self._timesteps]),
                compression="gzip",
            )

            cam_names = list(self._timesteps[0]["images"].keys())
            if cam_names:
                cam_grp = obs.create_group("images")
                for cam in cam_names:
                    frames = [t["images"][cam] for t in self._timesteps if cam in t["images"]]

                    if self._image_format == "raw":
                        cam_grp.create_dataset(
                            cam,
                            data=np.stack(frames).astype(np.uint8),
                            compression="gzip",
                        )
                    else:
                        dt = h5py.vlen_dtype(np.dtype("uint8"))
                        ds = cam_grp.create_dataset(cam, (num,), dtype=dt)
                        for i, t in enumerate(self._timesteps):
                            ds[i] = t["images"].get(cam, b"")

            f.create_dataset(
                "action",
                data=np.stack([t["action"] for t in self._timesteps]),
                compression="gzip",
            )

            f.attrs["episode_id"] = episode_idx
            f.attrs["num_steps"] = num

        self._timesteps.clear()
