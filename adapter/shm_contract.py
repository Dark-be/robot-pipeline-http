# Confidential Information of Motphys. Not for disclosure or distribution without Motphys's prior
# written consent.
#
# This software contains code, techniques and know-how which is confidential and proprietary to
# Motphys.
#
# Product and Trade Secret source code contains trade secrets of Motphys.
#
# Copyright (C) 2020-2026 Motphys Technology Co., Ltd. All Rights Reserved.
#
# This software belongs to the Intellectual Property of Motphys. Use of this software is subject to
# the terms and conditions in the license file accompanying. You may not use this software except
# in compliance with the license file.

"""shm_contract —— adapter ↔ 机器人 SDK 进程的共享内存观测契约。

观测方向固定：**SDK 进程产出 → adapter 读取**（模拟 / 真实机器人图像 + 关节数据）。
指令方向（action 下发、采集命令）走 HTTP，契约见
[`http_contract`](./http_contract.py)（``scripts/test_robot_sdk.py`` 服务器 /
``test_adapter`` 客户端），不经过共享内存。

布局（单个共享内存块）：

- ``header``：固定 numpy dtype（magic / 几何 / 动态状态），记录各数据区偏移。
- ``qpos``：``float64[dim]`` 关节数据。
- ``images``：``uint8[N][H][W][3]`` **raw RGB**（N 张相机帧连续排布）。

一致性：无锁、单写者单读者。写者先写数据区、最后更新 header 动态字段
（``timestamp`` / ``running`` / ``capturing`` / ``frame_seq``）；读者用
``frame_seq`` 乒乓校验，读到撕裂帧（写入中途）时返回 None 丢弃。

> 本模块是 Edge adapter（``ObsShmReader``）与 SDK 进程（``ObsShmWriter``）共享的
> **共享内存布局契约**——SDK 侧可直接复用本模块保证布局与 adapter 读取逻辑统一。
"""

from __future__ import annotations

import time
from multiprocessing import resource_tracker, shared_memory

import numpy as np

# 观测共享内存头：固定布局（小端），几何字段在 create 时写入、之后不变；
# 动态字段（timestamp / running / capturing / frame_seq）由写者每帧更新。
OBS_SHM_HEADER = np.dtype(
    [
        ("magic", "<i8"),
        ("version", "<i8"),
        ("frame_seq", "<i8"),
        ("timestamp", "<f8"),
        ("image_count", "<i8"),
        ("image_height", "<i8"),
        ("image_width", "<i8"),
        ("image_channels", "<i8"),
        ("qpos_dim", "<i8"),
        ("qpos_offset", "<i8"),
        ("image_data_offset", "<i8"),
        ("image_data_size", "<i8"),
        ("running", "<i8"),
        ("capturing", "<i8"),
    ]
)

_OBS_SHM_MAGIC = 0x4D4F544F4E474F  # "MOTONGO"
_OBS_SHM_VERSION = 1


def _layout(image_count: int, image_size: tuple[int, int], qpos_dim: int) -> dict:
    """计算共享内存块布局（各数据区偏移 / 大小 / 总大小）。"""
    w, h = image_size
    channels = 3
    header_size = OBS_SHM_HEADER.itemsize
    qpos_offset = header_size
    qpos_bytes = qpos_dim * np.dtype("<f8").itemsize
    image_data_offset = qpos_offset + qpos_bytes
    image_data_size = image_count * h * w * channels
    return {
        "header_size": header_size,
        "qpos_offset": qpos_offset,
        "image_data_offset": image_data_offset,
        "image_data_size": image_data_size,
        "total_size": image_data_offset + image_data_size,
    }


class ObsShmWriter:
    """共享内存观测写者（SDK 进程侧）：创建共享内存块并持续写入最新观测帧。

    - ``create`` 时初始化 header 几何字段；``write`` 每帧写 qpos + raw RGB 图像。
    - ``close()`` 释放本进程句柄；``unlink()`` 删除共享内存（由 SDK 进程退出时调用）。
    """

    def __init__(self, name: str, image_count: int, image_size: tuple[int, int], qpos_dim: int):
        self.name = name
        self.image_count = image_count
        self.image_size = tuple(image_size)  # (width, height)
        self.qpos_dim = qpos_dim
        self._layout = _layout(image_count, self.image_size, qpos_dim)

        self._shm = shared_memory.SharedMemory(name=name, create=True, size=self._layout["total_size"])
        self._header = np.ndarray((), dtype=OBS_SHM_HEADER, buffer=self._shm.buf)
        h = self._header
        h["magic"] = _OBS_SHM_MAGIC
        h["version"] = _OBS_SHM_VERSION
        h["frame_seq"] = 0
        h["timestamp"] = 0.0
        h["image_count"] = image_count
        h["image_height"], h["image_width"], h["image_channels"] = self.image_size[1], self.image_size[0], 3
        h["qpos_dim"] = qpos_dim
        h["qpos_offset"] = self._layout["qpos_offset"]
        h["image_data_offset"] = self._layout["image_data_offset"]
        h["image_data_size"] = self._layout["image_data_size"]
        h["running"] = 0
        h["capturing"] = 0

        self._qpos = np.ndarray((qpos_dim,), dtype="<f8", buffer=self._shm.buf, offset=self._layout["qpos_offset"])
        self._images = np.ndarray(
            (image_count, self.image_size[1], self.image_size[0], 3),
            dtype="<u1",
            buffer=self._shm.buf,
            offset=self._layout["image_data_offset"],
        )

    def write(self, qpos: np.ndarray, images: list[np.ndarray]) -> None:
        """写入最新观测帧：先写数据区，最后更新 header 动态字段（frame_seq 递增）。"""
        self._qpos[:] = np.asarray(qpos, dtype="<f8")
        for i, img in enumerate(images[: self.image_count]):
            self._images[i] = np.asarray(img, dtype="<u1")
        self._header["timestamp"] = time.time()
        self._header["frame_seq"] = self._header["frame_seq"] + 1

    def set_flags(self, running: bool | None = None, capturing: bool | None = None) -> None:
        """更新动态状态位（running / capturing），供 adapter 侧只读判断。"""
        if running is not None:
            self._header["running"] = 1 if running else 0
        if capturing is not None:
            self._header["capturing"] = 1 if capturing else 0

    def close(self) -> None:
        """释放本进程的共享内存句柄（不删除）。"""
        self._shm.close()

    def unlink(self) -> None:
        """删除共享内存（由创建者 / SDK 进程退出时调用）。"""
        try:
            self._shm.unlink()
        except FileNotFoundError:  # noqa: PERF203 已被删除（幂等）
            pass


class ObsShmReader:
    """共享内存观测读者（adapter 侧）：attach 已存在的共享内存块并读取最新观测。

    - 几何字段（图像数量 / 尺寸 / qpos 维度）从 header 读取（writer 已初始化）。
    - ``read()`` 返回 ``{KEY_QPOS, "images": [rgb, ...]}``；无帧或撕裂帧返回 None。
    - ``running`` / ``capturing`` 属性：只读 SDK 侧状态位。
    """

    def __init__(self, name: str):
        self.name = name
        try:
            self._shm = shared_memory.SharedMemory(name=name)  # attach，不 create
        except FileNotFoundError as exc:
            raise ValueError(
                f"Observation shared memory '{name}' not found (is the robot SDK process running?)"
            ) from exc
        # attach 端不拥有共享内存：解除 resource_tracker 追踪，避免本进程退出时误 unlink
        # （共享内存生命周期归 create 端 / SDK 进程）——同时消除 resource_tracker 泄漏警告。
        try:
            resource_tracker.unregister(self._shm._name, "shared_memory")
        except (AttributeError, KeyError):  # noqa: PERF203 已移除 / 无追踪（幂等）
            pass
        self._header = np.ndarray((), dtype=OBS_SHM_HEADER, buffer=self._shm.buf)
        h = self._header
        if h["magic"] != _OBS_SHM_MAGIC:
            self._shm.close()
            raise ValueError(f"Invalid observation shared memory '{name}' (bad magic)")
        self.image_count = int(h["image_count"])
        self.image_size = (int(h["image_width"]), int(h["image_height"]))  # (width, height)
        self.qpos_dim = int(h["qpos_dim"])
        self._qpos = np.ndarray((self.qpos_dim,), dtype="<f8", buffer=self._shm.buf, offset=int(h["qpos_offset"]))
        self._images = np.ndarray(
            (self.image_count, self.image_size[1], self.image_size[0], 3),
            dtype="<u1",
            buffer=self._shm.buf,
            offset=int(h["image_data_offset"]),
        )

    @property
    def running(self) -> bool:
        """SDK 进程是否在运行（产出数据）。"""
        return bool(self._header["running"])

    @property
    def capturing(self) -> bool:
        """SDK 侧是否正在采集。"""
        return bool(self._header["capturing"])

    def read(self) -> dict | None:
        """读取最新观测帧：``{KEY_QPOS, "images": [rgb, ...]}``；无帧 / 撕裂帧返回 None。

        帧一致性用 frame_seq 乒乓校验：写者先写数据后递增 seq，若读前后 seq
        不一致说明数据写入中途被读到，丢弃。
        """
        seq0 = int(self._header["frame_seq"])
        if seq0 == 0:
            return None  # 写者尚未产出第一帧
        qpos = self._qpos.copy()
        images = [self._images[i].copy() for i in range(self.image_count)]
        if int(self._header["frame_seq"]) != seq0:
            return None  # 撕裂帧（写入中途），丢弃
        return {"qpos": qpos, "images": images}

    def close(self) -> None:
        """释放本进程的共享内存句柄（reader 不 unlink，由 writer / SDK 进程删除）。"""
        self._shm.close()


__all__ = ["OBS_SHM_HEADER", "ObsShmReader", "ObsShmWriter"]
