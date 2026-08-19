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

"""TestRobotAdapter —— 测试 / 无硬件联调用适配器（HTTP + 共享内存**薄客户端**）。

机器人硬件初始化和连接由 SDK 进程（``scripts/test_robot_sdk.py``）自行维护；本适配器
只是 Edge 侧薄客户端，**不实现任何硬件 / 连接逻辑**：

- **指令走 HTTP**：``execute`` / ``rollout`` / ``safe_stop`` / ``reset`` 等经 HTTP POST
  转发给 SDK 进程；SDK 执行硬件动作，并把观测填充到共享内存。
- **observe() 走共享内存**：SDK 进程按 ``run_hz`` 持续把观测（raw RGB + 关节）写入
  共享内存（``ObsShmWriter``），本适配器经 ``ObsShmReader`` 读取，并把图像编码为
  JPEG（Edge 观测契约）返回。
- **health() 实时查询**：直接 ``GET /v1/health`` 反映进程状态（节点经 alive-check 驱动）。

身份由机器人进程 discover 解析（``name`` / ``id`` / ``type``）传入构造函数；能力与连接参数
**全部由类级常量定义**（自包含，不随 discover 传输、不接收 Edge 配置）。HTTP 客户端与
共享内存读取器**惰性建立**（首次指令 / 观测）。
"""

import cv2
import httpx
import numpy as np

from motrix_edge.adapter.base import (
    CAMERA_PREFIX,
    KEY_ACTION,
    KEY_QPOS,
    AdapterCapability,
    CaptureData,
    HealthStatus,
    RobotAdapter,
    RobotCapabilities,
)
from motrix_edge.adapter.http_contract import (
    FIELD_ACTION,
    FIELD_DATA_FILES,
    FIELD_OK,
    FIELD_SAVE_DIR,
    FIELD_TELEOP_ENABLED,
    PATH_CAPTURE_END,
    PATH_CAPTURE_START,
    PATH_DATA_STATUS,
    PATH_EXECUTE,
    PATH_HEALTH,
    PATH_RESET,
    PATH_ROLLOUT,
    PATH_SAFE_STOP,
    PATH_TELEOP,
)
from motrix_edge.adapter.shm_contract import ObsShmReader
from motrix_edge.utils.data_handler import debug_print


class TestRobotAdapter(RobotAdapter):
    # ---- 身份（discover 解析传入；缺省回退类常量）----
    NAME = "test_robot"  # 实例名（debug_print 前缀）
    ADAPTER_TYPE = "test_robot"  # 本 adapter 的 entry point 类型

    # ---- 能力 / 连接参数（类级常量，自包含，不随 discover 传输）----
    ROBOT_MODEL_ID = "test-robot"
    ROBOT_MODEL_VERSION = "0.0.0"
    ACTION_DIM = 14  # 动作维度
    # 相机布局：{相机名: 分辨率 (width, height)}（SDK 产出 raw RGB；observe 编码 JPEG 原图）
    IMAGES: dict[str, tuple[int, int]] = {
        "cam_head": (640, 480),
        "cam_left_wrist": (640, 480),
        "cam_right_wrist": (640, 480),
    }

    # 中间件连接参数（SDK 自维护硬件与连接）
    SDK_URL = "http://127.0.0.1:8090"  # SDK HTTP 服务地址（指令下行）
    SHM_NAME = "test_robot_obs"  # 共享内存名（观测上行）
    HTTP_TIMEOUT = 3.0  # HTTP 指令超时（秒）

    # 能力声明：采集 + 执行 + 视频流（模拟相机）
    CAPABILITIES: dict[AdapterCapability, bool] = {
        AdapterCapability.CAPTURE: True,
        AdapterCapability.EXECUTE: True,
        AdapterCapability.STREAMING: True,
    }

    def __init__(self, name: str = "", id: str = ""):
        """中间件实例：由 discover 解析出的身份（name / id）参数化。

        - ``name`` / ``id``：机器人进程 discover 解析出的身份（缺省回退类常量）。
        - ``type`` 由类常量 ``ADAPTER_TYPE`` 确定（entry point 类型，不随 discover 传输）。
        - 能力（动作维度 / 相机布局）与连接参数（HTTP 地址 / 共享内存名）**全部由类级
          常量定义**，不随 discover 传输、不接收 Edge 配置。
        """
        super().__init__(name=name, id=id)
        self.name = name or self.NAME
        self.id = id or self.ADAPTER_TYPE
        # 能力：类级常量（自包含，不随 discover 传输）
        self.action_dim = self.ACTION_DIM
        self.images = list(self.IMAGES)  # 相机名列表（IMAGES 字典的键）
        self.robot_model_id = self.ROBOT_MODEL_ID
        self.robot_model_version = self.ROBOT_MODEL_VERSION
        self._capabilities = dict(self.CAPABILITIES)

        # 中间件连接参数：类级常量（不接收 Edge 配置）
        self.sdk_url = self.SDK_URL.rstrip("/")
        self.shm_name = self.SHM_NAME
        self.http_timeout = self.HTTP_TIMEOUT

        # 惰性连接资源：首次指令 / 观测时建立（SDK 自维护硬件与连接）
        self._http: httpx.Client | None = None  # SDK HTTP 客户端（指令下行）
        self._shm: ObsShmReader | None = None  # 共享内存观测读者（观测上行）
        self._running = False  # 机器人进程最近一次确认是否运行（health 实时刷新）

        # 测试断言用本地记录（指令均经 HTTP 转发，计数保留在本地便于测试）
        self.executed: list = []
        self.rollout_calls = 0
        self.safe_stop_calls = 0
        self.reset_calls = 0
        self.teleop_enabled = False  # 遥操作开关（本地回显；SDK 侧状态自维护）

    # ---- health（实时查询 SDK 进程状态；硬件由 SDK 自维护）----------------------
    def health(self) -> HealthStatus:
        """健康检查：实时 ``GET /v1/health``（SDK 自维护硬件；Edge 只查询）。"""
        try:
            resp = self._client().get(PATH_HEALTH)
            ok = resp.status_code == 200 and bool(resp.json().get(FIELD_OK, False))
        except Exception as exc:  # noqa: BLE001 进程失联
            debug_print(self.name, f"health check failed: {exc}", "WARNING")
            ok = False
        self._running = ok
        return HealthStatus(ok=ok)

    @property
    def running(self) -> bool:
        """机器人进程最近一次确认是否运行（health 实时刷新）。"""
        return self._running

    def _client(self) -> httpx.Client:
        """惰性建立 SDK HTTP 客户端（首次指令 / 查询时）。"""
        if self._http is None:
            self._http = httpx.Client(base_url=self.sdk_url, timeout=self.http_timeout)
        return self._http

    def release(self):
        """释放 Edge 侧本地资源：关闭共享内存读者 / HTTP 客户端（SDK 连接由进程自维护）。"""
        if self._shm is not None:
            self._shm.close()
            self._shm = None
        if self._http is not None:
            self._http.close()
            self._http = None
        debug_print(self.name, "TestRobotAdapter released.", "INFO")

    # ---- capabilities ----------------------------------------------------------
    @property
    def capabilities(self) -> RobotCapabilities:
        obs_keys = [KEY_QPOS] + [f"{CAMERA_PREFIX}{img}" for img in self.images]
        return RobotCapabilities(
            robot_model_id=self.robot_model_id,
            robot_model_version=self.robot_model_version,
            action_dim=self.action_dim,
            observation_keys=obs_keys,
            capabilities=dict(self._capabilities),
        )

    # ---- 指令（经 HTTP 转发 SDK 进程）--------------------------------------------
    def reset(self):
        """程序复位到 home（非阻塞）：HTTP 转发 SDK 进程。"""
        self.reset_calls += 1
        self._client().post(PATH_RESET)

    # ---- 采集数据状态（经 HTTP 查询 SDK；数据由进程自维护，无回合控制）-------------
    def data_status(self) -> CaptureData | None:
        """采集数据状态：数据保存路径 + 本次采集得到的数据列表（查询 SDK 进程）。

        数据采集（录制写盘）由 SDK 进程自维护；本方法只查询 / 上报结果。
        """
        try:
            resp = self._client().get(PATH_DATA_STATUS)
            resp.raise_for_status()
            body = resp.json()
        except Exception as exc:  # noqa: BLE001
            debug_print(self.name, f"data status query failed: {exc}", "WARNING")
            return None
        return CaptureData(
            save_dir=body.get(FIELD_SAVE_DIR),
            data_files=[str(f) for f in body.get(FIELD_DATA_FILES, [])],
        )

    # ---- 采集回合控制（capture episode start / end）----------------------------
    def start_capture(self):
        """开始一轮采集：HTTP 转发 SDK 进程（episode 开始）。"""
        debug_print(self.name, "capture episode start", "INFO")
        self._client().post(PATH_CAPTURE_START)

    def end_capture(self):
        """结束一轮采集：HTTP 转发 SDK 进程（episode 结束）。"""
        debug_print(self.name, "capture episode end", "INFO")
        self._client().post(PATH_CAPTURE_END)

    # ---- observe（共享内存观测上行：SDK 进程产出 → adapter 读取）-------------------
    def observe(self) -> dict:
        """读取共享内存最新观测帧（SDK 进程产出），图像编码为 JPEG（Edge 契约）。

        SDK 进程把观测填充到共享内存；observe 只读取、不推进 SDK 运行。
        """
        if self._shm is None:
            self._shm = ObsShmReader(self.shm_name)  # 惰性 attach（首次观测时）
        frame = self._shm.read()
        if frame is None:
            return {}  # SDK 尚未产出首帧
        qpos = np.asarray(frame["qpos"], dtype=np.float32)
        obs = {
            KEY_QPOS: qpos,
            KEY_ACTION: qpos.copy(),  # 测试：action = 当前执行位置
        }
        for img, name in zip(frame["images"], self.images):
            obs[f"{CAMERA_PREFIX}{name}"] = self._encode_jpeg(img)
        return obs

    @staticmethod
    def _encode_jpeg(rgb: np.ndarray) -> bytes:
        """RGB ndarray → JPEG bytes（观测缓存图像编码；SDK 产出原图尺寸）。"""
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError("Failed to encode image as JPEG")
        return buf.tobytes()

    # ---- execute ------------------------------------------------------------------
    def execute(self, action):
        """直接下发动作指令（raw）：本地记录 + HTTP 转发 SDK 进程。

        校验维度（与 rollout 一致）再发送：action 解析为 float64 数组，维度须等于
        ``action_dim``，否则抛 ``ValueError``（不发送）。
        """
        target = np.asarray(action, dtype=np.float64)
        if target.shape[0] != self.action_dim:
            raise ValueError(f"execute action dim {target.shape[0]} != action_dim {self.action_dim}")
        self.executed.append(target.tolist())
        debug_print(self.name, f"execute sent: {target.tolist()}", "INFO")
        self._client().post(PATH_EXECUTE, json={FIELD_ACTION: target.tolist()})

    # ---- teleop ------------------------------------------------------------------
    def set_teleop(self, enabled: bool):
        """设置遥操作开关（true=遥操作 / false=程控）：本地记录 + HTTP 转发 SDK 进程。"""
        self.teleop_enabled = bool(enabled)
        debug_print(self.name, f"teleop set to {self.teleop_enabled}", "INFO")
        self._client().post(PATH_TELEOP, json={FIELD_TELEOP_ENABLED: self.teleop_enabled})

    # ---- rollout --------------------------------------------------------------------
    def rollout(self, action):
        """推理闭环：本地校验维度 + HTTP 转发（SDK 侧设为限速目标并逐帧靠近）。"""
        target = np.asarray(action, dtype=np.float64)
        if target.shape[0] != self.action_dim:
            raise ValueError(f"rollout action dim {target.shape[0]} != action_dim {self.action_dim}")
        self.rollout_calls += 1
        self._client().post(PATH_ROLLOUT, json={FIELD_ACTION: target.tolist()})

    # ---- safe_stop -------------------------------------------------------------------
    def safe_stop(self):
        """安全停止（幂等、失败安全）：本地记录 + HTTP 转发 SDK 进程。"""
        self.safe_stop_calls += 1
        try:
            self._client().post(PATH_SAFE_STOP)
        except Exception as exc:  # noqa: BLE001
            debug_print(self.name, f"safe_stop failed: {exc}", "ERROR")
