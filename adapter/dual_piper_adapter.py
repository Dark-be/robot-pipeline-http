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

"""DualPiperAdapter —— 双臂 Piper 机器人适配器（骨架 / 预留接口，未接入实机 SDK）。

基于 ``TestRobotAdapter`` 的**薄客户端**模式：Edge 侧只做「指令下行（HTTP）+ 观测上行
（共享内存）」，硬件初始化与连接由机器人 SDK 进程自维护。本适配器只**预留接口与
身份 / 能力声明**，具体硬件动作映射 / 观测键 / SDK 协议**待接入真实 Piper 双臂 SDK 时
实现**（各方法体为 ``NotImplementedError`` 占位）。

接入指引（对照 TestRobotAdapter 模板）：
- 配置 ``SDK_URL`` / ``SHM_NAME`` 指向真实 Piper 双臂 SDK 进程（类级常量，自包含）。
- 实现 ``execute`` / ``rollout``：动作维度 = 双臂（左 + 右，各 6 关节 + 夹爪 = 7）共 14；
  需把模型动作拆分到左右臂。
- 实现 ``observe``：从共享内存读取 qpos + 相机帧（``cam_head`` / 左右腕相机），编码 JPEG。
- 实现 ``health`` / ``data_status``：查询 SDK 进程状态与采集数据状态。

身份由机器人进程 discover 解析（``name`` / ``id`` / ``type``）传入构造函数；能力与连接参数
**全部由类级常量定义**（自包含，不随 discover 传输、不接收 Edge 配置）。
"""

import httpx

from motrix_edge.adapter.base import (
    AdapterCapability,
    CaptureData,
    HealthStatus,
    RobotAdapter,
    RobotCapabilities,
)
from motrix_edge.adapter.shm_contract import ObsShmReader
from motrix_edge.utils.data_handler import debug_print


class DualPiperAdapter(RobotAdapter):
    # ---- 身份（discover 解析传入；缺省回退类常量）----
    NAME = "dual_piper"  # 实例名（debug_print 前缀）
    ADAPTER_TYPE = "dual_piper"  # 本 adapter 的 entry point 类型

    # ---- 能力 / 连接参数（类级常量，自包含，不随 discover 传输）----
    ROBOT_MODEL_ID = "dual-piper"
    ROBOT_MODEL_VERSION = "0.0.0"
    # 双臂 Piper：左 + 右臂，各 6 关节 + 1 夹爪 = 7，共 14
    ACTION_DIM = 14
    # 相机布局：{相机名: 分辨率 (width, height)}（SDK 产出 raw RGB；observe 编码 JPEG 原图）
    IMAGES: dict[str, tuple[int, int]] = {
        "cam_head": (640, 480),
        "cam_left_wrist": (640, 480),
        "cam_right_wrist": (640, 480),
    }

    # 中间件连接参数（真实 SDK 进程地址 / 共享内存名接入时配置）
    SDK_URL = "http://127.0.0.1:8090"  # SDK HTTP 服务地址（指令下行；TODO 接入真实双臂 SDK）
    SHM_NAME = "dual_piper_obs"  # 共享内存名（观测上行；TODO 接入真实双臂 SDK）
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
        - 能力（动作维度 / 相机布局）与连接参数**全部由类级常量定义**，不随 discover 传输、
          不接收 Edge 配置。
        """
        super().__init__(name=name, id=id)
        self.id = id or self.ADAPTER_TYPE
        self.name = name or self.NAME
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
        """释放 Edge 侧本地资源（SDK 连接由进程自维护）。"""
        if self._shm is not None:
            self._shm.close()
            self._shm = None
        if self._http is not None:
            self._http.close()
            self._http = None
        debug_print(self.name, "DualPiperAdapter released.", "INFO")

    # ---- capabilities ----------------------------------------------------------
    @property
    def capabilities(self) -> RobotCapabilities:
        obs_keys = ["observations/qpos"] + [f"observations/images/{img}" for img in self.images]
        return RobotCapabilities(
            robot_model_id=self.robot_model_id,
            robot_model_version=self.robot_model_version,
            action_dim=self.action_dim,
            observation_keys=obs_keys,
            capabilities=dict(self._capabilities),
        )

    # ---- health（预留：查询真实双臂 SDK 进程状态；硬件由 SDK 自维护）-------------
    def health(self) -> HealthStatus:
        """健康检查：查询 SDK 进程状态（TODO 接入真实双臂 SDK 的 health 协议）。"""
        raise NotImplementedError("DualPiperAdapter.health: 预留接口，接入真实 Piper 双臂 SDK 时实现")

    # ---- 指令（预留：经 HTTP 转发真实双臂 SDK；动作拆分到左右臂）-----------------
    def reset(self):
        """程序复位到 home（非阻塞）：TODO 实现双臂复位指令。"""
        raise NotImplementedError("DualPiperAdapter.reset: 预留接口，接入真实 Piper 双臂 SDK 时实现")

    def execute(self, action: dict):
        """直接下发动作指令（raw）：TODO 把动作拆分到左右臂并下发 SDK。"""
        raise NotImplementedError("DualPiperAdapter.execute: 预留接口，接入真实 Piper 双臂 SDK 时实现")

    def set_teleop(self, enabled: bool):
        """设置遥操作开关：TODO 转发真实双臂 SDK。"""
        raise NotImplementedError("DualPiperAdapter.set_teleop: 预留接口，接入真实 Piper 双臂 SDK 时实现")

    def rollout(self, action):
        """推理闭环：TODO 接收模型 action（维度校验），拆分左右臂设限速目标。"""
        raise NotImplementedError("DualPiperAdapter.rollout: 预留接口，接入真实 Piper 双臂 SDK 时实现")

    def safe_stop(self):
        """安全停止（幂等、失败安全）：TODO 实现双臂急停指令。"""
        raise NotImplementedError("DualPiperAdapter.safe_stop: 预留接口，接入真实 Piper 双臂 SDK 时实现")

    # ---- 采集数据状态（预留：查询 SDK 进程数据保存路径 + 数据列表）----------------
    def data_status(self) -> CaptureData | None:
        """采集数据状态：TODO 查询真实双臂 SDK 进程的数据状态。"""
        raise NotImplementedError("DualPiperAdapter.data_status: 预留接口，接入真实 Piper 双臂 SDK 时实现")

    # ---- 采集回合控制（预留：转发真实双臂 SDK 开启 / 结束一轮采集）---------------
    def start_capture(self):
        """开始一轮采集：TODO 转发真实双臂 SDK。"""
        raise NotImplementedError("DualPiperAdapter.start_capture: 预留接口，接入真实 Piper 双臂 SDK 时实现")

    def end_capture(self):
        """结束一轮采集：TODO 转发真实双臂 SDK。"""
        raise NotImplementedError("DualPiperAdapter.end_capture: 预留接口，接入真实 Piper 双臂 SDK 时实现")

    # ---- observe（预留：共享内存读取双臂观测，图像编码 JPEG）----------------------
    def observe(self) -> dict:
        """读取共享内存最新观测帧（双臂 qpos + 相机），编码 JPEG（Edge 契约）。"""
        raise NotImplementedError("DualPiperAdapter.observe: 预留接口，接入真实 Piper 双臂 SDK 时实现")
