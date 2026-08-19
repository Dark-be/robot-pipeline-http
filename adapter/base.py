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

"""RobotAdapter —— 机器人硬件抽象层（HAL）契约。

核心（session / server / CLI）只依赖本接口与 entry point 发现，不引用具体机器人实现。
具体机器人（及 controller / sensor / profile）由外部 SDK / 包实现本接口并通过
``motrix_edge.adapters`` entry point 注册接入。

职责面（角色）：
  discover/health  发现并检查硬件（discover / health / ready / release）
  capabilities    声明能力（动作维度 / 观测布局 / 相机）
  observe         读取最新观测缓存（JPEG 图像 + qpos；**不推进 / 不影响适配器运行**）
  execute         执行动作指令（直接下发）
  data_status     采集数据状态（数据保存路径 + 本次采集得到的数据列表；采集会话预留）
  rollout         推理闭环（被推理任务消费）
  safe_stop       安全停止（幂等、失败安全）

设计取舍：
  - **适配器独立运行**：适配器自身持续运行（常驻运行线程 / 硬件控制循环）推进运动并更新
    最新观测缓存；``observe()`` **只读取缓存**（JPEG 图像 + qpos），不推进、不驱动适配器。
    ``rollout()`` 设置目标，由适配器运行循环限速靠近。
  - **观测图像为 JPEG**：观测缓存中的摄像头帧为 **JPEG 编码**（adapter 提供，如 640x480）；
    Edge 侧可解码 / 降采样后用于预览与 WebRTC 推流。
  - **采集下沉、无回合控制**：数据采集（录制写盘）由适配器 / 机器人进程自维护——Edge
    进入采集会话后只读共享内存观测并展示，**不驱动回合**。adapter 只预留一个**数据状态**
    接口（``data_status()``）返回数据保存路径 + 本次采集得到的数据列表，供 server 状态
    上报。观测键契约（KEY_QPOS / KEY_ACTION / CAMERA_PREFIX）在此单点定义。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

# ---- 观测键契约（standard_obs 字典的键名，与 ACT 采集格式一致）----------------
KEY_QPOS = "observations/qpos"
KEY_ACTION = "action"
CAMERA_PREFIX = "observations/images/"


class AdapterCapability(str, Enum):
    """适配器能力标识（能力描述 dict 的键）。

    会话按能力选择适配器：CaptureSession 要求 CAPTURE，InferSession 要求 EXECUTE。
    """

    CAPTURE = "capture"  # 支持数据采集（数据生产者，被采集控制）
    EXECUTE = "execute"  # 支持动作执行（推理闭环，被推理任务消费）
    STREAMING = "streaming"  # 支持视频流（遥操作预览 / 只读流，供后续 WebRTC 使用）


@dataclass
class HealthStatus:
    """健康检查结果：ok=False 时 detail 说明原因。"""

    ok: bool
    detail: str = ""


@dataclass
class CaptureData:
    """采集数据状态（采集会话预留）：数据保存路径 + 本次采集得到的数据列表。

    数据采集（录制写盘）由适配器 / 机器人进程自维护；Edge 只读共享内存观测并展示，
    **不驱动回合**。adapter 通过 ``data_status()`` 返回本状态，供 server 状态上报。
    """

    save_dir: str | None = None  # 数据保存路径（未启用 / 未知 → None）
    data_files: list[str] = field(default_factory=list)  # 本次采集得到的数据列表


@dataclass
class RobotCapabilities:
    """适配器声明的能力（数据布局声明）。"""

    robot_model_id: str = "unknown"
    robot_model_version: str = "0.0.0"
    action_dim: int = 0
    # 观测键（如 observations/qpos、observations/images/cam_head）
    observation_keys: list[str] = field(default_factory=list)
    # 能力描述 dict：capability -> 是否支持（见 AdapterCapability：CAPTURE / EXECUTE / STREAMING）
    capabilities: dict[AdapterCapability, bool] = field(
        default_factory=lambda: {
            AdapterCapability.CAPTURE: True,
            AdapterCapability.EXECUTE: True,
        }
    )

    @property
    def image_names(self) -> list[str]:
        """从观测键推导相机名（observations/images/<name>）。"""
        return [k[len(CAMERA_PREFIX) :] for k in self.observation_keys if k.startswith(CAMERA_PREFIX)]

    def supports(self, cap: AdapterCapability) -> bool:
        """该适配器是否支持给定能力。"""
        return self.capabilities.get(cap, False)


@dataclass
class DiscoveredRobot:
    """机器人进程 discover 结果 —— **只保留身份（id / name / type）**。

    身份用于实例化 adapter：``type`` 为 adapter 类 entry point 名（加载并实例化）。
    能力（动作维度 / 相机 / capabilities）与连接参数（SDK 地址 / 共享内存名）**全部由
    adapter 内部类常量定义**，不随 discover 传输——discover 只回答「找到了哪个类型的
    机器人进程」。
    """

    id: str
    name: str
    type: str  # adapter 类型（entry point 名，用于加载 adapter 类）


class RobotAdapter(ABC):
    """机器人硬件抽象层接口。

    由 **身份参数**（discover 解析出的 ``name`` / ``id``，`type` 由类常量确定）参数化；
    能力与连接参数（SDK 地址 / 共享内存名）由 adapter 内部类常量定义。adapter 只负责
    **连接进程**并转发指令 / 读取观测——不接收 Edge 配置、不自带 discover / probe（发现
    由 ``discover_adapter`` 完成）。
    """

    # 本 adapter 的 entry point 类型（类确定，用于匹配 discover 的 type 加载类）；子类覆盖。
    ADAPTER_TYPE: str = ""

    # 能力声明（类级 dict，供「不实例化」按能力列出 / 过滤 adapter；子类覆盖）。
    # 实例 ``capabilities`` 属性把本声明并入 RobotCapabilities.capabilities。
    CAPABILITIES: dict[AdapterCapability, bool] = {}

    def __init__(self, name: str = "", id: str = ""):
        """身份参数化：``name`` / ``id`` 由 discover 赋予（缺省为空 = 进程内测试）。

        ``type`` 由类常量 ``ADAPTER_TYPE`` 确定（不随 discover 传输）。
        能力与连接参数由子类类常量定义，不在此接收。
        """
        self.name = name
        self.id = id
        self.type = self.ADAPTER_TYPE

    # ---- health / release（硬件由 SDK 进程自维护，Edge 只查询 / 释放本地资源）-----
    def release(self) -> None:
        """释放资源（原 disconnect）。默认 no-op，子类按需实现。"""
        pass

    @abstractmethod
    def health(self) -> HealthStatus:
        """健康检查：就绪 / 状态 / 错误详情。"""
        raise NotImplementedError

    @property
    def ready(self) -> bool:
        """是否就绪（可开始任务）。默认取 health().ok。"""
        return self.health().ok

    # ---- capabilities（声明能力）--------------------------------------------
    @property
    @abstractmethod
    def capabilities(self) -> RobotCapabilities:
        """声明能力：动作维度 / 观测布局 / 相机。"""
        raise NotImplementedError

    # ---- observe（读取最新观测缓存，被预览 / policy 推理消费）------------------
    @abstractmethod
    def observe(self) -> dict:
        """返回适配器维护的「最新观测缓存」（standard_obs，含 ``action``）。

        - 图像为 **JPEG 编码**（adapter 提供，如 640x480）；qpos / action 为状态缓存。
        - **observe 不推进 / 不影响适配器运行**——适配器自身持续运行（控制循环 /
          采集程序）更新缓存，observe 只是取出缓存。
        - 被「预览（摄像头 + 状态）」与「policy 推理」消费；不采集。
        键见模块级契约（KEY_QPOS / CAMERA_PREFIX / KEY_ACTION）。
        """
        raise NotImplementedError

    # ---- execute（执行动作指令）-----------------------------------------------
    @abstractmethod
    def execute(self, action: dict) -> None:
        """直接下发动作指令（raw 指令，立即执行）。"""
        raise NotImplementedError

    # ---- teleop（遥操作开关）--------------------------------------------------
    def set_teleop(self, enabled: bool) -> None:
        """设置遥操作开关（``True``=遥操作 / ``False``=程控 / 推理控制）。

        默认 no-op；支持遥操作的子类按需覆盖（如经 HTTP 转发机器人进程 /v1/teleop）。
        """
        pass

    # ---- 采集数据状态（采集会话预留：数据由适配器 / 进程自维护，无回合控制）-----
    # Edge 进入采集会话后只读共享内存观测并展示，不驱动回合；adapter 只预留一个
    # data_status() 接口返回数据保存路径 + 本次采集得到的数据列表（供 server 上报）。
    def data_status(self) -> CaptureData | None:
        """采集数据状态：数据保存路径 + 本次采集得到的数据列表。

        数据采集（录制写盘）由适配器 / 机器人进程自维护；本方法只查询 / 上报结果。
        未启用采集 / 数据未知 → 返回 ``None``。子类按需覆盖。
        """
        return None

    # ---- 采集回合控制（capture episode start / end）---------------------------
    def start_capture(self) -> None:
        """开始一轮采集（episode 开始）：通知适配器 / 机器人进程开启录制。

        默认 no-op；采集由适配器 / 机器人进程自维护的适配器按需覆盖（如经 HTTP 转发
        机器人进程 /v1/capture/start）。
        """
        pass

    def end_capture(self) -> None:
        """结束一轮采集（episode 结束）：通知适配器 / 机器人进程停止录制。

        默认 no-op；采集由适配器 / 机器人进程自维护的适配器按需覆盖（如经 HTTP 转发
        机器人进程 /v1/capture/end）。
        """
        pass

    # ---- rollout（推理闭环，被推理任务消费）-----------------------------------
    @abstractmethod
    def rollout(self, action) -> None:
        """推理闭环：接收模型 action（按 capabilities.action_dim 解析为限速目标）并推进一帧。"""
        raise NotImplementedError

    # ---- safe_stop（安全停止）-------------------------------------------------
    @abstractmethod
    def safe_stop(self) -> None:
        """安全停止（幂等、失败安全）。"""
        raise NotImplementedError

    # ---- 生命周期辅助 ----------------------------------------------------------
    def reset(self) -> None:
        """程序复位到 home（非阻塞）：设置 home 目标，由后续 observe()/rollout() 推进。"""
        pass
