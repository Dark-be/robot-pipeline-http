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

"""BaseRobot —— 机器人基类（**无 profile**；obs/action 形态由各机器人类常量固定）。

机器人是「会动」的一层：持有 ``action``（当前指令）与 ``target_action``（目标），由运行
环境（env）每帧调用 ``step()`` 限速接近 target 并执行。本基类**不依赖 motrix_edge.profile**
——每个机器人的 obs/action 形态（动作维度 / 相机布局）由子类**类常量固定声明**，
与对应 adapter 协定一致。

约定：
- **扁平动作**：一维数组，维度由 ``QPOS`` 声明（各臂关节 + 夹爪拼接）。
  例：双臂 6 关节 + 1 夹爪 → ``QPOS`` = 14，动作 ``[左6关节, 左夹爪, 右6关节, 右夹爪]``。
- **原始观测 ``get_observation()``**：``{qpos, images, action}``（**无契约键**）——观测键
  （qpos 布局、``observations/images/<cam>``）由 robot server（contract_server）按
  adapter 约定组装并写入共享内存。
- **控制**：``reset()`` / ``execute()`` / ``rollout()`` / ``safe_stop()`` 只修改
  ``target_action``，实际运动由 env 主循环每帧 ``step()`` 限速推进（单一目标模型）。
- **遥操作**：``teleop_enabled`` 默认 **False**（adapter 通讯控制中暂时均处于 false）；
  接入位由子类在 ``_get_teleop_target()`` 实现，``step()`` 仅在开启时刷新 target_action。
"""

import numpy as np
import time

from utils.base.data_handler import debug_print


class BaseRobot:
    # ---- 身份 / 能力（对齐 adapter 契约；子类覆盖）----
    NAME = "base_robot"
    ADAPTER_TYPE = "base_robot"  # 本机器人对应的 adapter 类型（entry point 名）
    ROBOT_MODEL_ID = "base-robot"
    ROBOT_MODEL_VERSION = "0.0.0"

    # ---- 布局 / 特性（子类覆盖）----
    QPOS = 14  # 扁平动作维度（各臂关节 + 夹爪拼接）
    IMAGE_NAMES: list[str] = []  # 相机名（observations/images/<name>）
    IMAGES: dict[str, tuple[int, int]] = {name: (640, 480) for name in IMAGE_NAMES}  # 相机名 -> (w, h)
    SHM_NAME = "robot_obs"  # 观测共享内存名（server 侧发布）

    # ---- 观测键（adapter 契约约定，与 Edge 侧 / robot server 保持一致；勿改）----
    KEY_QPOS = "observations/qpos"  # 关节 qpos 键
    CAMERA_PREFIX = "observations/images/"  # 相机图像键前缀（<prefix><cam_name>）

    def __init__(self, robot_config: dict | None = None):
        self.robot_config = dict(robot_config or {})
        self.name = str(self.robot_config.get("name", self.NAME))
        # 每帧最大关节增量（rad）：限速插值步长，可经配置 step_rad 修改
        self.step_rad = float(self.robot_config.get("step_rad", 0.1))
        self.ready = False
        self.last_error = None

        # 控制状态：action 当前执行位置；target_action 目标（reset/execute/rollout 都只覆盖它）
        self.action: np.ndarray | None = None
        self.target_action: np.ndarray | None = None
        # 复位目标：默认全零（子类可按配置 init_qpos 覆盖）
        self.init_qpos = self.robot_config.get("init_qpos")
        if self.init_qpos is None:
            raise ValueError(f"Robot {self.name} init_qpos is not set in robot_config.")

        # 遥操作开关：adapter 通讯控制中暂时均处于 False（真实接入位见子类）
        self.teleop_enabled = False

        # 帧计数（get_observation() 每帧自增；server 组装 standard_obs 时附带）
        self.seq = 0

        # 控制器 / 传感器（真实机器人填充；虚拟机器人可为空）
        self.controllers: dict = {}
        self.sensors: dict = {}

    # ---- 布局 / 解析（无 profile；obs/action 形态由类常量固定）------------------
    @classmethod
    def action_dim(cls) -> int:
        return cls.QPOS  # 扁平动作维度（各臂关节 + 夹爪拼接）

    # ---- 控制：HTTP（reset/execute/rollout/safe_stop）都只修改 target_action -----------
    def reset(self):
        """程序复位到 home（非阻塞）：target_action = init_qpos（env 每帧限速接近）。"""
        self.disable_teleop()
        self.set_target_action(self.init_qpos)

    def execute(self, action: np.ndarray):
        """直接下发动作指令（raw）：扁平动作 → target_action。"""
        self.set_target_action(action)

    def rollout(self, action: np.ndarray):
        """推理闭环：当前与 execute 一致（都只修改唯一 target_action）。"""
        self.execute(action)

    def safe_stop(self):
        """安全停止（幂等、失败安全）：退出遥操作并清空目标，step() 不再推进。"""
        self.teleop_enabled = False
        self.target_action = None
        debug_print(self.name, "Safe stop executed.", "WARNING")

    def set_target_action(self, target_action: np.ndarray | None):
        """设置目标动作：step() 每帧把 action 朝 target_action 限速移动。"""
        self.target_action = None if target_action is None else np.asarray(target_action, dtype=np.float64)

    # ---- 遥操作（接入位；adapter 通讯控制中暂时均为 False）-------------------------
    def enable_teleop(self):
        """启用遥操作：step() 把 target_action 刷新为主臂目标。"""
        self.teleop_enabled = True

    def disable_teleop(self):
        """关闭遥操作：回到程序控制，step() 不再刷新 target_action。"""
        self.teleop_enabled = False

    def _get_teleop_target(self) -> np.ndarray | None:
        """遥操作目标源（主臂 → 从臂）；默认 None（无遥操作源）。"""
        return None

    # ---- 每帧推进 --------------------------------------------------------------------
    def step(self):
        """每帧：把 action 朝 target_action 限速接近并执行（子类实现执行 / 状态更新）。"""
        if self.teleop_enabled:
            target = self._get_teleop_target()
            if target is not None:
                self.set_target_action(target)
        if self.target_action is None:
            return
        if self.action is None:
            self.action = self._init_action_from_qpos()

        self.action = self._step_toward(self.action, self.target_action, self.step_rad)
        
        self._apply_action(self.action)

    def _init_action_from_qpos(self) -> np.ndarray:
        """以当前实际状态初始化 action（避免开始时跳变）。"""
        qpos = self.get_observation_qpos()
        return qpos

    def _apply_action(self, action: np.ndarray):
        """把 action 下发 / 应用到硬件（子类实现；虚拟机器人同步到合成状态）。"""
        raise NotImplementedError

    # ---- 每帧观测（**原始数据，无契约键**；由 robot server 组装 standard_obs + 写共享内存）----
    def get_observation(self) -> dict:
        """读取当前帧原始观测（契约键：observations/qpos + 每相机 observations/images/<cam> + action）。

        seq 自增，供 server 上报。子类实现 get_observation_qpos() / get_observation_images()
        ——直接从控制器 / 传感器「手搓」取数据；本方法负责按契约键组装。
        """
        self.seq += 1
        qpos = self.get_observation_qpos()
        action = self.get_action()
        timestamp = time.time()
        if action is None:
            action = qpos  # 无指令时以当前 qpos 作为 action（保证观测含有效 action）
        obs = {self.KEY_QPOS: qpos, "action": action, "timestamp": timestamp}
        # 相机成员：observations/images/<cam_name>（顺序对齐 IMAGE_NAMES）
        for name, img in zip(self.IMAGE_NAMES, self.get_observation_images()):
            obs[f"{self.CAMERA_PREFIX}{name}"] = img
        return obs

    def get_observation_qpos(self) -> np.ndarray:
        """读取当前帧原始观测的 qpos（扁平 QPOS 维；子类实现）。"""
        raise NotImplementedError

    def get_observation_images(self) -> list:
        """读取各相机 raw RGB 帧（list，顺序对齐 IMAGE_NAMES；子类实现）。

        由 get_observation() 组装为 observations/images/<cam_name> 成员。
        """
        raise NotImplementedError

    def get_action(self) -> np.ndarray | None:
        """当前执行中的 action（无则 None）。"""
        return self.action.copy() if self.action is not None else None


    # ---- 状态（供 server 上报）---------------------------------------------------------
    def data_status(self) -> dict:
        """采集数据状态（预留：数据保存路径 + 本次采集得到的数据列表）。"""
        return {"data_dir": None, "episodes": []}

    # ---- 生命周期（env / server 调用）----------------------------------------------------
    def connect(self):
        """连接硬件（子类实现；虚拟机器人直接 ready）。"""
        self.ready = True

    def disconnect(self):
        """断开硬件 / 释放资源（子类实现）。"""
        pass

    # ---- 通用限位插值 ------------------------------------------------------------------
    @staticmethod
    def _step_toward(current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
        """限位插值：每步最多向目标靠近 max_step，防止关节数据跳变。"""
        current = np.asarray(current, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        delta = target - current
        step = np.clip(delta, -max_step, max_step)
        return np.where(np.abs(delta) <= max_step, target, current + step)
