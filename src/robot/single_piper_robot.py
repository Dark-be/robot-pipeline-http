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

"""SinglePiperRobot —— 单臂双 Piper 遥操作机器人（Leader 主臂 + Follower 从臂）。

由**两个 Piper 机械臂**组成：一个为 Leader（主臂，只读取，经 ``set_leader_mode()`` 进入
主臂模式，用 ``get_leader_joint_angles()`` 读关节），一个为 Follower（从臂，执行动作）。
**无相机**（obs 只有 qpos，无 images）。

- 动作维度：Follower 单臂 6 关节 + 1 夹爪 = 7（``QPOS=7``），扁平 ``[j1..j6, gripper]``。
- 遥操作：``_get_teleop_target()`` 读 Leader 主臂关节角度（``get_leader_joint_angles()``）
  + 夹爪（``get_gripper()``），映射到 Follower 从臂（同构，角度可直接下发）。
- 硬件 SDK（pyAgxArm）**仅在机器人端安装**；obs/action 形态由类常量固定（无 profile）。
"""

import numpy as np

from robot.base_robot import BaseRobot
from robot.controller.piper_controller import PiperController  # noqa: E402
from utils.base.data_handler import debug_print  # noqa: E402


class SinglePiperRobot(BaseRobot):
    NAME = "single_piper"
    ADAPTER_TYPE = "single_piper"
    ROBOT_MODEL_ID = "single-piper"
    ROBOT_MODEL_VERSION = "0.0.0"

    QPOS = 7  # Follower 单臂：6 关节 + 1 夹爪
    IMAGE_NAMES: list[str] = []  # 无相机
    IMAGES: dict[str, tuple[int, int]] = {}
    SHM_NAME = "single_piper_obs"

    # 默认复位目标（config 未提供 init_qpos 时使用）：6 关节 0 + 夹爪张开 1
    DEFAULT_INIT_QPOS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    def __init__(self, robot_config: dict | None = None):
        robot_config = dict(robot_config or {})
        robot_config.setdefault("init_qpos", self.DEFAULT_INIT_QPOS)
        super().__init__(robot_config)
        # 控制器：Leader 主臂（只读取，遥操作输入）+ Follower 从臂（执行）
        # 端口 / 固件版本可经 robot_config 配置（leader_port / follower_port / *_firmware）
        self.leader_port = str(self.robot_config.get("leader_port", "can_left"))
        self.follower_port = str(self.robot_config.get("follower_port", "can_right"))
        self.leader_firmware = str(self.robot_config.get("leader_firmware", "v188"))
        self.follower_firmware = str(self.robot_config.get("follower_firmware", "v188"))
        self.controllers: dict = {
            'leader': PiperController('leader'),  # Leader 主臂，只读取
            'follower': PiperController('follower'),  # Follower 从臂，执行
        }

    def connect(self):
        """连接：Leader 主臂（role=leader）+ Follower 从臂（role=follower，执行）。"""
        self.controllers['leader'].connect(
            port=self.leader_port, role="leader", firmware=self.leader_firmware)
        self.controllers['follower'].connect(
            port=self.follower_port, role="follower", firmware=self.follower_firmware)
        debug_print(self.name, "Setup controllers done", "INFO")
        self.ready = True

    def disconnect(self):
        """断开 Leader/Follower 控制器（幂等，重复调用安全）。"""

        self.ready = False
        for name, ctrl in self.controllers.items():
            ctrl.disconnect()
            debug_print(self.name, f"Disconnect controller {name} done", "INFO")

    def get_observation_qpos(self) -> np.ndarray:
        """读取当前帧原始观测的 qpos（扁平 QPOS 维）——从 Follower 从臂控制器「手搓」。

        布局（对齐 QPOS=7）：6 关节 + 夹爪。
        """
        joint = self.controllers["follower"].get_joint()
        gripper = self.controllers["follower"].get_gripper()
        if joint is None or gripper is None:
            raise RuntimeError("SinglePiperRobot.get_observation_qpos: 从臂控制器读取失败（返回 None）")
        return np.concatenate([joint, [gripper]]).astype(np.float32)

    def get_observation_images(self) -> list:
        """无相机（IMAGE_NAMES 为空），返回空列表。"""
        return []

    def _apply_action(self, action: np.ndarray):
        """把目标 action 下发到 Follower 从臂（限速已在 step() 内完成）。

        布局：6 关节 + 夹爪（对齐 QPOS=7）。
        set_joint 内部会 clip 关节角度（原地修改），故传 copy 避免改动 self.action。
        """
        self.controllers["follower"].set_joint(np.asarray(action[:6], dtype=np.float64).copy())
        self.controllers["follower"].set_gripper(float(action[6]))

    def _get_teleop_target(self) -> np.ndarray | None:
        """遥操作接入位：Leader 主臂 → Follower 从臂。

        读 Leader 主臂关节角度（get_leader_joint_angles）+ 夹爪（get_gripper），返回扁平
        QPOS 维 np.ndarray；主臂读取失败/未连接时返回 None（step() 保持原 target_action 不刷新）。
        """
        joint = self.controllers["leader"].get_leader_joint_angles()
        # gripper = self.controllers["leader"].get_gripper()
        gripper = 1
        if joint is None or gripper is None:
            return None
        return np.concatenate([joint, [gripper]]).astype(np.float32)
