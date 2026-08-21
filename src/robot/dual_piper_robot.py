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

"""DualPiperRobot —— 真实双臂机器人骨架（**接入位预留，未接入硬件**）。

结构对齐原 ``alicia_piper_teleop_robot``：master（alicia 主手）+ slave（piper 从手）
双控制器 + 3 相机。本骨架只声明布局 / 控制状态机（**无 profile**，obs/action 形态由
类常量固定，与 DualPiperAdapter 协定一致）；硬件 SDK（alicia_d_sdk / pyAgxArm /
pyrealsense2 / v4l2）**仅在机器人端安装**，接入时在 ``connect()`` / ``get_observation_qpos()`` /
``get_observation_images()`` / ``_apply_action()`` 中填充。

遥操作：``teleop_enabled`` 默认 False（adapter 通讯控制中暂时均处于 false）；
接入位见 ``_get_teleop_target()``（master 主手 → slave 从手）。
"""

import numpy as np

from robot.base_robot import BaseRobot
from robot.controller.piper_controller import PiperController  # noqa: E402
from robot.controller.alicia_teach_controller import AliciaTeachController  # noqa: E402
from robot.sensor.realsense_sensor import RealsenseSensor  # noqa: E402
from robot.sensor.v4l2_sensor import V4l2Sensor  # noqa: E402
from utils.base.data_handler import debug_print  # noqa: E402
import cv2

class DualPiperRobot(BaseRobot):
    NAME = "dual_piper"
    ADAPTER_TYPE = "dual_piper"
    ROBOT_MODEL_ID = "dual-piper"
    ROBOT_MODEL_VERSION = "0.0.0"

    QPOS = 14
    IMAGE_NAMES = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
    IMAGES = { name: (640, 480) for name in IMAGE_NAMES }
    SHM_NAME = "dual_piper_obs"

    def __init__(self, robot_config: dict | None = None):
        super().__init__(robot_config)
        # ---- 控制器 / 传感器接入位（硬件 SDK 仅在机器人端；接入时实例化）----
        # 参照原 src/robot/alicia_piper_teleop_robot.py：
        #   left/right_master = AliciaTeachController（主手，遥操作输入）
        #   left/right_arm    = PiperController（从手，执行）
        #   cam_head = RealsenseSensor；cam_left/right_wrist = V4l2Sensor
        self.controllers: dict = {
            'left_arm': PiperController('left'),  # slave 从手
            'right_arm': PiperController('right'),  # slave 从手
            'left_master': AliciaTeachController('left_master'),  # master 主手
            'right_master': AliciaTeachController('right_master'),  # master 主手
        }
        self.sensors: dict = {
            'cam_head': RealsenseSensor('cam_head'),
            'cam_left_wrist': V4l2Sensor('cam_left_wrist'),
            'cam_right_wrist': V4l2Sensor('cam_right_wrist'),
        }

    def connect(self):
        """连接真实双臂 SDK：master 主手（遥操作输入）+ slave 从手（执行） + 相机。

        主手只作输入、不下发指令；slave 从手执行。硬件 SDK 仅在机器人端安装。
        """
        self.controllers['left_master'].connect(port='/dev/ttyACM0')
        self.controllers['right_master'].connect(port='/dev/ttyACM1')
        self.controllers['left_arm'].connect(port='can_left')
        self.controllers['right_arm'].connect(port='can_right')
        debug_print(self.name, "Setup controllers done", "INFO")
        self.sensors['cam_head'].connect(device="261222074970", is_jpeg=True)
        self.sensors['cam_left_wrist'].connect(device="/dev/left-camera", is_jpeg=True)
        self.sensors['cam_right_wrist'].connect(device="/dev/right-camera", is_jpeg=True)
        debug_print(self.name, "Setup sensors done", "INFO")
        self.ready = True

    def disconnect(self):
        """断开 master/slave 控制器与相机（幂等，重复调用安全）。"""

        self.ready = False
        for name, ctrl in self.controllers.items():
            ctrl.disconnect()
            debug_print(self.name, f"Disconnect controller {name} done", "INFO")
        for name, sensor in self.sensors.items():
            sensor.disconnect()
            debug_print(self.name, f"Disconnect sensor {name} done", "INFO")

    def get_observation_qpos(self) -> np.ndarray:
        """读取当前帧原始观测的 qpos（扁平 QPOS 维）——从 slave 从臂控制器「手搓」。

        布局（对齐 QPOS=14 / init_qpos）：左 6 关节 + 左夹爪 + 右 6 关节 + 右夹爪。
        """
        left_joint = self.controllers["left_arm"].get_joint()
        left_gripper = self.controllers["left_arm"].get_gripper()
        right_joint = self.controllers["right_arm"].get_joint()
        right_gripper = self.controllers["right_arm"].get_gripper()
        if left_joint is None or left_gripper is None or right_joint is None or right_gripper is None:
            raise RuntimeError("DualPiperRobot.get_observation_qpos: 从臂控制器读取失败（返回 None）")
        return np.concatenate([
            left_joint, [left_gripper],
            right_joint, [right_gripper],
        ]).astype(np.float32)

    def get_observation_images(self) -> list:
        """读取当前帧原始观测的 images——从相机传感器直接取 color。

        顺序对齐 IMAGE_NAMES（cam_head / cam_left_wrist / cam_right_wrist）；任一无帧时报错。
        """
        images = []
        for name in self.IMAGE_NAMES:
            info = self.sensors[name].get_information()
            color = info.get("color") if info else None
            if color is None:
                raise RuntimeError(f"DualPiperRobot.get_observation_images: 相机 {name} 无帧")
            decoded = cv2.imdecode(color, cv2.IMREAD_COLOR)
            if decoded is None:
                # raise RuntimeError(f"DualPiperRobot.get_observation_images: 相机 {name} JPEG 解码失败")
                debug_print(self.name, f"相机 {name} JPEG 解码失败，返回空帧", "WARNNING")
                decoded = np.zeros((self.IMAGES[name][1], self.IMAGES[name][0], 3), dtype=np.uint8)
            images.append(decoded[:, :, ::-1])  # BGR → RGB
        return images

    def _apply_action(self, action: np.ndarray):
        """把目标 action 拆分到 slave 从臂下发（限速已在 step() 内完成）。

        布局：左 6 关节 + 左夹爪 + 右 6 关节 + 右夹爪（对齐 QPOS=14）。
        set_joint 内部会 clip 关节角度（原地修改），故传 copy 避免改动 self.action。
        """
        left = self.controllers["left_arm"]
        right = self.controllers["right_arm"]
        left.set_joint(np.asarray(action[:6], dtype=np.float64).copy())
        left.set_gripper(float(action[6]))
        right.set_joint(np.asarray(action[7:13], dtype=np.float64).copy())
        right.set_gripper(float(action[13]))

    def _get_teleop_target(self) -> np.ndarray | None:
        """遥操作接入位：master 主手 → slave 从手（左→左、右→右）。

        从 master 主手读取并返回扁平 QPOS 维 np.ndarray；主手读取失败/未连接时
        返回 None（step() 保持原 target_action 不刷新）。
        """
        left_joint = self.controllers["left_master"].get_joint()
        left_gripper = self.controllers["left_master"].get_gripper()
        right_joint = self.controllers["right_master"].get_joint()
        right_gripper = self.controllers["right_master"].get_gripper()
        if left_joint is None or left_gripper is None or right_joint is None or right_gripper is None:
            return None
        print(f"Teleop target: left_gripper={left_gripper}, right_gripper={right_gripper}")
        return np.concatenate([
            left_joint, [left_gripper],
            right_joint, [right_gripper],
        ]).astype(np.float32)
