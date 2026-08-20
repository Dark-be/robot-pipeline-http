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

"""TestRobot —— 虚拟测试机器人（**无硬件依赖**，离线联调 Edge 侧 TestRobotAdapter）。

参考 DualPiperRobot 的结构：由 **test 控制器 + test 视觉传感器**组装——动作经
``_apply_action`` 下发到控制器，qpos 从控制器「手搓」读取，图像从传感器读取并解码
为 raw RGB。

- **控制器**：左/右臂 ``TestArmController``（内部有界随机游走，模拟关节反馈）。
- **视觉传感器**：3 相机 ``TestVisionSensor``（每帧生成移动 RGB 渐变，JPEG 输出）。
- 身份 / 共享内存名（``type="test_robot"`` / ``SHM="test_robot_obs"``）对齐
  ``TestRobotAdapter``；换真实 SDK 进程即可无缝替换。
"""

import cv2
import numpy as np

from robot.base_robot import BaseRobot
from robot.controller.test_arm_controller import TestArmController  # noqa: E402
from robot.sensor.test_vision_sensor import TestVisionSensor  # noqa: E402
from utils.base.data_handler import debug_print  # noqa: E402


class TestRobot(BaseRobot):
    NAME = "test_robot"
    ADAPTER_TYPE = "test_robot"
    ROBOT_MODEL_ID = "test-robot"
    ROBOT_MODEL_VERSION = "0.0.0"
    # 双臂：左 + 右臂，各 6 关节 + 1 夹爪 = 7，共 14
    QPOS = 14
    IMAGE_NAMES = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
    IMAGES = {name: (640, 480) for name in IMAGE_NAMES}
    SHM_NAME = "test_robot_obs"

    def __init__(self, robot_config: dict | None = None):
        super().__init__(robot_config)
        # 控制器：左/右臂 test 控制器（执行）；传感器：3 个 test 视觉传感器（相机）
        # 结构对齐 DualPiperRobot（无 master 主手：测试机器人不做遥操作）
        self.controllers: dict = {
            'left_arm': TestArmController('left_arm'),  # 左臂，执行
            'right_arm': TestArmController('right_arm'),  # 右臂，执行
        }
        self.sensors: dict = {
            'cam_head': TestVisionSensor('cam_head'),
            'cam_left_wrist': TestVisionSensor('cam_left_wrist'),
            'cam_right_wrist': TestVisionSensor('cam_right_wrist'),
        }

    def connect(self):
        """连接 test 控制器与视觉传感器（虚拟，无硬件）。"""
        for ctrl in self.controllers.values():
            ctrl.connect()
        for sensor in self.sensors.values():
            sensor.connect(is_jpeg=True)
        self.ready = True
        debug_print(self.name, "TestRobot connected (virtual).", "INFO")

    def disconnect(self):
        """断开控制器与传感器（幂等，重复调用安全）。"""
        self.ready = False
        for name, ctrl in self.controllers.items():
            ctrl.disconnect()
            debug_print(self.name, f"Disconnect controller {name} done", "INFO")
        for name, sensor in self.sensors.items():
            sensor.disconnect()
            debug_print(self.name, f"Disconnect sensor {name} done", "INFO")

    # ---- 控制 / 观测 ----
    def _apply_action(self, action: np.ndarray):
        """把目标 action 拆分到左/右臂控制器下发（限速已在 step() 内完成）。

        布局：左 6 关节 + 左夹爪 + 右 6 关节 + 右夹爪（对齐 QPOS=14）。
        """
        left = self.controllers["left_arm"]
        right = self.controllers["right_arm"]
        left.set_joint(np.asarray(action[:6], dtype=np.float64).copy())
        left.set_gripper(float(action[6]))
        right.set_joint(np.asarray(action[7:13], dtype=np.float64).copy())
        right.set_gripper(float(action[13]))

    def get_observation_qpos(self) -> np.ndarray:
        """读取当前帧原始观测的 qpos（扁平 QPOS 维）——从控制器「手搓」。

        布局（对齐 QPOS=14）：左 6 关节 + 左夹爪 + 右 6 关节 + 右夹爪。
        """
        left_joint = self.controllers["left_arm"].get_joint()
        left_gripper = self.controllers["left_arm"].get_gripper()
        right_joint = self.controllers["right_arm"].get_joint()
        right_gripper = self.controllers["right_arm"].get_gripper()
        if (left_joint is None or left_gripper is None
                or right_joint is None or right_gripper is None):
            raise RuntimeError("TestRobot.get_observation_qpos: 控制器读取失败（返回 None）")
        return np.concatenate([
            np.asarray(left_joint).ravel(),
            np.asarray(left_gripper).ravel(),
            np.asarray(right_joint).ravel(),
            np.asarray(right_gripper).ravel(),
        ]).astype(np.float32)

    def get_observation_images(self) -> list:
        """读取各相机 raw RGB 帧（顺序对齐 IMAGE_NAMES）。

        从 test 视觉传感器读 color（JPEG），解码为 raw RGB——与真实 piper 接入位一致。
        """
        images = []
        for name in self.IMAGE_NAMES:
            info = self.sensors[name].get_information()
            color = info.get("color") if info else None
            if color is None:
                raise RuntimeError(f"TestRobot.get_observation_images: 相机 {name} 无帧")
            decoded = cv2.imdecode(color, cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError(f"TestRobot.get_observation_images: 相机 {name} JPEG 解码失败")
            images.append(decoded[:, :, ::-1])  # BGR → RGB
        return images
