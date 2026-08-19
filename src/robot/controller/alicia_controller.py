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

import alicia_d_sdk
import numpy as np
from alicia_d_sdk.utils.logger import BeautyLogger, LogLevel

from .arm_controller import ArmController
from utils.base.data_handler import debug_print

# 只显示 ERROR 级别的日志
logger = BeautyLogger(log_dir="logs", log_name="app.log", min_level=LogLevel.ERROR)


class AliciaController(ArmController):
    def __init__(self, name="alicia_controller"):
        super().__init__(name)
        self.robot = None
        self.port: str = "/dev/ttyACM0"

    def connect(self, port: str):
        self.port = port

        self.robot = alicia_d_sdk.create_robot(
            port=port,
            gripper_type="50mm",
            debug_mode=False,
            auto_connect=True,
        )
        self.robot.torque_control("off")
        debug_print(self.name, f"Connected to Alicia on port {port}", "INFO")

    def disconnect(self):
        if self.robot is None:
            return
        try:
            self.robot.disconnect()
        finally:
            self.robot = None

    def _get_state(self):
        if self.robot is None:
            raise RuntimeError(f"{self.name}: controller is not set up (robot is None)")

        state = self.robot.get_robot_state("joint_gripper")
        if state is None:
            return {"joint": None, "gripper": None, "pose": None}
        offset = [0, -2.0, 0.4, 0, 0.6, 0]  # SDK 角度偏移（弧度），需要根据实际情况调整
        multiplier = [1, -1.3, -1, -1, -1, -1]

        joint = np.asarray(state.angles, dtype=float)
        joint = np.array([(joint[i] + offset[i]) * multiplier[i] for i in range(6)], dtype=float)
        gripper_raw = float(state.gripper)  # SDK: 0-1000
        gripper = max(0.0, min(1.0, gripper_raw / 1000.0))

        return {"joint": joint, "gripper": gripper}
