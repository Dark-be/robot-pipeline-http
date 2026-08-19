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

import numpy as np

from .controller import Controller


class ArmController(Controller):
    def __init__(self, name="arm_controller"):
        super().__init__(name)
        self.state = {
            "joint": None,
            "gripper": None,
            "pose": None,
        }  # 机械臂状态信息，包含关节位置、末端位置和夹爪状态等

    def execute(self, action: dict):
        for key, value in action.items():
            if key == "joint":
                self.set_joint(np.array(value))
            elif key == "gripper":
                self.set_gripper(value)
            elif key == "pose":
                self.set_position(np.array(value))

    def get_joint(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_position(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_gripper(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def set_joint(self, joint: np.ndarray):
        raise NotImplementedError("Subclasses should implement this method.")

    def set_position(self, position: np.ndarray):
        raise NotImplementedError("Subclasses should implement this method.")

    def set_gripper(self, gripper: float):
        raise NotImplementedError("Subclasses should implement this method.")
