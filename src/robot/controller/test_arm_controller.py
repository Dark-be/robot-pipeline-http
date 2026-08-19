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

from .arm_controller import ArmController
from utils.base.data_handler import debug_print


# 基于ArmController的测试机械臂控制器类，包含机械臂状态获取和控制方法的简单实现，用于测试和调试
class TestArmController(ArmController):
    def __init__(self, name="test_arm"):
        super().__init__(name)

    def connect(self):
        debug_print(self.name, "setup success", "INFO")

    def disconnect(self):
        debug_print(self.name, "disconnect success", "INFO")

    def _walk(self, key, shape):
        """对 self.state[key] 做一次有界随机游走：随机加减 0.05 以内，限制在 [0, 2]。"""
        cur = self.state.get(key)
        if cur is None:
            cur = np.zeros(shape, dtype=np.float64)
        cur = np.asarray(cur, dtype=np.float64)
        delta = np.random.uniform(-0.05, 0.05, size=cur.shape)
        self.state[key] = np.clip(cur + delta, 0.0, 2.0).copy()
        return self.state[key]

    def get_joint(self):
        return self._walk("joint", (6,))

    def get_position(self):
        return self._walk("pose", (6,))

    def get_gripper(self):
        return self._walk("gripper", (1,))

    def set_position(self, position: np.ndarray):
        if position.shape[0] == 6:
            debug_print(self.name, f"using EULER set position to {position}", "DEBUG")
        elif position.shape[0] == 7:
            debug_print(self.name, f"using QUATERNION set position to {position}", "DEBUG")
        else:
            debug_print(self.name, "set_position input size should be 6 -> EULER or 7 -> QUATERNION", "ERROR")

        self.state["pose"] = position

    def set_joint(self, joint: np.ndarray):
        debug_print(self.name, f"set joint to {joint}", "DEBUG")

        self.state["joint"] = joint

    # The input gripper value is in the range [0, 1], representing the degree of opening.
    def set_gripper(self, gripper: float):
        if isinstance(gripper, (int, float, complex, np.ndarray)) and not isinstance(gripper, bool):
            if 1 >= gripper >= 0:
                debug_print(self.name, f"set gripper to {gripper}", "DEBUG")
            else:
                debug_print(self.name, f"gripper better be 0~1, but get number {gripper}", "WARNING")
        else:
            debug_print(self.name, f"gripper should be a number 0~1, but get type {type(gripper)}", "ERROR")

        self.state["gripper"] = gripper
