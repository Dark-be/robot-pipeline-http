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

"""DualPiperEnv —— 真实双臂机器人运行环境（接入位预留，未接入硬件）。"""

from env.base_env import BaseEnv
from robot.dual_piper_robot import DualPiperRobot


class DualPiperEnv(BaseEnv):
    """真实双臂运行环境：DualPiperRobot + 30Hz 主循环 + 共享内存发布（接入位预留）。"""

    def __init__(self, robot_config: dict | None = None, capture_config: dict | None = None):
        super().__init__(robot=DualPiperRobot(robot_config), capture_config=capture_config)
