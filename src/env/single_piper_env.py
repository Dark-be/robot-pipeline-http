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

"""SinglePiperEnv —— 单臂双 Piper 遥操作机器人运行环境（Leader 主臂 + Follower 从臂）。"""

from env.base_env import BaseEnv
from robot.single_piper_robot import SinglePiperRobot


class SinglePiperEnv(BaseEnv):
    """真实单臂双 Piper 运行环境：SinglePiperRobot + 30Hz 主循环 + 共享内存发布。"""

    def __init__(self, robot_config: dict | None = None, capture_config: dict | None = None):
        super().__init__(robot=SinglePiperRobot(robot_config), capture_config=capture_config)
