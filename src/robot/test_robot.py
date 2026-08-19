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

- **合成关节**：限速接近 target_action（step_toward）并直接同步到状态（无伺服延迟）。
- **合成图像**：每帧移动的 RGB 渐变 + 方块（raw RGB，与 shm_contract 布局一致）。
- 身份 / 共享内存名（``type="test_robot"`` / ``SHM="test_robot_obs"``）对齐
  ``TestRobotAdapter``；换真实 SDK 进程即可无缝替换。
"""

import numpy as np

from robot.base_robot import BaseRobot
from utils.base.data_handler import debug_print


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
        # 合成状态：扁平 QPOS 维动作（init_qpos 来自 config，可能为 list → 转 np 数组）
        self.qpos = np.asarray(self.init_qpos, dtype=np.float64).copy()
        self._frame = 0  # 合成图像帧计数器（每帧移动方块）

    def connect(self):
        """虚拟机器人：无需硬件，直接就绪。"""
        self.ready = True
        debug_print(self.name, "TestRobot connected (virtual).", "INFO")

    # ---- 控制 / 观测 ----
    def _apply_action(self, action: np.ndarray):
        """虚拟：状态直接同步到命令（合成无伺服延迟）。"""
        self.qpos = action.copy()

    def get_observation_qpos(self) -> np.ndarray:
        """当前扁平关节 qpos（QPOS 维，float32）。"""
        return self.qpos.astype(np.float32)

    def get_observation_images(self) -> list:
        """合成各相机 raw RGB 帧（顺序对齐 IMAGE_NAMES）。"""
        return [self._synthetic_image(i) for i in range(len(self.IMAGE_NAMES))]

    def _synthetic_image(self, index: int) -> np.ndarray:
        """合成相机帧：RGB 渐变 + 移动方块（raw RGB，HxWx3 uint8）。"""
        w, h = self.IMAGES[self.IMAGE_NAMES[index]]
        t = self._frame * 0.04 + index * 1.7
        self._frame += 1
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for x in range(w):
            img[:, x, 0] = np.cos(t + x * 0.02) * 127 + 128  # 红色
            img[:, x, 1] = np.cos(t + 3.14 + x * 0.02) * 127 + 128  # 绿色
        cx = int((np.sin(t) * 0.5 + 0.5) * (w - 40))
        cy = int((np.cos(t * 1.3) * 0.5 + 0.5) * (h - 40))
        img[cy : cy + 40, cx : cx + 40] = (255, 200, 100)
        return img
