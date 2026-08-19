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

import cv2
import numpy as np

from .sensor import Sensor
from utils.base.data_handler import debug_print


class TestVisionSensor(Sensor):
    def __init__(self, name="test_vision_sensor"):
        super().__init__(name)
        self.timestep = 0
        self.width = 640
        self.height = 480
        self.is_jpeg = True

    def connect(self, is_jpeg=True):
        self.is_jpeg = is_jpeg

    def get_information(self):
        """读取完整观测（color），不做 collect_info 过滤。"""
        image = {}

        self.timestep += 1  # 每帧增加
        t = self.timestep * 0.05
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for x in range(self.width):
            # 开始是红色，随着时间变化，颜色在红、绿、蓝之间循环
            # 红色通道：正弦变化
            r = int(128 + 127 * np.cos(t + x * 0.02))
            # 绿色通道：余弦变化（偏移）
            g = int(128 + 127 * np.cos(t + 3.14 + x * 0.02))
            # 蓝色通道：另一个相位
            # b = int(128 + 127 * np.cos(t + 3.14 + x * 0.02))
            b = 0

            img[:, x, 0] = r
            img[:, x, 1] = g
            img[:, x, 2] = b
        if self.is_jpeg:
            # 生成随机 JPEG 图像
            img = img[:, :, [2, 1, 0]]  # 转换为 BGR 格式，cv2 使用 BGR，而不是 RGB
            _, jpeg_image = cv2.imencode(".jpg", img)
            image["color"] = jpeg_image
        else:
            image["color"] = img

        return image

    def disconnect(self):
        debug_print(self.name, "disconnect success", "INFO")
