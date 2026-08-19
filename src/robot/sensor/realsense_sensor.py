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

import time

import cv2
import numpy as np
import pyrealsense2 as rs

from .sensor import Sensor
from utils.base.data_handler import debug_print


def find_device_by_serial(devices, serial):
    for i, dev in enumerate(devices):
        if dev.get_info(rs.camera_info.serial_number) == serial:
            return i
    return None


class RealsenseSensor(Sensor):
    def __init__(self, name):
        super().__init__(name)
        self.enable_depth = False
        self.is_jpeg = False
        self.context = None
        self.devices = None
        self.pipeline = None
        self.config = None

    def connect(self, device, is_jpeg=False, enable_depth=False):
        self.enable_depth = enable_depth
        self.is_jpeg = is_jpeg

        self.context = rs.context()
        self.devices = list(self.context.query_devices())

        if not self.devices:
            raise RuntimeError("No RealSense devices found")

        serial = device
        device_idx = find_device_by_serial(self.devices, serial)
        if device_idx is None:
            raise RuntimeError(f"Could not find camera with serial number {serial}")

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_device(serial)
        # self.config.disable_all_streams()
        # Enable color stream only
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        if enable_depth:
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        try:
            self.pipeline.start(self.config)
            debug_print(self.name, f"Started camera: {self.name} (SN: {serial})", "INFO")
        except RuntimeError as e:
            raise RuntimeError(f"Error starting camera: {str(e)}")

    def get_information(self):
        """读取完整观测（color 恒取；depth 仅 enable_depth=True 时取），不做 collect_info 过滤。"""
        image = {}
        frame = self.pipeline.wait_for_frames()

        color_frame = frame.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to get color frame.")
        tmp_img = np.asanyarray(color_frame.get_data())
        if self.is_jpeg:
            # 不需要转换为 BGR 格式，因为 RealSense 输出的已经是 BGR
            image["color"] = cv2.imencode(".jpg", tmp_img)[1]
        else:
            image["color"] = tmp_img[:, :, ::-1]

        if self.enable_depth:
            depth_frame = frame.get_depth_frame()
            if not depth_frame:
                raise RuntimeError("Failed to get depth frame.")
            image["depth"] = np.asanyarray(depth_frame.get_data()).copy()

        return image

    def disconnect(self):
        try:
            self.pipeline.stop()
        except Exception as e:
            debug_print(self.name, f"Pipeline stop failed: {e}", "ERROR")


if __name__ == "__main__":
    cam = RealsenseSensor("test")
    cam.connect("419522071856")
    cam_list = []
    for i in range(1000):
        print(i)
        data = cam.get_information()["color"]
        cam_list.append(data)
        time.sleep(0.1)
