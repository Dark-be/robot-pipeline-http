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
    # 支持的 color 输出格式（connect() 用 pixel_format 选择）
    PIXEL_FORMAT_JPG = "jpg"  # color 返回 JPEG 编码 bytes（1 维 ndarray，调用方 imdecode）
    PIXEL_FORMAT_RAW = "raw"  # color 返回原始 RGB ndarray（HxWx3）

    def __init__(self, name):
        super().__init__(name)
        self.enable_depth = False
        self.pixel_format = self.PIXEL_FORMAT_RAW  # 默认 raw（硬件 color 流恒为 BGR8）
        self.context = None
        self.devices = None
        self.pipeline = None
        self.config = None

    def connect(self, device, pixel_format="raw", enable_depth=False):
        """连接 RealSense，选择 color 输出格式（jpg | raw）。

        - ``"jpg"`` / ``"jpeg"`` / ``"mjpeg"`` → ``color`` 返回 JPEG 编码 bytes（1 维 ndarray）；
        - ``"raw"`` → ``color`` 返回原始 RGB ndarray（HxWx3）。
        硬件 color 流恒为 BGR8（rs.pipeline），``pixel_format`` 只决定输出编码。
        """
        pixel_format = str(pixel_format).lower()
        if pixel_format in ("jpg", "jpeg", "mjpeg"):
            self.pixel_format = self.PIXEL_FORMAT_JPG
        elif pixel_format in ("raw", "rgb"):
            self.pixel_format = self.PIXEL_FORMAT_RAW
        else:
            raise ValueError(f"Unsupported pixel_format: {pixel_format!r} (expected 'jpg' or 'raw')")
        self.enable_depth = enable_depth

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
        if self.pixel_format == self.PIXEL_FORMAT_JPG:
            # 不需要转换为 BGR 格式，因为 RealSense 输出的已经是 BGR
            image["color"] = cv2.imencode(".jpg", tmp_img)[1]
        else:
            image["color"] = tmp_img[:, :, ::-1]  # BGR → RGB

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
