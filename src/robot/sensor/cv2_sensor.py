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


class Cv2Sensor(Sensor):
    """基于 ``cv2.VideoCapture`` 的相机 sensor，接口对齐 ``V4l2Sensor``。

    与 ``V4l2Sensor`` 一致：``connect(device, pixel_format)`` 选择 color 输出格式
    （``"jpg"`` → JPEG bytes / ``"raw"`` → RGB ndarray），底层采集格式由本 sensor
    内部用 ``CAP_PROP_FOURCC`` 尽力协商（jpg → MJPG，raw → YUYV）。

    适用场景：
    - ``cv2.VideoCapture`` 能直接打开的相机设备（路径或索引），作为 V4L2 实现的
      对照 / 备选；
    - 需要 OpenCV 内部自动做格式转换（read() 恒返回 BGR）的简易接入。

    注意：``cap.read()`` 恒返回 BGR ndarray（OpenCV 已自动转换底层格式），因此
    ``pixel_format`` 只决定 ``color`` 的输出编码，与 V4l2Sensor 语义一致。
    """

    # 输出格式（与 V4l2Sensor 对齐；消费端按 pixel_format != jpg 判定为 RGB）
    PIXEL_FORMAT_JPG = "jpg"  # color 返回 JPEG bytes（1 维 ndarray）；底层协商 MJPG
    PIXEL_FORMAT_RAW = "raw"  # color 返回 RGB ndarray（HxWx3）；底层协商 YUYV

    def __init__(self, name):
        super().__init__(name)
        self.cap = None
        self.width = 640
        self.height = 480
        self.fps = 30
        self.pixel_format = self.PIXEL_FORMAT_JPG  # 默认 jpg（压缩，color 为 JPEG bytes）

    def connect(self, device, pixel_format="jpg", width=None, height=None, fps=None):
        """连接相机（cv2.VideoCapture），选择 color 输出格式（jpg | raw）。

        - ``device``：设备路径（``"/dev/video0"``）或整数索引（``0``, ``1``, ...）；
        - ``pixel_format``：``"jpg"``/``"jpeg"``/``"mjpeg"`` → color 返回 JPEG bytes，
          底层协商 ``MJPG``；``"raw"``/``"yuyv"``/``"yuy2"``/``"rgb"`` → color 返回
          RGB ndarray，底层协商 ``YUYV``；
        - ``width``/``height``/``fps``：覆盖默认 640x480@30（底层不一定支持，失败忽略）。
        """
        pixel_format = str(pixel_format).lower()
        if pixel_format in ("jpg", "jpeg", "mjpeg"):
            self.pixel_format = self.PIXEL_FORMAT_JPG
        elif pixel_format in ("raw", "yuyv", "yuy2", "rgb"):
            self.pixel_format = self.PIXEL_FORMAT_RAW
        else:
            raise ValueError(f"Unsupported pixel_format: {pixel_format!r} (expected 'jpg' or 'raw')")

        if self.cap is not None:
            debug_print(self.name, "Already connected, disconnecting first", "WARNING")
            self.disconnect()

        if width is not None:
            self.width = int(width)
        if height is not None:
            self.height = int(height)
        if fps is not None:
            self.fps = int(fps)

        # 优先 V4L2 后端，失败回退 CAP_ANY
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            self.cap = None
            raise RuntimeError(f"Cv2Sensor: failed to open camera {device}")

        # 底层采集格式协商（尽力而为，部分后端忽略 FOURCC）
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, self._cv2_fourcc())
        except Exception as e:  # noqa: BLE001 后端不支持时忽略
            debug_print(self.name, f"set FOURCC failed (ignored): {e}", "WARNING")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        debug_print(self.name, f"Connected to {device} with pixel_format={self.pixel_format}", "INFO")

    def _cv2_fourcc(self) -> int:
        """底层采集四字符码（CAP_PROP_FOURCC）：raw → YUYV，jpg → MJPG。"""
        if self.pixel_format == self.PIXEL_FORMAT_RAW:
            return cv2.VideoWriter_fourcc(*"YUYV")
        return cv2.VideoWriter_fourcc(*"MJPG")

    def disconnect(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            debug_print(self.name, "Disconnected", "INFO")

    def get_information(self):
        """读取一帧：color 为 JPEG bytes（jpg）或 RGB ndarray（raw）；无帧返回 None。"""
        image = {}
        if self.cap is None or not self.cap.isOpened():
            image["color"] = None
            debug_print(self.name, "Camera not opened", "ERROR")
            return image

        ret, frame = self.cap.read()  # read() 恒返回 BGR ndarray
        if not ret or frame is None:
            image["color"] = None
            debug_print(self.name, "Failed to read frame", "WARNING")
            return image

        if self.pixel_format == self.PIXEL_FORMAT_JPG:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            image["color"] = buf if ok else None
        else:
            image["color"] = frame[:, :, ::-1]  # BGR → RGB
        return image


if __name__ == "__main__":
    cam = Cv2Sensor("test_cv2_1")
    # pixel_format: "jpg"（color 返回 JPEG bytes）或 "raw"（color 返回 RGB ndarray）
    cam.connect(0, pixel_format="raw")
