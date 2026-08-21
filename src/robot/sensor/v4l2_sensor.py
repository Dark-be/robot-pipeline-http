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

import fcntl
import mmap
import os
import select

import cv2
import numpy as np
import v4l2

from .sensor import Sensor
from utils.base.data_handler import debug_print


class V4l2Sensor(Sensor):
    # 输出格式（connect() 用 pixel_format 选择；底层 V4L2 像素格式由本 sensor 内部协商）
    PIXEL_FORMAT_JPG = "jpg"  # color 返回原始 JPEG bytes（调用方 imdecode）；底层协商 MJPEG
    PIXEL_FORMAT_RAW = "raw"  # color 返回解码后的 RGB ndarray（HxWx3）；底层协商 YUYV 4:2:2

    def __init__(self, name):
        super().__init__(name)
        self.fd = None
        self.buffers = []
        self.width = 640
        self.height = 480
        self.pixel_format = self.PIXEL_FORMAT_JPG  # 默认 jpg（压缩，color 为 JPEG bytes）

    def connect(self, device: str, pixel_format="jpg"):
        """连接 V4L2 相机，选择 color 输出格式（jpg | raw）。

        - ``"jpg"`` / ``"jpeg"`` / ``"mjpeg"`` → 底层协商 ``V4L2_PIX_FMT_MJPEG``，
          ``color`` 返回原始 JPEG bytes（1 维 ndarray，由调用方 ``cv2.imdecode`` 解码）；
        - ``"raw"`` → 底层协商 ``V4L2_PIX_FMT_YUYV``，``color`` 返回解码后的
          RGB ndarray（HxWx3）。
        """
        pixel_format = str(pixel_format).lower()
        if pixel_format in ("jpg", "jpeg", "mjpeg"):
            self.pixel_format = self.PIXEL_FORMAT_JPG
        elif pixel_format in ("raw", "yuyv", "yuy2", "rgb"):
            self.pixel_format = self.PIXEL_FORMAT_RAW
        else:
            raise ValueError(f"Unsupported pixel_format: {pixel_format!r} (expected 'jpg' or 'raw')")

        if self.fd is not None:
            debug_print(self.name, "Already connected, disconnecting first", "WARNING")
            self.disconnect()

        self.fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)

        # 设置视频格式（按 pixel_format 选择 V4L2 像素格式）
        fmt = v4l2.v4l2_format()
        fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.fmt.pix.width = self.width
        fmt.fmt.pix.height = self.height
        fmt.fmt.pix.pixelformat = self._v4l2_pixel_format()
        fmt.fmt.pix.field = v4l2.V4L2_FIELD_NONE
        fcntl.ioctl(self.fd, v4l2.VIDIOC_S_FMT, fmt)

        # 请求缓冲区
        req = v4l2.v4l2_requestbuffers()
        req.count = 8
        req.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        req.memory = v4l2.V4L2_MEMORY_MMAP
        fcntl.ioctl(self.fd, v4l2.VIDIOC_REQBUFS, req)
        actual_count = req.count

        # 映射缓冲区并入列
        self.buffers = []
        for i in range(req.count):
            buf = v4l2.v4l2_buffer()
            buf.type = req.type
            buf.memory = v4l2.V4L2_MEMORY_MMAP
            buf.index = i
            fcntl.ioctl(self.fd, v4l2.VIDIOC_QUERYBUF, buf)

            mm = mmap.mmap(self.fd, buf.length, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, offset=buf.m.offset)
            self.buffers.append(mm)

            fcntl.ioctl(self.fd, v4l2.VIDIOC_QBUF, buf)

        buf_type = v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE)
        fcntl.ioctl(self.fd, v4l2.VIDIOC_STREAMON, buf_type)

        debug_print(self.name, f"Connected to {device} with pixel_format={self.pixel_format}", "INFO")

    def _v4l2_pixel_format(self) -> int:
        """底层 V4L2 像素格式（由本 sensor 内部协商）：raw → YUYV，jpg → MJPEG。"""
        if self.pixel_format == self.PIXEL_FORMAT_RAW:
            return v4l2.V4L2_PIX_FMT_YUYV
        return v4l2.V4L2_PIX_FMT_MJPEG

    def disconnect(self):
        if self.fd is None:
            return
        try:
            buf_type = v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE)
            fcntl.ioctl(self.fd, v4l2.VIDIOC_STREAMOFF, buf_type)
        except Exception as e:
            debug_print(self.name, f"STREAMOFF failed: {e}", "ERROR")

        for mm in self.buffers:
            try:
                mm.close()
            except Exception as e:
                debug_print(self.name, f"mmap close failed: {e}", "ERROR")
        self.buffers.clear()

        try:
            os.close(self.fd)
        except Exception as e:
            debug_print(self.name, f"fd close failed: {e}", "ERROR")

        self.fd = None

    # 返回 ndarray：pixel_format=jpg → color 为 JPEG bytes 的 1 维 ndarray（调用方 imdecode）；
    # pixel_format=raw → color 为解码后的 RGB ndarray（HxWx3）。
    def get_information(self):
        image = {}

        r, _, _ = select.select([self.fd], [], [], 0.2)
        if not r:
            image["color"] = None
            debug_print(self.name, "Timeout waiting for frame", "ERROR")
            return image

        buf = v4l2.v4l2_buffer()
        buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = v4l2.V4L2_MEMORY_MMAP
        fcntl.ioctl(self.fd, v4l2.VIDIOC_DQBUF, buf)

        data = self.buffers[buf.index][: buf.bytesused]

        fcntl.ioctl(self.fd, v4l2.VIDIOC_QBUF, buf)

        raw = np.frombuffer(data, dtype=np.uint8)
        if self.pixel_format == self.PIXEL_FORMAT_JPG:
            if len(raw) > 4 and raw[0] == 0xFF and raw[1] == 0xD8:
                image["color"] = raw
            else:
                image["color"] = None
                debug_print(self.name, "Invalid JPEG data", "WARNING")
        else:  # raw → YUYV 解码为 RGB（规避相机劣质 MJPEG 编码器；帧过短则丢弃）
            expected = self.height * self.width * 2
            # print(f"[{self.name}] YUYV frame size = {raw.size} bytes, expected = {expected} bytes")
            if raw.size < expected:
                image["color"] = None
                # debug_print(self.name, "YUYV frame too short, dropped", "WARNING")
                return image
            yuyv_3d = raw[:expected].reshape((self.height, self.width, 2))
            image["color"] = cv2.cvtColor(yuyv_3d, cv2.COLOR_YUV2RGB_YUY2)

        return image


if __name__ == "__main__":
    cam1 = V4l2Sensor("test_v4l2_1")
    # pixel_format: "jpg"（color 返回 JPEG bytes）或 "raw"（color 返回 RGB ndarray）
    cam1.connect("/dev/video0", pixel_format="jpg")
