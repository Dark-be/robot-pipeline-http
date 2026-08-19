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
    def __init__(self, name):
        super().__init__(name)
        self.fd = None
        self.buffers = []
        self.width = 640
        self.height = 480
        self.is_jpeg = True  # 默认使用 JPEG 格式

    def connect(self, device: str, is_jpeg=True):
        self.is_jpeg = is_jpeg

        if self.fd is not None:
            debug_print(self.name, "Already connected, disconnecting first", "WARNING")
            self.disconnect()

        self.fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)

        # 设置视频格式
        fmt = v4l2.v4l2_format()
        fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.fmt.pix.width = self.width
        fmt.fmt.pix.height = self.height
        fmt.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_MJPEG
        fmt.fmt.pix.field = v4l2.V4L2_FIELD_NONE
        fcntl.ioctl(self.fd, v4l2.VIDIOC_S_FMT, fmt)

        # 请求缓冲区
        req = v4l2.v4l2_requestbuffers()
        req.count = 4
        req.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        req.memory = v4l2.V4L2_MEMORY_MMAP
        fcntl.ioctl(self.fd, v4l2.VIDIOC_REQBUFS, req)

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

        debug_print(self.name, f"Connected to {device} with is_jpeg={self.is_jpeg}", "INFO")

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

    # 返回 ndarray，如果是 is_jpeg=True，则返回的是 JPEG bytes 的 1 维 bytes
    # 通过维度校验
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
        if self.is_jpeg:
            image["color"] = raw
        else:
            tmp_img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            image["color"] = tmp_img[:, :, ::-1]  # 转为 RGB

        return image


if __name__ == "__main__":
    cam1 = V4l2Sensor("test_v4l2_1")
    cam1.connect("/dev/video0", is_jpeg=True)
