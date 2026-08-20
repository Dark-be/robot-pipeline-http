"""MCAP-format collector（Foxglove，**ROS2 官方消息格式**）。

接收 ``standard_obs`` dict（来自 ``Robot.get_standard_obs()``），把每一条 episode
写成一个 MCAP 文件：:

    episode_{idx}.mcap

按 ROS2 官方消息格式、每个信号一个 topic（CDR 编码，mcap_ros2）：:

    topic                        schema                               内容
    ---------------------------  -----------------------------------  --------------------------
    observations/qpos            std_msgs/msg/Float64MultiArray       qpos（float64[]）
    observations/images/<cam>    sensor_msgs/msg/CompressedImage      JPEG（format='jpeg'）
    action                       std_msgs/msg/Float64MultiArray       动作（float64[]）

帧时间以 obs 内 ``timestamp``（秒）为准：作消息 ``log_time``/``publish_time``（ns），
并填入 CompressedImage 的 ``header.stamp``。可在 Foxglove 中按时间轴查看。

**流式写入**：``start()`` 打开文件 → 每次 ``collect()`` 直接落盘（不缓冲整段 episode，
内存开销只随单帧大小）→ ``finish()`` 写入 footer 并返回文件路径：:

    c = ActMcapCollector({"save_dir": "./data"})
    c.start(episode_idx=0)    # 开启 episode（打开 .mcap 文件）
    c.collect(standard_obs)   # 每帧直接落盘
    c.finish()                # -> 已写入的 .mcap 文件 Path
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from utils.base.data_handler import debug_print

# standard_obs keys (see Robot.get_standard_obs())
KEY_QPOS = "observations/qpos"
KEY_ACTION = "action"
CAMERA_PREFIX = "observations/images/"
KEY_TIMESTAMP = "timestamp"  # obs 内帧采集时刻（秒，float；由 Robot.get_observation() 提供）

# ROS2 官方消息定义（.msg 展开文本，register_msgdef 使用）
MSG_FLOAT64_MULTI_ARRAY = """\
std_msgs/MultiArrayLayout layout
float64[] data
================================================================================
MSG: std_msgs/MultiArrayLayout
MultiArrayDimension[] dim
uint32 data_offset
================================================================================
MSG: std_msgs/MultiArrayDimension
string label
uint32 size
uint32 stride
"""

MSG_COMPRESSED_IMAGE = """\
std_msgs/Header header
string format
uint8[] data
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""

# ROS2 datatype 名
TYPE_FLOAT64_MULTI_ARRAY = "std_msgs/msg/Float64MultiArray"
TYPE_COMPRESSED_IMAGE = "sensor_msgs/msg/CompressedImage"

# topic 名（每信号一个）
TOPIC_QPOS = "observations/qpos"
TOPIC_ACTION = "action"
TOPIC_IMAGE_PREFIX = "observations/images/"


class ActMcapCollector:
    def __init__(self, collector_config: dict):
        self.name = "ActMcapCollector"
        self._save_dir = Path(collector_config.get("save_dir", "./data"))
        self._save_dir.mkdir(parents=True, exist_ok=True)
        # 输入侧兼容 image_format；MCAP 输出恒为 JPEG（CompressedImage format='jpeg'）
        self._image_format = collector_config.get("image_format", "jpeg")
        # 流式状态：start() 打开 writer，collect() 直接落盘，finish() 关闭
        self._writer = None              # 当前 episode 的 mcap writer
        self._fp = None                  # 对应文件句柄
        self._schema_qpos = None         # Float64MultiArray schema
        self._schema_img = None          # CompressedImage schema
        self._path: Path | None = None   # 当前 episode 文件路径
        self._step_count = 0             # 当前 episode 已写帧数（sequence）
        debug_print(
            self.name,
            f"Initialized with save_dir={self._save_dir}, image_format={self._image_format}",
            "INFO",
        )

    # -- public API ---------------------------------------------------------
    def set_save_dir(self, path):
        self._save_dir = Path(path)

    @property
    def save_dir(self) -> Path:
        """当前数据保存目录（供 server 上报 data_status）。"""
        return self._save_dir

    def _image_to_jpeg(self, val) -> bytes:
        """把单帧图像统一成 JPEG bytes（CompressedImage format='jpeg'）。

        obs 中已是 JPEG bytes（uint8 一维）时直接使用；raw RGB 帧按 ACT/HDF5 约定
        （RGB → BGR）编码为 JPEG。
        """
        import cv2

        is_jpeg = isinstance(val, np.ndarray) and val.ndim == 1 and val.dtype == np.uint8
        if is_jpeg:
            return val.tobytes()
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(val), cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError("Failed to encode image as JPEG")
        return buf.tobytes()

    def start(self, episode_idx: int):
        """开启新一轮 episode 的 mcap 文件（流式：collect 直接落盘）。

        若上一轮未 finish，先防御性关闭（写 footer）再开新文件。
        """
        self._close_writer()
        from mcap_ros2.writer import Writer

        self._path = self._save_dir / f"episode_{episode_idx}.mcap"
        self._fp = open(self._path, "wb")
        self._writer = Writer(self._fp)
        self._schema_qpos = self._writer.register_msgdef(
            TYPE_FLOAT64_MULTI_ARRAY, MSG_FLOAT64_MULTI_ARRAY
        )
        self._schema_img = self._writer.register_msgdef(
            TYPE_COMPRESSED_IMAGE, MSG_COMPRESSED_IMAGE
        )
        self._step_count = 0
        debug_print(
            self.name,
            f"Collector start: episode {episode_idx} -> {self._path}",
            "INFO",
        )

    def collect(self, standard_obs: dict):
        """写入一帧观测（**直接落盘**，不缓冲内存）。需先 ``start()``。

        帧时间以 obs 内 ``timestamp``（秒）为准，转 ns 作 MCAP log_time / header.stamp；
        obs 缺失 timestamp 时回退本地时钟。
        """
        if self._writer is None:
            raise RuntimeError("ActMcapCollector not started; call start(episode_idx) first")

        timestamp_s = standard_obs.get(KEY_TIMESTAMP)
        if timestamp_s is None:
            timestamp_s = time.time()
        ts_ns = int(timestamp_s * 1e9)
        sec, nanosec = divmod(ts_ns, 1_000_000_000)
        qpos = np.asarray(standard_obs[KEY_QPOS], dtype=np.float64).tolist()
        action = np.asarray(standard_obs[KEY_ACTION], dtype=np.float64).tolist()
        seq = self._step_count
        self._step_count += 1

        # qpos / action：std_msgs/msg/Float64MultiArray
        for topic, data in ((TOPIC_QPOS, qpos), (TOPIC_ACTION, action)):
            self._writer.write_message(
                topic=topic,
                schema=self._schema_qpos,
                message={"layout": {"dim": [], "data_offset": 0}, "data": data},
                log_time=ts_ns,
                publish_time=ts_ns,
                sequence=seq,
            )
        # images：每相机一个 topic，sensor_msgs/msg/CompressedImage（jpeg）
        for key, val in standard_obs.items():
            if key.startswith(CAMERA_PREFIX) and val is not None:
                cam_name = key[len(CAMERA_PREFIX):]
                self._writer.write_message(
                    topic=f"{TOPIC_IMAGE_PREFIX}{cam_name}",
                    schema=self._schema_img,
                    message={
                        "header": {
                            "stamp": {"sec": sec, "nanosec": nanosec},
                            "frame_id": "",
                        },
                        "format": "jpeg",
                        "data": self._image_to_jpeg(val),
                    },
                    log_time=ts_ns,
                    publish_time=ts_ns,
                    sequence=seq,
                )

    def finish(self, episode_idx: int | None = None) -> Path | None:
        """结束当前 episode：写入 MCAP footer 并返回文件路径；未 start 过返回 None。"""
        path = self._close_writer()
        if path is not None:
            debug_print(
                self.name,
                f"Collector finish: episode finished, {self._step_count} steps -> {path}",
                "INFO",
            )
        return path

    def _close_writer(self) -> Path | None:
        """关闭当前 writer（写 footer）；无打开 writer 时返回 None。"""
        if self._writer is None:
            return None
        try:
            self._writer.finish()
        finally:
            self._writer = None
            if self._fp is not None:
                self._fp.close()
                self._fp = None
        path, self._path = self._path, None
        self._schema_qpos = None
        self._schema_img = None
        return path
