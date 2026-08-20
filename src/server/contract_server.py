#!/usr/bin/env python
"""edge 契约 HTTP 服务器 + 契约（合并）：为任意机器人运行环境（env）提供 /v1 指令接口，
并在 server 侧**直接写入观测共享内存**。

**一个 adapter 对应一个 robot server**：观测键 / HTTP 端点 / 共享内存布局都由 adapter
（参考实现）规定；本模块**复用**这些定义（importlib 加载 adapter 下三个自包含文件：
base / http_contract / shm_contract，绕过 adapter/__init__ 的 httpx / motrix_edge 依赖），
并直接写共享内存（ObsShmWriter）。env 不碰 HTTP / 共享内存，只负责控制 robot。

env（BaseEnv 子类）只控制 robot：30Hz 主循环、状态切换、接收控制方法（robot_reset /
robot_execute ...）。本模块把 /v1 指令桥接到 env，并在每帧回调（env.on_frame）中读取
env 保留的观测副本（30Hz 由 env 调用 robot.get_observation() 产生）组装 standard_obs、
写入共享内存、缓存供 /observe 调试。

端点（前缀 /v1）:
    POST /v1/discover      自描述探活
    GET  /v1/health        {ok, detail}
    POST /v1/reset         复位到 home（非阻塞）
    POST /v1/execute       raw 动作 {action}
    POST /v1/rollout       推理动作 {action}
    POST /v1/teleop        遥操作开关 {enabled: bool}
    POST /v1/safe_stop     急停
    GET  /v1/data_status   {save_dir, data_files, running}
    POST /v1/capture/start 开始一轮采集（episode 开始）
    POST /v1/capture/end   结束一轮采集（episode 结束）

入口见 server/robot_server.py（按 config 自动匹配机器人，由 robot.type 选择虚拟/真实接入位）。
"""

from __future__ import annotations

import base64
import importlib.util
import os
import sys
from contextlib import asynccontextmanager
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config._GLOBAL_CONFIG import ROOT_DIR
from utils.base.data_handler import debug_print

# ----------------------------------------------------------------------------
# 契约（复用 adapter 包定义：观测键 / HTTP 端点 / 共享内存布局）
# ----------------------------------------------------------------------------
# adapter 目录位于项目根（不在 src 包内）；用 ROOT_DIR 定位，不依赖 __file__ 层级
_ADAPTER_DIR = Path(ROOT_DIR) / "adapter"


def _load_contract(module_name: str, file_name: str):
    """直接加载 adapter 下的自包含契约文件（绕过 adapter/__init__ 的 httpx / motrix_edge 依赖）。"""
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = _ADAPTER_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_contract("adapter.base", "base.py")
_http = _load_contract("adapter.http_contract", "http_contract.py")
_shm = _load_contract("adapter.shm_contract", "shm_contract.py")

# 观测键（adapter/base.py）
KEY_QPOS = _base.KEY_QPOS
KEY_ACTION = _base.KEY_ACTION
CAMERA_PREFIX = _base.CAMERA_PREFIX

# HTTP 端点 / 字段 / 状态值（adapter/http_contract.py；显式导出，避免动态注入致静态分析误报）
PATH_DISCOVER = _http.PATH_DISCOVER
PATH_HEALTH = _http.PATH_HEALTH
PATH_RESET = _http.PATH_RESET
PATH_EXECUTE = _http.PATH_EXECUTE
PATH_ROLLOUT = _http.PATH_ROLLOUT
PATH_SAFE_STOP = _http.PATH_SAFE_STOP
PATH_DATA_STATUS = _http.PATH_DATA_STATUS
PATH_TELEOP = _http.PATH_TELEOP
PATH_CAPTURE_START = _http.PATH_CAPTURE_START
PATH_CAPTURE_END = _http.PATH_CAPTURE_END
FIELD_TELEOP_ENABLED = _http.FIELD_TELEOP_ENABLED
FIELD_STATUS = _http.FIELD_STATUS
FIELD_OK = _http.FIELD_OK
FIELD_DETAIL = _http.FIELD_DETAIL
FIELD_ROBOT = _http.FIELD_ROBOT
FIELD_ID = _http.FIELD_ID
FIELD_NAME = _http.FIELD_NAME
FIELD_TYPE = _http.FIELD_TYPE
FIELD_SUPPORTED_ADAPTERS = _http.FIELD_SUPPORTED_ADAPTERS
FIELD_ROBOT_MODEL_ID = _http.FIELD_ROBOT_MODEL_ID
FIELD_ROBOT_MODEL_VERSION = _http.FIELD_ROBOT_MODEL_VERSION
FIELD_ACTION_DIM = _http.FIELD_ACTION_DIM
FIELD_OBSERVATION_KEYS = _http.FIELD_OBSERVATION_KEYS
FIELD_CONTROLLERS = _http.FIELD_CONTROLLERS
FIELD_SENSORS = _http.FIELD_SENSORS
FIELD_CAPABILITIES = _http.FIELD_CAPABILITIES
FIELD_ENDPOINT = _http.FIELD_ENDPOINT
FIELD_SHM_NAME = _http.FIELD_SHM_NAME
FIELD_RUNNING = _http.FIELD_RUNNING
FIELD_SAVE_DIR = _http.FIELD_SAVE_DIR
FIELD_DATA_FILES = _http.FIELD_DATA_FILES
VALUE_STATUS_ACCEPTED = _http.VALUE_STATUS_ACCEPTED

# 共享内存观测写者（adapter/shm_contract.py）
ObsShmWriter = _shm.ObsShmWriter

_DEFAULT_HOST = "0.0.0.0"  # 对齐 Edge 侧 adapter.SDK_HOST
_DEFAULT_PORT = 8090  # 对齐 Edge 侧 adapter.SDK_URL


class ActionRequest(BaseModel):
    action: list[float] = Field(..., description="动作 [左6关节, 左夹爪, 右6关节, 右夹爪]")


class TeleopRequest(BaseModel):
    enabled: bool = Field(..., description="是否启用遥操作（true=遥操作 / false=程控）")


class _ShmPublisher:
    """server 侧观测发布器：组装 standard_obs 并写入共享内存（env 不碰共享内存）。

    env 每帧回调（on_frame）把 30Hz 保留的观测副本传入 publish()；本类按 adapter 契约
    组装 standard_obs、写入 ObsShmWriter，并缓存供 /observe 调试。
    """

    def __init__(self, robot):
        self.robot = robot
        self._writer: ObsShmWriter | None = None
        self.last_obs: dict = {}  # 最新 standard_obs（/observe 调试用）

    def publish(self, obs):
        """发布一帧观测（obs 来自 env 30Hz 保留的副本，键已按契约：observations/qpos + images/<cam>）。"""
        if obs is None or obs.get(KEY_QPOS) is None:
            return
        standard = {
            KEY_QPOS: obs[KEY_QPOS],
            KEY_ACTION: obs.get("action") if obs.get("action") is not None else obs[KEY_QPOS].copy(),
            "seq": getattr(self.robot, "seq", 0),
        }
        for name in self.robot.IMAGE_NAMES:
            standard[f"{CAMERA_PREFIX}{name}"] = obs[f"{CAMERA_PREFIX}{name}"]
        self.last_obs = standard
        if self._writer is None:
            self._writer = self._create_writer()
        self._writer.write(
            qpos=standard[KEY_QPOS],
            images=[standard[f"{CAMERA_PREFIX}{n}"] for n in self.robot.IMAGE_NAMES],
        )

    def _create_writer(self) -> ObsShmWriter:
        # 相机尺寸（假设各相机一致）；无相机机器人（IMAGES 为空）用占位尺寸，image_count=0 无图像数据
        image_size = next(iter(self.robot.IMAGES.values()), (640, 480))
        try:
            writer = ObsShmWriter(
                name=self.robot.SHM_NAME,
                image_count=len(self.robot.IMAGE_NAMES),
                image_size=image_size,
                qpos_dim=self.robot.action_dim(),
            )
        except FileExistsError:
            # 上次进程残留：attach 后 unlink 再重建（幂等清理）
            stale = shared_memory.SharedMemory(name=self.robot.SHM_NAME)
            stale.close()
            stale.unlink()
            writer = ObsShmWriter(
                name=self.robot.SHM_NAME,
                image_count=len(self.robot.IMAGE_NAMES),
                image_size=image_size,
                qpos_dim=self.robot.action_dim(),
            )
        writer.set_flags(running=True)
        return writer

    def close(self):
        """释放共享内存写者（server 停止时调用）。"""
        if self._writer is not None:
            try:
                self._writer.set_flags(running=False)
            finally:
                try:
                    self._writer.unlink()
                finally:
                    self._writer.close()
                    self._writer = None


def create_app(env, host: str | None = None, port: int | None = None) -> FastAPI:
    """为已构造的机器人运行环境构建 /v1 契约应用（服务器不读配置；主循环在 env 内）。

    ``host`` / ``port`` 用于 /v1/discover 上报 endpoint；None 时回退默认值。
    实际监听由 ``serve()``（或 uvicorn）决定，可来自配置 server 段或命令行。
    """
    host = host or _DEFAULT_HOST
    port = int(port or _DEFAULT_PORT)
    robot = env.robot
    # server 直接写共享内存：挂到 env 的每帧回调（env 不碰共享内存 / 契约）
    publisher = _ShmPublisher(robot)
    env.on_frame = publisher.publish

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        env.start()
        if not env.health().get("ready"):
            debug_print("SERVER", f"{robot.NAME} not ready — check config / SDK.", "WARNING")
        debug_print("SERVER", f"{robot.NAME} process server started (pid={os.getpid()})", "INFO")
        yield
        debug_print("SERVER", "shutting down ...", "INFO")
        env.stop()
        publisher.close()

    app = FastAPI(title=f"{robot.NAME} robot process server", version="0.1.0", lifespan=lifespan)
    app.state.robot = robot  # 供测试/中间件访问内部机器人
    app.state.env = env

    def _require_ready():
        if not robot.ready:
            raise HTTPException(status_code=503, detail="robot not ready (connect first)")

    # ---------------------------------------------------------------- 信息（调试）
    @app.get("/")
    def root():
        return {"name": f"{robot.NAME}_process_server", "action_dim": robot.action_dim(),
                "endpoints": [PATH_DISCOVER, PATH_HEALTH, PATH_RESET, PATH_EXECUTE,
                              PATH_ROLLOUT, PATH_TELEOP, PATH_SAFE_STOP, PATH_DATA_STATUS,
                              PATH_CAPTURE_START, PATH_CAPTURE_END]}

    # ---------------------------------------------------------------- 自描述探活
    @app.post(PATH_DISCOVER)
    def discover():
        """自描述探活（不初始化）：Edge discover_adapter 消费（id / name / type / running）。"""
        return {
            FIELD_STATUS: VALUE_STATUS_ACCEPTED,
            FIELD_ROBOT: {
                FIELD_ID: robot.NAME,
                FIELD_NAME: robot.NAME,
                FIELD_TYPE: robot.ADAPTER_TYPE,
                FIELD_RUNNING: True,
                FIELD_SUPPORTED_ADAPTERS: [robot.ADAPTER_TYPE],
                FIELD_ROBOT_MODEL_ID: robot.ROBOT_MODEL_ID,
                FIELD_ROBOT_MODEL_VERSION: robot.ROBOT_MODEL_VERSION,
                FIELD_ACTION_DIM: robot.action_dim(),
                FIELD_OBSERVATION_KEYS: [KEY_QPOS] + [f"{CAMERA_PREFIX}{n}" for n in robot.IMAGE_NAMES],
                FIELD_CONTROLLERS: list(robot.controllers),
                FIELD_SENSORS: list(robot.sensors),
                FIELD_CAPABILITIES: {"capture": True, "execute": True, "streaming": True},
                FIELD_ENDPOINT: f"http://{host}:{port}",
                FIELD_SHM_NAME: robot.SHM_NAME,
            },
        }

    # ---------------------------------------------------------------- 健康检查
    @app.get(PATH_HEALTH)
    def health():
        """健康检查 {ok, detail}：Edge adapter.health 消费。"""
        h = env.health()
        ok = bool(h.get("ready") and h.get("loop_alive"))
        detail = "" if ok else (h.get("last_error") or "robot not ready")
        return {FIELD_OK: ok, FIELD_DETAIL: detail}

    # ---------------------------------------------------------------- 指令
    @app.post(PATH_RESET)
    def reset():
        """程序复位到 home（非阻塞）：与 execute 一样只修改唯一目标。"""
        _require_ready()
        try:
            env.robot_reset()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e))
        return {FIELD_STATUS: VALUE_STATUS_ACCEPTED}

    def _apply_action(req: ActionRequest) -> dict:
        _require_ready()
        try:
            env.robot_execute(req.action)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return {FIELD_STATUS: VALUE_STATUS_ACCEPTED}

    @app.post(PATH_EXECUTE)
    def execute(req: ActionRequest):
        return _apply_action(req)

    @app.post(PATH_ROLLOUT)
    def rollout(req: ActionRequest):
        # 当前与 execute 一致：都只修改唯一目标，主循环限速跟踪
        return _apply_action(req)

    @app.post(PATH_TELEOP)
    def teleop(req: TeleopRequest):
        """设置遥操作开关（true=遥操作 / false=程控）。"""
        _require_ready()
        env.robot_set_teleop(req.enabled)
        return {FIELD_STATUS: VALUE_STATUS_ACCEPTED}

    @app.post(PATH_SAFE_STOP)
    def safe_stop():
        env.robot_safe_stop()
        return {FIELD_STATUS: VALUE_STATUS_ACCEPTED}

    # ---------------------------------------------------------------- 采集（episode）
    @app.post(PATH_CAPTURE_START)
    def capture_start():
        """开始一轮采集（episode 开始）：env 置 capturing=True，主循环记录观测。"""
        _require_ready()
        env.robot_capture_start()
        return {FIELD_STATUS: VALUE_STATUS_ACCEPTED}

    @app.post(PATH_CAPTURE_END)
    def capture_end():
        """结束一轮采集（episode 结束）：env 置 capturing=False，保存为一条 episode。"""
        _require_ready()
        env.robot_capture_end()
        return {FIELD_STATUS: VALUE_STATUS_ACCEPTED}

    # ---------------------------------------------------------------- 采集数据状态
    @app.get(PATH_DATA_STATUS)
    def data_status():
        """采集数据状态 {save_dir, data_files, running}：Edge adapter.data_status 消费。"""
        ds = env.data_status()
        return {FIELD_SAVE_DIR: ds.get("data_dir"), FIELD_DATA_FILES: ds.get("episodes", []),
                FIELD_RUNNING: True}

    # ---------------------------------------------------------------- 调试（非契约）
    @app.get("/observe")
    def observe_debug():
        """调试用：最新观测 qpos + 相机 JPEG(base64)（Edge 侧实际经共享内存读观测）。"""
        obs = publisher.last_obs
        if not obs:
            return {"ready": False, "data": None}
        out = {"ready": True, "qpos": obs[KEY_QPOS].tolist(), "seq": obs.get("seq")}
        for name in robot.IMAGE_NAMES:
            rgb = obs.get(f"{CAMERA_PREFIX}{name}")
            if rgb is None:
                out[f"images/{name}"] = None
                continue
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))
            out[f"images/{name}"] = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
        return out

    return app


def serve(app_obj, host: str | None = None, port: int | None = None) -> None:
    host = host or _DEFAULT_HOST
    port = int(port or _DEFAULT_PORT)
    debug_print("SERVER", f"uvicorn: http://{host}:{port}", "INFO")
    uvicorn.run(app_obj, host=host, port=port, log_level="info")
