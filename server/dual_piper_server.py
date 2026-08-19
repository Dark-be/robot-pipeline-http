#!/usr/bin/env python
"""真实双臂机器人进程服务器入口（**接入位预留，未接入硬件**）。

结构与 test_robot_server 一致：DualPiperRobot + DualPiperEnv + 通用 /v1 契约服务器。
硬件 SDK（alicia_d_sdk / pyAgxArm / pyrealsense2 / v4l2）仅在机器人端安装；当前骨架
接入后即可运行（未接入时 /v1/health 会报告主循环异常）。

启动方式（二选一）:
  1) uv run uvicorn server.dual_piper_server:app --host 0.0.0.0 --port 8090
  2) uv run python server/dual_piper_server.py --host 0.0.0.0 --port 8090
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config._GLOBAL_CONFIG import CONFIG_DIR  # noqa: E402
from env.dual_piper_env import DualPiperEnv  # noqa: E402
from server.contract_server import create_app, serve  # noqa: E402
from utils.base.load_file import load_yaml  # noqa: E402


_DEFAULT_CFG = "dual_piper_server.yml"


def build_app(config_name: str | None = None):
    """按机器人控制配置构造运行环境并构建 /v1 契约应用（服务器本身不读配置）。"""
    name = config_name or _DEFAULT_CFG
    cfg = load_yaml(os.path.join(CONFIG_DIR, f"{name}"))
    os.environ.setdefault("INFO_LEVEL", cfg.get("INFO_LEVEL", "INFO"))
    return create_app(DualPiperEnv(
        robot_config=cfg.get("robot", cfg),
        capture_config=cfg.get("collector"),
    ))


# uvicorn CLI 入口：`uvicorn pipeline.dual_piper_server:app`
app = build_app()


def main():
    parser = argparse.ArgumentParser(description="Dual piper robot process server")
    parser.add_argument("--config", default=None,
                        help="config file name under config; "
                             "default: $ROBOT_SERVER_CFG or dual_piper_server")
    parser.add_argument("--host", default=None, help="override host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="override port (default 8090)")
    args = parser.parse_args()
    app_obj = build_app(args.config) if args.config else app
    serve(app_obj, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
