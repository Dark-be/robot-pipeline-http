#!/usr/bin/env python
"""单臂双 Piper 机器人进程服务器入口（Leader 主臂 + Follower 从臂）。

结构与 test_robot_server / dual_piper_server 一致：SinglePiperRobot + SinglePiperEnv +
通用 /v1 契约服务器。硬件 SDK（pyAgxArm）仅在机器人端安装；配置见
config/single_piper_server.yml（端口 / 固件版本 / init_qpos）。

启动方式（二选一）:
  1) uv run uvicorn server.single_piper_server:app --host 0.0.0.0 --port 8090
  2) uv run python server/single_piper_server.py --host 0.0.0.0 --port 8090
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
from env.single_piper_env import SinglePiperEnv  # noqa: E402
from server.contract_server import create_app, serve  # noqa: E402
from utils.base.load_file import load_yaml  # noqa: E402

_DEFAULT_CFG = "single_piper_server.yml"


def build_app(config_name: str | None = None):
    """按机器人控制配置构造运行环境并构建 /v1 契约应用（服务器本身不读配置）。"""
    name = config_name or _DEFAULT_CFG
    cfg = load_yaml(os.path.join(CONFIG_DIR, f"{name}"))
    os.environ.setdefault("INFO_LEVEL", cfg.get("INFO_LEVEL", "INFO"))
    return create_app(SinglePiperEnv(
        robot_config=cfg.get("robot", cfg),
        capture_config=cfg.get("collector"),
    ))


# uvicorn CLI 入口：`uvicorn server.single_piper_server:app`
app = build_app()


def main():
    parser = argparse.ArgumentParser(description="Single piper (leader+follower) robot process server")
    parser.add_argument("--config", default=None,
                        help="config file name under config; "
                             "default: $ROBOT_SERVER_CFG or single_piper_server")
    parser.add_argument("--host", default=None, help="override host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="override port (default 8090)")
    args = parser.parse_args()
    app_obj = build_app(args.config) if args.config else app
    serve(app_obj, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
