#!/usr/bin/env python
"""虚拟测试机器人进程服务器入口（无硬件，离线联调 Edge 侧 TestRobotAdapter）。

配置只与机器人控制有关（config/test_robot_server.yml）；服务器 host/port 用
--host/--port 或环境变量 ROBOT_SERVER_HOST/PORT（默认 0.0.0.0:8090）。

启动方式（二选一）:
  1) uv run uvicorn server.test_robot_server:app --host 0.0.0.0 --port 8090
  2) uv run python server/test_robot_server.py --host 0.0.0.0 --port 8090

接入真实 SDK：以真实机器人进程（同样 /v1 + 共享内存契约）替换本进程即可。
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
from env.test_env import TestEnv  # noqa: E402
from server.contract_server import create_app, serve  # noqa: E402
from utils.base.load_file import load_yaml  # noqa: E402

_DEFAULT_CFG = "test_robot_server.yml"


def build_app(config_name: str | None = None):
    """按机器人控制配置构造运行环境并构建 /v1 契约应用（服务器本身不读配置）。"""
    name = config_name or _DEFAULT_CFG
    cfg = load_yaml(os.path.join(CONFIG_DIR, f"{name}"))
    os.environ.setdefault("INFO_LEVEL", cfg.get("INFO_LEVEL", "INFO"))
    return create_app(TestEnv(
        robot_config=cfg.get("robot", cfg),
        capture_config=cfg.get("collector"),
    ))



# uvicorn CLI 入口：`uvicorn server.test_robot_server:app`
app = build_app()


def main():
    parser = argparse.ArgumentParser(description="Test robot process server")
    parser.add_argument("--config", default=None,
                        help="config file name under config;"
                             "default: $ROBOT_SERVER_CFG or test_robot_server")
    parser.add_argument("--host", default=None, help="override host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="override port (default 8090)")
    args = parser.parse_args()
    app_obj = build_app(args.config) if args.config else app
    serve(app_obj, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
