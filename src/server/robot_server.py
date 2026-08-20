#!/usr/bin/env python
"""通用机器人进程服务器入口（**按 config 自动匹配机器人**）。

单一入口取代 per-robot 的 xxx_server.py，全部由配置驱动：
- 配置文件名：``--config <name>`` 或环境变量 ``ROBOT_SERVER_CFG``
  （默认 ``test_robot_server.yml``，虚拟机器人、无硬件，最安全）。
- 机器人：由配置 ``robot.type`` 经 robot 注册表懒加载实例化（``get_robot``）。
- 运行环境：由 ``get_env(cfg)`` 自动构造（BaseEnv + 对应机器人 + 采集配置）。
- HTTP / 共享内存契约：``server.contract_server``（create_app / serve）。

新增机器人只需在 ``ROBOT_REGISTRY`` 注册 + 写一个 yml 配置，无需新增 server/env 文件。

启动方式（二选一）:
  1) ROBOT_SERVER_CFG=test_robot_server.yml uv run uvicorn server.robot_server:app --host 0.0.0.0 --port 8090
  2) uv run python src/server/robot_server.py --config test_robot_server.yml --host 0.0.0.0 --port 8090
"""

from __future__ import annotations

import argparse
import os

from config._GLOBAL_CONFIG import CONFIG_DIR  # noqa: E402
from env import get_env  # noqa: E402
from server.contract_server import create_app, serve  # noqa: E402
from utils.base.load_file import load_yaml  # noqa: E402

_DEFAULT_CFG = "test_robot_server.yml"


def build_app(config_name: str | None = None):
    """按机器人控制配置构造运行环境并构建 /v1 契约应用（服务器本身不读配置）。

    监听地址取自配置 ``server`` 段（host/port），存入 ``app.state`` 供启动时读取；
    discover 上报的 endpoint 也使用该地址。
    """
    name = config_name or _DEFAULT_CFG
    cfg = load_yaml(os.path.join(CONFIG_DIR, f"{name}"))
    os.environ.setdefault("INFO_LEVEL", cfg.get("INFO_LEVEL", "INFO"))
    server_cfg = cfg.get("server") or {}
    host = server_cfg.get("host")
    port = server_cfg.get("port")
    app = create_app(get_env(cfg), host=host, port=port)
    app.state.host = host
    app.state.port = port
    return app


# uvicorn CLI 入口：`ROBOT_SERVER_CFG=... uv run uvicorn server.robot_server:app`
app = build_app()


def main():
    parser = argparse.ArgumentParser(
        description="Robot process server (config-driven, auto-matches robot by robot.type)"
    )
    parser.add_argument("--config", default=None,
                        help="config file name under config; "
                             "default: $ROBOT_SERVER_CFG or test_robot_server")
    parser.add_argument("--host", default=None,
                        help="override host (default: config server.host or 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None,
                        help="override port (default: config server.port or 8090)")
    args = parser.parse_args()
    app_obj = build_app(args.config) if args.config else app
    host = args.host or getattr(app_obj.state, "host", None)
    port = args.port or getattr(app_obj.state, "port", None)
    serve(app_obj, host=host, port=port)


if __name__ == "__main__":
    main()
