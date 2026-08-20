#!/bin/bash
# 启动机器人进程服务器（通用入口，按 config 自动匹配机器人）。
#
# 用法:
#   bash scripts/start.sh                                  # 默认 test_robot_server.yml（虚拟机器人）
#   bash scripts/start.sh test_robot_server                # 配置名（.yml 后缀可省略）
#   bash scripts/start.sh dual_piper_server.yml --host 0.0.0.0 --port 8090
#   bash scripts/start.sh --help                           # 查看 server 参数说明（用默认配置）
#
# 说明:
#   - 配置放在 config/ 下；首个非“-”开头的参数视为配置名，其余参数透传给 server
#   - 等价命令: uv run python src/server/robot_server.py --config <name> [args...]

set -euo pipefail

# 定位项目根（本脚本位于 <root>/scripts/）
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 首个非选项参数为配置名；默认 test_robot_server.yml
CONFIG_NAME="test_robot_server.yml"
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
    CONFIG_NAME="$1"
    shift
fi

# 自动补全 .yml 后缀
case "$CONFIG_NAME" in
    *.yml) ;;
    *) CONFIG_NAME="$CONFIG_NAME.yml" ;;
esac

# 校验配置文件存在
if [ ! -f "$ROOT_DIR/config/$CONFIG_NAME" ]; then
    echo "错误：找不到配置文件 config/$CONFIG_NAME" >&2
    echo "可用配置：" >&2
    ls "$ROOT_DIR"/config/*.yml 2>/dev/null | xargs -n1 basename >&2
    exit 1
fi

echo "==> 启动机器人服务器：config=$CONFIG_NAME"
echo "==> 命令: uv run python src/server/robot_server.py --config $CONFIG_NAME $*"
exec uv run python src/server/robot_server.py --config "$CONFIG_NAME" "$@"
