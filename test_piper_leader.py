#!/usr/bin/env python
"""用 PiperController 复现 pyAgxArm 官方 Leader 示例。

等价逻辑（官方示例 / 可正常运行）：
    connect(role="leader")  →  内部 set_leader_mode()
    while True:
        get_leader_joint_angles()  →  打印主臂关节角度
    Ctrl+C 退出

用法（项目根目录运行）：
    python test_piper_leader.py [can端口] [firmware]   # 默认 can_right / default
"""

import os
import sys
import time

import numpy as np

# 把 src 加入模块搜索路径（脚本位于项目根目录）
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("INFO_LEVEL", "INFO")

from robot.controller.piper_controller import PiperController  # noqa: E402


def main(port: str, firmware: str):
    controller = PiperController("right_piper_leader")
    # 与官方示例一致：role=leader（内部 set_leader_mode）、固件版本 default
    controller.connect(port=port, role="leader", firmware=firmware)
    try:
        while True:
            angles = controller.get_leader_joint_angles()
            if angles is not None:
                print("joint:", np.round(angles, 4).tolist())
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        controller.disconnect()


if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "can_right"
    firmware = sys.argv[2] if len(sys.argv) > 2 else "default"
    main(port, firmware)