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

"""robot 包 —— 机器人实现（无 profile；obs/action 形态由各机器人类常量固定）。

通过 ROBOT_REGISTRY 注册机器人实现，由 get_robot(base_cfg) 依据配置 robot.type
选择性实例化。

机器人（BaseRobot 子类）是「会动」的一层：持有 action/target_action，由 env 主循环每帧
step() 限速推进并产出原始观测 get_observation()；与 adapter 协定的 obs/action 形态由类
常量（QPOS / IMAGE_NAMES / IMAGES）固定声明，不依赖 motrix_edge.profile。

真实硬件机器人（如 DualPiperRobot）依赖硬件 SDK（alicia_d_sdk / pyAgxArm /
pyrealsense2 / v4l2），这些 SDK 只安装在对应机器人上；注册表持有「模块路径 + 类名」，
在 get_robot() 被选中时才 import，避免缺 SDK 的机器上导入失败。
"""

import importlib

# 注册表：机器人类型名 -> (模块路径, 类名)
# 懒加载：仅当 get_robot() 选中该类型时才 import 对应模块。
ROBOT_REGISTRY = {
    "test_robot": ("robot.test_robot", "TestRobot"),  # 虚拟（无硬件，离线联调）
    "dual_piper_robot": ("robot.dual_piper_robot", "DualPiperRobot"),  # 真实双臂接入位
}


def get_robot(base_cfg):
    """工厂：从注册表按需懒加载并实例化机器人。

    配置段：
      robot:
        type: <注册表键>
        ...
    """
    robot_config = base_cfg.get("robot")
    robot_type = robot_config.get("type")

    if robot_type not in ROBOT_REGISTRY:
        available = list(ROBOT_REGISTRY.keys())
        raise ValueError(f"Can't find robot type '{robot_type}'. Available types are: {available}")

    module_path, class_name = ROBOT_REGISTRY[robot_type]
    module = importlib.import_module(module_path)  # 此刻才 import，加载该机器人及其 SDK
    robot_cls = getattr(module, class_name)

    return robot_cls(robot_config=robot_config)


def robot_adapters():
    """列出所有已注册的机器人适配器（不触发 SDK 导入）。

    返回 [(type, class_name, module_path), ...]，保持注册顺序。
    """
    return [(rtype, cls_name, module_path) for rtype, (module_path, cls_name) in ROBOT_REGISTRY.items()]
