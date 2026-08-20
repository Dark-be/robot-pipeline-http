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

"""env 包 —— 机器人运行环境（server 进程侧；无硬件依赖，可离线导入）。

层次：
  server (HTTP) → env (30Hz 主循环 + 共享内存发布) → robot (action/target_action/step 限速)

- ``base_env.py``：唯一运行环境（只控制 robot）。机器人实现在 ``robot`` 包
  （``src/robot/``：BaseRobot 基类 + 各机器人实现，均无 profile，obs/action 形态由类常量固定）。
- ``get_env(base_cfg)``：**按配置自动匹配**——依据 ``robot.type`` 经 robot 注册表懒加载
  实例化机器人，包进 ``BaseEnv`` 返回。新增机器人只需在 ``ROBOT_REGISTRY`` 注册 + 写配置，
  无需新增 env。
- 契约（/v1 HTTP 端点 + 观测键 + 共享内存布局）与共享内存发布都归 **robot server**
  （``server/contract_server.py``，已合并契约；env 不碰 HTTP / 共享内存）。
"""

from env.base_env import BaseEnv
from robot import get_robot


def get_env(base_cfg):
    """工厂：按配置自动构造运行环境（BaseEnv + 对应机器人）。

    依据配置 ``robot.type`` 经 ``ROBOT_REGISTRY`` 懒加载实例化机器人（见 robot.get_robot），
    并绑定采集配置。配置段：:

        robot:
          type: test_robot        # ROBOT_REGISTRY 键
          init_qpos: [...]

        collector:                # 可选
          type: act_mcap          # 当前支持 act_mcap（act_hdf5 暂不可用）
          save_dir: ./data
    """
    robot = get_robot(base_cfg)
    return BaseEnv(robot=robot, capture_config=base_cfg.get("collector"))

