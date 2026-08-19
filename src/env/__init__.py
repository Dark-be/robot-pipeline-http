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

- ``base_env.py`` / ``test_env.py`` / ``dual_piper_env.py``：运行环境（只控制 robot）。
- **机器人实现在 ``robot`` 包**（``src/robot/``：BaseRobot 基类 + TestRobot 虚拟 + DualPiperRobot
  真实接入位，均无 profile，obs/action 形态由类常量固定）；env 只负责控制频率 / 状态切换 /
  控制方法（robot_reset / robot_execute ...）。
- 契约（/v1 HTTP 端点 + 观测键 + 共享内存布局）与共享内存发布都归 **robot server**
  （``server/contract_server.py``，已合并契约；env 不碰 HTTP / 共享内存）。
"""
