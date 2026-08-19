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

# execute 控制器执行动作的函数 需要子类实现
# action: Dict[str, Any] 包含控制器需要执行的动作信息的字典
# is_delta: bool 表示action中的动作信息是否为增量
from utils.base.data_handler import debug_print


class Controller:
    def __init__(self, name="controller"):
        self.name = name

    def emergency_stop(self):
        """紧急停止：立即停止控制器动作。"""
        debug_print(self.name, "Emergency stop executed.", "WARNING")

    def connect(self):
        raise NotImplementedError("This method should be implemented by the subclass")

    def disconnect(self):
        raise NotImplementedError("This method should be implemented by the subclass")

    def execute(self, action):
        raise NotImplementedError("This method should be implemented by the subclass")
