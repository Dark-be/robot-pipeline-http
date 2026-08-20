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

import functools
import time

import numpy as np
from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config

from .arm_controller import ArmController
from utils.base.data_handler import debug_print

MIT_CTRL_CFG = [
    {"vel_ref": 0.0, "kp": 4.0, "kd": 0.8, "t_ref": 0.0},
    {"vel_ref": 0.0, "kp": 6.0, "kd": 1.2, "t_ref": 0.0},
    {"vel_ref": 0.0, "kp": 4.0, "kd": 0.9, "t_ref": 0.0},
    {"vel_ref": 0.0, "kp": 4.0, "kd": 0.5, "t_ref": 0.0},
    {"vel_ref": 0.0, "kp": 6.0, "kd": 0.5, "t_ref": 0.0},
    {"vel_ref": 0.0, "kp": 4.0, "kd": 0.4, "t_ref": 0.0},
]


def _require_robot(method):
    """方法装饰器：调用前统一校验 self.robot 已连接，未连接则抛出统一错误。

    避免每个方法里重复 ``if self.robot is None: raise ...``。
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.robot is None:
            raise RuntimeError(f"{self.name}: Piper is not set up. Call connect() first.")
        return method(self, *args, **kwargs)
    return wrapper


class PiperController(ArmController):
    def __init__(self, name="piper_controller"):
        super().__init__(name)
        self.robot = None
        self.gripper = None
        self.port: str = "can0"
        self.ctrl_mode: str = "mit"  # 控制模式，支持 "joint" "mit" 或 "pose"
        self.role: str = "follower"  # 角色，支持 "leader" 或 "follower"

    def connect(self, port: str = "can0", ctrl_mode: str = "mit", role: str = "follower", firmware: str = PiperFW.V188):
        self.port = port
        self.ctrl_mode = ctrl_mode

        cfg = create_agx_arm_config(robot=ArmModel.PIPER, firmeware_version=firmware, channel=port)
        self.robot = AgxArmFactory.create_arm(cfg)
        self.robot.set_joint_limits_enabled(True)

        self.gripper = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)

        self.robot.connect(start_read_thread=True)
        self.role = role

        if role == "leader":
            self.robot.set_leader_mode()
            debug_print(self.name, f"Connected to Piper on port {port} as LEADER", "INFO")
        elif role == "follower":
            self.robot.set_follower_mode()
            while not self.robot.enable():
                time.sleep(0.01)
            debug_print(self.name, f"Connected to Piper on port {port} as FOLLOWER", "INFO")
        else:
            debug_print(self.name, f"Connected to Piper on port {port}, ctrl_mode={ctrl_mode}", "INFO")

    @_require_robot
    def set_leader_mode(self):
        """切换为 Leader（主臂）模式：之后用 get_leader_joint_angles() 读取主臂关节。"""
        self.robot.set_leader_mode()
        self.role = "leader"
        debug_print(self.name, f"Piper on port {self.port} switched to leader mode", "INFO")

    @_require_robot
    def set_follower_mode(self):
        """切换为 Follower（从臂）模式：接收总线上 Leader（主臂）的角度用于跟随。"""
        self.robot.set_follower_mode()
        self.role = "follower"
        debug_print(self.name, f"Piper on port {self.port} switched to follower mode", "INFO")

    @_require_robot
    def get_leader_joint_angles(self):
        angles = self.robot.get_leader_joint_angles()
        if angles is None:
            debug_print(self.name, "Failed to get leader joint angles")
            return None
        return np.array(angles.msg)

    @_require_robot
    def move_leader_to_home(self):
        self.robot.move_leader_to_home()
        debug_print(self.name, "Leader moving to home (0 point)", "INFO")

    @_require_robot
    def restore_leader_drag_mode(self):
        self.robot.restore_leader_drag_mode()
        debug_print(self.name, "Leader restored to zero-gravity drag mode", "INFO")

    @_require_robot
    def set_speed_percent(self, percent: int = 100):
        """设置运行速度百分比（位置速度模式 move_j/move_p/move_l/move_c 生效，范围 0~100）。"""
        self.robot.set_speed_percent(percent)
        debug_print(self.name, f"Set speed percent to {percent}", "INFO")

    @_require_robot
    def get_arm_status(self):
        return self.robot.get_arm_status()

    def disconnect(self):
        if self.robot is not None:
            self.robot.disconnect()
            debug_print(self.name, f"Disconnected from Piper on port {self.port}", "INFO")
            self.robot = None

    @_require_robot
    def get_joint(self):
        if self.role == "leader":
            return self.get_leader_joint_angles()
        
        joint_angles = self.robot.get_joint_angles()
        if joint_angles is None:
            debug_print(self.name, "Failed to get joint angles")
            return None
        return np.array(joint_angles.msg)

    @_require_robot
    def get_position(self):
        flange_pose = self.robot.get_flange_pose()
        if flange_pose is None:
            debug_print(self.name, "Failed to get flange pose")
            return None
        return np.array(flange_pose.msg)

    @_require_robot
    def get_gripper(self):
        gripper_state = self.gripper.get_gripper_status()
        if gripper_state is None:
            debug_print(self.name, "Failed to get gripper status")
            return None
        return gripper_state.msg.value * 10

    @_require_robot
    def set_joint(self, joint: np.ndarray):
        if joint.shape[0] != 6:
            debug_print(self.name, "set_joint() input size should be 6", "ERROR")
        else:
            # 限制关节角度范围，防止机械臂损坏
            joint[2] = np.clip(joint[2], -2.96706, 0.0)

            if self.ctrl_mode == "mit":
                for i in range(6):
                    self.robot.move_mit(
                        joint_index=i + 1,
                        p_des=joint[i],
                        v_des=MIT_CTRL_CFG[i]["vel_ref"],
                        kp=MIT_CTRL_CFG[i]["kp"],
                        kd=MIT_CTRL_CFG[i]["kd"],
                        t_ff=MIT_CTRL_CFG[i]["t_ref"],
                    )

            elif self.ctrl_mode == "joint":
                # pose = self.robot.fk(joint.tolist())
                # self.robot.move_p(pose)
                self.robot.move_j(joint.tolist())

            debug_print(self.name, f"set joint to {joint}", "DEBUG")

    @_require_robot
    def set_pose(self, pose: np.ndarray):
        self.robot.move_p(pose.tolist())
        debug_print(self.name, f"set pose to {pose}", "DEBUG")

    @_require_robot
    def set_gripper(self, gripper: float):
        if not (1 >= gripper >= 0):
            gripper = np.clip(gripper, 0, 1)
            debug_print(self.name, f"gripper better be 0~1, but get number {gripper}", "WARNING")
        else:
            gripper_cmd = gripper / 10
            self.gripper.move_gripper_m(gripper_cmd)
            debug_print(self.name, f"set gripper to {gripper}", "DEBUG")

    @_require_robot
    def emergency_stop(self):
        self.robot.electronic_emergency_stop()


if __name__ == "__main__":
    import os

    os.environ["INFO_LEVEL"] = "DEBUG"

    controller = PiperController("robot_controller")
    controller.connect(port="can_left")

    state = controller.get_joint()
    print(f"Initial joint: {state}")

    controller.set_gripper(0.99)
    for i in range(40):
        controller.set_joint(np.array([0.036, 0.046, -0.407, -0.081, 0.471, 0.216]))
        time.sleep(0.05)  # Wait for the commands to take effect

    controller.set_joint(np.array([0.036, 0.046, -0.407, -0.081, 0.471, 0.216]))
    time.sleep(1)  # Wait for the commands to take effect
    state = controller.get_joint()
    print(f"Joint after commands: {state}")
