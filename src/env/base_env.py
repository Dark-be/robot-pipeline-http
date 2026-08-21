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

"""BaseEnv —— 机器人运行环境（server 进程内，30Hz 主循环；**只控制 robot**）。

职责（不碰 HTTP / 共享内存 / 契约——那些归 robot server / contract_server）：
- **主循环**：以 ``HZ``（可修改）频率驱动 ``robot.step()``，每帧限速接近 target。
- **命令队列**：HTTP 线程（server handler）调用 ``robot_reset`` / ``robot_execute`` /
  ``robot_rollout`` / ``robot_safe_stop`` **只把指令入队**，由主循环每帧统一取出执行——
  robot 只有主循环线程一个写者，HTTP 线程不直接碰 robot 状态，避免跨线程竞态。
- **帧钩子**：``on_frame``（server 设置）每帧后回调；server 据此读取观测并发布共享内存。

层次：server (HTTP + 共享内存) → env (控制频率 / 命令队列) → robot (action/target_action/step 限速)

env 收到采集开关的请求启动采集开关，从收到采集开始，到收到采集结束，会保存其中的一段episode（mcap）
env 收到启动遥操作请求后，robot就会进入遥操作模式，robot会自行开始跟随遥操作。
"""

import queue
import threading
import time

from collector import get_collector
from utils.base.data_handler import debug_print


class BaseEnv:
    HZ = 30.0  # 运动控制频率（Hz，可修改；改小/大即调整限速节奏）

    def __init__(self, robot, capture_config: dict | None = None):
        self.robot = robot
        self._running = False
        self._thread: threading.Thread | None = None
        self.loop_alive = False  # 主循环是否存活（health 上报）
        self.last_error = None  # 主循环最近一次异常（health 上报）
        self.on_frame = None  # 每帧回调（server 设置：读取观测并发布共享内存）
        self.observation = {}  # 30Hz 保留的最新原始观测副本（其他消费方只读）
        self.commands: queue.Queue = queue.Queue()  # 命令队列（HTTP 线程入队，主循环消费）

        # 采集：capturing=True 时 step 记录观测；False 时保存为一条 episode
        # collector 类型由 capture_config 里的 `type` 决定（当前支持 act_mcap，默认 mcap）
        self.capturing = False
        self._collector = get_collector(
            capture_config or {"type": "act_mcap", "save_dir": "./data", "image_format": "jpeg"}
        )
        self._episode_idx = 0  # 下一个 episode 编号
        self._episode_open = False  # 当前是否有未关闭的 episode
        self._episodes: list[str] = []  # 已保存的 episode 文件路径

    # ---- 控制方法（HTTP 线程调用：只入队，不直接碰 robot；主循环统一执行）---------
    def _check_action_dim(self, flat_action):
        """同步校验动作维度（HTTP 线程即时反馈，不触碰 robot 状态）。"""
        dim = self.robot.action_dim()
        if len(flat_action) != dim:
            raise ValueError(f"action dim {len(flat_action)} != ACTION_DIM {dim}")

    def robot_reset(self):
        """程序复位到 home（非阻塞）：入队，由主循环执行。"""
        debug_print(self.robot.name, "HTTP 收到命令: reset（复位到 home）", "INFO")
        self.commands.put(("reset", None))

    def robot_execute(self, flat_action):
        """直接下发动作指令（raw）：入队，由主循环执行（同步校验维度）。"""
        self._check_action_dim(flat_action)
        debug_print(self.robot.name, f"HTTP 收到命令: execute dim={len(flat_action)}", "INFO")
        self.commands.put(("execute", flat_action))

    def robot_rollout(self, flat_action):
        """推理闭环：入队，由主循环执行（同步校验维度）。"""
        self._check_action_dim(flat_action)
        debug_print(self.robot.name, f"HTTP 收到命令: rollout dim={len(flat_action)}", "INFO")
        self.commands.put(("rollout", flat_action))

    def robot_safe_stop(self):
        """安全停止（幂等、失败安全）：入队，由主循环执行。"""
        debug_print(self.robot.name, "HTTP 收到命令: safe_stop（急停）", "INFO")
        self.commands.put(("safe_stop", None))

    def robot_set_teleop(self, enabled: bool):
        """设置遥操作开关（True=遥操作 / False=程控）：入队，由主循环执行。"""
        debug_print(self.robot.name, f"HTTP 收到命令: teleop enabled={bool(enabled)}", "INFO")
        self.commands.put(("teleop", bool(enabled)))

    def robot_capture_start(self):
        """开始一轮采集（episode 开始）：入队，由主循环置 capturing=True。"""
        debug_print(self.robot.name, "HTTP 收到命令: capture/start（开始采集）", "INFO")
        self.commands.put(("capture_start", None))

    def robot_capture_end(self):
        """结束一轮采集（episode 结束）：入队，由主循环置 capturing=False 并保存。"""
        debug_print(self.robot.name, "HTTP 收到命令: capture/end（结束采集）", "INFO")
        self.commands.put(("capture_end", None))

    def health(self) -> dict:
        """健康检查：就绪（硬件 + 主循环存活 + 无错误）+ 最近错误。"""
        err = self.last_error or self.robot.last_error
        return {
            "ready": self.robot.ready and self.loop_alive and err is None,
            "loop_alive": self.loop_alive,
            "last_error": err,
        }

    def data_status(self) -> dict:
        """采集数据状态：数据保存目录 + 已保存 episode 文件列表。"""
        return {
            "data_dir": str(self._collector.save_dir),
            "episodes": list(self._episodes),
        }

    def observe(self) -> dict:
        """最新原始观测副本（30Hz 由主循环保留；只读，不推进）。

        get_observation() **只应在 env 主循环中调用**（30Hz）；共享内存发布等其它消费方
        一律读本副本，避免重复触发机器人取数。
        """
        return self.observation

    # ---- 生命周期 ----------------------------------------------------------------------
    def start(self):
        """启动：连接机器人并拉起 30Hz 主循环线程（server 的 lifespan 调用）。"""
        if self._running:
            return
        self.robot.connect()
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"{self.robot.name}-env"
        )
        self._thread.start()

    def stop(self):
        """停止：退出主循环并断开机器人（共享内存写者由 server 关闭）。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.robot.disconnect()

    def _loop(self):
        """主循环：以 HZ 频率驱动 step()；单次异常不退出，记录 last_error 继续。"""
        interval = 1.0 / self.HZ
        self.loop_alive = True
        debug_print(self.robot.name, f"env loop started @ {self.HZ:.1f}Hz", "INFO")
        try:
            while self._running:
                t0 = time.perf_counter()
                try:
                    self.step()
                except Exception as exc:  # noqa: BLE001 单帧异常不拖垮循环
                    # self.last_error = exc
                    debug_print(self.robot.name, f"env step error: {exc}", "ERROR")
                dt = time.perf_counter() - t0
                sleep = interval - dt
                if sleep > 0:
                    time.sleep(sleep)
        finally:
            self.loop_alive = False

    def _drain_commands(self):
        """取出本帧所有待执行命令并统一应用到 robot（主循环线程内，单写者）。"""
        while True:
            try:
                cmd, payload = self.commands.get_nowait()
            except queue.Empty:
                break
            try:
                if cmd == "reset":
                    self.robot.reset()
                elif cmd == "execute":
                    self.robot.execute(payload)
                elif cmd == "rollout":
                    self.robot.rollout(payload)
                elif cmd == "safe_stop":
                    self.robot.safe_stop()
                elif cmd == "teleop":
                    if payload:
                        self.robot.enable_teleop()
                    else:
                        self.robot.disable_teleop()
                elif cmd == "capture_start":
                    self.capturing = True
                elif cmd == "capture_end":
                    self.capturing = False
            except Exception as exc:  # noqa: BLE001 单条命令失败不阻断其余
                self.last_error = exc
                debug_print(self.robot.name, f"command '{cmd}' failed: {exc}", "ERROR")

    def step(self):
        """每帧：先执行本帧命令（队列）→ 驱动机器人限速运动 → 30Hz 唯一调用
        get_observation() 保留副本 → 通知帧回调。

        get_observation() 只在本方法调用（30Hz）；帧回调（如共享内存发布）读取
        self.observation 副本，不重复触发机器人取数。
        """
        self._drain_commands()
        self.robot.step()
        self.observation = self.robot.get_observation().copy()  # 30Hz 保留最新原始观测副本
        if self.on_frame is not None:
            self.on_frame(self.observation)
        self._update_capture()

    def _update_capture(self):
        """按 capturing 记录观测：True→记录本帧；False→结束并保存为一条 episode。

        流式 collector（实现了 start()，如 ActMcapCollector）在 capture 开始时由 env
        调用 start(episode_idx) 打开文件，之后每帧 collect 直接落盘；缓冲式 collector
        （如缓冲式 collector）无 start，仍按 collect→finish 一次性保存。
        """
        if self.capturing:
            if not self._episode_open:
                self._episode_open = True
                start = getattr(self._collector, "start", None)
                if start is not None:
                    start()  # 不传编号：由 collector 自动续接 save_dir 下最大 episode 编号
            self._collector.collect(self.observation)
        elif self._episode_open:
            self._episode_open = False
            saved = self._collector.finish(self._episode_idx)
            if saved is not None:
                self._episodes.append(str(saved))
            self._episode_idx += 1
