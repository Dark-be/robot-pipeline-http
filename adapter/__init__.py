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

"""adapter 包 —— Robot Adapter（硬件抽象层）：discover 驱动 + entry point 发现。

**发现 + 实例化一步完成**：Edge 配置不声明 adapter 身份（id / name / type），只配置
「在哪里找」（``discover`` 段 host/port）；``discover_adapter(host, port)`` 向固定端口
发 ``POST /v1/discover`` 主动寻找机器人进程（响应**只含身份** id / name / type），找到则
按 ``type`` 经 **Python entry points**（``motrix_edge.adapters``，名 = adapter 类型）
懒加载并实例化——解析出的 ``name`` / ``id`` / ``type`` 传入构造函数，能力与连接参数由
adapter **内部类常量**定义。node 只解析 host/port 并调用，不接触 ``DiscoveredRobot`` /
``get_adapter``。

本包不内置任何具体机器人实现；具体机器人由外部 SDK / 包实现 RobotAdapter 并按同一机制
注册接入（adapter 只负责把 Edge 指令转发给进程、读取进程观测）。
"""

from importlib.metadata import entry_points

import httpx

from motrix_edge.utils.data_handler import debug_print

from .base import (
    CAMERA_PREFIX,
    KEY_ACTION,
    KEY_QPOS,
    AdapterCapability,
    CaptureData,
    DiscoveredRobot,
    HealthStatus,
    RobotAdapter,
    RobotCapabilities,
)
from .http_contract import (
    FIELD_ID,
    FIELD_NAME,
    FIELD_ROBOT,
    FIELD_RUNNING,
    FIELD_TYPE,
    PATH_DISCOVER,
)

# entry point group：外部包在此组下注册适配器（名 = 适配器类型）
ADAPTER_EP_GROUP = "motrix_edge.adapters"

# discover 段缺省（固定端口；机器人进程监听此端口响应 discover）
DEFAULT_DISCOVER_HOST = "127.0.0.1"
DEFAULT_DISCOVER_PORT = 8090


def discover_adapter(
    host: str = DEFAULT_DISCOVER_HOST,
    port: int = DEFAULT_DISCOVER_PORT,
    required_capability: AdapterCapability | None = None,
) -> RobotAdapter | None:
    """向 ``host:port`` 发 ``POST /v1/discover`` 寻找机器人进程，找到则**实例化 adapter**。

    **发现 + 实例化一步完成**：进程启动后响应身份（id / name / type），按 ``type`` 经
    entry point 懒加载并实例化（能力与连接参数由 adapter 类常量定义，不随 discover
    传输）。进程不可达（未启动 / 网络错误）/ 未运行 → ``None``；实例化失败（缺 SDK /
    类型未注册）→ 打印 ERROR 后返回 ``None``，由节点持续重试等待上线（不进 ERROR）。
    """
    url = f"http://{host}:{port}"
    try:
        with httpx.Client(base_url=url, timeout=1.0) as client:
            resp = client.post(PATH_DISCOVER)
            resp.raise_for_status()
            robot = resp.json().get(FIELD_ROBOT, {})
    except Exception:  # noqa: BLE001 进程不可达 / 网络错误 → 视为未发现
        return None
    if not robot.get(FIELD_RUNNING, False):
        return None
    discovered = _robot_from_mapping(robot)
    try:
        return get_adapter(discovered, required_capability=required_capability)
    except Exception as exc:  # noqa: BLE001 缺 SDK / 类型未注册 / 实例化失败 → 持续重试
        debug_print("adapter", f"discover {discovered.id} failed: {exc}", "ERROR")
        return None


def _robot_from_mapping(robot: dict) -> DiscoveredRobot:
    """discover 响应 ``robot`` 块 → ``DiscoveredRobot``（只取身份 id / name / type）。"""
    return DiscoveredRobot(
        id=str(robot.get(FIELD_ID, "")),
        name=str(robot.get(FIELD_NAME, "")),
        type=str(robot.get(FIELD_TYPE, "")),
    )


def _entry_points() -> dict:
    """读取所有已注册的适配器 entry point：{类型名: EntryPoint}。"""
    return {ep.name: ep for ep in entry_points(group=ADAPTER_EP_GROUP)}


def get_adapter(
    discovered: DiscoveredRobot,
    required_capability: AdapterCapability | None = None,
) -> RobotAdapter:
    """工厂：按机器人进程身份的 ``type`` 懒加载并实例化 adapter（**只做实例化**）。

    **发现与实例化分离**：discover 由调用方完成（``discover_adapter`` 发 ``POST
    /v1/discover`` 返回 ``DiscoveredRobot``，只含身份）；此处按 ``discovered.type``
    （entry point 名）``load()`` 实例化，并把 discover 解析出的身份
    ``cls(name=..., id=...)`` 传入构造函数——``type`` 由 adapter 类常量 ``ADAPTER_TYPE``
    确定（实例化类名 = entry point 名），不随 discover 传入；能力与连接参数由 adapter
    **内部类常量**定义，**不接收 Edge 配置**。
    - 仅在被选中时才 import 对应类（连带其硬件 SDK），避免导入 motrix_edge 时因缺少
      未选中适配器的 SDK 而报 ModuleNotFoundError。

    required_capability: 指定所需能力（如 AdapterCapability.CAPTURE / EXECUTE），
    实例化后校验 adapter.capabilities.supports()，不支持则抛 ValueError——
    会话据此保证「采集只用采集型、推理只用执行型」适配器。
    """
    eps = _entry_points()
    if discovered.type not in eps:
        available = sorted(eps)
        raise ValueError(f"Can't find adapter type '{discovered.type}'. Available types are: {available}")

    adapter_cls = eps[discovered.type].load()  # 此刻才 import，加载该适配器及其 SDK
    adapter = adapter_cls(name=discovered.name, id=discovered.id)
    if required_capability is not None and not adapter.capabilities.supports(required_capability):
        raise ValueError(
            f"Adapter '{discovered.type}' does not support capability '{required_capability.value}'. "
            f"Supported: {[c.value for c, ok in adapter.capabilities.capabilities.items() if ok]}"
        )
    return adapter


def robot_adapters(required_capability: AdapterCapability | None = None) -> list:
    """列出已注册的机器人适配器。

    - ``required_capability=None``：列出全部（**不触发类加载**，供 adapters list / health）。
    - 给定能力：仅列出支持该能力的适配器（需加载类以读取类级 ``CAPABILITIES``，
      属选择期操作；缺失 SDK 的适配器跳过不中断）。

    返回 [(type, class_name, module), ...]，保持注册顺序。
    """
    eps = entry_points(group=ADAPTER_EP_GROUP)
    if required_capability is None:
        return [(ep.name, ep.attr, ep.module) for ep in eps]

    result = []
    for ep in eps:
        try:
            cls = ep.load()
        except Exception:  # noqa: BLE001 缺失 SDK / 导入失败 → 跳过，不中断列表
            continue
        if cls.CAPABILITIES.get(required_capability, False):
            result.append((ep.name, ep.attr, ep.module))
    return result


def adapter_details() -> list[dict]:
    """列出**所有已注册**的机器人适配器（静态，不 discover / 不探活）。

    遍历 ``motrix_edge.adapters`` entry points，无参实例化读取类常量能力；缺失 SDK /
    导入失败 → 跳过（不中断）。``available`` = 类可实例化（**非机器人进程探活结果**——
    探活职责归节点：IDLE 探测 / READY 心跳）。

    **只列 ``type`` / ``available`` / ``capabilities``**：``id`` / ``name`` 由 discover
    赋予（机器人进程身份），静态列表无 discover 故不列出。``type`` = entry point 名
    （adapter 类确定，用于实例化类名）。

    返回 ``[{type, available, capabilities}]``，供「查看全部适配器」展示（前端手动触发，
    非轮询；**与 discover 无关，SDK 未启动也应列出全部**）。
    """
    result = []
    for ep in entry_points(group=ADAPTER_EP_GROUP):
        try:
            adapter = ep.load()()  # 无参实例化（身份缺省 → 类常量回退）
        except Exception:  # noqa: BLE001 缺失 SDK / 实例化失败 → 跳过，不中断
            continue
        caps = adapter.capabilities
        result.append(
            {
                "type": ep.name,
                "available": True,
                "capabilities": {
                    "robot_model_id": caps.robot_model_id,
                    "robot_model_version": caps.robot_model_version,
                    "action_dim": caps.action_dim,
                    "observation_keys": caps.observation_keys,
                    "image_names": caps.image_names,
                    "capabilities": {c.value: ok for c, ok in caps.capabilities.items()},
                },
            }
        )
    return result


__all__ = [
    "ADAPTER_EP_GROUP",
    "CAMERA_PREFIX",
    "KEY_ACTION",
    "KEY_QPOS",
    "AdapterCapability",
    "CaptureData",
    "DiscoveredRobot",
    "HealthStatus",
    "RobotAdapter",
    "RobotCapabilities",
    "adapter_details",
    "discover_adapter",
    "get_adapter",
    "robot_adapters",
]
