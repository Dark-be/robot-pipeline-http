# Robot Pipeline

基于Python的机器人数据采集、回放和可视化工具

## 安装

````bash
uv sync
uv pip install -e third_party/xxx

## 接入机器人
1. 在src/robot/controller中接入机械臂，在src/robot/sensor中接入传感器
2. 在src/robot中组装机器人，并设置采集数据类型
3. 在src/robot/\_\_init\_\_.py包的ROBOT_REGISTRY中注册类（类型名 -> 模块+类名），
   配置yml中["robot"]["type"]即按此自动匹配

机器人具体配置在config.yml["robot"]中定义
注意：config.yml中["robot"]["type"]需要是ROBOT_REGISTRY中的键

## 启动机器人服务

单一入口 `src/server/robot_server.py`，**按配置自动匹配**（无需为每种机器人写 env/server）：

- 机器人：由配置 `robot.type` 经 `ROBOT_REGISTRY` 懒加载实例化（`env.get_env` 自动构造环境）
- 数据格式：由配置 `collector.type` 决定（见下节）
- 默认配置 `test_robot_server.yml`（虚拟机器人，无硬件，最安全），可用 `--config` 或
  环境变量 `ROBOT_SERVER_CFG` 指定其它配置文件

```bash
# 指定配置文件（两种方式任选其一）
uv run python src/server/robot_server.py --config dual_piper_server.yml --host 0.0.0.0 --port 8090
ROBOT_SERVER_CFG=dual_piper_server.yml uv run python src/server/robot_server.py

# uvicorn 方式（app 默认按 ROBOT_SERVER_CFG / test_robot_server.yml 构建）
ROBOT_SERVER_CFG=test_robot_server.yml uv run uvicorn server.robot_server:app --host 0.0.0.0 --port 8090
```

## 数据采集

采集数据具体配置在config.yml["collector"]中定义，`type` 字段可选：

- `act_mcap`（当前唯一/默认）：Foxglove MCAP（**ROS2 官方消息格式**，CDR 编码），每条
  episode 保存为 `episode_{idx}.mcap`；每信号一个 topic——`observations/qpos` / `action`
  用 `std_msgs/msg/Float64MultiArray`，`observations/images/<cam>` 用
  `sensor_msgs/msg/CompressedImage`（JPEG），帧时间取 obs 内 `timestamp` 作 `log_time`，
  可在 Foxglove 中按时间轴查看。**流式写入**：`start`→`collect` 直接落盘→`finish`
- `act_hdf5`：ACT 格式 HDF5（**已移除**，不再支持）

```bash
bash scripts/collect.sh config-name
````

## 数据回放

回放保存数据中的第replay_index个episode

```bash
bash scripts/replay.sh robot-config-name replay_index
```

## 策略部署

部署策略具体配置在config.yml["deploy"]中定义
目前支持模仿学习：ACT

```bash
bash scripts/deploy_act_policy.sh config-name
```

## 可视化

分为数据采集过程可视化和hdf5数据可视化

```bash
uv run pipeline/rerun_visual.py path/to/hdf5
```

Piper mit 控制参数在 controller.piper_controller 中定义
