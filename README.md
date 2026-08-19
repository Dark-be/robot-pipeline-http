# Robot Pipeline
基于Python的机器人数据采集、回放和可视化工具

## 安装
```bash
uv sync
uv pip install -e third_party/xxx

## 接入机器人
1. 在src/robot/controller中接入机械臂，在src/robot/sensor中接入传感器
2. 在src/robot中组装机器人，并设置采集数据类型
3. 在src/robot/\_\_init\_\_.py包中注册类，由yml文件中读取类名，实例化机器人

机器人具体配置在config.yml["robot"]中定义
注意：config.yml中["robot"]["type"]需要与机器人类名对应

## 数据采集

采集数据具体配置在config.yml["collect"]中定义

```bash
bash scripts/collect.sh config-name
```

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