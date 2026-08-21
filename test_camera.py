#!/usr/bin/env python3
"""V4l2Sensor raw 帧读取复现 demo。

专门用于复现腕相机（V4L2, pixel_format="raw" → 底层 YUYV）帧读取的问题：
  - **帧过短**：raw.size < H*W*2，sensor 内部丢弃（get_information() 返回 None，快速返回）；
  - **超时**：select 0.2s 内无帧（返回 None，~0.2s）；
  - **卡帧 / 花屏**：连续帧完全相同（冻结）或全黑 / JPEG 解码失败（jpg 模式）。

用时间差区分两种 None：
  - 快速 None（< 0.15s）→ 帧过短被丢弃；
  - 慢速 None（≈ 0.2s）→ select 超时。

用法：
    uv run python test_camera.py                                   # 默认 /dev/left-camera, raw, 300 帧
    uv run python test_camera.py --device /dev/right-camera
    uv run python test_camera.py --pixel-format raw --frames 500 --hz 30
    uv run python test_camera.py --pixel-format jpg                # 顺带对比 JPEG 坏帧率
"""

import argparse
import sys
import time
from pathlib import Path

# 使 src 布局可导入（import robot / import utils）
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from robot.sensor.v4l2_sensor import V4l2Sensor  # noqa: E402


def _fourcc_str(code: int) -> str:
    """V4L2 四字符码 int → 可读字符串（如 0x56595559 → 'YUYV'）。"""
    return "".join(chr((code >> s) & 0xFF) for s in (0, 8, 16, 24))


def parse_args():
    p = argparse.ArgumentParser(description="V4l2Sensor raw 帧读取复现 demo")
    p.add_argument("--device", default="/dev/left-camera", help="V4L2 设备节点（默认 /dev/left-camera）")
    p.add_argument("--pixel-format", default="raw", choices=["raw", "jpg"], help="sensor 输出格式（默认 raw）")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--frames", type=int, default=300, help="读取帧数")
    p.add_argument("--hz", type=float, default=30, help="读取节奏（模拟 env 30Hz；0 = 尽快读）")
    return p.parse_args()


def main():
    args = parse_args()

    expected_size = args.width * args.height * 2  # YUYV 每像素 2 字节
    print(f"[demo] connect {args.device} pixel_format={args.pixel_format} "
          f"({args.width}x{args.height}), frames={args.frames}, hz={args.hz or 'max'}")
    print(f"[demo] expected YUYV frame size = {expected_size} bytes")

    cam = V4l2Sensor("test_raw")
    cam.width = args.width
    cam.height = args.height
    try:
        cam.connect(device=args.device, pixel_format=args.pixel_format)
    except Exception as e:  # noqa: BLE001 复现工具：连接失败直接给出可读信息
        print(f"[demo][FATAL] connect 失败: {e}")
        return 1
    print(f"[demo] negotiated V4L2 format = {_fourcc_str(cam._v4l2_pixel_format())} "
          f"(pixel_format={cam.pixel_format})")

    interval = 1.0 / args.hz if args.hz > 0 else 0.0

    total = good = short_drop = timeout_drop = 0
    min_dt = 1e9
    max_dt = 0.0
    sum_dt = 0.0
    # 冻结检测：连续相同帧（采样）最大次数
    last_sample = None
    freeze_run = 0
    max_freeze = 0
    all_zero = 0
    jpeg_bad = 0
    first_shape = None

    print("[demo] start reading ... (Ctrl-C 提前结束)")
    t_start = time.perf_counter()
    try:
        for i in range(args.frames):
            loop_start = time.perf_counter()
            info = cam.get_information()
            dt = time.perf_counter() - loop_start
            total += 1
            min_dt = min(min_dt, dt)
            max_dt = max(max_dt, dt)
            sum_dt += dt

            color = info.get("color") if info else None
            if color is None:
                # 快 None → 帧过短被 sensor 丢弃；慢 None（≈0.2s）→ select 超时
                if dt >= 0.15:
                    timeout_drop += 1
                else:
                    short_drop += 1
                if i % 50 == 0:
                    print(f"  [{i}] None: short_drop={short_drop}, timeout={timeout_drop}")
                continue

            good += 1
            arr = np.asarray(color)
            if first_shape is None:
                first_shape = arr.shape
            if arr.size > 0 and np.count_nonzero(arr) == 0:
                all_zero += 1

            if arr.ndim == 3:  # raw → RGB (H, W, 3)
                sample = arr[::8, ::8].tobytes()  # 采样网格做冻结检测，避免全图哈希
            else:  # jpg → 1 维 JPEG bytes，顺带统计解码失败
                sample = arr[:4096].tobytes()
                if cv2.imdecode(arr, cv2.IMREAD_COLOR) is None:
                    jpeg_bad += 1

            if sample == last_sample:
                freeze_run += 1
            else:
                freeze_run = 0
                last_sample = sample
            max_freeze = max(max_freeze, freeze_run)

            if i % 50 == 0:
                print(f"  [{i}] ok shape={arr.shape} dt={dt*1000:.1f}ms "
                      f"(short_drop={short_drop}, timeout={timeout_drop})")

            if interval > 0:
                sleep = interval - (time.perf_counter() - loop_start)
                if sleep > 0:
                    time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n[demo] interrupted by user")
    finally:
        cam.disconnect()

    # ---- 汇总 ----
    elapsed = time.perf_counter() - t_start
    print("\n========== 汇总 ==========")
    print(f"设备            : {args.device}")
    print(f"pixel_format    : {cam.pixel_format}（底层 {_fourcc_str(cam._v4l2_pixel_format())}）")
    print(f"总帧            : {total}  耗时 {elapsed:.2f}s  "
          f"(平均 {total/elapsed:.1f} fps, 目标 {args.hz or 'max'}Hz)")
    print(f"好帧            : {good} ({good/total*100:.1f}%)  shape={first_shape}")
    print(f"丢弃帧(None)    : {total-good} ({100-good/total*100:.1f}%)")
    print(f"  ├ 帧过短丢弃   : {short_drop}（raw.size < {expected_size}）")
    print(f"  └ select 超时  : {timeout_drop}（0.2s 无帧）")
    print(f"单帧读耗时      : min {min_dt*1000:.2f}ms / max {max_dt*1000:.2f}ms / "
          f"avg {sum_dt/total*1000:.2f}ms")
    print(f"连续相同帧(冻结): 最大 {max_freeze} 帧")
    print(f"全黑帧          : {all_zero}")
    if args.pixel_format == "jpg":
        print(f"JPEG 解码失败   : {jpeg_bad}")

    print("\n判定建议（对照）：")
    if short_drop > 0:
        print("  * 存在帧过短丢弃 → 底层 YUYV 帧不完整（相机/驱动/带宽），sensor 已容错丢弃；"
              "若频繁需查 USB/相机。")
    if timeout_drop > 0:
        print("  * 存在 select 超时 → 相机产出跟不上读取节奏或流异常。")
    if max_freeze > 5:
        print("  * 连续相同帧偏多 → 疑似卡帧/画面冻结。")
    if all_zero > 0:
        print("  * 出现全黑帧 → 解码后内容为空。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
