import h5py
import cv2
import os
import sys
import fnmatch
import termios
import tty
import select
import datetime
import numpy as np
from typing import *
from typing import Dict, Any, List
# from skimage.metrics import structural_similarity as ssim

from config._GLOBAL_CONFIG import LOG_PATH

def _get_log_file():
    log_dir = LOG_PATH
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_FILE = os.path.join(log_dir, f"log_{timestamp}.txt")
    return _LOG_FILE

def get_item(Dict_data: Dict, item):
    if isinstance(item, str):
        keys = item.split(".")
        data = Dict_data
        for key in keys:
            data = data[key]
    elif isinstance(item, list):
        key_item = None
        for it in item:
            now_data = get_item(Dict_data, it)
            # import pdb;pdb.set_trace()
            if key_item is None:
                key_item = now_data
            else:
                key_item = np.column_stack((key_item, now_data))
        data = key_item
    else:
        raise ValueError(f"input type is not allow!")
    return data

def hdf5_to_dict(h5obj):
    if isinstance(h5obj, h5py.Dataset):
        return h5obj[()]
    elif isinstance(h5obj, h5py.Group):
        return {k: hdf5_to_dict(v) for k, v in h5obj.items()}
    else:
        return None


def load_hdf5_as_dict(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        return hdf5_to_dict(f)
        
def hdf5_groups_to_dict(hdf5_path):
    """
    读取 HDF5 文件，返回真正的嵌套 dict
    - dict.keys() 只包含第一层
    - 子 group / dataset 保持原始层级
    """
    import h5py

    def read_group(group):
        out = {}
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                out[key] = item[()]
            elif isinstance(item, h5py.Group):
                out[key] = read_group(item)
        return out

    with h5py.File(hdf5_path, "r") as f:
        result = read_group(f)

    return result

def get_files(directory, extension):
    """使用pathlib获取所有匹配的文件"""
    file_paths = []
    for root, _, files in os.walk(directory):
            for filename in fnmatch.filter(files, extension):
                file_path = os.path.join(root, filename)
                file_paths.append(file_path)
    return file_paths

def get_array_length(data: Dict[str, Any]) -> int:
    """获取最外层np.array的长度"""
    for value in data.values():
        if isinstance(value, dict):
            return get_array_length(value)
        elif isinstance(value, np.ndarray):
            return value.shape[0]
        elif isinstance(value, list):
            return len(value)
    raise ValueError("No np.ndarray found in data.")

def split_nested_dict(data: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """提取每一帧的子结构"""
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = split_nested_dict(value, idx)
        elif isinstance(value, np.ndarray):
            result[key] = value[idx]
        elif isinstance(value, list):
            result[key] = value[idx]
        else:
            raise TypeError(f"Unsupported type: {type(value)} at key {key}")
    return result

def dict_to_list(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    length = get_array_length(data)
    return [split_nested_dict(data, i) for i in range(length)]

def debug_print(name, info, level="INFO", end="\n", flush=True):
    levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
    if level not in levels.keys():
        debug_print("DEBUG_PRINT", f"level setting error : {level}", "ERROR")
        return
    env_level = os.getenv("INFO_LEVEL", "INFO").upper()
    env_level_value = levels.get(env_level, 20)

    msg_level_value = levels.get(level.upper(), 20)

    if msg_level_value < env_level_value:
        return

    colors = {
        "DEBUG": "\033[94m",   # blue
        "INFO": "\033[92m",    # green
        "WARNING": "\033[93m", # yellow
        "ERROR": "\033[91m",   # red
        "ENDC": "\033[0m",
    }
    color = colors.get(level.upper(), "")
    endc = colors["ENDC"]
    msg = f"[{level}][{name}] {info}"
    print(f"{color}{msg}{endc}", end=end, flush=flush)

    # 写入日志文件 (INFO及以上级别)
    if msg_level_value >= 20:  # 20 is INFO
        log_file_path = _get_log_file()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        try:
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}]{msg}\n")
        except Exception as e:
            # 避免递归调用
            print(f"\033[91m[ERROR][DEBUG_PRINT] Failed to write log to file: {e}\033[0m")

KEY_DICT = {
    "START": 's', # 开始采集的按键
    "STOP": 'e', # 结束采集的按键
    "QUIT": 'q', # 退出程序的按键
    "RESET": 'r', # 重置机器人的按键
    "CONTINUE": 'c', # 继续下一步的按键
}

def read_key():
    # old_settings = termios.tcgetattr(sys.stdin)
    # try:
    #     tty.setraw(sys.stdin)
    #     ch = sys.stdin.read(1)
    #     if ch.isalpha():
    #         return ch
    #     return None
    # finally:
    #     termios.tcsetattr(sys.stdin, termios.TCSANOW, old_settings)
    while select.select([sys.stdin], [], [], 0)[0]:
        ch = sys.stdin.read(1)
        # 处理回车
        if ch == '\n' or ch == '\r':
            continue
        # 若存在有效输入，则清空输入缓冲区并返回该字符
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.read(1)
        return ch
    return None

def vis_video(data_path, picture_key, save_path=None, fps=30):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    episode = dict_to_list(hdf5_groups_to_dict(data_path))
    
    video_writer = None
    
    for idx, ep in enumerate(episode):
        img_data = ep[picture_key]["color"]
        
        if isinstance(img_data, (bytes, bytearray)) or (isinstance(img_data, np.ndarray) and img_data.ndim == 1):
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            img = img_data 
        
        # RGB -> BGR
        img = img[:,:,::-1]
        if save_path:
            if video_writer is None:
                h, w = img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 编码
                video_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
            
            video_writer.write(img)
        else:
            cv2.imshow(f"{picture_key}", img)
            cv2.waitKey(int(1000 / fps)) 

    if video_writer:
        video_writer.release()
        debug_print("vis_video", f"save video at: {save_path} .", "INFO")

# def jpeg_test(img_raw, jpeg_data):
#     jpeg_bytes = jpeg_data.rstrip(b"\0")
#     nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
#     img_dec = cv2.imdecode(nparr, 1)

#     def mse(img1, img2):
#         return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    
#     result = {}
#     result["PSNR"] = cv2.PSNR(img_raw, img_dec)
#     result["MSE"] = mse(img_raw, img_dec)
#     result["SSIM"] = ssim(img_raw, img_dec, channel_axis=-1, data_range=255)

#     return result

class DataBuffer:
    '''
    一个用于共享存储不同组件采集的数据的信息的类
    输入:
    manager: 创建的一个独立的控制器, multiprocessing::Manager
    '''
    def __init__(self, manager):
        self.manager = manager
        self.buffer = manager.dict()

    def collect(self, name, data):
        if name not in self.buffer:
            self.buffer[name] = self.manager.list()
        self.buffer[name].append(data)

    def get(self):
        return dict(self.buffer)

class KalmanFilter2D:
    """
    二阶卡尔曼滤波器（实时更新版本）
    状态向量: [位置, 速度]
    支持每次输入一个新观测值，实时输出滤波后的位置和速度
    """
    
    def __init__(self, process_noise_pos=1e-5, process_noise_vel=1e-1, 
                 measurement_noise=1e-2, dt=1.0):
        """
        初始化二阶卡尔曼滤波器
        
        参数:
            process_noise_pos: 位置的过程噪声（Q矩阵中位置分量）
            process_noise_vel: 速度的过程噪声（Q矩阵中速度分量）
            measurement_noise: 测量噪声（只测量位置）
            dt: 采样时间间隔
        """
        # 状态向量: [位置, 速度]
        self.x = np.array([0.0, 0.0])
        
        # 状态协方差矩阵
        self.P = np.eye(2) * 1.0
        
        # 状态转移矩阵: [1, dt; 0, 1]
        self.F = np.array([[1.0, dt],
                          [0.0, 1.0]])
        
        # 观测矩阵: 只观测位置 [1, 0]
        self.H = np.array([[1.0, 0.0]])
        
        # 过程噪声协方差矩阵
        q_pos = process_noise_pos
        q_vel = process_noise_vel
        self.Q = np.array([[dt**3/3 * q_pos + dt**2/2 * q_vel, 
                            dt**2/2 * q_pos + dt * q_vel],
                           [dt**2/2 * q_pos + dt * q_vel, 
                            dt * q_pos + q_vel]])
        
        # 测量噪声协方差
        self.R = np.array([[measurement_noise]])
        
        self.dt = dt
        self.is_initialized = False
    
    def init_first_measurement(self, measurement):
        """
        用第一个观测值初始化滤波器
        
        参数:
            measurement: 第一个观测值
        """
        self.x = np.array([measurement, 0.0])
        self.P = np.eye(2) * 1.0
        self.is_initialized = True
    
    def predict(self):
        """预测步骤"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0]
    
    def update(self, measurement):
        """
        更新步骤
        
        参数:
            measurement: 当前时刻的位置测量值
        
        返回:
            filtered_pos: 滤波后的位置
            filtered_vel: 滤波后的速度
        """
        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 状态更新
        y = measurement - self.H @ self.x
        self.x = self.x + K.flatten() * y
        
        # 协方差更新
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x[0], self.x[1]
    
    def process(self, measurement):
        """
        处理单个新观测值（主接口）
        
        参数:
            measurement: 新到达的观测值
        
        返回:
            filtered_pos: 滤波后的位置
            filtered_vel: 滤波后的速度
        """
        # 如果是第一个观测值，先初始化
        if not self.is_initialized:
            self.init_first_measurement(measurement)
            return self.x[0], self.x[1]
        
        # 预测 + 更新
        self.predict()
        pos, vel = self.update(measurement)
        
        return pos, vel
    
    def reset(self, initial_position=None, initial_velocity=None):
        """重置滤波器"""
        if initial_position is not None:
            self.x = np.array([initial_position, initial_velocity if initial_velocity is not None else 0.0])
        else:
            self.x = np.array([0.0, 0.0])
        self.P = np.eye(2) * 1.0
        self.is_initialized = initial_position is not None