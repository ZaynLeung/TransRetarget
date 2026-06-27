#!/usr/bin/env python3
"""
Vision Pro 手部跟踪 → 多款灵巧手实物控制
支持手型: L20 Lite, L21 (Linker), Inspire Hand
采用多进程并行推理

sudo chmod 666 /dev/ttyUSB0
"""

import os
import sys
import multiprocessing
import time
import numpy as np
import torch
import argparse
from collections import deque

PROJECT_ROOT = '/home/ub/TransHandR'
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'yumi_gym'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'linkerhand_20_lite_test'))

from avp_stream import VisionProStreamer
from model.model_poseformer import PoseTransformer

import importlib.util
spec = importlib.util.spec_from_file_location(
    "variables_define", 
    os.path.join(PROJECT_ROOT, "config/variables_define.py")
)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)

# 从 config 模块导入需要的变量
receptive_field = config_module.receptive_field
num_joints = config_module.num_joints
scaling_factor = config_module.scaling_factor
embed_dim_ratio = config_module.embed_dim_ratio
spatial_depth = config_module.spatial_depth
temporal_depth = config_module.temporal_depth
spatial_mlp_ratio = config_module.spatial_mlp_ratio
temporal_mlp_ratio = config_module.temporal_mlp_ratio
num_heads = config_module.num_heads
qkv_bias = config_module.qkv_bias
drop_path_rate = config_module.drop_path_rate

# 导入真机SDK
REAL_ROBOT_AVAILABLE = {}
try:
    from linkerhand_python_sdk.LinkerHand.linker_hand_api import LinkerHandApi
    REAL_ROBOT_AVAILABLE['l20'] = True
    REAL_ROBOT_AVAILABLE['l21'] = True
    print("✓ LinkerHand SDK 可用")
except ImportError:
    REAL_ROBOT_AVAILABLE['l20'] = False
    REAL_ROBOT_AVAILABLE['l21'] = False
    print("✗ LinkerHand SDK 不可用")

try:
    from inspire_hand import InspireHand
    REAL_ROBOT_AVAILABLE['inspire'] = True
    print("✓ Inspire Hand SDK 可用")
except ImportError:
    REAL_ROBOT_AVAILABLE['inspire'] = False
    print("✗ Inspire Hand SDK 不可用")


# ==================== 各手型配置 ====================

class HandConfig:
    def __init__(self):
        self.name = None
        self.model_path = None
        self.out_num_joint = None
        self.angle_limit = None
        self.scaling_factor_rb = None
    
    def angles_to_control(self, angles):
        raise NotImplementedError


class L20LiteConfig(HandConfig):
    """L20 Lite 灵巧手配置"""
    def __init__(self):
        super().__init__()
        self.name = 'l20'
        self.model_path = "/home/ub/TransHandR/checkpoint/models/Optimized/l20_lite_right/model_final.pth"
        self.out_num_joint = 20
        self.scaling_factor_rb = 1.0 / 0.064
        
        self.angle_limit = [
            [0, 1.03], [0, 1.40], [0, 0.52], [0, 0.72], [0, 0.78],
            [0, 0.19], [0, 1.36], [0, 1.78], [0, 0.62],
            [0, 1.36], [0, 1.78], [0, 0.62],
            [0, 0.20], [0, 1.36], [0, 1.78], [0, 0.62],
            [0, 0.30], [0, 1.36], [0, 1.78], [0, 0.62],
        ]
        
        self.pitch_ranges = {
            'Thumb_Pitch': 0.52 + 0.72 + 0.78,
            'Index_Pitch': 1.36 + 1.78 + 0.62,
            'Middle_Pitch': 1.36 + 1.78 + 0.62,
            'Ring_Pitch': 1.36 + 1.78 + 0.62,
            'Little_Pitch': 1.36 + 1.78 + 0.62,
        }
        
        self.yaw_ranges = {
            'thumb_cmc_yaw': 1.40, 'index_mcp_roll': 0.19,
            'ring_mcp_roll': 0.20, 'pinky_mcp_roll': 0.30, 'thumb_cmc_roll': 1.03,
        }
    
    def angles_to_control(self, angles):
        if len(angles) != 20:
            raise ValueError(f"L20需要20维输入")
        
        modbus_values = [0] * 10
        thumb_pitch_sum = angles[2] + angles[3] + angles[4]
        modbus_values[0] = 255 - int(np.clip(thumb_pitch_sum / self.pitch_ranges['Thumb_Pitch'] * 255, 0, 255))
        modbus_values[1] = 255 - int(np.clip(angles[1] / self.yaw_ranges['thumb_cmc_yaw'] * 255, 0, 255))
        index_pitch_sum = angles[6] + angles[7] + angles[8]
        modbus_values[2] = 255 - int(np.clip(index_pitch_sum / self.pitch_ranges['Index_Pitch'] * 255, 0, 255))
        middle_pitch_sum = angles[9] + angles[10] + angles[11]
        modbus_values[3] = 255 - int(np.clip(middle_pitch_sum / self.pitch_ranges['Middle_Pitch'] * 255, 0, 255))
        ring_pitch_sum = angles[13] + angles[14] + angles[15]
        modbus_values[4] = 255 - int(np.clip(ring_pitch_sum / self.pitch_ranges['Ring_Pitch'] * 255, 0, 255))
        little_pitch_sum = angles[17] + angles[18] + angles[19]
        modbus_values[5] = 255 - int(np.clip(little_pitch_sum / self.pitch_ranges['Little_Pitch'] * 255, 0, 255))
        modbus_values[6] = int(np.clip(angles[5] / self.yaw_ranges['index_mcp_roll'] * 255, 0, 255))
        modbus_values[7] = int(np.clip(angles[12] / self.yaw_ranges['ring_mcp_roll'] * 255, 0, 255))
        modbus_values[8] = int(np.clip(angles[16] / self.yaw_ranges['pinky_mcp_roll'] * 255, 0, 255))
        modbus_values[9] = 255 - int(np.clip(angles[0] / self.yaw_ranges['thumb_cmc_roll'] * 255, 0, 255))
        
        return modbus_values


class L21Config(HandConfig):
    def __init__(self):
        super().__init__()
        self.name = 'l21'
        self.model_path = "/home/ub/TransHandR/checkpoint/models/Optimized/linker/model_final.pth"
        self.out_num_joint = 18
        self.scaling_factor_rb = 1.0 / 0.064
        
        self.angle_limit = [
            [0.0, 0.0], [-0.18, 0.18], [0.0, 1.57], [0.0, 1.57],
            [-0.18, 0.18], [0.0, 1.57], [0.0, 1.57], [-0.18, 0.18],
            [0.0, 1.57], [0.0, 1.57], [-0.18, 0.18], [0.0, 1.57],
            [0.0, 1.57], [-0.6, 0.6], [0.0, 1.6], [0.0, 1.0],
            [0.0, 1.57], [0.0, 1.57]
        ]
        
        self.drive_to_joint = {
            0: 15, 1: 2, 2: 5, 3: 8, 4: 11, 5: 14, 6: 1, 7: 4, 8: 7, 9: 10,
            10: 13, 11: 0, 12: 0, 13: 0, 14: 0, 15: 16, 16: 0, 17: 0, 18: 0,
            19: 0, 20: 17, 21: 3, 22: 6, 23: 9, 24: 12
        }
    
    def angles_to_control(self, angles):
        if len(angles) != 18:
            raise ValueError(f"L21需要18维输入")
        
        angle_normalized = [0] * 18
        for i in range(18):
            low, high = self.angle_limit[i]
            if high > low:
                angle_val = angles[i]
                is_roll = i in [1, 4, 7, 10]
                if not is_roll:
                    if angle_val > high * 0.95:
                        angle_val = high
                    if angle_val < low + 0.05:
                        angle_val = low
                norm = (angle_val - low) / (high - low)
                angle_normalized[i] = int(np.clip(norm * 255, 0, 255))
            else:
                angle_normalized[i] = 0
        
        modbus_values = [0] * 25
        for drive_idx, joint_idx in self.drive_to_joint.items():
            if joint_idx < 18:
                modbus_values[drive_idx] = angle_normalized[joint_idx]
        
        for i in range(25):
            if i not in [6, 7, 8, 9]:
                modbus_values[i] = 255 - modbus_values[i]
        
        modbus_values[8] = modbus_values[8] // 2
        modbus_values[9] = modbus_values[9] // 2
        
        return [max(0, min(255, v)) for v in modbus_values]


class InspireConfig(HandConfig):
    def __init__(self):
        super().__init__()
        self.name = 'inspire'
        self.model_path = "/home/ub/TransHandR/checkpoint/models/Optimized/inspire/model_final.pth"
        self.out_num_joint = 13
        self.scaling_factor_rb = 1.0 / 0.1
        
        self.angle_limit = [
            [0.0, 0.0], [0.0, 1.308], [0.0, 0.6], [0.0, 0.8], [0.0, 0.4],
            [0.0, 1.47], [-0.04545, 1.56], [0.0, 1.47], [-0.04545, 1.56],
            [0.0, 1.47], [-0.04545, 1.56], [0.0, 1.47], [-0.04545, 1.56],
        ]
        
        self.pitch_ranges = {
            'Thumb_Pitch': 0.6 + 0.8 + 0.4, 'Index_Pitch': 1.47 + 1.56,
            'Middle_Pitch': 1.47 + 1.56, 'Ring_Pitch': 1.47 + 1.56,
            'Pinky_Pitch': 1.47 + 1.56,
        }
        self.yaw_ranges = {'thumb_yaw': 1.308}
    
    def angles_to_control(self, angles):
        if len(angles) != 13:
            raise ValueError(f"Inspire需要13维输入")
        
        control_values = [0] * 6
        control_values[0] = 1000 - int(np.clip((angles[11] + angles[12]) / self.pitch_ranges['Pinky_Pitch'] * 1000, 0, 1000))
        control_values[1] = 1000 - int(np.clip((angles[9] + angles[10]) / self.pitch_ranges['Ring_Pitch'] * 1000, 0, 1000))
        control_values[2] = 1000 - int(np.clip((angles[7] + angles[8]) / self.pitch_ranges['Middle_Pitch'] * 1000, 0, 1000))
        control_values[3] = 1000 - int(np.clip((angles[5] + angles[6]) / self.pitch_ranges['Index_Pitch'] * 1000, 0, 1000))
        control_values[4] = 1000 - int(np.clip((angles[2] + angles[3] + angles[4]) / self.pitch_ranges['Thumb_Pitch'] * 1000, 0, 1000))
        control_values[5] = 1000 - int(np.clip(max(0, angles[1]) / self.yaw_ranges['thumb_yaw'] * 1000, 0, 1000))
        
        return control_values


# ==================== 模型加载 ====================
def load_model(config, device):
    model = PoseTransformer(
        num_frame=receptive_field,
        in_num_joints=num_joints,
        in_chans=3,
        out_num_joint=config.out_num_joint,
        out_chans=1,
        embed_dim_ratio=embed_dim_ratio,
        spatial_depth=spatial_depth,
        temporal_depth=temporal_depth,
        spatial_mlp_ratio=spatial_mlp_ratio,
        temporal_mlp_ratio=temporal_mlp_ratio,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        qk_scale=None,
        drop_path_rate=drop_path_rate,
        angle_limit_rad=config.angle_limit
    )
    
    print(f"[{config.name.upper()}] 加载模型: {os.path.basename(config.model_path)}")
    checkpoint = torch.load(config.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_pos'], strict=False)
    model = model.to(device)
    model.eval()
    
    return model


# ==================== 真机控制器 ====================

class L20LiteController:
    """L20 Lite 真机控制器"""
    def __init__(self, config, can_port='can0'):
        self.config = config
        self.can_port = can_port
        self.hand = None
        self.init_success = False
    
    def init(self):
        try:
            self.hand = LinkerHandApi(hand_joint="L10", hand_type="right", can=self.can_port)
            self.hand.set_speed(speed=[255] * 10)
            self.hand.set_torque(torque=[255] * 10)
            self.init_success = True
            print(f"[L20] ✓ 初始化成功 (CAN: {self.can_port})")
        except Exception as e:
            print(f"[L20] ✗ 初始化失败: {e}")
    
    def send_angles(self, angles):
        if not self.init_success or self.hand is None:
            return False
        try:
            modbus_values = self.config.angles_to_control(angles)
            self.hand.finger_move(pose=modbus_values)
            return True
        except Exception as e:
            return False
    
    def close(self):
        if self.hand:
            try:
                self.hand.close_can()
            except:
                pass


class L21Controller:
    def __init__(self, config, can_port='can0'):
        self.config = config
        self.can_port = can_port
        self.hand = None
        self.init_success = False
    
    def init(self):
        try:
            self.hand = LinkerHandApi(hand_joint="L21", hand_type="right", can=self.can_port)
            self.hand.set_speed(speed=[60, 220, 220, 220, 220])
            self.hand.set_torque(torque=[255] * 5)
            self.init_success = True
            print(f"[L21] ✓ 初始化成功 (CAN: {self.can_port})")
        except Exception as e:
            print(f"[L21] ✗ 初始化失败: {e}")
    
    def send_angles(self, angles):
        if not self.init_success or self.hand is None:
            return False
        try:
            modbus_values = self.config.angles_to_control(angles)
            self.hand.finger_move(pose=modbus_values)
            return True
        except Exception as e:
            return False
    
    def close(self):
        if self.hand:
            try:
                self.hand.close_can()
            except:
                pass

class InspireController:
    def __init__(self, config, serial_port='/dev/ttyUSB0'):
        self.config = config
        self.serial_port = serial_port
        self.hand = None
        self.init_success = False
    
    def init(self):
        try:
            self.hand = InspireHand(self.serial_port)
            self.hand.open()
            self.hand.set_all_finger_speeds(200)
            self.init_success = True
            print(f"[Inspire] ✓ 初始化成功 (串口: {self.serial_port}, 批量写入模式)")
        except Exception as e:
            print(f"[Inspire] ✗ 初始化失败: {e}")
    
    def send_angles(self, angles):
        if not self.init_success or self.hand is None:
            return False
        try:
            control_values = self.config.angles_to_control(angles)
            # 直接使用 Modbus 批量写入所有手指角度
            # Register.ANGLE_SET = 1486
            self.hand.modbus.write_multiple_registers(1486, control_values)
            return True
        except Exception as e:
            print(f"[Inspire] 发送错误: {e}")
            return False
    
    def close(self):
        if self.hand:
            try:
                self.hand.close()
            except:
                pass

# ==================== Vision Pro 数据获取进程 ====================
def vision_pro_data_process(shared_data, stop_event):
    """Vision Pro数据获取进程"""
    print("[Vision Pro] 数据获取进程启动")
    
    avp_ip = shared_data.get('avp_ip', '10.242.129.189')
    streamer = VisionProStreamer(ip=avp_ip)
    
    frame_buffer = []
    frame_count = 0
    last_log = time.time()
    
    while not stop_event.is_set():
        try:
            r = streamer.get_latest()
            fingers = r.get('right_fingers', r.get('left_fingers', None))
            
            if fingers is None:
                time.sleep(0.001)
                continue
            
            if isinstance(fingers, (np.ndarray, list)) and len(fingers) >= 25:
                coordinates = []
                for i in range(25):
                    mat = fingers[i]
                    x = -mat[1][3]
                    y = mat[2][3]
                    z = -mat[0][3]
                    coordinates.append([x, y, z])
                
                coordinates = np.array(coordinates)
                wrist_pos = coordinates[0]
                relative_coordinates = coordinates - wrist_pos
                
                frame_buffer.append(relative_coordinates)
                if len(frame_buffer) > 3:
                    frame_buffer.pop(0)
                
                if len(frame_buffer) == 3:
                    vision_data = np.stack(frame_buffer, axis=1)
                    shared_data['vision_data'] = vision_data
                    frame_count += 1
                    
                    if time.time() - last_log >= 1.0:
                        print(f"[Vision Pro] {frame_count} fps")
                        frame_count = 0
                        last_log = time.time()
            
            time.sleep(0.001)
            
        except Exception as e:
            print(f"[Vision Pro] 错误: {e}")
            time.sleep(0.1)
    
    print("[Vision Pro] 进程结束")


# ==================== 独立推理进程 ====================
def inference_process_single(shared_data, stop_event, hand_type, config):
    """单个手型的独立推理进程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{hand_type.upper()}] 推理进程启动, 设备: {device}")
    
    model = load_model(config, device)
    
    frame_count = 0
    inference_times = deque(maxlen=100)
    last_log = time.time()
    last_frame_hash = 0
    
    while not stop_event.is_set():
        vision_data = shared_data.get('vision_data', None)
        
        if vision_data is not None:
            # 简单的帧变化检测
            frame_hash = hash(vision_data.tobytes()) if isinstance(vision_data, np.ndarray) else 0
            
            if frame_hash != last_frame_hash:
                try:
                    input_data = np.transpose(vision_data, (1, 0, 2))
                    input_data = np.expand_dims(input_data, axis=0)
                    input_data *= scaling_factor
                    input_tensor = torch.from_numpy(input_data).float().to(device)
                    
                    start_time = time.time()
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                    
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
                    
                    predicted_angles = output.cpu().numpy()[0]
                    shared_data[f'angles_{hand_type}'] = predicted_angles
                    
                    frame_count += 1
                    last_frame_hash = frame_hash
                    
                    if time.time() - last_log >= 2.0:
                        avg_time = np.mean(inference_times) if inference_times else 0
                        fps = 1.0 / avg_time if avg_time > 0 else 0
                        print(f"[{hand_type.upper()}] 推理: {frame_count}帧, {avg_time*1000:.2f}ms ({fps:.1f} FPS)")
                        frame_count = 0
                        last_log = time.time()
                    
                except Exception as e:
                    print(f"[{hand_type.upper()}] 推理错误: {e}")
        
        time.sleep(0.001)
    
    print(f"[{hand_type.upper()}] 推理进程结束")


def real_robot_process(shared_data, stop_event, hand_type, config, serial_port=None):
    """真机控制进程"""
    print(f"[{hand_type.upper()}] 控制进程启动")
    
    if hand_type == 'l20':
        # L20 使用 can0
        controller = L20LiteController(config, can_port='can0')
    elif hand_type == 'l21':
        # L21 使用 can1（新插上的）
        controller = L21Controller(config, can_port='can1')
    elif hand_type == 'inspire':
        controller = InspireController(config, serial_port=serial_port or '/dev/ttyUSB0')
    else:
        print(f"[{hand_type}] 未知手型")
        return
    
    controller.init()
    
    if not controller.init_success:
        print(f"[{hand_type.upper()}] 初始化失败，退出")
        return
    
    frame_count = 0
    success_count = 0
    last_angles_hash = 0
    last_log = time.time()
    
    while not stop_event.is_set():
        angles = shared_data.get(f'angles_{hand_type}', None)
        
        if angles is not None:
            angles_hash = hash(angles.tobytes())
            if angles_hash != last_angles_hash:
                success = controller.send_angles(angles)
                frame_count += 1
                if success:
                    success_count += 1
                last_angles_hash = angles_hash
                
                if time.time() - last_log >= 5.0:
                    success_rate = success_count / frame_count * 100 if frame_count > 0 else 0
                    print(f"[{hand_type.upper()}] 已发送 {frame_count} 帧, 成功率: {success_rate:.1f}%")
                    last_log = time.time()
                    frame_count = 0
                    success_count = 0
        
        time.sleep(0.02)
    
    controller.close()
    print(f"[{hand_type.upper()}] 控制进程结束")


# ==================== 调试进程 ====================
def debug_process(shared_data, stop_event, hand_types):
    """调试进程"""
    print(f"[调试] 进程启动")
    
    frame_count = 0
    
    # 为每个手型创建临时配置用于显示
    configs = {}
    for hand_type in hand_types:
        if hand_type == 'l20':
            configs[hand_type] = L20LiteConfig()
        elif hand_type == 'l21':
            configs[hand_type] = L21Config()
        elif hand_type == 'inspire':
            configs[hand_type] = InspireConfig()
    
    while not stop_event.is_set():
        time.sleep(0.5)
        frame_count += 1
        
        for hand_type in hand_types:
            angles = shared_data.get(f'angles_{hand_type}', None)
            if angles is not None:
                print(f"\n[调试] {hand_type.upper()} 第 {frame_count} 帧")
                print(f"  角度范围: [{angles.min():.3f}, {angles.max():.3f}]")
                
                if hand_type in configs:
                    controls = configs[hand_type].angles_to_control(angles)
                    if hand_type == 'inspire':
                        print(f"  PWM (0-1000): {controls}")
                    elif hand_type == 'l20':
                        print(f"  Modbus (0-255): {controls}")
                    elif hand_type == 'l21':
                        print(f"  Modbus (0-255) 前5个: {controls[:5]}...")


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(
        description='多款灵巧手实时手部跟踪控制 (多进程并行推理)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main_visionpro_multi_hand.py --hand l20               # 只控制 L20 Lite
  python main_visionpro_multi_hand.py --hand l21               # 只控制 L21
  python main_visionpro_multi_hand.py --hand inspire           # 只控制 Inspire Hand
  python main_visionpro_multi_hand.py --hand l20 l21           # 同时控制 L20 和 L21
  python main_visionpro_multi_hand.py --hand all               # 同时控制所有手型
  python main_visionpro_multi_hand.py --hand all --debug       # 调试模式
        """
    )
    parser.add_argument('--hand', type=str, nargs='+', required=True,
                        choices=['l20', 'l21', 'inspire', 'all'],
                        help='要控制的手型')
    parser.add_argument('--avp_ip', type=str, default='10.242.129.189',
                        help='Vision Pro IP地址')
    parser.add_argument('--serial_port', type=str, default='/dev/ttyUSB0',
                        help='Inspire Hand 串口设备路径')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')
    
    args = parser.parse_args()
    
    # 处理 hand 参数 - 修改这里支持 all 包含所有三个手型
    if 'all' in args.hand:
        hand_types = ['l20', 'l21', 'inspire']  # 包含所有三个手型
    else:
        hand_types = args.hand
    
    # 只保留可用的手型
    available_hands = [h for h in hand_types if REAL_ROBOT_AVAILABLE.get(h, False)]
    
    if not available_hands:
        print("错误: 没有可用的手型")
        print("请检查 SDK 安装和硬件连接")
        return
    
    print("=" * 60)
    print("多款灵巧手实时手部跟踪控制 (多进程并行推理)")
    print(f"控制手型: {available_hands}")
    print(f"Vision Pro IP: {args.avp_ip}")
    if 'inspire' in available_hands:
        print(f"Inspire 串口: {args.serial_port}")
    print("=" * 60)
    
    # 使用 fork 而不是 spawn (Linux 下更快)
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    
    
    manager = multiprocessing.Manager()
    shared_data = manager.dict()
    shared_data['avp_ip'] = args.avp_ip
    shared_data['serial_port'] = args.serial_port
    
    stop_event = multiprocessing.Event()
    
    # 配置映射 - 添加 L20
    config_map = {
        'l20': L20LiteConfig(),
        'l21': L21Config(),
        'inspire': InspireConfig(),
    }
    
    processes = []
    
    # 1. Vision Pro 数据获取进程
    p1 = multiprocessing.Process(target=vision_pro_data_process, args=(shared_data, stop_event))
    processes.append(p1)
    
    # 2. 为每个手型创建独立的推理进程（并行）
    for hand_type in available_hands:
        if hand_type in config_map:
            p = multiprocessing.Process(
                target=inference_process_single,
                args=(shared_data, stop_event, hand_type, config_map[hand_type])
            )
            processes.append(p)
    
    # 3. 控制进程
    if not args.debug:
        for hand_type in available_hands:
            if hand_type in config_map:
                p = multiprocessing.Process(
                    target=real_robot_process,
                    args=(shared_data, stop_event, hand_type, config_map[hand_type], args.serial_port)
                )
                processes.append(p)
    else:
        p = multiprocessing.Process(
            target=debug_process, args=(shared_data, stop_event, available_hands)
        )
        processes.append(p)
    
    # 启动所有进程
    for p in processes:
        p.start()
    
    print(f"\n已启动 {len(processes)} 个进程")
    print("  - 1个 Vision Pro 数据采集进程")
    print(f"  - {len(available_hands)} 个独立推理进程 (并行)")
    print(f"  - {len(available_hands) if not args.debug else 1} 个控制/调试进程")
    print("\n按 Ctrl+C 停止程序...\n")
    
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n用户中断程序")
        stop_event.set()
        
        for p in processes:
            if p.is_alive():
                p.join(timeout=2)
                if p.is_alive():
                    p.terminate()
    
    print("程序已退出")


if __name__ == '__main__':
    main()