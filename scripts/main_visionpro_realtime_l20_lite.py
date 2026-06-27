#!/usr/bin/env python3
"""
Vision Pro 手部跟踪 → L20 Lite 灵巧手控制
支持仿真环境和真机控制
20维仿真关节 → 10个Modbus值（0-255）

映射规则：
- 弯曲关节：三个关节角度加权求和后归一化
- 横摆关节：直接归一化
- Thumb_Roll: 需要取反
- 其他电机: 不需要取反
"""

import os
import sys
import multiprocessing
import time
import numpy as np
import torch
import argparse

# 添加项目根目录到 Python 路径
sys.path.insert(0, '/home/ub/TransHandR')
sys.path.insert(0, '/home/ub/TransHandR/yumi_gym')
sys.path.insert(0, '/home/ub/TransHandR/linkerhand_20_lite_test')

from avp_stream import VisionProStreamer
from model.model_poseformer import PoseTransformer
from config.variables_define import *

# 尝试导入仿真模块
try:
    import gym
    import pybullet as p
    import yumi_gym
    SIMULATION_AVAILABLE = True
except ImportError:
    SIMULATION_AVAILABLE = False
    print("警告: 仿真环境模块未找到")

# 尝试导入真机SDK
try:
    from linkerhand_python_sdk.LinkerHand.linker_hand_api import LinkerHandApi
    REAL_ROBOT_AVAILABLE = True
except ImportError:
    REAL_ROBOT_AVAILABLE = False
    print("警告: 真机SDK模块未找到")


# ==================== 20维仿真关节索引定义 ====================
# 根据 URDF 配置文件中的关节顺序
JOINT_INDICES = {
    # 拇指 (0-4)
    'thumb_cmc_roll': 0,      # 对应电机9 Thumb_Roll
    'thumb_cmc_yaw': 1,       # 对应电机1 Thumb_Yaw
    'thumb_cmc_pitch': 2,     # 电机0 的一部分
    'thumb_mcp': 3,           # 电机0 的一部分
    'thumb_ip': 4,            # 电机0 的一部分
    # 食指 (5-8)
    'index_mcp_roll': 5,      # 对应电机6 Index_Yaw
    'index_mcp_pitch': 6,     # 电机2 的一部分
    'index_pip': 7,           # 电机2 的一部分
    'index_dip': 8,           # 电机2 的一部分
    # 中指 (9-11)
    'middle_mcp_pitch': 9,    # 电机3 的一部分
    'middle_pip': 10,         # 电机3 的一部分
    'middle_dip': 11,         # 电机3 的一部分
    # 无名指 (12-15)
    'ring_mcp_roll': 12,      # 对应电机7 Ring_Yaw
    'ring_mcp_pitch': 13,     # 电机4 的一部分
    'ring_pip': 14,           # 电机4 的一部分
    'ring_dip': 15,           # 电机4 的一部分
    # 小指 (16-19)
    'pinky_mcp_roll': 16,     # 对应电机8 Little_Yaw
    'pinky_mcp_pitch': 17,    # 电机5 的一部分
    'pinky_pip': 18,          # 电机5 的一部分
    'pinky_dip': 19,          # 电机5 的一部分
}

# 各弯曲关节的最大角度和（弧度）
PITCH_RANGES = {
    'Thumb_Pitch': 0.52 + 0.72 + 0.78,   # = 2.02 rad
    'Index_Pitch': 1.36 + 1.78 + 0.62,   # = 3.76 rad
    'Middle_Pitch': 1.36 + 1.78 + 0.62,  # = 3.76 rad
    'Ring_Pitch': 1.36 + 1.78 + 0.62,    # = 3.76 rad
    'Little_Pitch': 1.36 + 1.78 + 0.62,  # = 3.76 rad
}

# 各横摆关节的最大角度（弧度）
YAW_RANGES = {
    'thumb_cmc_yaw': 1.40,
    'index_mcp_roll': 0.19,
    'ring_mcp_roll': 0.20,
    'pinky_mcp_roll': 0.30,
    'thumb_cmc_roll': 1.03,
}

# L20 Lite 配置
hand_brand = 'l20_lite_right'
out_num_joint = 20
scaling_factor_rb = 1.0 / 0.064


def map_20d_to_10d_modbus(angles_20d):
    """
    将20维仿真关节角度映射到10个 Modbus 值 (0-255)
    
    映射规则：
    - 弯曲关节：三个关节角度加权求和后归一化
    - 横摆关节：直接归一化
    - Thumb_Roll (电机9): 需要取反
    - 其他电机: 不需要取反
    
    参数:
        angles_20d: 长度为20的numpy数组，单位弧度
    
    返回:
        modbus_values: 长度为10的列表，每个值在0-255之间
    """
    if len(angles_20d) != 20:
        raise ValueError(f"需要20维输入，实际{len(angles_20d)}维")
    
    modbus_values = [0] * 10
    
    # ===== 电机0: Thumb_Pitch (大拇指弯曲) =====
    # 关节: thumb_cmc_pitch + thumb_mcp + thumb_ip
    thumb_pitch_sum = (
        angles_20d[JOINT_INDICES['thumb_cmc_pitch']] +
        angles_20d[JOINT_INDICES['thumb_mcp']] +
        angles_20d[JOINT_INDICES['thumb_ip']]
    )
    modbus_values[0] = 255 - int(np.clip(
        thumb_pitch_sum / PITCH_RANGES['Thumb_Pitch'] * 255, 0, 255
    ))
    
    # ===== 电机1: Thumb_Yaw (大拇指横摆) =====
    # 关节: thumb_cmc_yaw，需要取反
    modbus_values[1] = 255 - int(np.clip(
        angles_20d[JOINT_INDICES['thumb_cmc_yaw']] / YAW_RANGES['thumb_cmc_yaw'] * 255, 0, 255
    ))
    
    # ===== 电机2: Index_Pitch (食指弯曲) =====
    # 关节: index_mcp_pitch + index_pip + index_dip
    index_pitch_sum = (
        angles_20d[JOINT_INDICES['index_mcp_pitch']] +
        angles_20d[JOINT_INDICES['index_pip']] +
        angles_20d[JOINT_INDICES['index_dip']]
    )
    modbus_values[2] = 255 - int(np.clip(
        index_pitch_sum / PITCH_RANGES['Index_Pitch'] * 255, 0, 255
    ))
    
    # ===== 电机3: Middle_Pitch (中指弯曲) =====
    middle_pitch_sum = (
        angles_20d[JOINT_INDICES['middle_mcp_pitch']] +
        angles_20d[JOINT_INDICES['middle_pip']] +
        angles_20d[JOINT_INDICES['middle_dip']]
    )
    modbus_values[3] = 255 - int(np.clip(
        middle_pitch_sum / PITCH_RANGES['Middle_Pitch'] * 255, 0, 255
    ))
    
    # ===== 电机4: Ring_Pitch (无名指弯曲) =====
    ring_pitch_sum = (
        angles_20d[JOINT_INDICES['ring_mcp_pitch']] +
        angles_20d[JOINT_INDICES['ring_pip']] +
        angles_20d[JOINT_INDICES['ring_dip']]
    )
    modbus_values[4] = 255 - int(np.clip(
        ring_pitch_sum / PITCH_RANGES['Ring_Pitch'] * 255, 0, 255
    ))
    
    # ===== 电机5: Little_Pitch (小拇指弯曲) =====
    little_pitch_sum = (
        angles_20d[JOINT_INDICES['pinky_mcp_pitch']] +
        angles_20d[JOINT_INDICES['pinky_pip']] +
        angles_20d[JOINT_INDICES['pinky_dip']]
    )
    modbus_values[5] = 255 - int(np.clip(
        little_pitch_sum / PITCH_RANGES['Little_Pitch'] * 255, 0, 255
    ))
    
    # ===== 电机6: Index_Yaw (食指横摆) =====
    # 关节: index_mcp_roll，不需要取反
    modbus_values[6] = int(np.clip(
        angles_20d[JOINT_INDICES['index_mcp_roll']] / YAW_RANGES['index_mcp_roll'] * 255, 0, 255
    ))
    
    # ===== 电机7: Ring_Yaw (无名指横摆) =====
    # 关节: ring_mcp_roll，不需要取反
    modbus_values[7] = int(np.clip(
        angles_20d[JOINT_INDICES['ring_mcp_roll']] / YAW_RANGES['ring_mcp_roll'] * 255, 0, 255
    ))
    
    # ===== 电机8: Little_Yaw (小拇指横摆) =====
    # 关节: pinky_mcp_roll，不需要取反
    modbus_values[8] = int(np.clip(
        angles_20d[JOINT_INDICES['pinky_mcp_roll']] / YAW_RANGES['pinky_mcp_roll'] * 255, 0, 255
    ))
    
    # ===== 电机9: Thumb_Roll (大拇指横滚) =====
    # 关节: thumb_cmc_roll，需要取反！
    # 小值向掌心靠拢，大值远离掌心，所以取反
    thumb_roll_rad = angles_20d[JOINT_INDICES['thumb_cmc_roll']]
    max_thumb_roll = YAW_RANGES['thumb_cmc_roll']
    modbus_values[9] = 255 - int(np.clip(
        thumb_roll_rad / max_thumb_roll * 255, 0, 255
    ))
    
    return modbus_values


# ==================== 模型加载 ====================
def load_model(model_path, device):
    """加载预训练模型"""
    model = PoseTransformer(
        num_frame=receptive_field,
        in_num_joints=num_joints,
        in_chans=3,
        out_num_joint=out_num_joint,
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
        angle_limit_rad=angle_limit_rob
    )
    
    print(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_pos'], strict=False)
    model = model.to(device)
    model.eval()
    
    print("模型加载成功！")
    return model


# ==================== Vision Pro 数据获取进程 ====================
def vision_pro_data_process(shared_dict, stop_event):
    """Vision Pro数据获取进程 - 使用右手数据"""
    print("Vision Pro数据获取进程启动 (右手)")
    
    avp_ip = shared_dict.get('avp_ip', '10.242.129.189')
    print(f"连接 Vision Pro: {avp_ip}")
    streamer = VisionProStreamer(ip=avp_ip)
    
    frame_buffer = []
    frame_count = 0
    
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
                    shared_dict['vision_data'] = vision_data
                    frame_count += 1
                    if frame_count % 100 == 0:
                        print(f"Vision Pro: 已获取 {frame_count} 帧")
            
            time.sleep(0.001)
            
        except Exception as e:
            print(f"Vision Pro 错误: {e}")
            time.sleep(0.1)
    
    print("Vision Pro数据获取进程结束")


# ==================== 模型推理进程 ====================
def inference_process(shared_dict, stop_event):
    """模型推理进程"""
    print("推理进程启动")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model_path = "/home/ub/TransHandR/checkpoint/models/Optimized/l20_lite_right/model_final.pth"
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 - {model_path}")
        stop_event.set()
        return
    
    model = load_model(model_path, device)
    
    frame_count = 0
    inference_times = []
    
    while not stop_event.is_set():
        vision_data = shared_dict.get('vision_data', None)
        
        if vision_data is not None:
            try:
                input_data = np.transpose(vision_data, (1, 0, 2))
                input_data = np.expand_dims(input_data, axis=0)
                input_data *= scaling_factor
                
                input_tensor = torch.from_numpy(input_data).float().to(device)
                
                start_time = time.time()
                with torch.no_grad():
                    output = model(input_tensor)
                inference_time = time.time() - start_time
                
                predicted_angles = output.cpu().numpy()[0]
                shared_dict['robot_angles'] = predicted_angles
                
                inference_times.append(inference_time)
                if len(inference_times) > 100:
                    inference_times.pop(0)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    avg_time = np.mean(inference_times)
                    print(f"推理: {frame_count} 帧, {avg_time*1000:.2f}ms ({1/avg_time:.1f} FPS)")
                
            except Exception as e:
                print(f"推理错误: {e}")
        
        time.sleep(0.001)
    
    print("推理进程结束")


# ==================== 仿真环境进程 ====================
def simulation_process(shared_dict, stop_event):
    """仿真环境进程 - 接收20维关节角度"""
    print("仿真进程启动")
    
    if not SIMULATION_AVAILABLE:
        print("错误: 仿真环境不可用")
        return
    
    env = gym.make('yumi-v0')
    env.reset()
    
    camera_distance = 1.0
    camera_yaw = 90
    camera_pitch = -20
    camera_target_position = [0, 0, 0.1]
    paused = False
    frame_count = 0
    
    print("=" * 50)
    print("仿真控制说明:")
    print("  w/s : 缩放相机距离")
    print("  a/d : 旋转相机左右")
    print("  q/e : 旋转相机上下")
    print("  空格: 暂停/继续")
    print("  ESC : 退出程序")
    print("=" * 50)
    
    while not stop_event.is_set():
        try:
            env.render()
            
            keys = p.getKeyboardEvents()
            for k, v in keys.items():
                if v & p.KEY_WAS_TRIGGERED:
                    if k == ord('w'):
                        camera_distance -= 0.3
                    elif k == ord('s'):
                        camera_distance += 0.3
                    elif k == ord('a'):
                        camera_yaw -= 10
                    elif k == ord('d'):
                        camera_yaw += 10
                    elif k == ord('q'):
                        camera_pitch -= 10
                    elif k == ord('e'):
                        camera_pitch += 10
                    elif k == ord(' '):
                        paused = not paused
                        print(f"{'暂停' if paused else '继续'}")
                    elif k == 27:
                        print("ESC 按下，退出程序")
                        stop_event.set()
                        return
            
            if paused:
                time.sleep(0.02)
                continue
            
            robot_angles = shared_dict.get('robot_angles', None)
            
            if robot_angles is not None:
                action = robot_angles.tolist() if isinstance(robot_angles, np.ndarray) else robot_angles
                
                p.resetDebugVisualizerCamera(
                    cameraDistance=camera_distance,
                    cameraYaw=camera_yaw,
                    cameraPitch=camera_pitch,
                    cameraTargetPosition=camera_target_position
                )
                
                env.step(action)
                frame_count += 1
                
                if frame_count % 500 == 0:
                    print(f"仿真: 已执行 {frame_count} 帧")
            
            time.sleep(0.02)
            
        except Exception as e:
            print(f"仿真错误: {e}")
    
    env.close()
    print("仿真进程结束")


# ==================== 真机控制进程 ====================
def real_robot_process(shared_dict, stop_event):
    """真机控制进程 - 20维关节角度 → 10个Modbus值 → 发送到真机"""
    print("真机控制进程启动")
    
    if not REAL_ROBOT_AVAILABLE:
        print("错误: 真机SDK不可用")
        return
    
    try:
        hand = LinkerHandApi(hand_joint="L10", hand_type="right", can="can0")
        hand.set_speed(speed=[255] * 10)
        hand.set_torque(torque=[255] * 10)
        print("L20 Lite 真机控制器初始化完成")
        print("  速度: 200, 扭矩: 255")
    except Exception as e:
        print(f"真机初始化失败: {e}")
        return
    
    frame_count = 0
    
    while not stop_event.is_set():
        robot_angles = shared_dict.get('robot_angles', None)
        
        if robot_angles is not None:
            try:
                modbus_values = map_20d_to_10d_modbus(robot_angles)
                hand.finger_move(pose=modbus_values)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"真机: 已发送 {frame_count} 帧")
                    print(f"  Modbus值: {modbus_values}")
                
            except Exception as e:
                print(f"真机控制错误: {e}")
        
        time.sleep(0.02)
    
    try:
        hand.close_can()
    except:
        pass
    
    print("真机控制进程结束")


# ==================== 调试进程 ====================
def debug_process(shared_dict, stop_event):
    """调试进程 - 打印20维角度和10维Modbus值"""
    print("调试进程启动")
    
    frame_count = 0
    
    while not stop_event.is_set():
        robot_angles = shared_dict.get('robot_angles', None)
        
        if robot_angles is not None:
            frame_count += 1
            if frame_count % 50 == 0:
                modbus_vals = map_20d_to_10d_modbus(robot_angles)
                print(f"\n[调试] 第 {frame_count} 帧")
                print(f"  20维角度范围: [{robot_angles.min():.3f}, {robot_angles.max():.3f}]")
                print(f"  10维Modbus值: {modbus_vals}")
        
        time.sleep(0.5)
    
    print("调试进程结束")


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(
        description='L20 Lite 实时手部跟踪控制',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main_visionpro_realtime_l20_lite_real.py --mode sim      # 仅仿真模式
  python main_visionpro_realtime_l20_lite_real.py --mode real     # 仅真机模式
  python main_visionpro_realtime_l20_lite_real.py --mode both     # 同时运行仿真和真机
  pythonmain_visionpro_realtime_l20_lite_real.py --mode debug    # 调试模式（只打印，不控制）
  python main_visionpro_realtime_l20_lite_real.py --mode sim --avp_ip 192.168.1.100  # 指定Vision Pro IP
        """
    )
    parser.add_argument('--mode', type=str, choices=['sim', 'real', 'both', 'debug'],
                        default='sim', help='运行模式')
    parser.add_argument('--avp_ip', type=str, default='10.242.129.189',
                        help='Vision Pro IP地址')
    args = parser.parse_args()
    
    print("=" * 60)
    print("L20 Lite 实时手部跟踪控制")
    print(f"运行模式: {args.mode}")
    print(f"Vision Pro IP: {args.avp_ip}")
    print("20维仿真关节 → 10个Modbus值 (0-255)")
    print("映射规则: 弯曲关节加权求和, Thumb_Roll取反, 其他直接映射")
    print("=" * 60)
    
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()
    shared_dict['avp_ip'] = args.avp_ip
    
    stop_event = multiprocessing.Event()
    
    # 基础进程：Vision Pro 数据获取和模型推理
    processes = [
        multiprocessing.Process(target=vision_pro_data_process, args=(shared_dict, stop_event)),
        multiprocessing.Process(target=inference_process, args=(shared_dict, stop_event)),
    ]
    
    # 根据模式添加控制进程
    if args.mode in ['sim', 'both']:
        if SIMULATION_AVAILABLE:
            processes.append(multiprocessing.Process(target=simulation_process, args=(shared_dict, stop_event)))
        else:
            print("警告: 仿真模式不可用，请检查 yumi_gym 模块")
    
    if args.mode in ['real', 'both']:
        if REAL_ROBOT_AVAILABLE:
            processes.append(multiprocessing.Process(target=real_robot_process, args=(shared_dict, stop_event)))
        else:
            print("警告: 真机模式不可用，请检查 linkerhand_python_sdk 模块")
    
    if args.mode == 'debug':
        processes.append(multiprocessing.Process(target=debug_process, args=(shared_dict, stop_event)))
    
    # 启动所有进程
    for p in processes:
        p.start()
    
    print("\n所有进程已启动")
    print("按 Ctrl+C 停止程序...\n")
    
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