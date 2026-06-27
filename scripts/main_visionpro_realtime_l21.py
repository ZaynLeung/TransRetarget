#!/usr/bin/env python3
"""
Vision Pro 手部跟踪 → LinkerHand L21 灵巧手控制
支持仿真环境和真机控制
18维仿真关节 → 25个Modbus值（0-255）
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


# ==================== LinkerHand L21 配置 ====================
hand_brand = 'linker'
out_num_joint = 18  # 模型输出维度（索引0-17）
fk_num_joints = 23  # FK需要维度（包含5个tip固定关节）
scaling_factor_rb = 1.0 / 0.064

# ==================== 关节角度限制（18个活动关节） ====================
# 顺序: 0:hand_base_link(固定), 1:index_mcp_roll, 2:index_mcp_pitch, 3:index_pip,
#       4:middle_mcp_roll, 5:middle_mcp_pitch, 6:middle_pip,
#       7:ring_mcp_roll, 8:ring_mcp_pitch, 9:ring_pip,
#       10:pinky_mcp_roll, 11:pinky_mcp_pitch, 12:pinky_pip,
#       13:thumb_cmc_roll, 14:thumb_cmc_yaw, 15:thumb_cmc_pitch,
#       16:thumb_mcp, 17:thumb_ip
angle_limit_rob = [
    [0.0, 0.0],           # 0: hand_base_link (固定)
    [-0.18, 0.18],        # 1: index_mcp_roll
    [0.0, 1.57],          # 2: index_mcp_pitch
    [0.0, 1.57],          # 3: index_pip
    [-0.18, 0.18],        # 4: middle_mcp_roll
    [0.0, 1.57],          # 5: middle_mcp_pitch
    [0.0, 1.57],          # 6: middle_pip
    [-0.18, 0.18],        # 7: ring_mcp_roll
    [0.0, 1.57],          # 8: ring_mcp_pitch
    [0.0, 1.57],          # 9: ring_pip
    [-0.18, 0.18],        # 10: pinky_mcp_roll
    [0.0, 1.57],          # 11: pinky_mcp_pitch
    [0.0, 1.57],          # 12: pinky_pip
    [-0.6, 0.6],          # 13: thumb_cmc_roll
    [0.0, 1.6],           # 14: thumb_cmc_yaw
    [0.0, 1.0],           # 15: thumb_cmc_pitch
    [0.0, 1.57],          # 16: thumb_mcp
    [0.0, 1.57]           # 17: thumb_ip
]

# ==================== 关节到驱动器的映射（25个驱动器） ====================
# key: 驱动器索引(0-24), value: 关节索引(0-17)
# 根据您修改后的 joint_map 定义
DRIVE_TO_JOINT = {
    0: 15,   # 驱动器0 -> thumb_cmc_pitch
    1: 2,    # 驱动器1 -> index_mcp_pitch
    2: 5,    # 驱动器2 -> middle_mcp_pitch
    3: 8,    # 驱动器3 -> ring_mcp_pitch
    4: 11,   # 驱动器4 -> pinky_mcp_pitch
    5: 14,   # 驱动器5 -> thumb_cmc_yaw
    6: 1,    # 驱动器6 -> index_mcp_roll 不取反
    7: 4,    # 驱动器7 -> middle_mcp_roll 不取反
    8: 7,    # 驱动器8 -> ring_mcp_roll 不取反
    9: 10,   # 驱动器9 -> pinky_mcp_roll 不取反
    10: 13,  # 驱动器10 -> thumb_cmc_roll 
    11: 0,   # 驱动器11 -> hand_base_link (固定)
    12: 0,   # 预留
    13: 0,   # 预留
    14: 0,   # 预留
    15: 16,  # 驱动器15 -> thumb_mcp
    16: 0,   # 预留
    17: 0,   # 预留
    18: 0,   # 预留
    19: 0,   # 预留
    20: 17,  # 驱动器20 -> thumb_ip
    21: 3,   # 驱动器21 -> index_pip
    22: 6,   # 驱动器22 -> middle_pip
    23: 9,   # 驱动器23 -> ring_pip
    24: 12,  # 驱动器24 -> pinky_pip
}

def expand_to_fk(predicted_angles):
    """
    将模型输出（18维）扩展到 FK 需要的维度（23维）
    模型输出: 索引0-17 (hand_base_link + 17个活动关节)
    FK需要: 索引0-22 (再加上5个tip固定关节)
    """
    batch_size = predicted_angles.shape[0] if len(predicted_angles.shape) > 1 else 1
    if len(predicted_angles.shape) == 1:
        predicted_angles = predicted_angles.reshape(1, -1)
        batch_size = 1
    
    fk_input = np.zeros((batch_size, fk_num_joints), dtype=np.float32)
    fk_input[:, :out_num_joint] = predicted_angles
    # 后5个tip关节保持为0
    return fk_input.squeeze(0) if batch_size == 1 else fk_input


def angles_to_modbus(angles_18d):
    """
    将18维仿真关节角度转换为25个 Modbus 值 (0-255)
    """
    if len(angles_18d) != 18:
        raise ValueError(f"需要18维输入，实际{len(angles_18d)}维")
    
    # 第一步：根据角度限制归一化到0-255，并做限位处理
    angle_normalized = [0] * 18
    for i in range(18):
        low, high = angle_limit_rob[i]
        if high > low:
            angle_val = angles_18d[i]
            
            # 弯曲关节（非横摆关节）做限位
            is_roll_joint = (i in [1, 4, 7, 10])
            
            if not is_roll_joint:
                # 弯曲关节：接近上限时钳位到上限
                if angle_val > high * 0.95:
                    angle_val = high
                # 接近下限时钳位到下限
                if angle_val < low + 0.05:
                    angle_val = low
            
            # 归一化
            norm = (angle_val - low) / (high - low)
            angle_normalized[i] = int(np.clip(norm * 255, 0, 255))
        else:
            angle_normalized[i] = 0
    
    # 第二步：映射到25个驱动器
    modbus_values = [0] * 25
    for drive_idx, joint_idx in DRIVE_TO_JOINT.items():
        if joint_idx < 18:
            modbus_values[drive_idx] = angle_normalized[joint_idx]
    
    # 第三步：取反逻辑
    # 驱动器6,7,8,9 是横摆关节，不取反
    for i in range(25):
        if i in [6, 7, 8, 9]:
            # 横摆关节：不取反
            continue
        else:
            # 弯曲关节：取反
            modbus_values[i] = 255 - modbus_values[i]
    
    # 第四步：无名指和小拇指的横摆范围映射（0-255 -> 0-128） 
    # 这两个关节动作范围小一点，在归0位时和人手较为相像
    modbus_values[8] = modbus_values[8] // 2
    modbus_values[9] = modbus_values[9] // 2
    
    # 确保所有值在0-255范围内
    modbus_values = [max(0, min(255, v)) for v in modbus_values]
    
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
    
    model_path = "/home/ub/TransHandR/checkpoint/models/Optimized/linker/model_final.pth"
    
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
                
                predicted_angles = output.cpu().numpy()[0]  # 18维
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
    """仿真环境进程 - 接收18维关节角度，扩展后传给仿真"""
    print("仿真进程启动 - LinkerHand L21")
    
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
                # 扩展到23维用于仿真
                action = expand_to_fk(robot_angles)
                action = action.tolist() if isinstance(action, np.ndarray) else action
                
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
    """真机控制进程 - 18维关节角度 → 25个Modbus值 → 发送到真机"""
    print("真机控制进程启动 - LinkerHand L21")
    
    if not REAL_ROBOT_AVAILABLE:
        print("错误: 真机SDK不可用")
        return
    
    try:
        hand = LinkerHandApi(hand_joint="L21", hand_type="right", can="can0")
        hand.set_speed(speed=[60, 220, 220, 220, 220])  # L21 速度配置
        hand.set_torque(torque=[255] * 5)
        print("LinkerHand L21 真机控制器初始化完成")
    except Exception as e:
        print(f"真机初始化失败: {e}")
        return
    
    frame_count = 0
    
    while not stop_event.is_set():
        robot_angles = shared_dict.get('robot_angles', None)
        
        if robot_angles is not None:
            try:
                modbus_values = angles_to_modbus(robot_angles)
                hand.finger_move(pose=modbus_values)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"真机: 已发送 {frame_count} 帧")
                    print(f"  Modbus值(前10): {modbus_values[:10]}")
                
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
    """调试进程 - 打印18维角度和转换后的Modbus值"""
    print("调试进程启动 - LinkerHand L21")
    
    frame_count = 0
    
    while not stop_event.is_set():
        robot_angles = shared_dict.get('robot_angles', None)
        
        if robot_angles is not None:
            frame_count += 1
            if frame_count % 50 == 0:
                modbus_vals = angles_to_modbus(robot_angles)
                print(f"\n[调试] 第 {frame_count} 帧")
                print(f"  18维角度范围: [{robot_angles.min():.3f}, {robot_angles.max():.3f}]")
                print(f"  18维角度: {robot_angles}")
                print(f"  25维Modbus值: {modbus_vals}")
        
        time.sleep(0.5)
    
    print("调试进程结束")


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(
        description='LinkerHand L21 实时手部跟踪控制',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main_visionpro_realtime_l21.py --mode sim      # 仅仿真模式
  python main_visionpro_realtime_l21.py --mode real     # 仅真机模式
  python main_visionpro_realtime_l21.py --mode both     # 同时运行仿真和真机
  python main_visionpro_realtime_l21.py --mode debug    # 调试模式（只打印，不控制）
  python main_visionpro_realtime_l21.py --mode sim --avp_ip 192.168.1.100  # 指定Vision Pro IP
        """
    )
    parser.add_argument('--mode', type=str, choices=['sim', 'real', 'both', 'debug'],
                        default='sim', help='运行模式')
    parser.add_argument('--avp_ip', type=str, default='10.242.129.189',
                        help='Vision Pro IP地址')
    args = parser.parse_args()
    
    print("=" * 60)
    print("LinkerHand L21 实时手部跟踪控制")
    print(f"运行模式: {args.mode}")
    print(f"Vision Pro IP: {args.avp_ip}")
    print("18维仿真关节 → 23维FK (仿真) / 25个Modbus值 (真机)")
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