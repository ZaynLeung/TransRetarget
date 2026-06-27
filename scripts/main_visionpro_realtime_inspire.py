#!/usr/bin/env python3
"""
Vision Pro 手部跟踪 → Inspire Hand 灵巧手控制
支持仿真环境和真机控制

真机控制说明：
- Inspire Hand 有 6 个控制通道（5个弯曲 + 1个拇指侧摆）
- 弯曲通道：由多个关节角度加权求和后归一化到 0-1000
- 拇指侧摆：单独归一化到 0-1000
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

# 尝试导入 Inspire Hand SDK
try:
    from inspire_hand import InspireHand
    REAL_ROBOT_AVAILABLE = True
except ImportError:
    REAL_ROBOT_AVAILABLE = False
    print("警告: Inspire Hand SDK 未找到")


# ==================== Inspire Hand 专用配置 ====================
hand_brand = 'inspire'
out_num_joint = 13  # 模型输出维度（R_base_link_joint + 12个活动关节）
fk_num_joints = 18  # FK需要维度（包含5个tip固定关节）
scaling_factor_rb = 1.0 / 0.1

# ==================== 20维仿真关节索引定义（用于仿真） ====================
# 仿真环境使用的是 URDF 中的20维关节
SIM_JOINT_INDICES = {
    # 拇指 (0-4)
    'thumb_cmc_roll': 0,
    'thumb_cmc_yaw': 1,
    'thumb_cmc_pitch': 2,
    'thumb_mcp': 3,
    'thumb_ip': 4,
    # 食指 (5-8)
    'index_mcp_roll': 5,
    'index_mcp_pitch': 6,
    'index_pip': 7,
    'index_dip': 8,
    # 中指 (9-11)
    'middle_mcp_pitch': 9,
    'middle_pip': 10,
    'middle_dip': 11,
    # 无名指 (12-15)
    'ring_mcp_roll': 12,
    'ring_mcp_pitch': 13,
    'ring_pip': 14,
    'ring_dip': 15,
    # 小指 (16-19)
    'pinky_mcp_roll': 16,
    'pinky_mcp_pitch': 17,
    'pinky_pip': 18,
    'pinky_dip': 19,
}

# Inspire Hand 真机控制参数
# 真机通道映射：
# 通道0: 小指弯曲 (关节17+18) → 合成
# 通道1: 无名指弯曲 (关节13+14) → 合成
# 通道2: 中指弯曲 (关节9+10) → 合成
# 通道3: 食指弯曲 (关节5+6) → 合成
# 通道4: 拇指弯曲 (关节2+3+4) → 合成
# 通道5: 拇指侧摆 (关节1) → 单独

# 各弯曲关节的最大角度和（弧度）用于真机归一化
# 从 angle_limit_rob 中提取
PITCH_RANGES = {
    'Thumb_Pitch': 0.6 + 0.8 + 0.4,   # thumb_pitch + intermediate + distal = 1.8 rad
    'Index_Pitch': 1.47 + 1.56,       # index_proximal + intermediate = 3.03 rad
    'Middle_Pitch': 1.47 + 1.56,      # middle_proximal + intermediate = 3.03 rad
    'Ring_Pitch': 1.47 + 1.56,        # ring_proximal + intermediate = 3.03 rad
    'Pinky_Pitch': 1.47 + 1.56,       # pinky_proximal + intermediate = 3.03 rad
}

# 侧摆关节的最大角度（弧度）
YAW_RANGES = {
    'thumb_yaw': 1.308,   # thumb_proximal_yaw
}

# 真机角度到PWM的映射
# 0 = 完全伸直, 1000 = 完全弯曲
# 弯曲通道：角度越大 → PWM越大
# 侧摆通道：角度越大（向外摆）→ PWM越大


def map_inspire_13d_to_6d_real(angles_13d):
    """
    将模型输出的13维角度映射到 Inspire Hand 真机的6个控制值 (0-1000)
    
    模型输出索引 (13维):
    0: R_base_link_joint (固定，忽略)
    1: R_thumb_proximal_yaw_joint (拇指侧摆)
    2: R_thumb_proximal_pitch_joint (拇指弯曲-近端)
    3: R_thumb_intermediate_joint (拇指弯曲-中间)
    4: R_thumb_distal_joint (拇指弯曲-远端)
    5: R_index_proximal_joint (食指弯曲-近端)
    6: R_index_intermediate_joint (食指弯曲-远端)
    7: R_middle_proximal_joint (中指弯曲-近端)
    8: R_middle_intermediate_joint (中指弯曲-远端)
    9: R_ring_proximal_joint (无名指弯曲-近端)
    10: R_ring_intermediate_joint (无名指弯曲-远端)
    11: R_pinky_proximal_joint (小指弯曲-近端)
    12: R_pinky_intermediate_joint (小指弯曲-远端)
    
    真机控制通道 (6个):
    0: 小指弯曲
    1: 无名指弯曲
    2: 中指弯曲
    3: 食指弯曲
    4: 拇指弯曲
    5: 拇指侧摆
    
    参数:
        angles_13d: 长度为13的numpy数组，单位弧度
    
    返回:
        control_values: 长度为6的列表，每个值在0-1000之间
    """
    if len(angles_13d) != 13:
        raise ValueError(f"需要13维输入，实际{len(angles_13d)}维")
    
    control_values = [0] * 6
    
    # ===== 通道0: 小指弯曲 (Pinky_Pitch) =====
    # 关节: R_pinky_proximal_joint (索引11) + R_pinky_intermediate_joint (索引12)
    pinky_pitch_sum = angles_13d[11] + angles_13d[12]
    control_values[0] = 1000 - int(np.clip(
        pinky_pitch_sum / PITCH_RANGES['Pinky_Pitch'] * 1000, 0, 1000
    ))
    
    # ===== 通道1: 无名指弯曲 (Ring_Pitch) =====
    # 关节: R_ring_proximal_joint (索引9) + R_ring_intermediate_joint (索引10)
    ring_pitch_sum = angles_13d[9] + angles_13d[10]
    control_values[1] = 1000 - int(np.clip(
        ring_pitch_sum / PITCH_RANGES['Ring_Pitch'] * 1000, 0, 1000
    ))
    
    # ===== 通道2: 中指弯曲 (Middle_Pitch) =====
    # 关节: R_middle_proximal_joint (索引7) + R_middle_intermediate_joint (索引8)
    middle_pitch_sum = angles_13d[7] + angles_13d[8]
    control_values[2] = 1000 - int(np.clip(
        middle_pitch_sum / PITCH_RANGES['Middle_Pitch'] * 1000, 0, 1000
    ))
    
    # ===== 通道3: 食指弯曲 (Index_Pitch) =====
    # 关节: R_index_proximal_joint (索引5) + R_index_intermediate_joint (索引6)
    index_pitch_sum = angles_13d[5] + angles_13d[6]
    control_values[3] = 1000 - int(np.clip(
        index_pitch_sum / PITCH_RANGES['Index_Pitch'] * 1000, 0, 1000
    ))
    
    # ===== 通道4: 拇指弯曲 (Thumb_Pitch) =====
    # 关节: R_thumb_proximal_pitch_joint (索引2) + R_thumb_intermediate_joint (索引3) + R_thumb_distal_joint (索引4)
    thumb_pitch_sum = angles_13d[2] + angles_13d[3] + angles_13d[4]
    control_values[4] = 1000 - int(np.clip(
        thumb_pitch_sum / PITCH_RANGES['Thumb_Pitch'] * 1000, 0, 1000
    ))
    
    # ===== 通道5: 拇指侧摆 (Thumb_Yaw) =====
    # 关节: R_thumb_proximal_yaw_joint (索引1)
    # 小值向掌心靠拢(0)，大值向外摆(1000)
    thumb_yaw_rad = max(0, angles_13d[1])  # 确保非负
    control_values[5] = 1000 - int(np.clip(
        thumb_yaw_rad / YAW_RANGES['thumb_yaw'] * 1000, 0, 1000
    ))
    
    return control_values


def expand_to_fk(predicted_angles, fk_num_joints=18):
    """
    将模型输出（13维）扩展到 FK 需要的维度（18维）
    用于仿真环境
    """
    batch_size = predicted_angles.shape[0] if len(predicted_angles.shape) > 1 else 1
    if len(predicted_angles.shape) == 1:
        predicted_angles = predicted_angles.reshape(1, -1)
        batch_size = 1
    
    fk_input = np.zeros((batch_size, fk_num_joints), dtype=np.float32)
    fk_input[:, :out_num_joint] = predicted_angles
    return fk_input.squeeze(0) if batch_size == 1 else fk_input


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
    
    # 打印模型实际输出维度
    test_input = torch.randn(1, 3, 25, 3).to(device)
    with torch.no_grad():
        test_output = model(test_input)
    print(f"模型输出维度: {test_output.shape[1]}")
    
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
    
    model_path = "/home/ub/TransHandR/checkpoint/models/Optimized/inspire/model_final.pth"
    
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
                
                predicted_angles = output.cpu().numpy()[0]  # 13维
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
    """仿真环境进程 - 使用20维仿真关节"""
    print("仿真进程启动 - Inspire Hand")
    
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
                # 扩展到18维用于仿真
                action_20d = expand_to_fk(robot_angles, fk_num_joints=18)
                action = action_20d.tolist() if isinstance(action_20d, np.ndarray) else action_20d
                
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
    """真机控制进程 - 13维角度 → 6个PWM值 → 发送到真机"""
    print("真机控制进程启动 - Inspire Hand")
    
    if not REAL_ROBOT_AVAILABLE:
        print("错误: Inspire Hand SDK 不可用")
        return
    
    try:
        # 打开串口连接
        serial_port = shared_dict.get('serial_port', '/dev/ttyUSB0')
        hand = InspireHand(serial_port)
        hand.open()
        
        # 设置所有手指速度
        hand.set_all_finger_speeds(200)
        print(f"Inspire Hand 真机控制器初始化完成")
        print(f"  串口: {serial_port}")
        print(f"  速度: 200")
        print(f"  控制范围: 0(伸直) - 1000(弯曲)")
    except Exception as e:
        print(f"真机初始化失败: {e}")
        return
    
    frame_count = 0
    
    while not stop_event.is_set():
        robot_angles = shared_dict.get('robot_angles', None)
        
        if robot_angles is not None:
            try:
                # 13维角度 → 6个控制值
                control_values = map_inspire_13d_to_6d_real(robot_angles)
                
                # 发送到真机
                for finger_id, angle in enumerate(control_values):
                    hand.set_finger_angle(finger_id, angle)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"真机: 已发送 {frame_count} 帧")
                    print(f"  控制值: {control_values}")
                
            except Exception as e:
                print(f"真机控制错误: {e}")
        
        time.sleep(0.02)  # 50Hz 控制频率
    
    # 清理
    try:
        hand.close()
    except:
        pass
    
    print("真机控制进程结束")


# ==================== 调试进程 ====================
def debug_process(shared_dict, stop_event):
    """调试进程 - 打印13维角度和6维控制值"""
    print("调试进程启动")
    
    frame_count = 0
    
    while not stop_event.is_set():
        robot_angles = shared_dict.get('robot_angles', None)
        
        if robot_angles is not None:
            frame_count += 1
            if frame_count % 50 == 0:
                control_vals = map_inspire_13d_to_6d_real(robot_angles)
                print(f"\n[调试] 第 {frame_count} 帧")
                print(f"  13维角度范围: [{robot_angles.min():.3f}, {robot_angles.max():.3f}]")
                print(f"  6维控制值 (0-1000): {control_vals}")
                print(f"    通道0(小指):{control_vals[0]:4d} 通道1(无名指):{control_vals[1]:4d}")
                print(f"    通道2(中指):{control_vals[2]:4d} 通道3(食指):{control_vals[3]:4d}")
                print(f"    通道4(拇指弯曲):{control_vals[4]:4d} 通道5(拇指侧摆):{control_vals[5]:4d}")
        
        time.sleep(0.5)
    
    print("调试进程结束")


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(
        description='Inspire Hand 实时手部跟踪控制',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main_visionpro_realtime_inspire.py --mode sim           # 仅仿真模式
  python main_visionpro_realtime_inspire.py --mode real          # 仅真机模式
  python main_visionpro_realtime_inspire.py --mode both          # 同时运行仿真和真机
  python main_visionpro_realtime_inspire.py --mode debug         # 调试模式（只打印，不控制）
  python main_visionpro_realtime_inspire.py --mode sim --avp_ip 192.168.1.100  # 指定Vision Pro IP
  python main_visionpro_realtime_inspire.py --mode real --serial_port /dev/ttyUSB1  # 指定串口
        """
    )
    parser.add_argument('--mode', type=str, choices=['sim', 'real', 'both', 'debug'],
                        default='sim', help='运行模式')
    parser.add_argument('--avp_ip', type=str, default='10.242.129.189',
                        help='Vision Pro IP地址')
    parser.add_argument('--serial_port', type=str, default='/dev/ttyUSB0',
                        help='Inspire Hand 串口设备路径')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Inspire Hand 实时手部跟踪控制")
    print(f"运行模式: {args.mode}")
    print(f"Vision Pro IP: {args.avp_ip}")
    if args.mode in ['real', 'both']:
        print(f"串口设备: {args.serial_port}")
    print("映射规则:")
    print("  真机: 13维角度 → 6通道 (5弯曲 + 1侧摆)")
    print("  仿真: 13维角度 → 18维FK → 仿真环境")
    print("=" * 60)
    
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()
    shared_dict['avp_ip'] = args.avp_ip
    shared_dict['serial_port'] = args.serial_port
    
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
            print("警告: 真机模式不可用，请检查 inspire_hand 模块")
    
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