#!/usr/bin/env python3
"""
Vision Pro 手部跟踪 → Leap Hand 灵巧手控制
模型输出顺序与仿真期望一致：食指、中指、无名指、拇指
每个手指内部顺序：[MCP, PIP, DIP, TIP]
"""

import os
import sys
import multiprocessing
import time
import numpy as np
import torch
import gym
import pybullet as p

sys.path.insert(0, '/home/ub/TransHandR')
sys.path.insert(0, '/home/ub/TransHandR/yumi_gym')

import yumi_gym
from avp_stream import VisionProStreamer
from model.model_poseformer import PoseTransformer
from config.variables_define import *

# ==================== Leap Hand 专用配置 ====================
hand_brand = 'leaphand'

out_num_joint = 16
fk_num_joints = 17
scaling_factor_rb = 1.0/0.09

print(f"Leap Hand 配置: out_num_joint={out_num_joint}, fk_num_joints={fk_num_joints}")

# ==================== 维度扩展函数 ====================
def expand_to_fk(predicted_angles, fk_num_joints=17):
    """
    将16维角度扩展到 FK 需要的17维
    FK需要: [base_joint(0), 16个活动关节角度]
    """
    batch_size = predicted_angles.shape[0] if len(predicted_angles.shape) > 1 else 1
    if len(predicted_angles.shape) == 1:
        predicted_angles = predicted_angles.reshape(1, -1)
        batch_size = 1
    
    fk_input = np.zeros((batch_size, fk_num_joints), dtype=np.float32)
    fk_input[:, 1:out_num_joint+1] = predicted_angles
    return fk_input.squeeze(0) if batch_size == 1 else fk_input

# ==================== 模型加载 ====================
def load_model(model_path, device):
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
    print("[VisionPro] 进程启动")
    avp_ip = shared_dict.get('avp_ip', '10.242.129.189')
    print(f"[VisionPro] 连接地址: {avp_ip}")
    
    try:
        streamer = VisionProStreamer(ip=avp_ip)
        print("[VisionPro] 流连接成功")
    except Exception as e:
        print(f"[VisionPro] 连接失败: {e}")
        stop_event.set()
        return
    
    frame_buffer = []
    frame_count = 0
    error_count = 0
    last_status_time = time.time()
    
    while not stop_event.is_set():
        try:
            r = streamer.get_latest()
            fingers = r.get('right_fingers', r.get('left_fingers', None))
            
            if fingers is None:
                error_count += 1
                if error_count % 100 == 0:
                    print(f"[VisionPro] 警告: 未获取到手指数据 ({error_count}次)")
                time.sleep(0.001)
                continue
            
            error_count = 0
            
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
                if len(frame_buffer) > receptive_field:
                    frame_buffer.pop(0)
                
                if len(frame_buffer) == receptive_field:
                    vision_data = np.stack(frame_buffer, axis=1)
                    shared_dict['vision_data'] = vision_data
                    frame_count += 1
                    
                    current_time = time.time()
                    if current_time - last_status_time > 5.0:
                        print(f"[VisionPro] 已获取 {frame_count} 帧数据")
                        last_status_time = current_time
            
            time.sleep(0.001)
        except Exception as e:
            print(f"[VisionPro] 错误: {e}")
            time.sleep(0.1)
    
    print(f"[VisionPro] 进程结束，共处理 {frame_count} 帧")

# ==================== 推理进程 ====================
def inference_process(shared_dict, stop_event):
    print("[推理] 进程启动")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[推理] 使用设备: {device}")
    
    model_path = "/home/ub/TransHandR/checkpoint/models/Optimized/leaphand/model_final.pth"
    
    if not os.path.exists(model_path):
        print(f"[推理] 错误: 模型文件不存在 - {model_path}")
        stop_event.set()
        return
    
    model = load_model(model_path, device)
    
    # ========== 检测模型实际输出维度 ==========
    test_input = torch.randn(1, receptive_field, num_joints, 3).to(device)
    with torch.no_grad():
        test_output = model(test_input)
    print(f"\n[模型输出维度检测]")
    print(f"  输入形状: {test_input.shape}")
    print(f"  输出形状: {test_output.shape}")
    print(f"  实际输出维度: {test_output.shape[1]}")
    print(f"  配置期望维度: {out_num_joint}")
    if test_output.shape[1] != out_num_joint:
        print(f"  ⚠️ 警告: 模型实际输出维度与配置不一致!")
        print(f"     将使用模型实际维度: {test_output.shape[1]}")
        actual_out_dim = test_output.shape[1]
    else:
        actual_out_dim = out_num_joint
    print("=" * 50 + "\n")
    # ========================================
    
    frame_count = 0
    inference_times = []
    nan_count = 0
    
    print("[推理] 等待 Vision Pro 数据...")
    wait_count = 0
    while 'vision_data' not in shared_dict and not stop_event.is_set() and wait_count < 300:
        time.sleep(0.1)
        wait_count += 1
        if wait_count % 50 == 0:
            print(f"[推理] 等待数据中... ({wait_count*0.1}s)")
    
    if 'vision_data' not in shared_dict:
        print("[推理] 超时: 未接收到 Vision Pro 数据")
        stop_event.set()
        return
    
    print("[推理] 开始推理循环")
    
    while not stop_event.is_set():
        vision_data = shared_dict.get('vision_data', None)
        
        if vision_data is not None:
            try:
                if np.isnan(vision_data).any() or np.isinf(vision_data).any():
                    nan_count += 1
                    if nan_count % 10 == 0:
                        print(f"[推理] 警告: 输入数据包含 NaN/Inf ({nan_count}次)")
                    time.sleep(0.001)
                    continue
                
                input_data = np.transpose(vision_data, (1, 0, 2))
                input_data = np.expand_dims(input_data, axis=0)
                input_data *= scaling_factor
                
                input_tensor = torch.from_numpy(input_data).float().to(device)
                
                start_time = time.time()
                with torch.no_grad():
                    output = model(input_tensor)
                inference_time = time.time() - start_time
                
                predicted_angles = output.cpu().numpy()[0]
                
                # 首帧打印
                if frame_count == 0:
                    print(f"\n[推理] 首帧输出:")
                    print(f"  输出形状: {predicted_angles.shape}")
                    print(f"  角度范围: [{np.min(predicted_angles):.4f}, {np.max(predicted_angles):.4f}]")
                    print(f"  食指 (0-3): {predicted_angles[0:4]}")
                    print(f"  中指 (4-7): {predicted_angles[4:8]}")
                    print(f"  无名指 (8-11): {predicted_angles[8:12]}")
                    print(f"  拇指 (12-15): {predicted_angles[12:16]}")
                    print("=" * 50)
                
                # 检查输出是否包含 NaN
                if np.isnan(predicted_angles).any():
                    print(f"[推理] 错误: 模型输出包含 NaN!")
                    predicted_angles = np.nan_to_num(predicted_angles, nan=0.0)
                
                # 直接扩展到 FK 维度（不裁剪）
                predicted_angles_expanded = expand_to_fk(predicted_angles, fk_num_joints)
                shared_dict['robot_angles'] = predicted_angles_expanded
                shared_dict['timestamp'] = time.time()
                
                inference_times.append(inference_time)
                if len(inference_times) > 100:
                    inference_times.pop(0)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    avg_time = np.mean(inference_times)
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    print(f"[推理] 帧数: {frame_count}, 平均推理时间: {avg_time*1000:.2f}ms, FPS: {fps:.1f}")
                
            except Exception as e:
                print(f"[推理] 错误: {e}")
                import traceback
                traceback.print_exc()
        
        time.sleep(0.001)
    
    print(f"[推理] 进程结束，共推理 {frame_count} 帧")

# ==================== 仿真环境进程 ====================
def simulation_process(shared_dict, stop_event):
    print("[仿真] 进程启动 - Leap Hand")
    
    try:
        env = gym.make('yumi-v0')
        env.reset()
        print("[仿真] 环境创建成功")
    except Exception as e:
        print(f"[仿真] 环境创建失败: {e}")
        stop_event.set()
        return
    
    # ========== 打印仿真环境关节信息 ==========
    print("\n[仿真] ========== 仿真环境关节信息 ==========")
    try:
        robot_id = env.robot.robot_id
        num_joints = p.getNumJoints(robot_id)
        print(f"  总关节数: {num_joints}")
        
        revolute_joints = []
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            joint_lower = info[8]
            joint_upper = info[9]
            
            if joint_type == p.JOINT_REVOLUTE:
                revolute_joints.append(i)
                print(f"  [{i:2d}]: {joint_name:20s} | 范围: [{joint_lower:.3f}, {joint_upper:.3f}]")
            elif joint_type == p.JOINT_FIXED:
                print(f"  [{i:2d}]: {joint_name:20s} (固定关节)")
        
        print(f"  活动关节数量: {len(revolute_joints)}")
        
        if len(revolute_joints) != fk_num_joints:
            print(f"  [警告] 环境活动关节数({len(revolute_joints)}) != 期望关节数({fk_num_joints})")
        else:
            print(f"  [OK] 关节数量匹配")
        
        print(f"\n[仿真] 动作空间信息:")
        print(f"  action_space: {env.action_space}")
        if hasattr(env.action_space, 'shape'):
            print(f"  action_space.shape: {env.action_space.shape}")
        
    except Exception as e:
        print(f"[仿真] 获取关节信息失败: {e}")
    print("=" * 50)
    
    # 相机控制参数
    camera_distance = 0.8
    camera_yaw = 90
    camera_pitch = -25
    camera_target_position = [0, 0, 0.1]
    paused = False
    frame_count = 0
    no_data_warning_count = 0
    
    print("\n" + "=" * 50)
    print("Leap Hand 仿真控制说明:")
    print("  w/s : 缩放相机距离")
    print("  a/d : 旋转相机左右")
    print("  q/e : 旋转相机上下")
    print("  空格: 暂停/继续")
    print("  ESC : 退出程序")
    print("=" * 50 + "\n")
    
    print("[仿真] 等待模型输出...")
    wait_count = 0
    while 'robot_angles' not in shared_dict and not stop_event.is_set() and wait_count < 300:
        time.sleep(0.1)
        wait_count += 1
        if wait_count % 50 == 0:
            print(f"[仿真] 等待模型输出中... ({wait_count*0.1}s)")
    
    if 'robot_angles' not in shared_dict:
        print("[仿真] 警告: 未接收到模型输出，将使用零角度")
    
    print("[仿真] 开始主循环")
    
    while not stop_event.is_set():
        try:
            env.render()
            
            # 处理键盘事件
            keys = p.getKeyboardEvents()
            for k, v in keys.items():
                if v & p.KEY_WAS_TRIGGERED:
                    if k == ord('w'):
                        camera_distance = max(0.2, camera_distance - 0.1)
                    elif k == ord('s'):
                        camera_distance = min(2.0, camera_distance + 0.1)
                    elif k == ord('a'):
                        camera_yaw = (camera_yaw - 10) % 360
                    elif k == ord('d'):
                        camera_yaw = (camera_yaw + 10) % 360
                    elif k == ord('q'):
                        camera_pitch = max(-90, camera_pitch - 10)
                    elif k == ord('e'):
                        camera_pitch = min(0, camera_pitch + 10)
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
                
                # 首帧打印
                if frame_count == 0:
                    print(f"\n[仿真] 首帧动作:")
                    print(f"  action 形状: {np.array(action).shape}")
                    print(f"  action 范围: [{np.min(action):.4f}, {np.max(action):.4f}]")
                    print(f"  action[1:5] (食指): {action[1:5]}")
                    print(f"  action[5:9] (中指): {action[5:9]}")
                    print(f"  action[9:13] (无名指): {action[9:13]}")
                    print(f"  action[13:17] (拇指): {action[13:17]}")
                    print("=" * 50)
                
                p.resetDebugVisualizerCamera(
                    cameraDistance=camera_distance,
                    cameraYaw=camera_yaw,
                    cameraPitch=camera_pitch,
                    cameraTargetPosition=camera_target_position
                )
                
                env.step(action)
                frame_count += 1
                
                if frame_count % 500 == 0:
                    print(f"[仿真] 已执行 {frame_count} 帧")
                
                if no_data_warning_count > 0:
                    no_data_warning_count = 0
            else:
                no_data_warning_count += 1
                if no_data_warning_count % 100 == 1:
                    print(f"[仿真] 警告: 未收到模型输出 ({no_data_warning_count}次)")
            
            time.sleep(0.02)
            
        except Exception as e:
            print(f"[仿真] 错误: {e}")
            time.sleep(0.1)
    
    env.close()
    print(f"[仿真] 进程结束，共执行 {frame_count} 帧")

# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("Leap Hand 实时手部跟踪控制")
    print("模型输出顺序: [食指, 中指, 无名指, 拇指]")
    print("每个手指内部顺序: [MCP, PIP, DIP, TIP]")
    print("=" * 60)
    
    if not os.path.exists('/home/ub/TransHandR/yumi_gym'):
        print("警告: yumi_gym 路径不存在，请确认路径配置")
    
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()
    shared_dict['avp_ip'] = '10.242.129.189'
    shared_dict['vision_data'] = None
    shared_dict['robot_angles'] = None
    shared_dict['timestamp'] = 0
    
    stop_event = multiprocessing.Event()
    
    processes = [
        multiprocessing.Process(target=vision_pro_data_process, args=(shared_dict, stop_event)),
        multiprocessing.Process(target=inference_process, args=(shared_dict, stop_event)),
        multiprocessing.Process(target=simulation_process, args=(shared_dict, stop_event))
    ]
    
    for p in processes:
        p.start()
    
    print("\n所有进程已启动")
    print("  - Vision Pro 数据获取进程")
    print("  - 模型推理进程")
    print("  - 仿真控制进程")
    print("\n按 Ctrl+C 停止...\n")
    
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
        stop_event.set()
        for p in processes:
            if p.is_alive():
                p.join(timeout=3)
                if p.is_alive():
                    p.terminate()
    
    print("\n程序已退出")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()