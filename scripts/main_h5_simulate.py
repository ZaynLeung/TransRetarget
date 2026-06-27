'''
调用模型将h5数据转换成虚拟环境机器手驱动
'''
import sys
import os
# 添加项目根目录和 yumi_gym 父目录到路径
sys.path.insert(0, '/home/ub/TransHandR')
sys.path.insert(0, '/home/ub/TransHandR/yumi_gym')

import gym, yumi_gym
import pybullet as p
import numpy as np
import h5py
import time
import math


def trans2realworld(angle):
    '''
    要将虚拟角度转换为真实角度,且检查是否超限,输入为弧度,下限,上限
    '''
    angle_real = angle
    return angle_real


hand_brand = 'l20_lite_right'
h5_file_path = f'/home/ub/TransHandR/output/h5/{hand_brand}/{hand_brand}_output.h5'

h5_file = h5py.File(h5_file_path, 'r')
r_glove_angles = h5_file['outputs'][:]
r_glove_angle_np = np.array(r_glove_angles)
print('数据格式', r_glove_angle_np.shape)
total_frames = r_glove_angle_np.shape[0]
h5_file.close()

# 初始化虚拟环境
env = gym.make('yumi-v0')
observation = env.reset()
camera_distance = 2
camera_yaw = 90
camera_pitch = -10
camera_roll = 0
camera_target_position = [0, 0, 0.05]
paused = False
v_rate = 1
stop = False

print("控制说明:")
print("  w/s: 缩放相机距离")
print("  a/d: 旋转相机左右")
print("  q/e: 旋转相机上下")
print("  空格: 暂停/继续")
print("  ESC: 退出仿真")

while not stop:
    env.render()
    for t in range(total_frames):
        # 检查 ESC 退出（放在每帧开始）
        keys = p.getKeyboardEvents()
        # ESC 键的键值是 27（十进制）
        if keys.get(27) == p.KEY_WAS_TRIGGERED:
            stop = True
            print("ESC 按下，退出仿真")
            break
        
        for i in range(2):
            R_robot_angle = trans2realworld(r_glove_angle_np[t, :]).tolist()
            action = R_robot_angle
            
            # 相机控制
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
                        print('切换暂停')
            
            if paused:
                time.sleep(0.02)
                continue
            
            p.resetDebugVisualizerCamera(cameraDistance=camera_distance,
                                          cameraYaw=camera_yaw,
                                          cameraPitch=camera_pitch,
                                          cameraTargetPosition=camera_target_position)
            observation, reward, done, info = env.step(action)
            time.sleep(0.02 * v_rate)
        
        print('当前帧数：', t)
        
        # 检查 stop 标志
        if stop:
            break
    
    # 如果正常播放完所有帧，退出循环
    if not stop:
        print("仿真完成！")
        stop = True

env.close()
print("仿真环境已关闭")