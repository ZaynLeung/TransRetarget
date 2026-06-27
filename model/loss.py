# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 适用于 l21 l20_lite inspire leap_hand - 完整版 20260615
# 新增 finger_segment_length_loss 约束每段手指长度

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from .angle2real import create_hand_kinematics
import threading
import queue
import time


def hand_loss(predicted_angle, source_3D, rb_dic, source_dic, pos_loss_function, 
              vec_loss_function, col_loss_function, reg_loss_function=None, 
              visualizer=None, hand_fk_model=None, logger=None, loss_weight=None):
    """
    手部损失函数 - 兼容版
    
    参数:
    pos_loss_function: MSELoss (用于位置/距离损失)
    vec_loss_function: 可能为 None 或 CosineEmbeddingLoss
    loss_weight: [vec, tip, col, thumb, tip_dist] 共5个权重
    """
    from config.variables_define import hand_brand, angle_limit_rob
    
    # 如果 vec_loss_function 是 None，使用一个替代方案
    if vec_loss_function is None:
        vec_loss_function = nn.MSELoss()
    
    device = predicted_angle.device
    source_3D = source_3D.to(device)
    
    # 转换维度
    b, frame, num_joints, dim = source_3D.shape
    source_3D = source_3D.view(b, num_joints, dim)
    
    # 正向运动学
    hand_fk = hand_fk_model
    fk_num_joints = hand_fk.num_joints
    model_num_joints = predicted_angle.shape[1]

    if model_num_joints != fk_num_joints:
        predicted_angle_for_fk = torch.zeros(b, fk_num_joints, device=device)
        predicted_angle_for_fk[:, :model_num_joints] = predicted_angle
    else:
        predicted_angle_for_fk = predicted_angle

    positions, orientations, global_positions = hand_fk.forward(predicted_angle_for_fk)
    
    # 可视化
    if visualizer is not None:
        actual_coords = source_3D[0].detach().cpu().numpy()
        robot_coords = global_positions[0].detach().cpu().numpy()
        visualizer.update_coordinates(actual_coords, robot_coords)
    
    # 各项损失（前5个由 loss_weight 控制权重）
    loss_1 = vec_inter_loss(global_positions, source_3D, pos_loss_function, rb_dic, source_dic) * loss_weight[0]
    loss_2 = tip_pos_loss(global_positions, source_3D, pos_loss_function, rb_dic, source_dic) * loss_weight[1]
    
    if loss_weight[2] > 0:
        loss_3 = col_loss_function(global_positions) * loss_weight[2]
    else:
        loss_3 = torch.tensor(0.0, device=device)
    
    if loss_weight[3] > 0:
        loss_4 = thumb_loss(global_positions, source_3D, pos_loss_function, rb_dic, source_dic) * loss_weight[3]
    else:
        loss_4 = torch.tensor(0.0, device=device)
    
    loss_5 = tip_distance_loss(global_positions, source_3D, pos_loss_function, rb_dic, source_dic) * loss_weight[4]
    
    # ===== 新增：每段手指长度损失 =====
    loss_segment = finger_segment_length_loss(global_positions, source_3D, pos_loss_function, rb_dic, source_dic) * 200.0

    # 总损失
    loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_segment

    # 返回7个值，最后一项为 segment loss（替代原有的 combined_reg_loss）
    return loss, loss_1, loss_2, loss_3, loss_4, loss_5, loss_segment


def finger_segment_length_loss(target_3D, source_3D, loss_function, rb_dic, source_dic):
    """
    约束每段手指长度：MCP-PIP, PIP-DIP, DIP-TIP
    对每根手指（拇指、食指、中指、无名指）分别计算长度 MSE
    """
    # 源数据关节索引（Vision Pro）
    src_mcp = source_dic['MCP_dic'][:4]   # [1,6,11,16]
    src_pip = source_dic['PIP_dic'][:4]   # [2,7,12,17]
    src_dip = source_dic['DIP_dic'][:4]   # [3,8,13,18]
    src_tip = source_dic['TIP_dic'][:4]   # [4,9,14,19]
    
    # 机器人手关节索引（rb_dic）
    tgt_mcp = rb_dic['MCP_dic']           # [13,1,5,9]
    tgt_pip = rb_dic['PIP_dic']           # [14,2,6,10]
    tgt_dip = rb_dic['DIP_dic']           # [15,3,7,11]
    tgt_tip = rb_dic['TIP_dic']           # [16,4,8,12]
    
    total_loss = 0.0
    segment_count = 0
    
    for i in range(4):  # 4根手指
        # 段1: MCP -> PIP
        tgt_len = torch.norm(target_3D[:, tgt_pip[i], :] - target_3D[:, tgt_mcp[i], :], dim=-1)
        src_len = torch.norm(source_3D[:, src_pip[i], :] - source_3D[:, src_mcp[i], :], dim=-1)
        loss = loss_function(tgt_len, src_len)
        total_loss += loss
        segment_count += 1
        
        # 段2: PIP -> DIP
        tgt_len = torch.norm(target_3D[:, tgt_dip[i], :] - target_3D[:, tgt_pip[i], :], dim=-1)
        src_len = torch.norm(source_3D[:, src_dip[i], :] - source_3D[:, src_pip[i], :], dim=-1)
        loss = loss_function(tgt_len, src_len)
        total_loss += loss
        segment_count += 1
        
        # 段3: DIP -> TIP
        tgt_len = torch.norm(target_3D[:, tgt_tip[i], :] - target_3D[:, tgt_dip[i], :], dim=-1)
        src_len = torch.norm(source_3D[:, src_tip[i], :] - source_3D[:, src_dip[i], :], dim=-1)
        loss = loss_function(tgt_len, src_len)
        total_loss += loss
        segment_count += 1
    
    return total_loss / segment_count


def vec_inter_loss(target_3D, source_3D, loss_function, rb_dic, source_dic):
    '''
    手指侧摆方向损失
    使用 MSE Loss（因为原训练脚本传的是 MSELoss）
    
    手指顺序: 食指、中指、无名指（排除拇指）
    '''
    # 获取源数据的手指索引（只用前4根手指，忽略小指）
    PIP_dic = source_dic['PIP_dic'][:4]  # [2, 7, 12, 17]
    MCP_dic = source_dic['MCP_dic'][:4]  # [1, 6, 11, 16]
    
    # 获取机器人手的手指索引（4根）
    DIP_dic_rb = rb_dic['DIP_dic']       # [15, 3, 7, 11]
    MCP_dic_rb = rb_dic['MCP_dic']       # [13, 1, 5, 9]
    
    # 排除拇指（索引0），只计算食指、中指、无名指的侧摆
    finger_indices = [1, 2, 3]  # 食指、中指、无名指
    
    if len(finger_indices) == 0:
        return torch.tensor(0.0, device=target_3D.device)
    
    # 计算向量
    target_vector = target_3D[:, [DIP_dic_rb[i] for i in finger_indices], :] - \
                    target_3D[:, [MCP_dic_rb[i] for i in finger_indices], :]
    source_vector = source_3D[:, [PIP_dic[i] for i in finger_indices], :] - \
                    source_3D[:, [MCP_dic[i] for i in finger_indices], :]
    
    # 只保留 YOZ 平面（侧摆方向）
    target_vector = target_vector.clone()
    source_vector = source_vector.clone()
    target_vector[:, :, 0] = 0
    source_vector[:, :, 0] = 0
    
    # 直接使用 MSE Loss 计算向量差异（不归一化）
    loss = loss_function(target_vector, source_vector)
    
    return loss


def tip_pos_loss(target_3D, source_3D, loss_function, rb_dic, source_dic):
    '''
    指尖方向损失
    使用 MSE Loss
    
    手指顺序: 拇指(0), 食指(1), 中指(2), 无名指(3)
    '''
    # 源数据（Vision Pro）取前4根手指
    TIP_dic = source_dic['TIP_dic'][:4]   # [4, 9, 14, 19]
    MCP_dic = source_dic['MCP_dic'][:4]   # [1, 6, 11, 16]
    
    # 机器人手
    TIP_dic_rb = rb_dic['TIP_dic']        # [16, 4, 8, 12]
    MCP_dic_rb = rb_dic['MCP_dic']        # [13, 1, 5, 9]
    
    num_fingers = 4
    
    total_loss = 0.0
    
    for i in range(num_fingers):
        # 计算指尖相对于 MCP 的向量
        target_vec = target_3D[:, TIP_dic_rb[i], :] - target_3D[:, MCP_dic_rb[i], :]
        source_vec = source_3D[:, TIP_dic[i], :] - source_3D[:, MCP_dic[i], :]
        
        # 直接使用 MSE Loss
        loss = loss_function(target_vec, source_vec)
        total_loss += loss
    
    return total_loss / num_fingers


def thumb_loss(target_3D, source_3D, loss_function, rb_dic, source_dic):
    '''
    拇指损失 - 使用 MSELoss
    单独处理拇指
    '''
    # 拇指关节顺序: [thumb_base(12), thumb_PIP(13), thumb_DIP(14), thumb_TIP(15)]
    target_pos = target_3D[:, 15, :] - target_3D[:, 14, :]
    source_pos = source_3D[:, 4, :] - source_3D[:, 3, :]
    loss1 = loss_function(target_pos, source_pos)
    
    # 第二段：DIP - PIP
    target_pos2 = target_3D[:, 14, :] - target_3D[:, 13, :]
    source_pos2 = source_3D[:, 3, :] - source_3D[:, 2, :]
    loss2 = loss_function(target_pos2, source_pos2)
    
    # 第三段：PIP - 基座
    target_pos3 = target_3D[:, 13, :] - target_3D[:, 12, :]
    source_pos3 = source_3D[:, 2, :] - source_3D[:, 1, :]
    loss3 = loss_function(target_pos3, source_pos3)
    
    return loss1 + loss2 + loss3


def tip_distance_loss(target_3D, source_3D, loss_function, rb_dic, source_dic):
    '''
    指尖距离损失 - 使用 MSELoss
    计算不同手指指尖之间的距离一致性
    
    手指: 拇指(0), 食指(1), 中指(2), 无名指(3)
    '''
    TIP_dic = source_dic['TIP_dic'][:4]
    TIP_dic_rb = rb_dic['TIP_dic']
    
    total_loss = 0.0
    
    # 定义指尖对
    tip_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]
    scaling_factor = 100.0
    
    for i, j in tip_pairs:
        target_dist = torch.norm(target_3D[:, TIP_dic_rb[i], :] - target_3D[:, TIP_dic_rb[j], :], dim=-1) * scaling_factor
        source_dist = torch.norm(source_3D[:, TIP_dic[i], :] - source_3D[:, TIP_dic[j], :], dim=-1) * scaling_factor
        
        loss = loss_function(target_dist, source_dist)
        total_loss += loss
    
    average_loss = total_loss / len(tip_pairs)
    return average_loss


class CollisionLoss(nn.Module):
    def __init__(self, threshold, rb_dic, excluded_points=None, excluded_pairs=None, mode='sphere-sphere', hand_type='right'):
        super(CollisionLoss, self).__init__()
        self.threshold = threshold
        self.mode = mode
        self.hand_type = hand_type
        self.rb_dic = rb_dic
        self.excluded_points = excluded_points if excluded_points is not None else [0]
        self.excluded_pairs = excluded_pairs if excluded_pairs is not None else []

    def forward(self, pos):
        threshold = self.threshold
        batch_size = pos.shape[0]
        num_nodes = pos.shape[1]
        
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)
        
        exclusion_mask = torch.ones_like(dist, dtype=torch.bool, device=pos.device)
        
        for point_idx in self.excluded_points:
            if point_idx < num_nodes:
                exclusion_mask[:, point_idx, :] = False
                exclusion_mask[:, :, point_idx] = False
        
        for pair in self.excluded_pairs:
            if len(pair) == 2:
                point_a, point_b = pair
                if point_a < num_nodes and point_b < num_nodes:
                    exclusion_mask[:, point_a, point_b] = False
                    exclusion_mask[:, point_b, point_a] = False
        
        self_mask = ~torch.eye(num_nodes, dtype=torch.bool, device=pos.device)
        mask = self_mask & exclusion_mask
        collision_mask = (dist < threshold) & mask

        distances = dist[collision_mask]
        normalized_distances = distances / threshold
        exp_losses = torch.exp(-(normalized_distances ** 2))

        num_collisions = torch.sum(collision_mask)
        if num_collisions > 0:
            total_loss = torch.mean(exp_losses)
        else:
            total_loss = torch.tensor(0.0, device=pos.device)

        total_loss += 1e-6
        return total_loss


class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, z):
        batch_size = z.shape[0]
        loss = torch.mean(torch.norm(z.view(batch_size, -1), dim=1).pow(2))
        return loss


class RealTimeVisualizer:
    def __init__(self, actual_connections=None, robot_connections=None):
        import matplotlib
        matplotlib.use('TkAgg')
        
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # 禁用工具栏，避免 PIL 图标加载问题
        plt.rcParams['toolbar'] = 'None'
        
        self.fig = plt.figure(figsize=(12, 6))
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax2 = self.fig.add_subplot(122, projection='3d')
        
        self.ax1.set_title('Actual Hand Coordinates')
        self.ax2.set_title('Robot Hand Coordinates')
        
        self.ax1.set_xlim([-1, 1])
        self.ax1.set_ylim([-1, 1])
        self.ax1.set_zlim([-1, 1])
        self.ax2.set_xlim([-1, 1])
        self.ax2.set_ylim([-1, 1])
        self.ax2.set_zlim([-1, 1])
        
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Y')
        self.ax2.set_zlabel('Z')
        
        self.ax1.grid(True)
        self.ax2.grid(True)
        
        if actual_connections is None:
            self.actual_connections = [
                [0, 1], [0, 5], [0, 10], [0, 15], [0, 20],
                [1, 2], [2, 3], [3, 4],
                [5, 6], [6, 7], [7, 8], [8, 9],
                [10, 11], [11, 12], [12, 13], [13, 14],
                [15, 16], [16, 17], [17, 18], [18, 19],
                [20, 21], [21, 22], [22, 23], [23, 24]
            ]
        else:
            self.actual_connections = actual_connections
            
        if robot_connections is None:
            self.robot_connections = [
                [0, 1], [0, 4], [0, 7], [0, 10], [0, 13],
                [1, 2], [2, 3], [3, 18],
                [4, 5], [5, 6], [6, 19],
                [7, 8], [8, 9], [9, 20],
                [10, 11], [11, 12], [12, 21],
                [13, 14], [14, 15], [15, 16], [16, 17], [17, 22]
            ]
        else:
            self.robot_connections = robot_connections
        
        self.actual_coords = None
        self.robot_coords = None
        self.should_update = False
        
        plt.ion()
        try:
            self.fig.show()
            self.fig.canvas.draw()
            print("可视化器初始化成功")
        except Exception as e:
            print(f"可视化器显示失败: {e}")
            print("使用后台模式")
            plt.ioff()
        
    def update_coordinates(self, actual_coords, robot_coords):
        self.actual_coords = actual_coords
        self.robot_coords = robot_coords
        self.should_update = True
    
    def update_plot(self):
        if self.should_update and self.actual_coords is not None and self.robot_coords is not None:
            try:
                self.ax1.clear()
                self.ax2.clear()
                
                if self.actual_coords is not None:
                    x, y, z = self.actual_coords[:, 0], self.actual_coords[:, 1], self.actual_coords[:, 2]
                    self.ax1.scatter(x, y, z, c='blue', label='Actual Hand Joints', s=30)
                    
                    for connection in self.actual_connections:
                        if connection[0] < len(x) and connection[1] < len(x):
                            conn_x = [x[connection[0]], x[connection[1]]]
                            conn_y = [y[connection[0]], y[connection[1]]]
                            conn_z = [z[connection[0]], z[connection[1]]]
                            self.ax1.plot(conn_x, conn_y, conn_z, 'b-', alpha=0.6, linewidth=1.5)
                    
                    tip_indices = [4, 9, 14, 19, 24]
                    valid_tips = [idx for idx in tip_indices if idx < len(self.actual_coords)]
                    if valid_tips:
                        tip_x = x[valid_tips]
                        tip_y = y[valid_tips]
                        tip_z = z[valid_tips]
                        self.ax1.scatter(tip_x, tip_y, tip_z, c='red', s=60, label='Tips', alpha=0.8)
                
                if self.robot_coords is not None:
                    x, y, z = self.robot_coords[:, 0], self.robot_coords[:, 1], self.robot_coords[:, 2]
                    self.ax2.scatter(x, y, z, c='green', label='Robot Hand Joints', s=30)
                    
                    for connection in self.robot_connections:
                        if connection[0] < len(x) and connection[1] < len(x):
                            conn_x = [x[connection[0]], x[connection[1]]]
                            conn_y = [y[connection[0]], y[connection[1]]]
                            conn_z = [z[connection[0]], z[connection[1]]]
                            self.ax2.plot(conn_x, conn_y, conn_z, 'g-', alpha=0.6, linewidth=1.5)
                    
                limits = [-1, 1]
                zlimits = [0, 3]
                self.ax1.set_xlim(limits)
                self.ax1.set_ylim(limits)
                self.ax1.set_zlim(zlimits)
                self.ax2.set_xlim(limits)
                self.ax2.set_ylim(limits)
                self.ax2.set_zlim(zlimits)
                
                self.ax1.set_xlabel('X')
                self.ax1.set_ylabel('Y')
                self.ax1.set_zlabel('Z')
                self.ax2.set_xlabel('X')
                self.ax2.set_ylabel('Y')
                self.ax2.set_zlabel('Z')
                
                self.ax1.grid(True)
                self.ax2.grid(True)
                
                self.ax1.legend()
                self.ax2.legend()
                
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                self.should_update = False
            except Exception as e:
                print(f"更新绘图失败: {e}")
                self.should_update = False


if __name__ == '__main__':
    source_vector = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    target_vector = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    mse_loss = nn.MSELoss()
    loss = mse_loss(source_vector, target_vector)
    print("MSE Loss (similar):", loss.item())
    
    source_vector = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    target_vector = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32)
    loss = mse_loss(source_vector, target_vector)
    print("MSE Loss (opposite):", loss.item())