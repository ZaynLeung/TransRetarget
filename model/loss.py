# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from .angle2real import create_hand_kinematics
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import threading
import queue
import time

# hand_brand = 'yumi'
# hand_brand = 'linker'
# if hand_brand == 'yumi':
#     hand_cfg = {
#         'joints_name': [
#             'yumi_link_7_r_joint',
#             'Link1',
#             'Link11',
#             'R_ring_tip_joint',

#             'Link2',
#             'Link22',
#             'R_middle_tip_joint',

#             'Link3',
#             'Link33',
#             'R_index_tip_joint',

#             'Link4',
#             'Link44',
#             'R_pinky_tip_joint',

#             'Link5',
#             'Link51',
#             'Link52',
#             'Link53',
#             'R_thumb_tip_joint',
#         ],
#         'edges': [
#             ['yumi_link_7_r_joint', 'Link1'],
#             ['Link1', 'Link11'],
#             ['Link11', 'R_ring_tip_joint'],
#             ['yumi_link_7_r_joint', 'Link2'],
#             ['Link2', 'Link22'],
#             ['Link22', 'R_middle_tip_joint'],
#             ['yumi_link_7_r_joint', 'Link3'],
#             ['Link3', 'Link33'],
#             ['Link33', 'R_index_tip_joint'],
#             ['yumi_link_7_r_joint', 'Link4'],
#             ['Link4', 'Link44'],
#             ['Link44', 'R_pinky_tip_joint'],
#             ['yumi_link_7_r_joint', 'Link5'],
#             ['Link5', 'Link51'],
#             ['Link51', 'Link52'],
#             ['Link52', 'Link53'],
#             ['Link53', 'R_thumb_tip_joint'],
#         ],
#         'root_name': 'yumi_link_7_r_joint',
#         'end_effectors': [
#             'R_index_tip_joint',
#             'R_middle_tip_joint',
#             'R_ring_tip_joint',
#             'R_pinky_tip_joint',
#             'R_thumb_tip_joint',
#         ],
#         # 'end_effectors': [
#         #     'Link11',
#         #     'Link22',
#         #     'Link33',
#         #     'Link44',
#         #     'Link53',
#         # ],
#         'elbows': [
#             'Link1',
#             'Link2',
#             'Link3',
#             'Link4',
#             'Link5',
#         ],
#     }
#     urdf_file = "D:\\2026\\code\\TransHandR\\dataset\\robot\\ur3\\robot(ur3).urdf"
# elif hand_brand == 'linker':
#     hand_cfg = {
#         'joints_name': [
#         'hand_base_link',
#         'index_mcp_roll',
#         'index_mcp_pitch',
#         'index_pip',
#         'middle_mcp_roll',
#         'middle_mcp_pitch',
#         'middle_pip',
#         'ring_mcp_roll',
#         'ring_mcp_pitch',
#         'ring_pip',
#         'pinky_mcp_roll',
#         'pinky_mcp_pitch',
#         'pinky_pip',
#         'thumb_cmc_roll',
#         'thumb_cmc_yaw',
#         'thumb_cmc_pitch',
#         'thumb_mcp',
#         'thumb_ip',

#         'index_tip',
#         'middle_tip',
#         'ring_tip',
#         'pinky_tip',
#         'thumb_tip'
#     ],
#     'edges': [
#         ['hand_base_link', 'index_mcp_roll'],
#         ['index_mcp_roll', 'index_mcp_pitch'],
#         ['index_mcp_pitch', 'index_pip'],
#         ['hand_base_link', 'middle_mcp_roll'],
#         ['middle_mcp_roll', 'middle_mcp_pitch'],
#         ['middle_mcp_pitch', 'middle_pip'],
#         ['hand_base_link', 'ring_mcp_roll'],
#         ['ring_mcp_roll', 'ring_mcp_pitch'],
#         ['ring_mcp_pitch', 'ring_pip'],
#         ['hand_base_link', 'pinky_mcp_roll'],
#         ['pinky_mcp_roll', 'pinky_mcp_pitch'],
#         ['pinky_mcp_pitch', 'pinky_pip'],
#         ['hand_base_link', 'thumb_cmc_roll'],
#         ['thumb_cmc_roll', 'thumb_cmc_yaw'],
#         ['thumb_cmc_yaw', 'thumb_cmc_pitch'],
#         ['thumb_cmc_pitch', 'thumb_mcp'],
#         ['thumb_mcp', 'thumb_ip'],

#         ['index_pip', 'index_tip'],
#         ['middle_pip', 'middle_tip'],
#         ['ring_pip', 'ring_tip'],
#         ['pinky_pip', 'pinky_tip'],
#         ['thumb_ip', 'thumb_tip']
#     ],
#     'root_name': 'hand_base_link',
#     'end_effectors': [
#         'index_pip',
#         'middle_pip',
#         'ring_pip',
#         'pinky_pip',
#         'thumb_ip'
#     ],
#     'elbows': [
#         'index_mcp_pitch',
#         'middle_mcp_pitch',
#         'ring_mcp_pitch',
#         'pinky_mcp_pitch',
#         'thumb_mcp'
#     ]
#     }
#     urdf_file = "D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\robot\\l21_right\\linkerhand_l21_right.urdf"

def hand_loss(predicted_angle, source_3D, rb_dic, source_dic, pos_loss_function, 
              vec_loss_function, col_loss_function, reg_loss_function=None, 
              visualizer=None,hand_fk_model=None,logger=None,loss_weight=None,):
    """
    angle:是一个角度序列，对应一只手的18+5个关节角度，source3D是3D关键点序列,pos_loss_function为损失函数类型
    """
    # 关键点定义

    #首先需要将角度通过运动学模块转换为3D关键点序列
    p_loss = pos_loss_function
    # loss_weight = [500, 50, 1000, 20, 10] # vec tip col thumb tip_distance
    # 获取输入张量的设备
    device = predicted_angle.device
    source_3D = source_3D.to(device)
    # 将target_3D 转换为 [batch_size, num_joints, 3]
    b, frame ,num_joints , dim = source_3D.shape
    # if frame == 1:
    source_3D = source_3D.view(b, num_joints, dim)
    # else:
    #     assert print("输入的序列帧长度不为1")
    # 创建hand_fk实例时指定设备
    hand_fk = hand_fk_model
    # hand_fk = create_hand_kinematics(urdf_file, hand_cfg, device)
    # print("运动学输入格式",predicted_angle.shape)
    predicted_angle_for_fk = predicted_angle
    positions, orientations, global_positions = hand_fk.forward(predicted_angle_for_fk)
    # print("运动学输出格式",global_positions.shape)
    # print("原始输入格式",source_3D.shape)
     # 如果提供了可视化器，则更新坐标显示
    if visualizer is not None:
        # 只显示批次中的第一个样本
        # print("角度格式", predicted_angle_for_fk.shape)
        # print(predicted_angle_for_fk[0])
        actual_coords = source_3D[0].detach().cpu().numpy()  # 实际手部坐标
        robot_coords = global_positions[0].detach().cpu().numpy()  # 机器人手坐标
        # print("实际手部坐标", actual_coords)
        # print("机器人手坐标", robot_coords)
        visualizer.update_coordinates(actual_coords, robot_coords)
    loss_1 = vec_inter_loss(global_positions, source_3D, pos_loss_function, rb_dic, source_dic) * loss_weight[0]
    loss_2 = tip_pos_loss(global_positions, source_3D, pos_loss_function, rb_dic, source_dic) * loss_weight[1]
    if loss_weight[2]>0:
        loss_3 = col_loss_function(global_positions) * loss_weight[2]
    else:
        loss_3 = torch.tensor(0.0, device=device)
    if loss_weight[3]>0:
        loss_4 = thumb_loss(global_positions, source_3D, pos_loss_function, rb_dic, source_dic) * loss_weight[3]
    else:
        loss_4 = torch.tensor(0.0, device=device)
    loss_5 = tip_distance_loss(global_positions, source_3D, pos_loss_function, rb_dic, source_dic) * loss_weight[4]
    # 添加正则化损失
    # loss_5 的形式 但给零
    loss_6 = torch.tensor(0.0, device=device)
    loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6

    # 添加调试信息
    # print("Predicted angles:", predicted_angle_for_fk[0][:])  # 打印前5个预测角度
    # print("FK computed positions shape:", global_positions.shape)
    # print("Source positions shape:", source_3D.shape)
    
    # # 检查指尖位置差异
    # tip_indices_source = source_dic['TIP_dic']
    # tip_indices_robot = rb_dic['TIP_dic']
    # fk_tip_pos = global_positions[:, tip_indices_robot, :]  # FK计算的机器人指尖
    # real_tip_pos = source_3D[:, tip_indices_source, :]     # 实际指尖位置
    
    # print("FK tip positions:", fk_tip_pos[0])
    # print("Real tip positions:", real_tip_pos[0])
    # print("Tip position differences:", torch.norm(fk_tip_pos - real_tip_pos, dim=-1)[0])

    return loss,loss_1,loss_2,loss_3,loss_4,loss_5,loss_6

def vec_inter_loss(target_3D, source_3D, loss_function, rb_dic, source_dic):
    '''
    interphalangeal_loss - 改进版
    此处只算手掌平面的向量即yoz平面上的向量,因为考虑到手指根只能侧摆,而且本函数就是计算侧摆的loss,所以只考虑yoz平面上的向量
    '''
    # TIP_dic = source_dic['TIP_dic']
    # DIP_dic = source_dic['DIP_dic']
    PIP_dic = source_dic['PIP_dic'].copy()
    MCP_dic = source_dic['MCP_dic'].copy()
    # TIP_dic_rb = rb_dic['TIP_dic']
    DIP_dic_rb = rb_dic['DIP_dic'].copy()
    PIP_dic_rb = rb_dic['PIP_dic'].copy()
    MCP_dic_rb = rb_dic['MCP_dic'].copy()

    # PIP_dic = source_dic['TIP_dic'].copy()
    # DIP_dic_rb = rb_dic['TIP_dic'].copy()
    # MCP_dic = source_dic['DIP_dic'].copy()
    # MCP_dic_rb = rb_dic['DIP_dic'].copy()

    # 不计算大拇指
    PIP_dic.pop(0)
    # PIP_dic_rb.pop(0)
    DIP_dic_rb.pop(0)
    MCP_dic.pop(0)
    MCP_dic_rb.pop(0)
    # 修改0的索引
    # MCP_dic[0] = 0
    # 计算向量
    target_vector = target_3D[:, DIP_dic_rb, :] - target_3D[:, MCP_dic_rb, :]
    source_vector = source_3D[:, PIP_dic, :] - source_3D[:, MCP_dic, :]
    # 将x置零 保留yoz平面上的向量
    # target_vector[:, :, 0] = 0
    # source_vector[:, :, 0] = 0
    # 向量归一化
    source_vector = F.normalize(source_vector, dim=-1)
    target_vector = F.normalize(target_vector, dim=-1)
    return loss_function(source_vector, target_vector)
def thumb_loss (target_3D, source_3D, loss_function, rb_dic, source_dic):
    '''
    pip_loss 计算手指近端关节相似度
    '''
    TIP_dic = source_dic['TIP_dic']
    DIP_dic = source_dic['DIP_dic']
    PIP_dic = source_dic['PIP_dic']
    MCP_dic = source_dic['MCP_dic']
    TIP_dic_rb = rb_dic['TIP_dic']
    DIP_dic_rb = rb_dic['DIP_dic']
    PIP_dic_rb = rb_dic['PIP_dic']
    MCP_dic_rb = rb_dic['MCP_dic']

    target_pos = target_3D[:, 4,:] - target_3D[:, 3 , :]
    source_pos = source_3D[:, 17, :] - source_3D[:, 16, :]
    # 这两个向量要归一化
    source_pos = F.normalize(source_pos, dim=-1)
    target_pos = F.normalize(target_pos, dim=-1)
    loss1 = loss_function(source_pos, target_pos)
    
    target_pos2 = target_3D[:, 3,:] - target_3D[:, 2, :]
    source_pos2 = source_3D[:, 16, :] - source_3D[:, 15, :]
    # 这两个向量要归一化
    source_pos2 = F.normalize(source_pos2, dim=-1)
    target_pos2 = F.normalize(target_pos2, dim=-1)
    loss2 = loss_function(source_pos2, target_pos2)
    loss3 = 0
    loss = loss1 + loss2 + loss3
    return loss

def tip_pos_loss (target_3D, source_3D, loss_function, rb_dic, source_dic):
    '''
    tip_loss
    源数据是人手数据
    目标数据是机器手数据
    # # 记录原始数据的特定关节索引
    # # 顺序是 大拇指 , 食指 , 中指 , 无名指 , 小指
    # TIP_dic = [4, 9, 14, 19, 24] # 指尖
    # DIP_dic = [3, 8, 13, 18, 23] # 远端  
    # PIP_dic = [2, 7, 12, 17, 22] # 近端
    # MCP_dic = [1, 5, 10, 15, 20] # 掌指
    # # 记录机器手的特定关节索引
    # TIP_dic_rb = [22, 18, 19 , 20, 21]
    # DIP_dic_rb = [17, 3, 6, 9, 12]
    # PIP_dic_rb = [15, 2, 5, 8, 11]
    # MCP_dic_rb = [14, 1, 4, 7, 10]
    '''
    TIP_dic = source_dic['TIP_dic']
    # DIP_dic = source_dic['DIP_dic']
    # PIP_dic = source_dic['PIP_dic']
    MCP_dic = source_dic['MCP_dic']
    TIP_dic_rb = rb_dic['TIP_dic']
    # DIP_dic_rb = rb_dic['DIP_dic']
    # PIP_dic_rb = rb_dic['PIP_dic']
    MCP_dic_rb = rb_dic['MCP_dic']
    # 直接比较指尖位置
    target_pos = target_3D[:, TIP_dic_rb, :] - target_3D[:, MCP_dic_rb, :]
    source_pos = source_3D[:, TIP_dic, :] - source_3D[:, MCP_dic, :]
    source_pos = F.normalize(source_pos, dim=-1)
    target_pos = F.normalize(target_pos, dim=-1)
    loss = loss_function(source_pos, target_pos)
    return loss

class CollisionLoss(nn.Module):
    def __init__(self, threshold, rb_dic, excluded_points=None, excluded_pairs=None, mode='sphere-sphere', hand_type='right'):
        """
        初始化碰撞检测损失函数
        
        参数:
        threshold -- 碰撞检测的距离阈值
        rb_dic -- 关节字典
        excluded_points -- 不参与碰撞检测的点索引列表，默认排除0号点
        excluded_pairs -- 不参与碰撞检测的点对列表，例如[(2,3), (5,6)]
        mode -- 碰撞检测模式，默认为'sphere-sphere'
               支持的模式包括：
               - 'sphere-sphere': 球体与球体之间的碰撞检测
               - 'sphere-capsule': 球体与胶囊体之间的碰撞检测  
               - 'capsule-capsule': 胶囊体与胶囊体之间的碰撞检测
        """
        super(CollisionLoss, self).__init__() # 继承自 nn.Module
        self.threshold = threshold  # 碰撞检测距离阈值
        self.mode = mode  # 碰撞检测模式
        self.hand_type = hand_type # left, right, both
        self.rb_dic = rb_dic
        # 默认排除0号点，也可以传入其他要排除的点
        self.excluded_points = excluded_points if excluded_points is not None else [0]
        # 要排除的特定点对
        self.excluded_pairs = excluded_pairs if excluded_pairs is not None else []

    def forward(self, pos):
        """
        前向传播计算碰撞损失
        
        参数:
        pos -- 关节位置 [batch_size, num_nodes, 3]
        """
        threshold = self.threshold
        batch_size = pos.shape[0]
        num_nodes = pos.shape[1]
        
        # 计算所有点之间的距离矩阵
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # [batch_size, num_nodes, num_nodes, 3]
        dist = torch.norm(diff, dim=-1)  # [batch_size, num_nodes, num_nodes]
        
        # 创建掩码来排除特定点和点对
        exclusion_mask = torch.ones_like(dist, dtype=torch.bool, device=pos.device)
        
        # 排除指定的点（整行整列都排除）
        for point_idx in self.excluded_points:
            if point_idx < num_nodes:
                exclusion_mask[:, point_idx, :] = False
                exclusion_mask[:, :, point_idx] = False
        
        # 排除指定的点对
        for pair in self.excluded_pairs:
            if len(pair) == 2:
                point_a, point_b = pair
                if point_a < num_nodes and point_b < num_nodes:
                    # 排除这对点的碰撞检测（双向）
                    exclusion_mask[:, point_a, point_b] = False
                    exclusion_mask[:, point_b, point_a] = False
        
        # 排除自身距离（对角线）
        self_mask = ~torch.eye(num_nodes, dtype=torch.bool, device=pos.device)
        
        # 组合所有掩码
        mask = self_mask & exclusion_mask
        
        # 计算碰撞损失（距离小于阈值且未被排除的部分）
        collision_mask = (dist < threshold) & mask

        # 使用 e^(-x²) 形式的损失函数，直接使用距离值
        distances = dist[collision_mask]

        # 对距离进行归一化处理，使其在合理范围内
        normalized_distances = distances / threshold  # 归一化到[0, 1]区间

        # 使用 exp(-x²) 函数，x是归一化的距离
        exp_losses = torch.exp(-(normalized_distances ** 2))

        # 计算平均损失
        num_collisions = torch.sum(collision_mask)
        if num_collisions > 0:
            total_loss = torch.mean(exp_losses)
        else:
            total_loss = torch.tensor(0.0, device=pos.device)


        # 给输出加上一个很小的常数以确保非零梯度
        total_loss += 1e-6
        return total_loss


    # def forward(self, pos, edge_index, rot, ee_mask):
    #     """
    #     Keyword arguments:
    #     pos -- joint positions [batch_size, num_nodes, 3]
    #     edge_index -- edge index [2, num_edges]
    #     """
    #     batch_size = pos.shape[0]
    #     num_nodes = pos.shape[1]
    #     num_edges = edge_index.shape[1]

    #     # sphere-sphere detection
    #     if self.mode == 'sphere-sphere':
    #         l_sphere = pos[:, :num_nodes // 2, :]
    #         r_sphere = pos[:, num_nodes // 2:, :]
    #         l_sphere = l_sphere.unsqueeze(1).expand(batch_size, num_nodes // 2, num_nodes // 2, 3)
    #         r_sphere = r_sphere.unsqueeze(2).expand(batch_size, num_nodes // 2, num_nodes // 2, 3)
    #         dist_square = torch.sum(torch.pow(l_sphere - r_sphere, 2), dim=-1)
    #         mask = (dist_square < self.threshold ** 2) & (dist_square > 0)
    #         loss = torch.sum(torch.exp(-1 * torch.masked_select(dist_square, mask))) / batch_size
def tip_distance_loss (target_3D, source_3D, loss_function, rb_dic, source_dic):
    '''
    tip_distance_loss 计算手指指尖间距相似度
    '''
    TIP_dic = source_dic['TIP_dic']
    TIP_dic_rb = rb_dic['TIP_dic']
    batch_size = target_3D.shape[0]
    total_loss = 0.0
    
    # 定义要计算的指尖对：(0,1) 和 (3,4)
    tip_pairs = [(0, 1),(0,2),(0,3),(0,4),(1,2),(2,3),(3, 4)]
    
    # 例如，将距离乘以一个缩放因子（如100），使其数值更大更易处理
    scaling_factor = 100.0
    
    for i, j in tip_pairs:
        # 计算机器人手第i和第j指尖间的距离
        target_dist = torch.norm(target_3D[:, TIP_dic_rb[i], :] - target_3D[:, TIP_dic_rb[j], :], dim=-1) * scaling_factor
        # 计算人手第i和第j指尖间的距离
        source_dist = torch.norm(source_3D[:, TIP_dic[i], :] - source_3D[:, TIP_dic[j], :], dim=-1) * scaling_factor
        # 计算距离差异的损失
        loss = loss_function(source_dist, target_dist)
        total_loss += loss
    
    # 计算平均损失，由于只有2对指尖，所以除以 (batch_size * 2)
    average_loss = total_loss / (batch_size * len(tip_pairs))
    return average_loss

class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, z):
        # calculate final loss
        batch_size = z.shape[0]
        loss = torch.mean(torch.norm(z.view(batch_size, -1), dim=1).pow(2))

        return loss
class RealTimeVisualizer:
    def __init__(self, actual_connections=None, robot_connections=None):
        # 设置matplotlib后端为TkAgg以支持多线程
        plt.switch_backend('TkAgg')
        self.fig = plt.figure(figsize=(12, 6))
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax2 = self.fig.add_subplot(122, projection='3d')
        
        self.ax1.set_title('Actual Hand Coordinates')
        self.ax2.set_title('Robot Hand Coordinates')
        
        # 设置坐标轴范围
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
        
        # 如果没有提供连接关系，则使用默认值
        if actual_connections is None:
            self.actual_connections = [
                # 手腕到手掌
                [0, 1], [0, 5], [0, 10], [0, 15], [0, 20],
                # 拇指
                [1, 2], [2, 3], [3, 4],
                # 食指
                [5, 6], [6, 7], [7, 8], [8, 9],
                # 中指
                [10, 11], [11, 12], [12, 13], [13, 14],
                # 无名指
                [15, 16], [16, 17], [17, 18], [18, 19],
                # 小指
                [20, 21], [21, 22], [22, 23], [23, 24]
            ]
        else:
            self.actual_connections = actual_connections
            
        if robot_connections is None:
            self.robot_connections = [
                # 手基座到各指根
                [0, 1], [0, 4], [0, 7], [0, 10], [0, 13],
                # 食指
                [1, 2], [2, 3], [3, 18],
                # 中指
                [4, 5], [5, 6], [6, 19],
                # 无名指
                [7, 8], [8, 9], [9, 20],
                # 小指
                [10, 11], [11, 12], [12, 21],
                # 大拇指
                [13, 14], [14, 15], [15, 16], [16, 17], [17, 22]
            ]
        else:
            self.robot_connections = robot_connections
        
        # 保存最新坐标数据
        self.actual_coords = None
        self.robot_coords = None
        self.should_update = False
        
        # 显示图形
        plt.ion()  # 开启交互模式
        self.fig.show()
        self.fig.canvas.draw()
        
    def update_coordinates(self, actual_coords, robot_coords):
        """更新坐标点（非阻塞）"""
        self.actual_coords = actual_coords
        self.robot_coords = robot_coords
        self.should_update = True
    
    def update_plot(self):
        """更新绘图（在主线程调用）"""
        if self.should_update and self.actual_coords is not None and self.robot_coords is not None:
            # 清除当前图形
            self.ax1.clear()
            self.ax2.clear()
            
            # 绘制实际手部坐标
            if self.actual_coords is not None:
                x, y, z = self.actual_coords[:, 0], self.actual_coords[:, 1], self.actual_coords[:, 2]
                
                # 绘制所有关节点
                self.ax1.scatter(x, y, z, c='blue', label='Actual Hand Joints', s=30)
                
                # 根据预定义的连接关系绘制线条
                for connection in self.actual_connections:
                    if connection[0] < len(x) and connection[1] < len(x):
                        conn_x = [x[connection[0]], x[connection[1]]]
                        conn_y = [y[connection[0]], y[connection[1]]]
                        conn_z = [z[connection[0]], z[connection[1]]]
                        self.ax1.plot(conn_x, conn_y, conn_z, 'b-', alpha=0.6, linewidth=1.5)
                
                # 标记指尖
                tip_indices = [4, 9, 14, 19, 24]  # 实际手部指尖索引
                valid_tips = [idx for idx in tip_indices if idx < len(self.actual_coords)]
                if valid_tips:
                    tip_x = x[valid_tips]
                    tip_y = y[valid_tips]
                    tip_z = z[valid_tips]
                    self.ax1.scatter(tip_x, tip_y, tip_z, c='red', s=60, label='Tips', alpha=0.8)
            
            # 绘制机器人手坐标
            if self.robot_coords is not None:
                x, y, z = self.robot_coords[:, 0], self.robot_coords[:, 1], self.robot_coords[:, 2]
                
                # 绘制所有关节点
                self.ax2.scatter(x, y, z, c='green', label='Robot Hand Joints', s=30)
                
                # 根据预定义的连接关系绘制线条
                for connection in self.robot_connections:
                    if connection[0] < len(x) and connection[1] < len(x):
                        conn_x = [x[connection[0]], x[connection[1]]]
                        conn_y = [y[connection[0]], y[connection[1]]]
                        conn_z = [z[connection[0]], z[connection[1]]]
                        self.ax2.plot(conn_x, conn_y, conn_z, 'g-', alpha=0.6, linewidth=1.5)
                
                # # 标记指尖
                # tip_indices_rb = [22, 18, 19, 20, 21]  # 机器人手部指尖索引
                # valid_tips_rb = [idx for idx in tip_indices_rb if idx < len(self.robot_coords)]
                # if valid_tips_rb:
                #     tip_x = x[valid_tips_rb]S
                #     tip_y = y[valid_tips_rb]
                #     tip_z = z[valid_tips_rb]
                #     self.ax2.scatter(tip_x, tip_y, tip_z, c='red', s=60, label='Tips', alpha=0.8)
            
            # 设置相同的坐标轴范围
            # limits = [-2, 2]
            # zlimits = [0,4]
            limits = [-0.3, 0.3]
            zlimits = [0,1]
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
            
            # 刷新显示
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.should_update = False

if __name__ == '__main__':
    source_vector = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    target_vector = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    cosine_loss = nn.CosineEmbeddingLoss()
    target_labels = torch.ones(source_vector.size(0), dtype=torch.float)
    loss = cosine_loss(source_vector, target_vector, target_labels)
    print("Cosine Loss (similar):", loss.item())
    #   新建两个张量计算相反
    source_vector = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    target_vector = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32)
    target_labels = -1 * torch.ones(source_vector.size(0), dtype=torch.float)
    loss = cosine_loss(source_vector, target_vector, target_labels)
    print("Cosine Loss (opposite):", loss.item())