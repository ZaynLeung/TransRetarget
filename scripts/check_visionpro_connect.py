from avp_stream import VisionProStreamer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
'''
总结:新x=-y,新y=z,新z=-x
'''
def visualize_hand_tracking(relative_coordinates, ax):
    """
    可视化手部跟踪数据
    
    参数:
    relative_coordinates: 相对于手腕的坐标数组 [num_joints, 3]
    ax: matplotlib 3D坐标轴对象
    """
    # 清除上一帧的图形
    ax.clear()
    
    # 手指关节连接关系
    connections = [
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
    
    # 提取x, y, z坐标
    x_coords = relative_coordinates[:, 0]
    y_coords = relative_coordinates[:, 1]
    z_coords = relative_coordinates[:, 2]
    
    # 绘制所有关节点
    ax.scatter(x_coords, y_coords, z_coords, c='blue', label='Hand Joints', s=50)
    
    # 根据连接关系绘制线条
    for connection in connections:
        if connection[0] < len(relative_coordinates) and connection[1] < len(relative_coordinates):
            conn_x = [relative_coordinates[connection[0]][0], relative_coordinates[connection[1]][0]]
            conn_y = [relative_coordinates[connection[0]][1], relative_coordinates[connection[1]][1]]
            conn_z = [relative_coordinates[connection[0]][2], relative_coordinates[connection[1]][2]]
            ax.plot(conn_x, conn_y, conn_z, 'b-', alpha=0.6, linewidth=1.5)
    
    # 标记指尖（索引4, 9, 14, 19, 24）
    tip_indices = [4, 9, 14, 19, 24]
    valid_tips = [idx for idx in tip_indices if idx < len(relative_coordinates)]
    if valid_tips:
        tip_x = x_coords[valid_tips]
        tip_y = y_coords[valid_tips]
        tip_z = z_coords[valid_tips]
        ax.scatter(tip_x, tip_y, tip_z, c='red', s=80, label='Tips', alpha=0.8)
    
    # 设置图形属性
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Right Hand Tracking (Relative to Wrist)')
    ax.legend()
    
    # 设置坐标轴范围以保持视角稳定
    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([-0.2, 0.2])
    ax.set_zlim([-0.2, 0.2])

# 初始化可视化
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

avp_ip = "10.242.129.189"  # Vision Pro IP (shown in the app)
s = VisionProStreamer(ip=avp_ip)

# Configure video streaming from robot camera
# s.configure_video(device="/dev/video0", format="v4l2", size="1280x720", fps=30)
# s.start_webrtc()
print("已经坐标变换")
while True:
    r = s.get_latest()
    
    # 提取右手手指跟踪数据
    right_fingers = r['right_fingers']
    # right_fingers = r['left_fingers']
    
    # 存储处理后的坐标
    coordinates = []
    
    # 遍历所有关节
    for i in range(len(right_fingers)):
        # 获取4x4变换矩阵
        transform_matrix = right_fingers[i]
        
        # 提取xyz坐标（从4x4矩阵的最后列获取平移部分）
        # x = transform_matrix[0][3]
        # y = transform_matrix[1][3]
        # z = transform_matrix[2][3]

        x = -transform_matrix[1][3]
        y = transform_matrix[2][3]
        z = -transform_matrix[0][3] # 右手

        # x = transform_matrix[1][3]
        # y = -transform_matrix[2][3]
        # z = transform_matrix[0][3]
        # 添加到坐标列表
        coordinates.append([x, y, z])
    
    # 转换为numpy数组以便处理
    coordinates = np.array(coordinates)
    
    # 获取手腕位置（0号点）
    wrist_pos = coordinates[0]
    
    # 将所有点相对于手腕位置进行变换（减去0号点）
    relative_coordinates = coordinates - wrist_pos
    
    # 使用可视化函数绘制手部姿态
    visualize_hand_tracking(relative_coordinates, ax)
    
    # 更新图形显示
    plt.pause(0.01)  # 短暂停顿以允许GUI更新

    