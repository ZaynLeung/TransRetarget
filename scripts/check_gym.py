import gym
import yumi_gym
import time
import pybullet as p

def test_gym_environment():
    """
    简单测试gym环境是否正常
    """
    print("开始测试gym环境...")
    
    try:
        # 创建环境
        env = gym.make('yumi-v0')
        print("环境创建成功")
        
        # 重置环境
        observation = env.reset()
        # 获取动作空间信息
        action_space = env.action_space
        print(f"动作空间: {action_space}")
        
        # 简单测试几个步骤
        for step in range(10):
            # 随机采样一个动作
            action = action_space.sample()
            
            # 执行动作
            observation, reward, done, info = env.step(action)
            
            if done:
                print("环境提前结束")
                break
            time.sleep(0.1)  # 短暂延时以便观察
        
        # 关闭环境
        env.close()
        print("环境测试完成")
        
    except Exception as e:
        print(f"环境测试失败: {e}")

if __name__ == "__main__":
    test_gym_environment()