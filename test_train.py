import gymnasium
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from test import UAVSecureEnv

def main():
    # 1. 创建环境实例
    env = UAVSecureEnv(num_waypoints=2)

    # 2. 初始化 PPO 模型，使用多层感知机策略 ("MlpPolicy")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=7e-4, gamma=0.98)

    # 3. 训练模型，设置总 timesteps 数
    total_timesteps = 2000000  # 根据任务复杂度可以调整
    model.learn(total_timesteps=total_timesteps)

    # 4. 测试训练结果，执行一个回合并渲染轨迹
    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.render()


if __name__ == '__main__':
    main()