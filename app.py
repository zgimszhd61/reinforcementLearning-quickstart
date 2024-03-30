#!pip install gym
import numpy as np
import gym
import random

# 创建环境
env = gym.make("FrozenLake-v1")

# 初始化Q表
action_size = env.action_space.n
state_size = env.observation_space.n
q_table = np.zeros((state_size, action_size))

# 设置参数
total_episodes = 10000        # 总迭代次数
learning_rate = 0.8           # 学习率
max_steps = 99                # 每个回合的最大步数
gamma = 0.95                  # 折扣因子

# 探索参数
epsilon = 1.0                 # 探索率
max_epsilon = 1.0             # 探索概率在开始时的最大值
min_epsilon = 0.01            # 最小探索概率
decay_rate = 0.005            # 探索概率的衰减率

# Q-Learning算法的实现
for episode in range(total_episodes):
    # 重置环境
    state = env.reset()
    step = 0
    done = False

    for step in range(max_steps):
        # 选择一个动作
        exp_exp_tradeoff = random.uniform(0, 1)

        # 如果这个数字大于epsilon --> 利用（选择最大Q值的动作）
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(q_table[state,:])

        # 否则 --> 探索
        else:
            action = env.action_space.sample()

        # 采取动作并观察结果
        new_state, reward, done, info = env.step(action)

        # 更新Q表
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])

        # 更新状态
        state = new_state

        # 如果游戏结束，则结束回合
        if done == True:
            break

    # 减少epsilon（因为我们需要越来越少的探索）
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

# 查看训练后的Q表
print(q_table)
