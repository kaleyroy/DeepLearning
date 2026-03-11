"""
经验回放缓冲区实现
包含标准回放缓冲区和优先经验回放缓冲区
"""

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    标准经验回放缓冲区
    
    功能：
    - 存储经验元组 (state, action, reward, next_state, done)
    - 随机采样一批经验用于训练
    - 打破样本相关性，提高数据效率
    """
    def __init__(self, capacity=10000):
        """
        初始化回放缓冲区
        
        Args:
            capacity: 缓冲区最大容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        存储一条经验
        
        Args:
            state: 当前状态
            action: 采取的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        随机采样一批经验
        
        Args:
            batch_size: 采样数量
        
        Returns:
            states, actions, rewards, next_states, dones
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        """返回缓冲区当前大小"""
        return len(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区（Prioritized Experience Replay）
    
    核心思想：
    - TD 误差大的经验更重要，应该有更高的采样概率
    - 使用 SumTree 实现高效的优先级采样
    
    优势：
    - 加速学习，特别是早期阶段
    - 更有效地利用重要经验
    """
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        初始化优先回放缓冲区
        
        Args:
            capacity: 缓冲区最大容量
            alpha: 优先级指数 (0=随机采样，1=完全按优先级)
            beta: 重要性采样权重初始值 (用于纠正偏差)
            beta_increment: beta 的增量 (逐步增加到 1)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        """
        存储一条经验，优先级设为最大值
        
        Args:
            state: 当前状态
            action: 采取的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
        """
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """
        按优先级采样一批经验
        
        Args:
            batch_size: 采样数量
        
        Returns:
            states, actions, rewards, next_states, dones, indices, weights
            (indices 和 weights 用于更新优先级)
        """
        # 计算采样概率
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # 按概率采样索引
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # 计算重要性采样权重 (纠正偏差)
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化
        
        # 增加 beta (逐步接近均匀采样)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 获取样本
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights
        )
    
    def update_priorities(self, indices, priorities):
        """
        更新指定经验的优先级
        
        Args:
            indices: 经验索引
            priorities: 新的优先级 (通常是 TD 误差)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        """返回缓冲区当前大小"""
        return self.size


# 测试代码
if __name__ == '__main__':
    print("测试标准回放缓冲区...")
    buffer = ReplayBuffer(capacity=1000)
    
    # 添加一些经验
    for i in range(100):
        state = np.random.randn(4)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = np.random.randint(0, 2)
        buffer.push(state, action, reward, next_state, done)
    
    print(f"缓冲区大小：{len(buffer)}")
    
    # 采样
    states, actions, rewards, next_states, dones = buffer.sample(32)
    print(f"采样形状：states={states.shape}, actions={actions.shape}")
    
    print("\n测试优先回放缓冲区...")
    pri_buffer = PrioritizedReplayBuffer(capacity=1000)
    
    # 添加经验
    for i in range(100):
        state = np.random.randn(4)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = np.random.randint(0, 2)
        pri_buffer.push(state, action, reward, next_state, done)
    
    print(f"缓冲区大小：{len(pri_buffer)}")
    
    # 采样
    states, actions, rewards, next_states, dones, indices, weights = pri_buffer.sample(32)
    print(f"采样形状：states={states.shape}, weights={weights.shape}")
    
    # 更新优先级
    td_errors = np.abs(np.random.randn(32))
    pri_buffer.update_priorities(indices, td_errors)
    
    print("\n✅ 所有回放缓冲区测试通过！")
