"""
DQN 智能体实现
包含标准 DQN、Double DQN、Dueling DQN 等多种变体
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from dqn_network import DQN, DuelingDQN
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent:
    """
    DQN 智能体
    
    功能：
    - 管理 Q 网络和目标网络
    - 实现 ε-greedy 探索策略
    - 经验回放和训练
    - 模型保存/加载
    """
    def __init__(self, state_dim, action_dim, config=None):
        """
        初始化 DQN 智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            config: 配置字典
        """
        # 默认配置
        self.config = {
            'lr': 0.001,              # 学习率
            'gamma': 0.99,            # 折扣因子
            'epsilon': 1.0,           # 初始探索率
            'epsilon_min': 0.01,      # 最小探索率
            'epsilon_decay': 0.995,   # 探索率衰减
            'tau': 0.001,             # 目标网络软更新系数
            'batch_size': 64,         # 批次大小
            'buffer_size': 10000,     # 回放缓冲区大小
            'update_target_every': 1, # 目标网络更新频率
            'use_double_dqn': False,  # 是否使用 Double DQN
            'use_dueling': False,     # 是否使用 Dueling DQN
            'use_per': False,         # 是否使用优先经验回放
            'grad_clip': 1.0,         # 梯度裁剪阈值
        }
        if config:
            self.config.update(config)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 设备选择
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备：{self.device}")
        
        # 初始化网络
        if self.config['use_dueling']:
            print("使用 Dueling DQN 架构")
            self.q_network = DuelingDQN(state_dim, action_dim).to(self.device)
        else:
            print("使用标准 DQN 架构")
            self.q_network = DQN(state_dim, action_dim).to(self.device)
        
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config['lr'])
        
        # 经验回放
        if self.config['use_per']:
            print("使用优先经验回放")
            self.replay_buffer = PrioritizedReplayBuffer(self.config['buffer_size'])
        else:
            self.replay_buffer = ReplayBuffer(self.config['buffer_size'])
        
        # 训练统计
        self.steps = 0
        self.episode_count = 0
    
    def select_action(self, state, training=True):
        """
        ε-greedy 策略选择动作
        
        Args:
            state: 当前状态
            training: 是否训练模式（训练时使用探索）
        
        Returns:
            选择的动作索引
        """
        if training and random.random() < self.config['epsilon']:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        存储经验到回放缓冲区
        
        Args:
            state: 当前状态
            action: 采取的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train(self):
        """
        训练一步：采样经验，计算损失，更新网络
        
        Returns:
            损失值（如果无法训练则返回 None）
        """
        if len(self.replay_buffer) < self.config['batch_size']:
            return None
        
        # 采样一批经验
        if self.config['use_per']:
            states, actions, rewards, next_states, dones, indices, weights = \
                self.replay_buffer.sample(self.config['batch_size'])
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.config['batch_size'])
            weights = torch.ones(self.config['batch_size']).to(self.device)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前 Q 值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标 Q 值（使用目标网络）
        with torch.no_grad():
            if self.config['use_double_dqn']:
                # Double DQN: 用主网络选动作，目标网络评估 Q 值
                next_action = self.q_network(next_states).max(1)[1]
                next_q = self.target_network(next_states).gather(1, next_action.unsqueeze(1))
            else:
                # 标准 DQN: 目标网络直接选最大 Q 值
                next_q = self.target_network(next_states).max(1)[0]
            
            target_q = rewards + (1 - dones) * self.config['gamma'] * next_q
        
        # 计算 TD 误差（用于优先经验回放更新）
        td_errors = (current_q.squeeze() - target_q).abs().detach().cpu().numpy()
        
        # 计算加权损失
        loss = (weights * (current_q.squeeze() - target_q) ** 2).mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            max_norm=self.config['grad_clip']
        )
        
        self.optimizer.step()
        
        # 更新优先经验回放的优先级
        if self.config['use_per']:
            self.replay_buffer.update_priorities(indices, td_errors + 1e-6)
        
        # 更新目标网络
        if self.steps % self.config['update_target_every'] == 0:
            self._soft_update_target()
        
        # 衰减探索率
        self.config['epsilon'] = max(
            self.config['epsilon_min'],
            self.config['epsilon'] * self.config['epsilon_decay']
        )
        
        self.steps += 1
        
        return loss.item()
    
    def _soft_update_target(self):
        """
        软更新目标网络参数
        Q_target = τ × Q_online + (1-τ) × Q_target
        """
        for target_param, online_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.config['tau'] * online_param.data +
                (1 - self.config['tau']) * target_param.data
            )
    
    def _hard_update_target(self):
        """
        硬更新目标网络参数（完全复制）
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'steps': self.steps,
            'episode_count': self.episode_count
        }, path)
        print(f"模型已保存到：{path}")
    
    def load(self, path):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.config = checkpoint['config']
        self.steps = checkpoint['steps']
        self.episode_count = checkpoint['episode_count']
        print(f"模型已加载：{path}")


# 测试代码
if __name__ == '__main__':
    print("测试 DQN 智能体...")
    
    # 创建智能体
    agent = DQNAgent(
        state_dim=4,
        action_dim=2,
        config={
            'lr': 0.001,
            'epsilon': 0.9,
            'epsilon_decay': 0.99,
            'use_double_dqn': True,
        }
    )
    
    # 模拟一些经验
    for i in range(100):
        state = np.random.randn(4)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = np.random.randint(0, 2)
        
        agent.store_transition(state, action, reward, next_state, done)
        
        # 训练
        loss = agent.train()
        if loss:
            print(f"Step {i+1}: Loss = {loss:.4f}, Epsilon = {agent.config['epsilon']:.3f}")
    
    # 保存模型
    agent.save('test_dqn.pth')
    
    # 加载模型
    agent.load('test_dqn.pth')
    
    print("\n✅ DQN 智能体测试通过！")
