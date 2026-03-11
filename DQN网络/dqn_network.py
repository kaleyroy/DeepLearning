"""
DQN 网络结构定义
包含标准 DQN 和 Dueling DQN 两种网络架构
"""

import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    标准 DQN 网络：输入状态，输出每个动作的 Q 值
    
    网络结构：
    Input -> Linear(128) -> ReLU -> Linear(128) -> ReLU -> Linear(action_dim)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入状态张量 [batch_size, state_dim]
        
        Returns:
            Q 值张量 [batch_size, action_dim]
        """
        return self.network(x)


class DuelingDQN(nn.Module):
    """
    Dueling DQN 网络：分离状态价值和动作优势
    
    优势：
    - 更好地学习状态价值，无需学习每个动作的价值
    - 在动作价值差异小的场景中表现更好
    
    网络结构：
    Input -> Shared Layers -> Value Stream -> V(s)
                          -> Advantage Stream -> A(s,a)
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()
        
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 价值流：输出状态价值 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 优势流：输出动作优势 A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入状态张量 [batch_size, state_dim]
        
        Returns:
            Q 值张量 [batch_size, action_dim]
        """
        # 共享特征
        shared = self.shared(x)
        
        # 计算状态价值
        value = self.value_stream(shared)  # [batch_size, 1]
        
        # 计算动作优势
        advantage = self.advantage_stream(shared)  # [batch_size, action_dim]
        
        # 合并：Q = V + (A - mean(A))
        # 减去均值确保优势的定义一致性
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class CNN_DQN(nn.Module):
    """
    卷积 DQN 网络：用于处理图像输入（如 Atari 游戏）
    
    参考 DeepMind 原始论文架构
    """
    def __init__(self, action_dim, input_channels=4):
        super(CNN_DQN, self).__init__()
        
        # 卷积层提取空间特征
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 计算卷积输出维度（假设输入 84x84）
        # 84 -> 20 -> 9 -> 7
        self.conv_output_dim = 64 * 7 * 7
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像张量 [batch_size, channels, height, width]
        
        Returns:
            Q 值张量 [batch_size, action_dim]
        """
        # 卷积特征提取
        features = self.conv_layers(x)
        
        # 展平
        features = features.view(features.size(0), -1)
        
        # 全连接输出 Q 值
        return self.fc_layers(features)


# 测试代码
if __name__ == '__main__':
    # 测试标准 DQN
    print("测试标准 DQN...")
    dqn = DQN(state_dim=4, action_dim=2)
    x = torch.randn(8, 4)  # batch_size=8, state_dim=4
    output = dqn(x)
    print(f"输入形状：{x.shape}, 输出形状：{output.shape}")
    
    # 测试 Dueling DQN
    print("\n测试 Dueling DQN...")
    dueling_dqn = DuelingDQN(state_dim=4, action_dim=2)
    output = dueling_dqn(x)
    print(f"输入形状：{x.shape}, 输出形状：{output.shape}")
    
    # 测试 CNN DQN
    print("\n测试 CNN DQN...")
    cnn_dqn = CNN_DQN(action_dim=6, input_channels=4)
    x_img = torch.randn(8, 4, 84, 84)  # batch_size=8, 4 帧 84x84 图像
    output = cnn_dqn(x_img)
    print(f"输入形状：{x_img.shape}, 输出形状：{output.shape}")
    
    print("\n✅ 所有网络测试通过！")
