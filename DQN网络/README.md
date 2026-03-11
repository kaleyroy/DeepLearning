# DQN 深度强化学习项目

基于 PyTorch 的 DQN（Deep Q-Network）实现，包含多种变体和完整训练流程。

---

## 📁 项目结构

```
dqn_project/
├── dqn_network.py      # DQN 网络结构（标准/Dueling/CNN）
├── replay_buffer.py    # 经验回放缓冲区（标准/优先）
├── dqn_agent.py        # DQN 智能体实现
├── train.py            # 训练脚本
├── requirements.txt    # 依赖列表
└── README.md           # 说明文档
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 基础训练（CartPole 环境）
python train.py --env CartPole-v1 --episodes 500

# 使用 Double DQN
python train.py --env CartPole-v1 --double-dqn

# 使用 Dueling DQN
python train.py --env CartPole-v1 --dueling

# 使用优先经验回放
python train.py --env CartPole-v1 --per

# 组合使用所有改进
python train.py --env CartPole-v1 --double-dqn --dueling --per
```

### 3. 测试模型

```bash
# 测试训练好的模型
python train.py --env CartPole-v1 --test ./models/dqn_CartPole-v1_final.pth
```

### 4. 关闭渲染（服务器环境）

```bash
python train.py --env CartPole-v1 --no-render
```

---

## 📊 网络架构

### 标准 DQN
```
Input(state_dim) -> Linear(128) -> ReLU -> Linear(128) -> ReLU -> Linear(action_dim)
```

### Dueling DQN
```
Input -> Shared Layers -> Value Stream -> V(s)
                     -> Advantage Stream -> A(s,a)
Q(s,a) = V(s) + (A(s,a) - mean(A))
```

### CNN DQN（用于图像输入）
```
Input(84x84x4) -> Conv(32,8,4) -> Conv(64,4,2) -> Conv(64,3,1) -> FC(512) -> Output
```

---

## ⚙️ 配置参数

在 `dqn_agent.py` 中可调整以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr` | 0.001 | 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `epsilon` | 1.0 | 初始探索率 |
| `epsilon_min` | 0.01 | 最小探索率 |
| `epsilon_decay` | 0.995 | 探索率衰减 |
| `tau` | 0.001 | 目标网络软更新系数 |
| `batch_size` | 64 | 批次大小 |
| `buffer_size` | 10000 | 回放缓冲区大小 |
| `grad_clip` | 1.0 | 梯度裁剪阈值 |

---

## 🎮 支持的环境

### Classic Control
- `CartPole-v1` - 平衡杆（推荐入门）
- `MountainCar-v0` - 爬山车
- `Acrobot-v1` - 双杆机器人

### Atari Games（需额外安装）
```bash
pip install gymnasium[atari] gymnasium[accept-rom-license]
```

- `ALE/Breakout-v5`
- `ALE/Pong-v5`
- `ALE/SpaceInvaders-v5`

---

## 📈 训练技巧

### 1. 学习率调整
```python
# 如果训练不稳定，降低学习率
config['lr'] = 0.0001
```

### 2. 探索率调整
```python
# 如果需要更多探索
config['epsilon_decay'] = 0.999  # 减慢衰减
config['epsilon_min'] = 0.1      # 提高最小值
```

### 3. 网络容量
```python
# 对于复杂环境，增加隐藏层维度
# 修改 dqn_network.py 中的 hidden_dim 参数
```

---

## 🔬 算法变体对比

| 变体 | 优势 | 适用场景 |
|------|------|----------|
| **DQN** | 基础版本 | 学习入门 |
| **Double DQN** | 解决 Q 值高估 | 通用改进 |
| **Dueling DQN** | 更好学习状态价值 | 动作价值差异小 |
| **PER** | 加速收敛 | 稀疏奖励环境 |
| **Rainbow** | 以上所有组合 | SOTA 性能 |

---

## 📝 输出示例

训练输出：
```
============================================================
开始训练 DQN
环境：CartPole-v1
状态维度：4, 动作维度：2
============================================================

Episode 10/500 | 平均奖励：45.20 | 平均长度：45.2 | ε: 0.951 | 损失：0.5234
Episode 20/500 | 平均奖励：78.50 | 平均长度：78.5 | ε: 0.904 | 损失：0.3421
...
✅ 环境已解决！用了 156 集
最后 100 集平均奖励：476.32
```

---

## 🐛 常见问题

### Q: 训练不收敛怎么办？
- 降低学习率（尝试 0.0001）
- 增加目标网络更新间隔
- 确保状态归一化
- 增加回放缓冲区大小

### Q: CUDA out of memory？
- 减小 batch_size
- 减小网络隐藏层维度
- 使用 CPU 训练

### Q: 如何自定义环境？
```python
# 创建自定义环境
import gymnasium as gym
env = gym.make('YourCustomEnv-v0')

# 获取维度
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 创建智能体
agent = DQNAgent(state_dim, action_dim)
```

---

## 📚 参考资料

- [Playing Atari with Deep Reinforcement Learning (2013)](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep reinforcement learning (Nature, 2015)](https://www.nature.com/articles/nature14236)
- [Double DQN (2015)](https://arxiv.org/abs/1509.06461)
- [Dueling DQN (2015)](https://arxiv.org/abs/1511.06581)
- [Prioritized Experience Replay (2015)](https://arxiv.org/abs/1511.05952)

---

## 📄 许可证

MIT License

---

**祝训练顺利！🚀**
