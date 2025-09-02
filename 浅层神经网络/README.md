# 浅层神经网络模块

本模块实现了一个浅层神经网络（Shallow Neural Network），用于解决非线性分类问题。该神经网络包含一个隐藏层，使用反向传播算法进行训练。

## 算法原理

### 神经网络结构

浅层神经网络由三层组成：
- **输入层**：接收原始特征数据
- **隐藏层**：使用非线性激活函数处理特征
- **输出层**：输出分类结果

### 前向传播

1. **隐藏层计算**：
   ```
   Z1 = X * W1 + b1
   A1 = activation(Z1)
   ```
   其中，activation可以是ReLU、Sigmoid或Tanh函数。

2. **输出层计算**：
   ```
   Z2 = A1 * W2 + b2
   A2 = softmax(Z2)
   ```
   Softmax函数将输出转换为概率分布。

### 反向传播

使用链式法则计算梯度：

1. **输出层梯度**：
   ```
   dZ2 = A2 - Y
   dW2 = A1^T * dZ2 / n + λ * W2
   db2 = ΣdZ2 / n
   ```

2. **隐藏层梯度**：
   ```
   dZ1 = (dZ2 * W2^T) ⊙ activation'(Z1)
   dW1 = X^T * dZ1 / n + λ * W1
   db1 = ΣdZ1 / n
   ```

### 参数更新

使用梯度下降算法更新参数：
```python
W = W - learning_rate * dW
b = b - learning_rate * db
```

## 文件结构

```
浅层神经网络/
├── README.md                    # 本文档
├── requirements.txt             # 依赖包列表
├── data_generation.py           # 数据生成模块
├── shallow_neural_network.py   # numpy实现的神经网络模型
├── train.py                     # numpy实现的训练脚本
├── pytorch_shallow_neural_network.py  # PyTorch实现的神经网络模型
├── pytorch_train.py             # PyTorch实现的训练脚本
├── test_pytorch_quick.py        # PyTorch快速测试脚本
├── data/                        # 数据文件夹
│   ├── X_train.npy             # 训练集特征
│   ├── X_val.npy               # 验证集特征
│   ├── X_test.npy              # 测试集特征
│   ├── y_train.npy             # 训练集标签
│   ├── y_val.npy               # 验证集标签
│   ├── y_test.npy              # 测试集标签
│   ├── dataset.json            # 完整数据集（JSON格式）
│   ├── data_visualization.png # 数据可视化图
│   ├── pytorch_data_visualization.png # PyTorch数据可视化图
│   └── ...                     # 其他数据文件
└── results/                     # 结果文件夹
    ├── loss_history.png        # numpy实现损失函数变化曲线
    ├── decision_boundary.png   # numpy实现决策边界图
    ├── confusion_matrix.png    # numpy实现混淆矩阵
    ├── model_params.npy        # numpy实现模型参数
    ├── performance_summary.json # numpy实现性能指标汇总
    ├── pytorch_training_history.png    # PyTorch实现训练历史曲线
    ├── pytorch_decision_boundary.png   # PyTorch实现决策边界图
    ├── pytorch_confusion_matrix.png    # PyTorch实现混淆矩阵
    ├── pytorch_model.pth        # PyTorch实现模型参数
    └── pytorch_performance_summary.json # PyTorch实现性能指标汇总
```

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行训练

#### NumPy实现
```bash
python train.py
```

#### PyTorch实现
```bash
python pytorch_train.py
```

#### 快速测试（PyTorch）
```bash
python test_pytorch_quick.py
```

### 3. 自定义训练参数

在 `train.py` 中，您可以修改以下参数：

```python
# 数据生成参数
data_gen = DataGenerator(
    n_samples=1000,    # 样本总数
    n_features=2,      # 特征数量
    n_classes=2,       # 类别数量
    random_state=42    # 随机种子
)

# 模型参数
model = ShallowNeuralNetwork(
    hidden_size=10,        # 隐藏层神经元数量
    learning_rate=0.01,    # 学习率
    n_iterations=2000,     # 训练迭代次数
    regularization=0.01,  # 正则化系数
    activation='relu'      # 激活函数类型
)
```

### 4. 单独使用各个模块

#### NumPy实现

##### 数据生成
```python
from data_generation import DataGenerator

# 创建数据生成器
data_gen = DataGenerator(n_samples=1000, n_features=2, n_classes=2)

# 生成非线性数据
X, y = data_gen.generate_nonlinear_data()

# 分割数据集
X_train, X_val, X_test, y_train, y_val, y_test = data_gen.split_and_save_data(X, y)
```

##### 模型训练
```python
from shallow_neural_network import ShallowNeuralNetwork

# 创建模型
model = ShallowNeuralNetwork(
    hidden_size=10,
    learning_rate=0.01,
    n_iterations=2000,
    activation='relu'
)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# 评估
metrics = model.evaluate(X_test, y_test)
print(f"准确率: {metrics['accuracy']:.4f}")
```

#### PyTorch实现

##### 数据生成（与NumPy实现相同）
```python
from data_generation import DataGenerator

# 创建数据生成器
data_gen = DataGenerator(random_state=42)

# 生成非线性数据
X, y = data_gen.generate_nonlinear_data()

# 分割数据集
X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test = data_gen.split_and_save_data(X, y)
```

##### 模型训练
```python
from pytorch_shallow_neural_network import PyTorchShallowNeuralNetwork, PyTorchNeuralNetworkTrainer, get_device

# 获取计算设备
device = get_device()

# 创建模型
model = PyTorchShallowNeuralNetwork(
    input_size=2,
    hidden_size=20,
    output_size=2,
    activation='relu'
)

# 创建训练器
trainer = PyTorchNeuralNetworkTrainer(
    model=model,
    learning_rate=0.01,
    weight_decay=0.001,
    device=device
)

# 训练模型
trainer.train(
    X_train=X_train_scaled,
    y_train=y_train,
    X_val=X_val_scaled,
    y_val=y_val,
    n_epochs=1000,
    batch_size=32,
    verbose=True
)

# 预测
y_pred = trainer.predict(X_test_scaled)
probabilities = trainer.predict_proba(X_test_scaled)

# 评估
metrics = trainer.get_metrics(X_test_scaled, y_test)
print(f"准确率: {metrics['accuracy']:.4f}")
```

## 激活函数选择

本模块支持三种激活函数：

### 1. ReLU (Rectified Linear Unit)
- **公式**：`ReLU(z) = max(0, z)`
- **特点**：计算简单，解决梯度消失问题
- **适用场景**：大多数情况下的首选

### 2. Sigmoid
- **公式**：`σ(z) = 1 / (1 + e^(-z))`
- **特点**：输出范围(0,1)，适合概率输出
- **适用场景**：二分类问题的输出层

### 3. Tanh
- **公式**：`tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))`
- **特点**：输出范围(-1,1)，均值为0
- **适用场景**：需要对称输出的隐藏层

## 参数调优建议

### 学习率 (learning_rate)
- **推荐范围**：0.001 - 0.1
- **过小**：训练速度慢
- **过大**：训练不稳定，可能发散

### 隐藏层神经元数量 (hidden_size)
- **推荐范围**：5 - 50
- **过少**：模型容量不足，欠拟合
- **过多**：计算成本高，可能过拟合

### 正则化系数 (regularization)
- **推荐范围**：0.001 - 0.1
- **过小**：正则化效果弱，可能过拟合
- **过大**：模型过于简单，欠拟合

### 训练迭代次数 (n_iterations / n_epochs)
- **推荐范围**：1000 - 5000
- **观察**：通过损失曲线判断是否收敛

## 实现对比

### NumPy实现 vs PyTorch实现

| 特性 | NumPy实现 | PyTorch实现 |
|------|-----------|-------------|
| 计算后端 | CPU | CPU/GPU |
| 自动求导 | 手动实现 | 自动 |
| 优化器 | 梯度下降 | Adam等 |
| 批量训练 | 全批量 | 小批量 |
| 正则化 | L2正则化 | L2正则化 + Dropout |
| 模型保存 | numpy数组 | PyTorch checkpoint |
| 训练速度 | 较慢 | 较快（GPU） |
| 代码复杂度 | 较高 | 较低 |
| 扩展性 | 有限 | 优秀 |

### 性能对比

在相同数据集上的性能表现：

| 指标 | NumPy实现 | PyTorch实现 |
|------|-----------|-------------|
| 验证集准确率 | 64.00% | 68.00% |
| 测试集准确率 | 54.50% | 67.50% |
| 训练时间 | 较短 | 较长（但支持GPU加速） |
| 模型容量 | 较小 | 较大 |

### 推荐使用场景

#### NumPy实现
- 学习神经网络原理
- 理解反向传播算法
- 资源受限环境
- 小规模数据集

#### PyTorch实现
- 生产环境部署
- 大规模数据集
- 需要GPU加速
- 复杂模型扩展
- 研究和实验

## 性能评估指标

本模块提供以下评估指标：

- **准确率 (Accuracy)**：正确预测的样本比例
- **精确率 (Precision)**：预测为正例中实际为正例的比例
- **召回率 (Recall)**：实际为正例中预测为正例的比例
- **F1分数 (F1 Score)**：精确率和召回率的调和平均

## 可视化结果

训练完成后，模块会生成以下可视化结果：

1. **损失函数变化曲线**：展示训练过程中损失值的变化
2. **决策边界图**：展示模型在特征空间中的决策边界
3. **混淆矩阵**：展示预测结果与真实标签的对应关系

## 注意事项

1. **数据标准化**：PyTorch实现会自动对数据进行标准化，NumPy实现也需要手动标准化
2. **设备选择**：PyTorch实现会自动检测并使用GPU（如果可用）
3. **随机种子**：为确保结果可重现，请设置相同的随机种子
4. **模型保存**：PyTorch模型保存为.pth文件，NumPy模型保存为.npy文件
5. **依赖包**：PyTorch实现需要安装额外的torch和torchvision包
6. **数值稳定性**：在计算中使用数值稳定的方法，避免溢出
7. **内存使用**：大数据集时注意内存使用情况

## 扩展功能

您可以通过以下方式扩展本模块：

1. **添加新的激活函数**：在 `ShallowNeuralNetwork` 类中添加新的激活函数方法
2. **实现其他优化算法**：如Adam、RMSprop等
3. **添加早停机制**：根据验证集性能提前停止训练
4. **实现批量归一化**：提高训练稳定性
5. **支持更多损失函数**：如均方误差、交叉熵等

## 参考资源

- [神经网络基础教程](https://www.coursera.org/learn/neural-networks-deep-learning)
- [深度学习书籍](https://www.deeplearningbook.org/)
- [Scikit-learn文档](https://scikit-learn.org/stable/)

## 作者信息

本模块为深度学习学习和研究而开发，欢迎提出改进建议和问题反馈。