# 线性分类模型

本项目实现了一个完整的线性分类模型，使用逻辑回归算法进行二分类和多分类任务。项目包含数据生成、模型训练、验证和测试的完整流程。

## 项目结构

```
线性分类/
├── data_generation.py      # 数据生成模块
├── linear_classifier.py    # 线性分类器核心实现
├── train.py               # 训练和评估脚本
├── requirements.txt       # 依赖包列表
├── data/                  # 数据文件夹（自动生成）
│   ├── X_train.npy       # 训练集特征 (npy格式)
│   ├── X_val.npy         # 验证集特征 (npy格式)
│   ├── X_test.npy        # 测试集特征 (npy格式)
│   ├── y_train.npy       # 训练集标签 (npy格式)
│   ├── y_val.npy         # 验证集标签 (npy格式)
│   ├── y_test.npy        # 测试集标签 (npy格式)
│   ├── X_train_raw.npy   # 原始训练集特征 (未标准化)
│   ├── X_val_raw.npy     # 原始验证集特征 (未标准化)
│   ├── X_test_raw.npy    # 原始测试集特征 (未标准化)
│   ├── dataset.json      # 完整数据集 (JSON格式)
│   ├── train_data.json   # 训练集数据 (JSON格式)
│   ├── val_data.json     # 验证集数据 (JSON格式)
│   ├── test_data.json    # 测试集数据 (JSON格式)
│   └── data_visualization.png  # 数据可视化图
└── results/              # 结果文件夹（自动生成）
    ├── loss_history.png          # 损失函数变化曲线
    ├── decision_boundary.png     # 决策边界图
    ├── confusion_matrix.png     # 混淆矩阵
    ├── model_params.npy         # 模型参数
    └── performance_summary.json # 性能指标汇总
```

## 算法原理

### 逻辑回归

逻辑回归是一种经典的线性分类算法，其核心思想是：

1. **线性组合**：将输入特征与权重进行线性组合
   \[ z = w^T x + b \]
   其中，$w$ 是权重向量，$x$ 是特征向量，$b$ 是偏置项

2. **激活函数**：使用Sigmoid函数将线性组合的结果映射到概率空间
   \[ \sigma(z) = \frac{1}{1 + e^{-z}} \]
   对于二分类问题，输出值表示样本属于正类的概率

3. **损失函数**：使用交叉熵损失函数
   \[ L = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(p_i) + (1-y_i) \log(1-p_i)] + \frac{\lambda}{2} ||w||^2 \]
   其中，$y_i$ 是真实标签，$p_i$ 是预测概率，$\lambda$ 是正则化系数

4. **梯度下降**：通过计算梯度并更新参数来最小化损失函数
   - 权重更新：$w = w - \eta \cdot \frac{\partial L}{\partial w}$
   - 偏置更新：$b = b - \eta \cdot \frac{\partial L}{\partial b}$
   其中，$\eta$ 是学习率

### 多分类扩展

对于多分类问题，使用Softmax函数扩展逻辑回归：

\[ \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} \]

其中，$K$ 是类别数量，$z_i$ 是第$i$类的线性组合结果。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 运行训练脚本

```bash
python train.py
```

这将执行以下步骤：
1. 生成训练、验证和测试数据集
2. 训练线性分类模型
3. 在验证集和测试集上评估模型性能
4. 生成可视化结果（损失曲线、决策边界、混淆矩阵）
5. 保存模型参数和性能指标

### 2. 自定义参数

可以在 `train.py` 中修改以下参数：

```python
# 数据生成参数
data_gen = DataGenerator(
    n_samples=1000,     # 样本总数
    n_features=2,        # 特征数量
    n_classes=2,        # 类别数量
    random_state=42      # 随机种子
)

# 模型参数
classifier = LinearClassifier(
    learning_rate=0.1,      # 学习率
    n_iterations=1000,      # 迭代次数
    regularization=0.01,   # 正则化系数
    random_state=42         # 随机种子
)
```

### 3. 多分类测试

要测试多分类功能，取消 `train.py` 中最后一行的注释：

```python
# 取消注释以运行多分类测试
test_multiclass_classification()
```

## 输出结果

### 1. 性能指标

程序会输出以下性能指标：
- **准确率 (Accuracy)**：正确分类的样本比例
- **精确率 (Precision)**：预测为正类的样本中实际为正类的比例
- **召回率 (Recall)**：实际为正类的样本中被正确预测的比例
- **F1分数 (F1 Score)**：精确率和召回率的调和平均数

### 2. 可视化结果

- **数据可视化图**：显示生成的数据分布
- **损失函数变化曲线**：展示训练过程中损失函数的变化
- **决策边界图**：显示分类器的决策边界（仅适用于2维特征）
- **混淆矩阵**：显示分类结果的详细统计

### 3. 数据文件格式

项目支持两种数据文件格式：

**NPY格式 (NumPy)**
- `X_*.npy`：特征数据，已标准化处理
- `y_*.npy`：标签数据
- `X_*_raw.npy`：原始特征数据，未标准化
- 优点：加载速度快，适合Python环境使用

**JSON格式**
- `dataset.json`：包含完整数据集信息和元数据
- `train_data.json`：训练集数据（标准化+原始）
- `val_data.json`：验证集数据（标准化+原始）
- `test_data.json`：测试集数据（标准化+原始）
- 优点：跨平台兼容，易于人工阅读，支持其他编程语言

### 4. 模型文件

- **model_params.npy**：保存训练好的模型参数
- **performance_summary.json**：保存详细的性能指标

## 核心代码说明

### DataGenerator 类

负责生成和预处理数据：
- `generate_linear_separable_data()`：生成线性可分的数据集
- `split_and_save_data()`：分割数据集并保存到文件
- `visualize_data()`：可视化生成的数据

### LinearClassifier 类

线性分类器的核心实现：
- `_sigmoid()`：Sigmoid激活函数
- `_softmax()`：Softmax激活函数（多分类）
- `_compute_loss_binary()`：二分类交叉熵损失
- `_compute_loss_multiclass()`：多分类交叉熵损失
- `fit()`：训练模型
- `predict()`：预测类别
- `evaluate()`：评估模型性能
- `plot_decision_boundary()`：绘制决策边界
- `plot_confusion_matrix()`：绘制混淆矩阵

## 算法特点

### 优点
1. **简单高效**：算法简单，计算效率高
2. **可解释性强**：权重向量可以反映特征的重要性
3. **概率输出**：提供类别概率，便于后续处理
4. **正则化支持**：内置L2正则化，防止过拟合

### 缺点
1. **线性假设**：只能处理线性可分的问题
2. **特征依赖**：对特征的缩放敏感，需要标准化处理
3. **类别不平衡**：在类别不平衡的情况下性能可能下降

## 扩展建议

1. **特征工程**：添加多项式特征以处理非线性问题
2. **优化算法**：实现更高级的优化算法（如Adam、RMSprop）
3. **正则化**：尝试L1正则化或ElasticNet
4. **交叉验证**：实现K折交叉验证以获得更稳定的性能评估
5. **超参数调优**：添加网格搜索或随机搜索来自动调优超参数

## 注意事项

1. 确保所有依赖包已正确安装
2. 数据文件夹和结果文件夹会自动创建
3. 决策边界可视化仅适用于2维特征
4. 对于大规模数据集，可能需要调整学习率和迭代次数
5. 建议在运行前备份重要数据

## 作者

本项目为深度学习学习项目，用于理解和实现线性分类算法。