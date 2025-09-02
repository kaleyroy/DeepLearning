# 线性回归模型实现

本项目实现了完整的线性回归模型，包括数据生成、模型训练和性能验证。

## 文件结构

- `data_generation.py`: 数据生成模块，创建训练、验证和测试数据集
- `linear_regression.py`: 线性回归模型核心实现（批量梯度下降）
- `minibatch_linear_regression.py`: 小批量梯度下降线性回归实现
- `train.py`: 批量梯度下降模型训练脚本
- `train_minibatch.py`: 小批量梯度下降模型训练脚本
- `evaluate.py`: 模型评估脚本
- `data_query_tool.py`: 数据查询工具，用于查询和分析JSON格式数据集
- `requirements.txt`: 项目依赖

## 数据集格式

本项目支持多种数据格式，便于不同场景的使用：

### 1. NPY格式（Python原生）
- **用途**: 适合Python程序快速加载和数值计算
- **文件**: `data/X_train.npy`, `data/y_train.npy`, `data/X_val.npy`, `data/y_val.npy`, `data/X_test.npy`, `data/y_test.npy`
- **特点**: 加载速度快，适合机器学习训练

### 2. JSON格式（通用交换格式）
- **用途**: 便于跨语言使用、实际查询和人类阅读
- **文件列表**:
  - `data/dataset.json`: 完整数据集（包含元数据）
  - `data/train_data.json`: 训练集（每条数据包含id, X, y）
  - `data/val_data.json`: 验证集
  - `data/test_data.json`: 测试集

#### JSON数据结构示例：
```json
{
  "metadata": {
    "description": "线性回归数据集",
    "true_weight": 2.5,
    "true_bias": 1.0,
    "noise_level": 0.3
  },
  "train": {
    "X": [0.1, 0.5, 1.2, ...],
    "y": [1.25, 2.25, 4.0, ...]
  }
}
```

#### 单条数据格式：
```json
{
  "id": 0,
  "X": 0.1234,
  "y": 1.3085
}
```

## 算法原理

线性回归是一种监督学习算法，用于建立自变量X和因变量y之间的线性关系：

y = wX + b

其中：
- w是权重向量
- b是偏置项

目标是最小化均方误差损失函数：

L = (1/2n) * Σ(y_pred - y_true)²

使用梯度下降法优化参数：
- w = w - α * ∂L/∂w
- b = b - α * ∂L/∂b

其中α是学习率。

## 小批量梯度下降算法

### 算法概述

小批量梯度下降（Mini-Batch Gradient Descent）是批量梯度下降和随机梯度下降的折中方案，在计算效率和收敛稳定性之间取得了良好的平衡。

### 核心概念

1. **Epoch（时代）**: 所有训练数据完整地通过模型一次
2. **Batch（批量）**: 一个epoch中用于计算梯度的一小部分数据
3. **Batch Size（批量大小）**: 每个batch包含的样本数量
4. **Iteration（迭代）**: 总迭代次数 = epochs × (n_samples // batch_size)

### 算法优势

- **计算效率**: 相比批量梯度下降，每次迭代计算量更小
- **收敛稳定性**: 相比随机梯度下降，梯度估计更稳定
- **内存友好**: 可以处理大规模数据集，不需要一次性加载所有数据
- **并行化**: 可以利用GPU并行计算小批量数据

### 算法步骤

```
1. 初始化参数 w, b
2. 对于每个 epoch:
   a. 将数据随机打乱
   b. 将数据分割为多个小批量
   c. 对于每个小批量:
      i.   前向传播：y_pred = X_batch * w + b
      ii.  计算损失：L = (1/2m) * Σ(y_pred - y_true)²
      iii. 计算梯度：
           ∂L/∂w = (1/m) * Σ_batch(y_pred - y_true) * X
           ∂L/∂b = (1/m) * Σ_batch(y_pred - y_true)
      iv.  更新参数：
           w = w - α * ∂L/∂w
           b = b - α * ∂L/∂b
3. 直到达到最大epoch数或收敛
```

### 使用方法

#### 基本使用

```python
from minibatch_linear_regression import MiniBatchGD

# 创建模型
model = MiniBatchGD(
    learning_rate=0.01,    # 学习率
    epochs=100,           # 训练轮数
    batch_size=32,        # 批量大小
    random_state=42       # 随机种子
)

# 训练模型
model.fit(X_train, y_train, X_val, y_val, verbose=True)

# 预测
predictions = model.predict(X_test)

# 评估
r2_score = model.calculate_r2_score(X_test, y_test)
```

#### 高级功能

```python
# 启用早停机制
model.fit(X_train, y_train, X_val, y_val, 
         early_stopping=True, patience=10)

# 可视化训练历史
model.plot_training_history()

# 获取训练统计信息
stats = model.get_training_stats()
print(f"总迭代次数: {stats['total_iterations']}")
print(f"最佳验证损失: {stats['best_val_loss']}")
```

### 批量大小选择建议

| 批量大小 | 特点 | 适用场景 |
|----------|------|----------|
| 8-16 | 小批量，梯度噪声大，收敛不稳定 | 小数据集，需要强正则化 |
| 32-64 | 标准选择，平衡稳定性和效率 | 大多数场景 |
| 128-256 | 大批量，梯度稳定，内存占用高 | 大数据集，GPU训练 |
| 512+ | 接近批量梯度下降 | 数据量极大，需要精确梯度 |

### 性能对比

| 方法 | 计算效率 | 收敛稳定性 | 内存使用 | 适用数据规模 |
|------|----------|------------|----------|--------------|
| 批量梯度下降 | 低 | 高 | 高 | 小数据集 |
| 小批量梯度下降 | 中 | 中 | 中 | 中大数据集 |
| 随机梯度下降 | 高 | 低 | 低 | 大数据集 |

### 运行训练脚本

```bash
# 训练小批量梯度下降模型
python train_minibatch.py
```

该脚本将：
1. 测试不同配置的小批量模型
2. 自动选择最佳模型
3. 生成详细的性能对比图表
4. 保存训练结果到 `minibatch_results/` 目录

### 结果文件

训练完成后，`minibatch_results/` 目录将包含：
- `best_model_weights.npy`: 最佳模型权重
- `best_model_bias.npy`: 最佳模型偏置
- `best_epoch_loss.npy`: Epoch损失历史
- `best_batch_loss.npy`: Batch损失历史
- `performance_summary.json`: 性能指标汇总
- `training_report.txt`: 详细训练报告
- `model_comparison.png`: 模型对比图表
- `gd_methods_comparison.png`: 梯度下降方法对比图表

## 数据查询工具

### 功能特点

`data_query_tool.py` 提供了强大的数据查询和分析功能：

1. **数据加载**: 自动加载JSON格式的数据集
2. **数据查询**: 
   - 按ID查询特定数据点
   - 按数值范围查询数据
3. **统计分析**: 计算各数据集的统计信息
4. **数据可视化**: 生成数据分布图表
5. **数据导出**: 支持导出为CSV格式
6. **报告生成**: 自动生成完整的数据分析报告

### 使用方法

```python
from data_query_tool import DataQueryTool

# 创建查询工具实例
query_tool = DataQueryTool()

# 加载数据
query_tool.load_data()

# 查看数据集信息
query_tool.get_dataset_info()

# 按ID查询数据
data_point = query_tool.query_data_by_id('train', 0)

# 按范围查询数据
filtered_data = query_tool.query_data_by_range('train', x_min=0, x_max=5)

# 获取统计信息
stats = query_tool.get_statistics('train')

# 可视化数据
query_tool.plot_data('train')

# 生成完整报告
query_tool.generate_report()

# 导出为CSV
query_tool.export_to_csv('train', 'my_train_data.csv')
```

### 运行演示

```bash
python data_query_tool.py
```

这将自动加载数据并生成完整的数据分析报告，包括统计信息和可视化图表。