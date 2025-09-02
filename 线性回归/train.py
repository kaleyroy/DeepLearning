import numpy as np
import matplotlib.pyplot as plt
from data_generation import DataGenerator
from linear_regression import LinearRegression
import os

def train_model():
    """
    完整的模型训练流程
    
    算法步骤：
    1. 生成和准备训练数据
    2. 初始化并训练线性回归模型
    3. 评估训练效果
    4. 可视化训练过程和结果
    5. 保存模型参数
    """
    print("=== 线性回归模型训练开始 ===")
    
    # 1. 数据准备
    print("\n1. 数据生成和预处理...")
    
    # 创建数据生成器
    # 参数说明：
    # - n_samples=1000: 生成1000个样本点
    # - noise=0.3: 添加标准差为0.3的高斯噪声
    # - random_state=42: 设置随机种子确保结果可复现
    data_gen = DataGenerator(n_samples=1000, noise=0.3, random_state=42)
    
    # 生成数据集
    # 算法原理：y = w*X + b + noise，其中w=2.5, b=1.0
    X, y = data_gen.generate_data()
    print(f"生成数据集大小: {X.shape[0]} 个样本")
    print(f"真实参数: weight={data_gen.true_weight}, bias={data_gen.true_bias}")
    
    # 划分数据集
    # 训练集70%，验证集15%，测试集15%
    # 这种划分确保模型有足够数据训练，同时有独立数据用于验证和测试
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_gen.split_data(
        X, y, train_ratio=0.7, val_ratio=0.15
    )
    
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 2. 模型训练
    print("\n2. 模型训练...")
    
    # 创建线性回归模型
    # 参数说明：
    # - learning_rate=0.01: 学习率，控制梯度下降的步长
    # - n_iterations=2000: 最大迭代次数
    model = LinearRegression(learning_rate=0.01, n_iterations=2000)
    
    # 训练模型
    # 算法原理：使用梯度下降法最小化均方误差
    # 在每次迭代中：
    # 1. 计算预测值 y_pred = X*w + b
    # 2. 计算损失 L = (1/2n)*Σ(y_pred-y_true)²
    # 3. 计算梯度 ∂L/∂w 和 ∂L/∂b
    # 4. 更新参数 w = w - α*∂L/∂w, b = b - α*∂L/∂b
    print("开始训练模型...")
    model.fit(X_train, y_train, verbose=True)
    
    # 3. 模型评估
    print("\n3. 模型评估...")
    
    # 获取学习到的参数
    learned_weights, learned_bias = model.get_parameters()
    print(f"学习到的权重: {learned_weights[0]:.4f} (真实值: {data_gen.true_weight})")
    print(f"学习到的偏置: {learned_bias:.4f} (真实值: {data_gen.true_bias})")
    
    # 计算各数据集上的性能指标
    # 训练集性能
    train_loss = model.calculate_loss(X_train, y_train)
    train_r2 = model.calculate_r2_score(X_train, y_train)
    print(f"\n训练集性能:")
    print(f"  损失值: {train_loss:.6f}")
    print(f"  R²决定系数: {train_r2:.4f}")
    
    # 验证集性能
    val_loss = model.calculate_loss(X_val, y_val)
    val_r2 = model.calculate_r2_score(X_val, y_val)
    print(f"\n验证集性能:")
    print(f"  损失值: {val_loss:.6f}")
    print(f"  R²决定系数: {val_r2:.4f}")
    
    # 测试集性能
    test_loss = model.calculate_loss(X_test, y_test)
    test_r2 = model.calculate_r2_score(X_test, y_test)
    print(f"\n测试集性能:")
    print(f"  损失值: {test_loss:.6f}")
    print(f"  R²决定系数: {test_r2:.4f}")
    
    # 4. 可视化结果
    print("\n4. 结果可视化...")
    
    # 绘制训练损失曲线
    # 这有助于判断模型是否收敛以及是否存在过拟合
    model.plot_loss_history()
    
    # 绘制回归线拟合结果
    # 直观展示模型学习到的线性关系与真实数据的匹配程度
    model.plot_regression_line(X_train, y_train, "训练集拟合结果")
    model.plot_regression_line(X_val, y_val, "验证集拟合结果")
    model.plot_regression_line(X_test, y_test, "测试集拟合结果")
    
    # 5. 保存模型和结果
    print("\n5. 保存模型和结果...")
    
    # 创建结果目录
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 保存模型参数
    np.save('results/model_weights.npy', learned_weights)
    np.save('results/model_bias.npy', learned_bias)
    
    # 保存训练历史
    np.save('results/loss_history.npy', np.array(model.loss_history))
    
    # 保存性能指标
    performance_metrics = {
        'train_loss': train_loss,
        'train_r2': train_r2,
        'val_loss': val_loss,
        'val_r2': val_r2,
        'test_loss': test_loss,
        'test_r2': test_r2,
        'true_weight': data_gen.true_weight,
        'true_bias': data_gen.true_bias,
        'learned_weight': learned_weights[0],
        'learned_bias': learned_bias
    }
    
    # 保存性能指标到文本文件
    with open('results/performance_metrics.txt', 'w', encoding='utf-8') as f:
        f.write("线性回归模型性能指标\n")
        f.write("="*50 + "\n\n")
        f.write(f"真实参数: weight={performance_metrics['true_weight']}, bias={performance_metrics['true_bias']}\n\n")
        f.write(f"学习参数: weight={performance_metrics['learned_weight']:.4f}, bias={performance_metrics['learned_bias']:.4f}\n\n")
        f.write("训练集性能:\n")
        f.write(f"  损失值: {performance_metrics['train_loss']:.6f}\n")
        f.write(f"  R²决定系数: {performance_metrics['train_r2']:.4f}\n\n")
        f.write("验证集性能:\n")
        f.write(f"  损失值: {performance_metrics['val_loss']:.6f}\n")
        f.write(f"  R²决定系数: {performance_metrics['val_r2']:.4f}\n\n")
        f.write("测试集性能:\n")
        f.write(f"  损失值: {performance_metrics['test_loss']:.6f}\n")
        f.write(f"  R²决定系数: {performance_metrics['test_r2']:.4f}\n")
    
    print("模型和结果已保存到results/目录")
    
    # 6. 训练总结
    print("\n=== 训练总结 ===")
    print(f"模型成功学习到线性关系: y = {learned_weights[0]:.4f} * x + {learned_bias:.4f}")
    print(f"真实关系: y = {data_gen.true_weight} * x + {data_gen.true_bias}")
    print(f"参数误差 - 权重: {abs(learned_weights[0] - data_gen.true_weight):.4f}")
    print(f"参数误差 - 偏置: {abs(learned_bias - data_gen.true_bias):.4f}")
    print(f"测试集R²决定系数: {test_r2:.4f} (越接近1越好)")
    
    if test_r2 > 0.9:
        print("✓ 模型性能优秀！")
    elif test_r2 > 0.8:
        print("✓ 模型性能良好！")
    elif test_r2 > 0.7:
        print("○ 模型性能一般，可考虑调整超参数")
    else:
        print("✗ 模型性能较差，建议检查数据或调整学习策略")
    
    return model, performance_metrics

if __name__ == "__main__":
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    # 执行训练流程
    model, metrics = train_model()
    
    print("\n训练完成！查看results/目录获取详细结果。")