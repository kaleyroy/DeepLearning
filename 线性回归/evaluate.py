import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
import os

def load_model_and_data():
    """
    加载训练好的模型和测试数据
    
    返回：
    model: 加载的线性回归模型
    X_test, y_test: 测试数据
    """
    # 检查模型文件是否存在
    if not os.path.exists('results/model_weights.npy'):
        raise FileNotFoundError("模型文件不存在，请先运行train.py训练模型")
    
    if not os.path.exists('data/X_test.npy'):
        raise FileNotFoundError("测试数据不存在，请先运行data_generation.py生成数据")
    
    # 加载模型参数
    weights = np.load('results/model_weights.npy')
    bias = np.load('results/model_bias.npy')
    
    # 创建模型并设置参数
    model = LinearRegression()
    model.weights = weights
    model.bias = bias
    
    # 加载测试数据
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    # 加载验证数据
    X_val = np.load('data/X_val.npy')
    y_val = np.load('data/y_val.npy')
    
    return model, X_test, y_test, X_val, y_val

def evaluate_model(model, X, y, dataset_name="数据集"):
    """
    评估模型在给定数据集上的性能
    
    算法原理：
    1. 计算预测值和真实值之间的误差
    2. 计算多种性能指标
    3. 分析误差分布
    
    参数：
    model: 训练好的模型
    X: 特征数据
    y: 真实标签
    dataset_name: 数据集名称
    
    返回：
    metrics: 包含各种性能指标的字典
    """
    print(f"\n=== {dataset_name}评估结果 ===")
    
    # 1. 基本性能指标
    # 计算预测值
    y_pred = model.predict(X)
    
    # 计算误差
    errors = y_pred - y
    absolute_errors = np.abs(errors)
    squared_errors = errors ** 2
    
    # 2. 计算各种性能指标
    
    # 均方误差 (MSE)
    # MSE = (1/n) * Σ(y_pred - y_true)²
    mse = np.mean(squared_errors)
    
    # 均方根误差 (RMSE)
    # RMSE = √MSE，与原始数据单位相同，更易解释
    rmse = np.sqrt(mse)
    
    # 平均绝对误差 (MAE)
    # MAE = (1/n) * Σ|y_pred - y_true|
    mae = np.mean(absolute_errors)
    
    # 平均绝对百分比误差 (MAPE)
    # MAPE = (100%/n) * Σ|(y_pred - y_true)/y_true|
    # 注意：避免除零错误
    non_zero_mask = y != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs(errors[non_zero_mask] / y[non_zero_mask])) * 100
    else:
        mape = float('inf')
    
    # R²决定系数
    # R² = 1 - (SS_res / SS_tot)
    # 衡量模型解释的方差比例
    r2_score = model.calculate_r2_score(X, y)
    
    # 3. 误差统计分析
    error_mean = np.mean(errors)
    error_std = np.std(errors)
    error_min = np.min(errors)
    error_max = np.max(errors)
    
    # 打印结果
    print(f"样本数量: {len(y)}")
    print(f"均方误差 (MSE): {mse:.6f}")
    print(f"均方根误差 (RMSE): {rmse:.6f}")
    print(f"平均绝对误差 (MAE): {mae:.6f}")
    print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
    print(f"R²决定系数: {r2_score:.4f}")
    print(f"\n误差统计:")
    print(f"  平均误差: {error_mean:.6f}")
    print(f"  误差标准差: {error_std:.6f}")
    print(f"  最小误差: {error_min:.6f}")
    print(f"  最大误差: {error_max:.6f}")
    
    # 返回所有指标
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2_score': r2_score,
        'error_mean': error_mean,
        'error_std': error_std,
        'error_min': error_min,
        'error_max': error_max,
        'y_pred': y_pred,
        'errors': errors
    }
    
    return metrics

def plot_prediction_analysis(model, X, y, metrics, dataset_name="数据集"):
    """
    绘制预测分析图表
    
    参数：
    model: 训练好的模型
    X: 特征数据
    y: 真实标签
    metrics: 评估指标
    dataset_name: 数据集名称
    """
    y_pred = metrics['y_pred']
    errors = metrics['errors']
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{dataset_name}预测分析', fontsize=16)
    
    # 1. 真实值vs预测值散点图
    axes[0, 0].scatter(y, y_pred, alpha=0.6, s=20)
    
    # 绘制理想对角线（完美预测线）
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测线')
    
    axes[0, 0].set_xlabel('真实值')
    axes[0, 0].set_ylabel('预测值')
    axes[0, 0].set_title('真实值 vs 预测值')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 误差分布直方图
    axes[0, 1].hist(errors, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='零误差线')
    axes[0, 1].axvline(x=metrics['error_mean'], color='g', linestyle='-', linewidth=2, label=f'平均误差 ({metrics["error_mean"]:.3f})')
    axes[0, 1].set_xlabel('预测误差')
    axes[0, 1].set_ylabel('频数')
    axes[0, 1].set_title('误差分布直方图')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 残差图（误差 vs 预测值）
    axes[1, 0].scatter(y_pred, errors, alpha=0.6, s=20)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2, label='零误差线')
    axes[1, 0].set_xlabel('预测值')
    axes[1, 0].set_ylabel('残差（误差）')
    axes[1, 0].set_title('残差图')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Q-Q图（检验误差正态性）
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q图（误差正态性检验）')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_regression_comparison(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    绘制三个数据集的回归拟合对比图
    
    参数：
    model: 训练好的模型
    X_train, y_train: 训练数据
    X_val, y_val: 验证数据
    X_test, y_test: 测试数据
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('线性回归模型在不同数据集上的拟合效果', fontsize=16)
    
    datasets = [
        (X_train, y_train, '训练集', axes[0]),
        (X_val, y_val, '验证集', axes[1]),
        (X_test, y_test, '测试集', axes[2])
    ]
    
    for X_data, y_data, title, ax in datasets:
        # 绘制数据点
        ax.scatter(X_data, y_data, alpha=0.6, s=15, label='真实数据')
        
        # 绘制回归线
        X_line = np.linspace(X_data.min(), X_data.max(), 100)
        y_line = model.predict(X_line.reshape(-1, 1))
        ax.plot(X_line, y_line, color='red', linewidth=2, label='回归线')
        
        # 计算并显示R²
        r2 = model.calculate_r2_score(X_data, y_data)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax.set_xlabel('X (自变量)')
        ax.set_ylabel('y (因变量)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def save_evaluation_results(val_metrics, test_metrics):
    """
    保存评估结果到文件
    
    参数：
    val_metrics: 验证集评估指标
    test_metrics: 测试集评估指标
    """
    # 确保结果目录存在
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 保存评估结果
    with open('results/evaluation_results.txt', 'w', encoding='utf-8') as f:
        f.write("线性回归模型评估结果\n")
        f.write("="*60 + "\n\n")
        
        f.write("验证集性能指标:\n")
        f.write("-"*30 + "\n")
        f.write(f"均方误差 (MSE): {val_metrics['mse']:.6f}\n")
        f.write(f"均方根误差 (RMSE): {val_metrics['rmse']:.6f}\n")
        f.write(f"平均绝对误差 (MAE): {val_metrics['mae']:.6f}\n")
        f.write(f"平均绝对百分比误差 (MAPE): {val_metrics['mape']:.2f}%\n")
        f.write(f"R²决定系数: {val_metrics['r2_score']:.4f}\n")
        f.write(f"误差均值: {val_metrics['error_mean']:.6f}\n")
        f.write(f"误差标准差: {val_metrics['error_std']:.6f}\n\n")
        
        f.write("测试集性能指标:\n")
        f.write("-"*30 + "\n")
        f.write(f"均方误差 (MSE): {test_metrics['mse']:.6f}\n")
        f.write(f"均方根误差 (RMSE): {test_metrics['rmse']:.6f}\n")
        f.write(f"平均绝对误差 (MAE): {test_metrics['mae']:.6f}\n")
        f.write(f"平均绝对百分比误差 (MAPE): {test_metrics['mape']:.2f}%\n")
        f.write(f"R²决定系数: {test_metrics['r2_score']:.4f}\n")
        f.write(f"误差均值: {test_metrics['error_mean']:.6f}\n")
        f.write(f"误差标准差: {test_metrics['error_std']:.6f}\n")
    
    print("评估结果已保存到results/evaluation_results.txt")

def main():
    """
    主函数：执行完整的模型评估流程
    """
    print("=== 线性回归模型评估开始 ===")
    
    try:
        # 1. 加载模型和数据
        print("\n1. 加载模型和测试数据...")
        model, X_test, y_test, X_val, y_val = load_model_and_data()
        print("模型和数据加载成功！")
        
        # 加载训练数据用于对比
        X_train = np.load('data/X_train.npy')
        y_train = np.load('data/y_train.npy')
        
        # 2. 评估验证集性能
        print("\n2. 评估验证集性能...")
        val_metrics = evaluate_model(model, X_val, y_val, "验证集")
        
        # 3. 评估测试集性能
        print("\n3. 评估测试集性能...")
        test_metrics = evaluate_model(model, X_test, y_test, "测试集")
        
        # 4. 可视化分析
        print("\n4. 生成可视化分析图表...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 验证集预测分析
        plot_prediction_analysis(model, X_val, y_val, val_metrics, "验证集")
        
        # 测试集预测分析
        plot_prediction_analysis(model, X_test, y_test, test_metrics, "测试集")
        
        # 三个数据集回归对比
        plot_regression_comparison(model, X_train, y_train, X_val, y_val, X_test, y_test)
        
        # 5. 保存评估结果
        print("\n5. 保存评估结果...")
        save_evaluation_results(val_metrics, test_metrics)
        
        # 6. 性能总结
        print("\n=== 评估总结 ===")
        print(f"验证集R²: {val_metrics['r2_score']:.4f}")
        print(f"测试集R²: {test_metrics['r2_score']:.4f}")
        print(f"测试集RMSE: {test_metrics['rmse']:.6f}")
        print(f"测试集MAE: {test_metrics['mae']:.6f}")
        
        # 性能评价
        if test_metrics['r2_score'] > 0.9:
            performance_level = "优秀"
        elif test_metrics['r2_score'] > 0.8:
            performance_level = "良好"
        elif test_metrics['r2_score'] > 0.7:
            performance_level = "一般"
        else:
            performance_level = "较差"
        
        print(f"\n模型性能评价: {performance_level}")
        
        # 检查过拟合
        r2_diff = abs(val_metrics['r2_score'] - test_metrics['r2_score'])
        if r2_diff > 0.1:
            print("⚠️  警告：验证集和测试集性能差异较大，可能存在过拟合")
        else:
            print("✓ 验证集和测试集性能一致，模型泛化能力良好")
        
    except Exception as e:
        print(f"评估过程中出现错误: {str(e)}")
        print("请确保已运行train.py完成模型训练")
        return
    
    print("\n评估完成！查看results/目录获取详细评估报告。")

if __name__ == "__main__":
    main()