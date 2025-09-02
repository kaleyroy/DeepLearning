import numpy as np
import matplotlib.pyplot as plt
from data_generation import DataGenerator
from minibatch_linear_regression import MiniBatchGD, compare_gradient_descent_methods, plot_comparison_results
import os
import time

def train_minibatch_model():
    """
    使用小批量梯度下降训练线性回归模型的完整流程
    
    算法步骤：
    1. 数据生成和预处理
    2. 模型训练（使用不同参数配置）
    3. 性能评估和比较
    4. 结果可视化和保存
    """
    print("=== 小批量梯度下降线性回归训练开始 ===")
    
    # 1. 数据准备
    print("\n1. 数据生成和预处理...")
    
    # 创建数据生成器
    data_gen = DataGenerator(n_samples=2000, noise=0.3, random_state=42)
    X, y = data_gen.generate_data()
    
    print(f"生成数据集大小: {X.shape[0]} 个样本")
    print(f"真实参数: weight={data_gen.true_weight}, bias={data_gen.true_bias}")
    
    # 划分数据集（训练集70%，验证集15%，测试集15%）
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_gen.split_data(
        X, y, train_ratio=0.7, val_ratio=0.15
    )
    
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 2. 模型训练 - 测试不同配置
    print("\n2. 模型训练（测试不同配置）...")
    
    # 定义不同的训练配置
    configs = [
        {'learning_rate': 0.01, 'epochs': 50, 'batch_size': 16, 'name': '小批量(16)'},
        {'learning_rate': 0.01, 'epochs': 50, 'batch_size': 32, 'name': '小批量(32)'},
        {'learning_rate': 0.01, 'epochs': 50, 'batch_size': 64, 'name': '小批量(64)'},
        {'learning_rate': 0.01, 'epochs': 50, 'batch_size': 128, 'name': '小批量(128)'},
        {'learning_rate': 0.005, 'epochs': 100, 'batch_size': 32, 'name': '小批量(慢学习率)'},
        {'learning_rate': 0.02, 'epochs': 30, 'batch_size': 32, 'name': '小批量(快学习率)'}
    ]
    
    models = {}
    training_times = {}
    
    for config in configs:
        print(f"\n训练 {config['name']} 模型...")
        
        # 创建模型
        model = MiniBatchGD(
            learning_rate=config['learning_rate'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            random_state=42
        )
        
        # 记录训练时间
        start_time = time.time()
        
        # 训练模型
        model.fit(X_train, y_train, X_val, y_val, verbose=False)
        
        # 记录训练时间
        training_time = time.time() - start_time
        training_times[config['name']] = training_time
        
        # 保存模型
        models[config['name']] = model
        
        print(f"  训练时间: {training_time:.2f} 秒")
        
        # 打印简要结果
        weights, bias = model.get_parameters()
        train_r2 = model.calculate_r2_score(X_train, y_train)
        val_r2 = model.calculate_r2_score(X_val, y_val)
        
        print(f"  学习参数: weight={weights[0]:.4f}, bias={bias:.4f}")
        print(f"  训练集R²: {train_r2:.4f}")
        print(f"  验证集R²: {val_r2:.4f}")
    
    # 3. 选择最佳模型
    print("\n3. 选择最佳模型...")
    
    best_model_name = None
    best_val_r2 = -float('inf')
    
    for name, model in models.items():
        val_r2 = model.calculate_r2_score(X_val, y_val)
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model_name = name
    
    best_model = models[best_model_name]
    print(f"最佳模型: {best_model_name}")
    print(f"最佳验证集R²: {best_val_r2:.4f}")
    
    # 4. 详细评估最佳模型
    print("\n4. 最佳模型详细评估...")
    
    # 在所有数据集上评估
    train_loss = best_model.calculate_loss(X_train, y_train)
    val_loss = best_model.calculate_loss(X_val, y_val)
    test_loss = best_model.calculate_loss(X_test, y_test)
    
    train_r2 = best_model.calculate_r2_score(X_train, y_train)
    val_r2 = best_model.calculate_r2_score(X_val, y_val)
    test_r2 = best_model.calculate_r2_score(X_test, y_test)
    
    print(f"\n{best_model_name} 模型性能:")
    print(f"训练集 - 损失: {train_loss:.6f}, R²: {train_r2:.4f}")
    print(f"验证集 - 损失: {val_loss:.6f}, R²: {val_r2:.4f}")
    print(f"测试集 - 损失: {test_loss:.6f}, R²: {test_r2:.4f}")
    
    # 获取模型参数
    weights, bias = best_model.get_parameters()
    print(f"\n学习参数:")
    print(f"权重: {weights[0]:.4f} (真实值: {data_gen.true_weight})")
    print(f"偏置: {bias:.4f} (真实值: {data_gen.true_bias})")
    print(f"权重误差: {abs(weights[0] - data_gen.true_weight):.4f}")
    print(f"偏置误差: {abs(bias - data_gen.true_bias):.4f}")
    
    # 打印训练统计信息
    best_model.print_training_summary()
    
    # 5. 可视化结果
    print("\n5. 生成可视化结果...")
    
    # 创建结果目录
    if not os.path.exists('minibatch_results'):
        os.makedirs('minibatch_results')
    
    # 5.1 训练历史可视化
    print("生成训练历史图表...")
    best_model.plot_training_history()
    
    # 5.2 拟合结果可视化
    print("生成拟合结果图表...")
    best_model.plot_regression_line(X_train, y_train, f"{best_model_name} - 训练集拟合")
    best_model.plot_regression_line(X_val, y_val, f"{best_model_name} - 验证集拟合")
    best_model.plot_regression_line(X_test, y_test, f"{best_model_name} - 测试集拟合")
    
    # 5.3 所有模型性能对比
    print("生成模型对比图表...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('小批量梯度下降不同配置性能对比', fontsize=16)
    
    model_names = list(models.keys())
    val_r2_scores = [models[name].calculate_r2_score(X_val, y_val) for name in model_names]
    test_r2_scores = [models[name].calculate_r2_score(X_test, y_test) for name in model_names]
    val_losses = [models[name].calculate_loss(X_val, y_val) for name in model_names]
    times = [training_times[name] for name in model_names]
    
    # 验证集R²对比
    axes[0, 0].bar(model_names, val_r2_scores, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('验证集R²对比')
    axes[0, 0].set_ylabel('R²分数')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 测试集R²对比
    axes[0, 1].bar(model_names, test_r2_scores, alpha=0.7, color='lightcoral')
    axes[0, 1].set_title('测试集R²对比')
    axes[0, 1].set_ylabel('R²分数')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 验证集损失对比
    axes[1, 0].bar(model_names, val_losses, alpha=0.7, color='lightgreen')
    axes[1, 0].set_title('验证集损失对比')
    axes[1, 0].set_ylabel('损失值')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 训练时间对比
    axes[1, 1].bar(model_names, times, alpha=0.7, color='gold')
    axes[1, 1].set_title('训练时间对比')
    axes[1, 1].set_ylabel('时间(秒)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('minibatch_results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5.4 与批量梯度下降对比
    print("与批量梯度下降方法对比...")
    comparison_results = compare_gradient_descent_methods(X_train, y_train, X_val, y_val)
    plot_comparison_results(comparison_results)
    plt.savefig('minibatch_results/gd_methods_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. 保存结果
    print("\n6. 保存训练结果...")
    
    # 保存最佳模型参数
    best_weights, best_bias = best_model.get_parameters()
    np.save('minibatch_results/best_model_weights.npy', best_weights)
    np.save('minibatch_results/best_model_bias.npy', best_bias)
    
    # 保存训练历史
    np.save('minibatch_results/best_epoch_loss.npy', np.array(best_model.epoch_loss_history))
    np.save('minibatch_results/best_batch_loss.npy', np.array(best_model.batch_loss_history))
    np.save('minibatch_results/best_train_loss.npy', np.array(best_model.train_loss_history))
    np.save('minibatch_results/best_val_loss.npy', np.array(best_model.val_loss_history))
    
    # 保存性能指标
    performance_summary = {
        'best_model_name': best_model_name,
        'best_model_config': {
            'learning_rate': best_model.learning_rate,
            'epochs': len(best_model.epoch_loss_history),
            'batch_size': best_model.batch_size
        },
        'model_parameters': {
            'learned_weight': float(best_weights[0]),
            'learned_bias': float(best_bias),
            'true_weight': data_gen.true_weight,
            'true_bias': data_gen.true_bias,
            'weight_error': abs(float(best_weights[0]) - data_gen.true_weight),
            'bias_error': abs(float(best_bias) - data_gen.true_bias)
        },
        'performance_metrics': {
            'train_loss': train_loss,
            'train_r2': train_r2,
            'val_loss': val_loss,
            'val_r2': val_r2,
            'test_loss': test_loss,
            'test_r2': test_r2
        },
        'training_stats': best_model.get_training_stats(),
        'all_models_comparison': {
            name: {
                'val_r2': models[name].calculate_r2_score(X_val, y_val),
                'test_r2': models[name].calculate_r2_score(X_test, y_test),
                'training_time': training_times[name]
            } for name in model_names
        }
    }
    
    # 保存为JSON格式
    import json
    with open('minibatch_results/performance_summary.json', 'w', encoding='utf-8') as f:
        json.dump(performance_summary, f, ensure_ascii=False, indent=2)
    
    # 保存为文本格式
    with open('minibatch_results/training_report.txt', 'w', encoding='utf-8') as f:
        f.write("小批量梯度下降线性回归训练报告\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"最佳模型: {best_model_name}\n")
        f.write(f"模型配置: 学习率={best_model.learning_rate}, Epochs={len(best_model.epoch_loss_history)}, Batch Size={best_model.batch_size}\n\n")
        
        f.write("模型参数:\n")
        f.write(f"  学习权重: {best_weights[0]:.6f} (真实值: {data_gen.true_weight})\n")
        f.write(f"  学习偏置: {best_bias:.6f} (真实值: {data_gen.true_bias})\n")
        f.write(f"  权重误差: {abs(best_weights[0] - data_gen.true_weight):.6f}\n")
        f.write(f"  偏置误差: {abs(best_bias - data_gen.true_bias):.6f}\n\n")
        
        f.write("性能指标:\n")
        f.write(f"  训练集 - 损失: {train_loss:.6f}, R²: {train_r2:.4f}\n")
        f.write(f"  验证集 - 损失: {val_loss:.6f}, R²: {val_r2:.4f}\n")
        f.write(f"  测试集 - 损失: {test_loss:.6f}, R²: {test_r2:.4f}\n\n")
        
        f.write("训练统计:\n")
        stats = best_model.get_training_stats()
        f.write(f"  总Epochs: {stats['total_epochs']}\n")
        f.write(f"  总迭代次数: {stats['total_iterations']}\n")
        f.write(f"  每Epoch批次数: {stats['batches_per_epoch']}\n")
        f.write(f"  Batch Size: {stats['batch_size']}\n\n")
        
        f.write("所有模型对比:\n")
        f.write(f"{'模型名称':<20} {'验证R²':<10} {'测试R²':<10} {'训练时间(秒)':<15}\n")
        f.write("-"*60 + "\n")
        for name in model_names:
            val_r2 = models[name].calculate_r2_score(X_val, y_val)
            test_r2 = models[name].calculate_r2_score(X_test, y_test)
            time_val = training_times[name]
            f.write(f"{name:<20} {val_r2:<10.4f} {test_r2:<10.4f} {time_val:<15.2f}\n")
    
    print("所有结果已保存到 minibatch_results/ 目录")
    
    # 7. 训练总结
    print("\n=== 训练总结 ===")
    print(f"最佳模型: {best_model_name}")
    print(f"模型性能: 测试集R² = {test_r2:.4f}")
    print(f"参数精度: 权重误差={abs(best_weights[0] - data_gen.true_weight):.4f}, 偏置误差={abs(best_bias - data_gen.true_bias):.4f}")
    
    if test_r2 > 0.95:
        print("✓ 模型性能优秀！")
    elif test_r2 > 0.90:
        print("✓ 模型性能良好！")
    elif test_r2 > 0.85:
        print("○ 模型性能一般")
    else:
        print("✗ 模型性能需要改进")
    
    return best_model, performance_summary


def demonstrate_minibatch_concepts():
    """
    演示小批量梯度下降的核心概念
    """
    print("\n" + "="*60)
    print("小批量梯度下降核心概念演示")
    print("="*60)
    
    # 生成小数据集便于演示
    np.random.seed(42)
    n_samples = 100
    X_demo = np.random.rand(n_samples, 1) * 5
    y_demo = 2.0 * X_demo.flatten() + 1.0 + np.random.normal(0, 0.2, n_samples)
    
    print(f"演示数据集: {n_samples} 个样本")
    print(f"数据范围: X=[{X_demo.min():.2f}, {X_demo.max():.2f}]")
    
    # 演示不同batch size的效果
    batch_sizes = [8, 16, 32]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('不同Batch Size的小批量划分演示', fontsize=16)
    
    for idx, batch_size in enumerate(batch_sizes):
        # 创建模型用于演示
        demo_model = MiniBatchGD(learning_rate=0.01, epochs=1, batch_size=batch_size, random_state=42)
        
        # 创建小批量
        mini_batches = demo_model._create_mini_batches(X_demo, y_demo)
        
        # 可视化数据划分
        axes[idx].scatter(X_demo, y_demo, alpha=0.6, s=30, c='gray', label='所有数据')
        
        # 用不同颜色标记每个batch
        colors = plt.cm.Set3(np.linspace(0, 1, len(mini_batches)))
        
        for i, (X_batch, y_batch) in enumerate(mini_batches):
            axes[idx].scatter(X_batch, y_batch, alpha=0.8, s=50, 
                           color=colors[i], label=f'Batch {i+1}')
        
        axes[idx].set_title(f'Batch Size = {batch_size}\n共 {len(mini_batches)} 个Batch')
        axes[idx].set_xlabel('X')
        axes[idx].set_ylabel('y')
        axes[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 演示训练过程中的损失变化
    print("\n演示训练过程中的损失变化...")
    
    model_demo = MiniBatchGD(learning_rate=0.01, epochs=20, batch_size=16, random_state=42)
    model_demo.fit(X_demo, y_demo, verbose=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Batch损失变化
    ax1.plot(model_demo.batch_loss_history, alpha=0.7, linewidth=1)
    ax1.set_title('Batch级别损失变化')
    ax1.set_xlabel('Batch迭代次数')
    ax1.set_ylabel('损失值')
    ax1.grid(True, alpha=0.3)
    
    # Epoch损失变化
    ax2.plot(model_demo.epoch_loss_history, 'b-o', linewidth=2, markersize=6)
    ax2.set_title('Epoch级别损失变化')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('损失值')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("演示完成！")


if __name__ == "__main__":
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 演示核心概念
    demonstrate_minibatch_concepts()
    
    # 执行完整训练流程
    best_model, summary = train_minibatch_model()
    
    print("\n小批量梯度下降训练完成！")
    print("查看 minibatch_results/ 目录获取详细结果。")