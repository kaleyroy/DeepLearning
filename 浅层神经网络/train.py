import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import json
from data_generation import DataGenerator
from shallow_neural_network import ShallowNeuralNetwork

def main():
    """
    主训练函数：完整的浅层神经网络训练流程
    
    算法原理:
    1. 生成非线性可分的数据集
    2. 初始化浅层神经网络模型
    3. 训练模型
    4. 在验证集和测试集上评估模型性能
    5. 可视化训练结果
    """
    print("=== 浅层神经网络训练开始 ===")
    
    # 第一步：生成训练数据
    print("\n1. 生成训练数据...")
    data_gen = DataGenerator(n_samples=1000, n_features=2, n_classes=2, random_state=42)
    
    # 生成非线性数据（螺旋形数据）
    X, y = data_gen.generate_nonlinear_data()
    
    # 分割并保存数据
    X_train, X_val, X_test, y_train, y_val, y_test = data_gen.split_and_save_data(X, y)
    
    # 可视化数据
    data_gen.visualize_data(X, y, save_path=os.path.join(os.path.dirname(__file__), 'data', 'data_visualization.png'))
    
    print(f"训练数据形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"验证数据形状: X_val={X_val.shape}, y_val={y_val.shape}")
    print(f"测试数据形状: X_test={X_test.shape}, y_test={y_test.shape}")
    
    # 第二步：初始化并训练模型
    print("\n2. 初始化并训练浅层神经网络...")
    
    # 创建浅层神经网络模型
    # 参数说明:
    # - hidden_size: 隐藏层神经元数量，影响模型容量
    # - learning_rate: 学习率，控制梯度下降步长
    # - n_iterations: 训练迭代次数
    # - regularization: L2正则化系数，防止过拟合
    # - activation: 激活函数类型 ('relu', 'sigmoid', 'tanh')
    model = ShallowNeuralNetwork(
        hidden_size=10,
        learning_rate=0.01,
        n_iterations=2000,
        regularization=0.01,
        activation='relu'
    )
    
    print(f"模型参数:")
    print(f"  - 隐藏层神经元数量: {model.hidden_size}")
    print(f"  - 学习率: {model.learning_rate}")
    print(f"  - 训练迭代次数: {model.n_iterations}")
    print(f"  - 正则化系数: {model.regularization}")
    print(f"  - 激活函数: {model.activation_type}")
    
    # 训练模型
    print("\n开始训练模型...")
    model.fit(X_train, y_train, verbose=True)
    
    print("\n模型训练完成！")
    
    # 第三步：在验证集上评估模型
    print("\n3. 在验证集上评估模型性能...")
    val_metrics = model.evaluate(X_val, y_val)
    
    print("验证集性能指标:")
    print(f"  - 准确率: {val_metrics['accuracy']:.4f}")
    print(f"  - 精确率: {val_metrics['precision']:.4f}")
    print(f"  - 召回率: {val_metrics['recall']:.4f}")
    print(f"  - F1分数: {val_metrics['f1_score']:.4f}")
    
    # 第四步：在测试集上评估模型
    print("\n4. 在测试集上评估模型性能...")
    test_metrics = model.evaluate(X_test, y_test)
    
    print("测试集性能指标:")
    print(f"  - 准确率: {test_metrics['accuracy']:.4f}")
    print(f"  - 精确率: {test_metrics['precision']:.4f}")
    print(f"  - 召回率: {test_metrics['recall']:.4f}")
    print(f"  - F1分数: {test_metrics['f1_score']:.4f}")
    
    # 第五步：生成详细的分类报告
    print("\n5. 生成详细分类报告...")
    y_test_pred = model.predict(X_test)
    
    print("\n测试集详细分类报告:")
    print(classification_report(y_test, y_test_pred))
    
    # 第六步：可视化结果
    print("\n6. 可视化训练结果...")
    
    # 创建结果文件夹
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 绘制损失函数变化曲线
    model.plot_loss_history(save_path=os.path.join(results_dir, 'loss_history.png'))
    
    # 绘制决策边界
    model.plot_decision_boundary(X_test, y_test, save_path=os.path.join(results_dir, 'decision_boundary.png'))
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('测试集混淆矩阵')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()
    
    print("可视化结果已保存到 results 文件夹")
    
    # 第七步：保存模型和性能指标
    print("\n7. 保存模型和性能指标...")
    
    # 保存模型参数
    model.save_model(os.path.join(results_dir, 'model_params.npy'))
    
    # 保存性能指标
    performance_summary = {
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'model_parameters': {
            'hidden_size': model.hidden_size,
            'learning_rate': model.learning_rate,
            'n_iterations': model.n_iterations,
            'regularization': model.regularization,
            'activation': model.activation_type
        },
        'final_loss': model.loss_history[-1] if model.loss_history else None,
        'data_info': {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        }
    }
    
    with open(os.path.join(results_dir, 'performance_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(performance_summary, f, ensure_ascii=False, indent=4)
    
    print("模型和性能指标已保存")
    
    # 第八步：打印训练总结
    print("\n=== 训练总结 ===")
    print(f"数据集信息:")
    print(f"  - 总样本数: {len(X)}")
    print(f"  - 特征数量: {X.shape[1]}")
    print(f"  - 类别数量: {len(np.unique(y))}")
    print(f"  - 训练集: {len(X_train)} 样本")
    print(f"  - 验证集: {len(X_val)} 样本")
    print(f"  - 测试集: {len(X_test)} 样本")
    
    print(f"\n模型结构:")
    print(f"  - 输入层: {X.shape[1]} 个神经元")
    print(f"  - 隐藏层: {model.hidden_size} 个神经元")
    print(f"  - 输出层: {len(np.unique(y))} 个神经元")
    print(f"  - 激活函数: {model.activation_type}")
    
    print(f"\n最终性能:")
    print(f"  - 验证集准确率: {val_metrics['accuracy']:.4f}")
    print(f"  - 测试集准确率: {test_metrics['accuracy']:.4f}")
    print(f"  - 最终损失值: {model.loss_history[-1]:.6f}")
    
    print("\n=== 浅层神经网络训练完成 ===")

if __name__ == "__main__":
    main()