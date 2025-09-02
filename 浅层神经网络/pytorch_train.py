import numpy as np
import matplotlib.pyplot as plt
import os
import json
from data_generation import DataGenerator
from pytorch_shallow_neural_network import (
    PyTorchShallowNeuralNetwork, 
    PyTorchNeuralNetworkTrainer, 
    get_device, 
    create_model
)
from sklearn.metrics import confusion_matrix
import seaborn as sns

def main():
    """
    PyTorch浅层神经网络训练主函数
    
    算法原理:
    1. 生成非线性可分数据集
    2. 创建PyTorch神经网络模型
    3. 使用PyTorch训练器进行模型训练
    4. 在验证集和测试集上评估模型性能
    5. 可视化训练结果和模型性能
    """
    print("=== PyTorch浅层神经网络训练开始 ===")
    
    # 创建必要的目录
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 1. 生成数据集
    print("\n1. 生成非线性可分数据集...")
    data_gen = DataGenerator(random_state=42)
    
    # 生成螺旋形非线性数据
    X, y = data_gen.generate_nonlinear_data()
    
    # 分割数据集
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test = data_gen.split_and_save_data(
        X, y, 
        test_size=0.2, 
        val_size=0.1
    )
    
    # 为了保持一致性，使用标准化后的数据
    X_train = X_train_scaled
    X_val = X_val_scaled
    X_test = X_test_scaled
    
    print(f"数据集生成完成:")
    print(f"  - 训练集: {X_train.shape[0]} 个样本")
    print(f"  - 验证集: {X_val.shape[0]} 个样本")
    print(f"  - 测试集: {X_test.shape[0]} 个样本")
    
    # 可视化数据
    data_gen.visualize_data(X, y, save_path='data/pytorch_data_visualization.png')
    
    # 2. 创建PyTorch模型
    print("\n2. 创建PyTorch浅层神经网络模型...")
    
    # 获取计算设备
    device = get_device()
    
    # 模型参数配置
    input_size = X_train.shape[1]  # 输入特征数量
    hidden_size = 20               # 隐藏层神经元数量
    output_size = len(np.unique(y)) # 输出类别数量
    activation = 'relu'            # 激活函数
    learning_rate = 0.01           # 学习率
    weight_decay = 0.001           # 权重衰减（L2正则化）
    n_epochs = 1000                # 训练轮数
    batch_size = 32                # 批量大小
    
    print(f"模型配置:")
    print(f"  - 输入层: {input_size} 个神经元")
    print(f"  - 隐藏层: {hidden_size} 个神经元")
    print(f"  - 输出层: {output_size} 个神经元")
    print(f"  - 激活函数: {activation}")
    print(f"  - 学习率: {learning_rate}")
    print(f"  - 权重衰减: {weight_decay}")
    print(f"  - 训练轮数: {n_epochs}")
    print(f"  - 批量大小: {batch_size}")
    print(f"  - 计算设备: {device}")
    
    # 创建模型
    model = create_model(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        activation=activation
    )
    
    # 创建训练器
    trainer = PyTorchNeuralNetworkTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device
    )
    
    # 3. 训练模型
    print("\n3. 开始训练模型...")
    trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbose=True
    )
    
    # 4. 评估模型性能
    print("\n4. 评估模型性能...")
    
    # 在验证集上评估
    val_loss, val_accuracy = trainer.evaluate(X_val, y_val)
    print(f"验证集性能:")
    print(f"  - 损失: {val_loss:.6f}")
    print(f"  - 准确率: {val_accuracy:.4f}")
    
    # 在测试集上评估
    test_loss, test_accuracy = trainer.evaluate(X_test, y_test)
    print(f"测试集性能:")
    print(f"  - 损失: {test_loss:.6f}")
    print(f"  - 准确率: {test_accuracy:.4f}")
    
    # 获取详细的评估指标
    val_metrics = trainer.get_metrics(X_val, y_val)
    test_metrics = trainer.get_metrics(X_test, y_test)
    
    print(f"\n验证集详细指标:")
    print(f"  - 准确率: {val_metrics['accuracy']:.4f}")
    print(f"  - 精确率: {val_metrics['precision']:.4f}")
    print(f"  - 召回率: {val_metrics['recall']:.4f}")
    print(f"  - F1分数: {val_metrics['f1_score']:.4f}")
    
    print(f"\n测试集详细指标:")
    print(f"  - 准确率: {test_metrics['accuracy']:.4f}")
    print(f"  - 精确率: {test_metrics['precision']:.4f}")
    print(f"  - 召回率: {test_metrics['recall']:.4f}")
    print(f"  - F1分数: {test_metrics['f1_score']:.4f}")
    
    # 5. 可视化结果
    print("\n5. 生成可视化结果...")
    
    # 绘制训练历史曲线
    trainer.plot_training_history(save_path='results/pytorch_training_history.png')
    
    # 绘制决策边界
    trainer.plot_decision_boundary(X_test, y_test, save_path='results/pytorch_decision_boundary.png')
    
    # 绘制混淆矩阵
    y_test_pred = trainer.predict(X_test)
    cm = confusion_matrix(y_test, y_test_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('PyTorch模型混淆矩阵')
    plt.tight_layout()
    plt.savefig('results/pytorch_confusion_matrix.png')
    plt.close()
    
    # 6. 保存模型和结果
    print("\n6. 保存模型和结果...")
    
    # 保存模型
    trainer.save_model('results/pytorch_model.pth')
    
    # 保存性能指标
    results = {
        'model_type': 'PyTorch Shallow Neural Network',
        'model_config': {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'activation': activation
        },
        'training_config': {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'device': device
        },
        'performance': {
            'validation': {
                'loss': val_loss,
                'accuracy': val_accuracy,
                'precision': val_metrics['precision'],
                'recall': val_metrics['recall'],
                'f1_score': val_metrics['f1_score']
            },
            'test': {
                'loss': test_loss,
                'accuracy': test_accuracy,
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'f1_score': test_metrics['f1_score']
            }
        }
    }
    
    with open('results/pytorch_performance_summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print("模型和结果已保存到 results/ 目录")
    
    # 7. 模型推理示例
    print("\n7. 模型推理示例...")
    
    # 选择几个测试样本进行预测
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    sample_X = X_test[sample_indices]
    sample_y_true = y_test[sample_indices]
    sample_y_pred = trainer.predict(sample_X)
    sample_y_proba = trainer.predict_proba(sample_X)
    
    print("样本预测结果:")
    for i, idx in enumerate(sample_indices):
        print(f"  样本 {idx}: 真实标签={sample_y_true[i]}, 预测标签={sample_y_pred[i]}, 概率={sample_y_proba[i]}")
    
    print("\n=== PyTorch浅层神经网络训练完成 ===")

if __name__ == "__main__":
    main()