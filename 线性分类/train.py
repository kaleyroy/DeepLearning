import numpy as np
import matplotlib.pyplot as plt
from data_generation import DataGenerator
from linear_classifier import LinearClassifier
import os

def main():
    """
    主训练函数：完成数据生成、模型训练、验证和测试的完整流程
    """
    print("=== 线性分类模型训练开始 ===")
    
    # 1. 生成训练数据
    print("\n1. 生成训练数据...")
    data_gen = DataGenerator(n_samples=1000, n_features=2, n_classes=2, random_state=42)
    X, y = data_gen.generate_linear_separable_data()
    
    # 分割数据集
    X_train, X_val, X_test, y_train, y_val, y_test = data_gen.split_and_save_data(X, y)
    
    # 可视化生成的数据
    # data_gen.visualize_data(X, y, save_path='data/data_visualization.png')
    
    # 2. 创建并训练模型
    print("\n2. 训练线性分类模型...")
    classifier = LinearClassifier(
        learning_rate=0.1,
        n_iterations=1000,
        regularization=0.01,
        random_state=42
    )
    
    # 训练模型
    classifier.fit(X_train, y_train)
    
    # 3. 在验证集上评估模型
    print("\n3. 在验证集上评估模型...")
    val_metrics = classifier.evaluate(X_val, y_val)
    
    print("验证集性能指标:")
    print(f"- 准确率: {val_metrics['accuracy']:.4f}")
    print(f"- 精确率: {val_metrics['precision']:.4f}")
    print(f"- 召回率: {val_metrics['recall']:.4f}")
    print(f"- F1分数: {val_metrics['f1_score']:.4f}")
    
    # 4. 在测试集上评估模型
    print("\n4. 在测试集上评估模型...")
    test_metrics = classifier.evaluate(X_test, y_test)
    
    print("测试集性能指标:")
    print(f"- 准确率: {test_metrics['accuracy']:.4f}")
    print(f"- 精确率: {test_metrics['precision']:.4f}")
    print(f"- 召回率: {test_metrics['recall']:.4f}")
    print(f"- F1分数: {test_metrics['f1_score']:.4f}")
    
    # 5. 可视化结果
    print("\n5. 生成可视化结果...")
    
    # 创建结果文件夹
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 绘制损失函数变化曲线
    classifier.plot_loss_history(save_path=os.path.join(results_dir, 'loss_history.png'))
    
    # 绘制决策边界（使用原始未标准化的数据）
    X_train_raw = np.load(os.path.join(os.path.dirname(__file__), 'data', 'X_train_raw.npy'))
    classifier.plot_decision_boundary(X_train_raw, y_train, 
                                    save_path=os.path.join(results_dir, 'decision_boundary.png'))
    
    # 绘制混淆矩阵
    classifier.plot_confusion_matrix(test_metrics['confusion_matrix'], 
                                   save_path=os.path.join(results_dir, 'confusion_matrix.png'))
    
    # 6. 保存模型
    print("\n6. 保存模型...")
    classifier.save_model(os.path.join(results_dir, 'model_params.npy'))
    
    # 7. 保存性能指标
    print("\n7. 保存性能指标...")
    performance_summary = {
        'validation_metrics': {
            'accuracy': float(val_metrics['accuracy']),
            'precision': float(val_metrics['precision']),
            'recall': float(val_metrics['recall']),
            'f1_score': float(val_metrics['f1_score'])
        },
        'test_metrics': {
            'accuracy': float(test_metrics['accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1_score': float(test_metrics['f1_score'])
        },
        'model_parameters': {
            'learning_rate': classifier.learning_rate,
            'n_iterations': classifier.n_iterations,
            'regularization': classifier.regularization,
            'final_loss': float(classifier.loss_history[-1])
        }
    }
    
    import json
    with open(os.path.join(results_dir, 'performance_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(performance_summary, f, ensure_ascii=False, indent=4)
    
    print("\n=== 训练完成 ===")
    print(f"所有结果已保存到 {results_dir} 文件夹")
    
    # 8. 打印最终总结
    print("\n=== 最终总结 ===")
    print(f"模型训练迭代次数: {classifier.n_iterations}")
    print(f"最终训练损失: {classifier.loss_history[-1]:.4f}")
    print(f"验证集准确率: {val_metrics['accuracy']:.4f}")
    print(f"测试集准确率: {test_metrics['accuracy']:.4f}")
    
    # 分析模型性能
    if test_metrics['accuracy'] > 0.9:
        print("✓ 模型性能优秀，测试准确率超过90%")
    elif test_metrics['accuracy'] > 0.8:
        print("✓ 模型性能良好，测试准确率超过80%")
    elif test_metrics['accuracy'] > 0.7:
        print("✓ 模型性能一般，测试准确率超过70%")
    else:
        print("⚠ 模型性能需要改进，测试准确率低于70%")
    
    # 检查过拟合
    val_accuracy = val_metrics['accuracy']
    test_accuracy = test_metrics['accuracy']
    overfitting_gap = val_accuracy - test_accuracy
    
    if overfitting_gap > 0.05:
        print(f"⚠ 可能存在过拟合，验证集与测试集准确率差距为 {overfitting_gap:.4f}")
        print("建议：增加正则化系数或减少模型复杂度")
    else:
        print("✓ 模型泛化性能良好，无明显过拟合现象")

def test_multiclass_classification():
    """
    测试多分类功能
    """
    print("\n=== 多分类测试 ===")
    
    # 生成多分类数据
    data_gen = DataGenerator(n_samples=1500, n_features=2, n_classes=3, random_state=42)
    X, y = data_gen.generate_linear_separable_data()
    
    # 分割数据集
    X_train, X_val, X_test, y_train, y_val, y_test = data_gen.split_and_save_data(X, y)
    
    # 创建多分类模型
    classifier = LinearClassifier(
        learning_rate=0.1,
        n_iterations=1000,
        regularization=0.01,
        random_state=42
    )
    
    # 训练模型
    classifier.fit(X_train, y_train, n_classes=3)
    
    # 评估模型
    test_metrics = classifier.evaluate(X_test, y_test)
    
    print("多分类测试结果:")
    print(f"测试集准确率: {test_metrics['accuracy']:.4f}")
    print(f"测试集精确率: {test_metrics['precision']:.4f}")
    print(f"测试集召回率: {test_metrics['recall']:.4f}")
    print(f"测试集F1分数: {test_metrics['f1_score']:.4f}")
    
    # 可视化多分类结果
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    X_train_raw = np.load(os.path.join(os.path.dirname(__file__), 'data', 'X_train_raw.npy'))
    classifier.plot_decision_boundary(X_train_raw, y_train, 
                                    save_path=os.path.join(results_dir, 'multiclass_decision_boundary.png'))
    
    classifier.plot_confusion_matrix(test_metrics['confusion_matrix'], 
                                   save_path=os.path.join(results_dir, 'multiclass_confusion_matrix.png'))
    
    print("多分类测试完成！")

if __name__ == "__main__":
    # 运行主训练流程
    main()
    
    # 可选：运行多分类测试
    # test_multiclass_classification()