import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
import os

class DataGenerator:
    """
    数据生成器类：用于生成线性回归的训练、验证和测试数据集
    
    算法原理：
    1. 生成符合线性关系的数据：y = w*x + b + noise
    2. 添加高斯噪声模拟真实数据的不确定性
    3. 按比例划分数据集，确保数据分布的一致性
    """
    
    def __init__(self, n_samples=1000, noise=0.1, random_state=42):
        """
        初始化数据生成器
        
        参数：
        n_samples: 样本总数
        noise: 噪声标准差，控制数据的离散程度
        random_state: 随机种子，确保结果可复现
        """
        self.n_samples = n_samples
        self.noise = noise
        self.random_state = random_state
        np.random.seed(random_state)
        
        # 真实参数（我们希望模型学习到的值）
        self.true_weight = 2.5  # 真实权重
        self.true_bias = 1.0     # 真实偏置
        
    def generate_data(self):
        """
        生成线性数据集
        
        算法步骤：
        1. 生成自变量X：在[0, 10]范围内均匀分布
        2. 计算因变量y：y = w*X + b + noise
        3. 添加高斯噪声模拟真实世界的数据不确定性
        
        返回：
        X: 自变量数组 (n_samples, 1)
        y: 因变量数组 (n_samples,)
        """
        # 生成自变量X：在[0, 10]范围内均匀分布
        X = np.random.uniform(0, 10, self.n_samples)
        
        # 计算真实线性关系：y = w*X + b
        y_true = self.true_weight * X + self.true_bias
        
        # 添加高斯噪声：噪声服从正态分布N(0, noise²)
        # 这样模拟了真实数据中的测量误差和随机扰动
        noise = np.random.normal(0, self.noise, self.n_samples)
        y = y_true + noise
        
        # 将X转换为列向量，便于矩阵运算
        X = X.reshape(-1, 1)
        
        return X, y
    
    def split_data(self, X, y, train_ratio=0.7, val_ratio=0.15):
        """
        将数据集划分为训练集、验证集和测试集
        
        算法原理：
        1. 首先划分出训练集
        2. 将剩余数据按比例划分为验证集和测试集
        3. 使用分层抽样确保各数据集分布一致
        
        参数：
        X: 自变量
        y: 因变量
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        
        返回：
        (X_train, y_train): 训练数据
        (X_val, y_val): 验证数据
        (X_test, y_test): 测试数据
        """
        # 计算测试集比例
        test_ratio = 1 - train_ratio - val_ratio
        
        # 第一步：划分训练集和临时集（验证+测试）
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_ratio + test_ratio), 
            random_state=self.random_state
        )
        
        # 第二步：将临时集划分为验证集和测试集
        # 计算验证集在临时集中的比例
        val_ratio_in_temp = val_ratio / (val_ratio + test_ratio)
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio_in_temp),
            random_state=self.random_state
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def visualize_data(self, X, y, title="数据分布"):
        """
        可视化数据分布
        
        参数：
        X: 自变量
        y: 因变量
        title: 图表标题
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, alpha=0.6, s=20)
        plt.xlabel('X (自变量)')
        plt.ylabel('y (因变量)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def save_data(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        保存数据集到文件（同时保存为npy和json格式）
        
        算法原理：
        1. 保存为npy格式：适合Python程序快速加载和数值计算
        2. 保存为json格式：便于跨语言使用和实际查询，人类可读
        
        参数：
        各数据集的X和y
        """
        # 确保数据目录存在
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # 1. 保存为npy格式（适合Python数值计算）
        np.save('data/X_train.npy', X_train)
        np.save('data/y_train.npy', y_train)
        np.save('data/X_val.npy', X_val)
        np.save('data/y_val.npy', y_val)
        np.save('data/X_test.npy', X_test)
        np.save('data/y_test.npy', y_test)
        
        # 2. 保存为json格式（便于实际查询和跨语言使用）
        # 将numpy数组转换为列表，因为json不支持numpy数据类型
        data_json = {
            'metadata': {
                'description': '线性回归数据集',
                'true_weight': self.true_weight,
                'true_bias': self.true_bias,
                'noise_level': self.noise,
                'total_samples': self.n_samples,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test)
            },
            'train': {
                'X': X_train.flatten().tolist(),  # 将二维数组转换为一维列表
                'y': y_train.tolist()
            },
            'validation': {
                'X': X_val.flatten().tolist(),
                'y': y_val.tolist()
            },
            'test': {
                'X': X_test.flatten().tolist(),
                'y': y_test.tolist()
            }
        }
        
        # 保存为json文件，使用ensure_ascii=False支持中文字符
        with open('data/dataset.json', 'w', encoding='utf-8') as f:
            json.dump(data_json, f, indent=2, ensure_ascii=False)
        
        # 3. 保存为单独的json文件，便于实际查询
        # 训练集
        train_data = []
        for i in range(len(X_train)):
            train_data.append({
                'id': i,
                'X': float(X_train[i][0]),  # 转换为Python float类型
                'y': float(y_train[i])
            })
        
        with open('data/train_data.json', 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        # 验证集
        val_data = []
        for i in range(len(X_val)):
            val_data.append({
                'id': i,
                'X': float(X_val[i][0]),
                'y': float(y_val[i])
            })
        
        with open('data/val_data.json', 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        # 测试集
        test_data = []
        for i in range(len(X_test)):
            test_data.append({
                'id': i,
                'X': float(X_test[i][0]),
                'y': float(y_test[i])
            })
        
        with open('data/test_data.json', 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        print("数据集已保存到data/目录")
        print("  - npy格式：适合Python程序快速加载")
        print("  - json格式：便于实际查询和跨语言使用")
        print("    * dataset.json: 完整数据集（包含元数据）")
        print("    * train_data.json: 训练集（每条数据包含id, X, y）")
        print("    * val_data.json: 验证集")
        print("    * test_data.json: 测试集")

if __name__ == "__main__":
    # 创建数据生成器实例
    data_gen = DataGenerator(n_samples=1000, noise=0.5)
    
    # 生成数据
    X, y = data_gen.generate_data()
    
    # 划分数据集
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_gen.split_data(X, y)
    
    # 打印数据集信息
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    print(f"真实权重: {data_gen.true_weight}, 真实偏置: {data_gen.true_bias}")
    
    # 可视化完整数据集
    data_gen.visualize_data(X, y, "完整数据集分布")
    
    # 可视化训练集
    data_gen.visualize_data(X_train, y_train, "训练集分布")
    
    # 创建数据目录并保存数据
    import os
    if not os.path.exists('data'):
        os.makedirs('data')
    
    data_gen.save_data(X_train, y_train, X_val, y_val, X_test, y_test)