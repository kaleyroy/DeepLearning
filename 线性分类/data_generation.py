import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json

class DataGenerator:
    """数据生成器类，用于生成线性分类问题的训练、验证和测试数据集"""
    
    def __init__(self, n_samples=1000, n_features=2, n_classes=2, random_state=42):
        """
        初始化数据生成器
        
        参数:
        - n_samples: 样本总数
        - n_features: 特征数量
        - n_classes: 类别数量
        - random_state: 随机种子
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_linear_separable_data(self):
        """
        生成线性可分的数据集
        
        算法原理:
        1. 为每个类别生成随机的中心点
        2. 在每个中心点周围生成服从高斯分布的数据点
        3. 添加一定的噪声使数据更真实
        4. 确保数据线性可分
        """
        # 为每个类别生成中心点
        centers = np.random.randn(self.n_classes, self.n_features) * 3
        
        # 生成每个类别的数据
        X = []
        y = []
        samples_per_class = self.n_samples // self.n_classes
        
        for class_idx in range(self.n_classes):
            # 在类别中心点周围生成高斯分布的数据点
            class_samples = np.random.randn(samples_per_class, self.n_features) * 0.5 + centers[class_idx]
            X.append(class_samples)
            y.extend([class_idx] * samples_per_class)
        
        X = np.vstack(X)
        y = np.array(y)
        
        return X, y
    
    def split_and_save_data(self, X, y, test_size=0.2, val_size=0.1):
        """
        将数据分割为训练集、验证集和测试集，并保存到文件
        
        参数:
        - X: 特征数据
        - y: 标签数据
        - test_size: 测试集比例
        - val_size: 验证集比例
        """
        # 创建数据文件夹
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # 首先分割出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # 从剩余数据中分割出验证集
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state, stratify=y_temp
        )
        
        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # 保存数据到文件（npy格式）
        np.save(os.path.join(data_dir, 'X_train.npy'), X_train_scaled)
        np.save(os.path.join(data_dir, 'X_val.npy'), X_val_scaled)
        np.save(os.path.join(data_dir, 'X_test.npy'), X_test_scaled)
        np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(data_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(data_dir, 'y_test.npy'), y_test)
        
        # 保存原始数据（未标准化）用于可视化
        np.save(os.path.join(data_dir, 'X_train_raw.npy'), X_train)
        np.save(os.path.join(data_dir, 'X_val_raw.npy'), X_val)
        np.save(os.path.join(data_dir, 'X_test_raw.npy'), X_test)
        
        # 保存数据到文件（JSON格式）
        # 将numpy数组转换为列表以便JSON序列化
        json_data = {
            'X_train': X_train_scaled.tolist(),
            'X_val': X_val_scaled.tolist(),
            'X_test': X_test_scaled.tolist(),
            'y_train': y_train.tolist(),
            'y_val': y_val.tolist(),
            'y_test': y_test.tolist(),
            'X_train_raw': X_train.tolist(),
            'X_val_raw': X_val.tolist(),
            'X_test_raw': X_test.tolist(),
            'data_info': {
                'n_samples': self.n_samples,
                'n_features': self.n_features,
                'n_classes': self.n_classes,
                'train_size': X_train.shape[0],
                'val_size': X_val.shape[0],
                'test_size': X_test.shape[0],
                'random_state': self.random_state
            }
        }
        
        # 保存为JSON文件
        with open(os.path.join(data_dir, 'dataset.json'), 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        
        # 分别保存各个数据集的JSON文件
        with open(os.path.join(data_dir, 'train_data.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'X': X_train_scaled.tolist(),
                'y': y_train.tolist(),
                'X_raw': X_train.tolist()
            }, f, ensure_ascii=False, indent=4)
        
        with open(os.path.join(data_dir, 'val_data.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'X': X_val_scaled.tolist(),
                'y': y_val.tolist(),
                'X_raw': X_val.tolist()
            }, f, ensure_ascii=False, indent=4)
        
        with open(os.path.join(data_dir, 'test_data.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'X': X_test_scaled.tolist(),
                'y': y_test.tolist(),
                'X_raw': X_test.tolist()
            }, f, ensure_ascii=False, indent=4)
        
        print(f"数据集已生成并保存到 {data_dir}")
        print(f"训练集大小: {X_train.shape[0]}")
        print(f"验证集大小: {X_val.shape[0]}")
        print(f"测试集大小: {X_test.shape[0]}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def visualize_data(self, X, y, save_path=None):
        """
        可视化生成的数据集
        
        参数:
        - X: 特征数据
        - y: 标签数据
        - save_path: 图片保存路径
        """
        if self.n_features != 2:
            print("只能可视化2维数据")
            return
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=(10, 8))
        
        # 为每个类别绘制散点图
        for class_idx in range(self.n_classes):
            mask = y == class_idx
            plt.scatter(X[mask, 0], X[mask, 1], label=f'类别 {class_idx}', alpha=0.7)
        
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.title('线性分类数据集可视化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"数据可视化图片已保存到 {save_path}")
        else:
            plt.show()
        
        plt.close()

def main():
    """主函数：生成数据集并可视化"""
    # 创建数据生成器
    data_gen = DataGenerator(n_samples=1000, n_features=2, n_classes=2, random_state=42)
    
    # 生成线性可分数据
    X, y = data_gen.generate_linear_separable_data()
    
    # 分割并保存数据
    X_train, X_val, X_test, y_train, y_val, y_test = data_gen.split_and_save_data(X, y)
    
    # 可视化数据
    data_gen.visualize_data(X, y, save_path='data/data_visualization.png')
    
    print("数据生成完成！")

if __name__ == "__main__":
    main()