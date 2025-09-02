import json
import numpy as np
import matplotlib.pyplot as plt

class DataQueryTool:
    """
    数据查询工具：用于查询和分析JSON格式的数据集
    
    功能特点：
    1. 加载JSON格式的数据集
    2. 提供数据查询和统计功能
    3. 支持数据可视化
    4. 生成数据报告
    """
    
    def __init__(self, data_dir='data'):
        """
        初始化数据查询工具
        
        参数：
        data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.dataset_info = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def load_data(self):
        """
        加载JSON格式的数据集
        
        算法原理：
        1. 加载完整数据集信息
        2. 分别加载训练集、验证集和测试集
        3. 验证数据完整性
        """
        try:
            # 加载完整数据集信息
            with open(f'{self.data_dir}/dataset.json', 'r', encoding='utf-8') as f:
                self.dataset_info = json.load(f)
            
            # 加载各数据集
            with open(f'{self.data_dir}/train_data.json', 'r', encoding='utf-8') as f:
                self.train_data = json.load(f)
            
            with open(f'{self.data_dir}/val_data.json', 'r', encoding='utf-8') as f:
                self.val_data = json.load(f)
            
            with open(f'{self.data_dir}/test_data.json', 'r', encoding='utf-8') as f:
                self.test_data = json.load(f)
            
            print("数据集加载成功！")
            return True
            
        except FileNotFoundError as e:
            print(f"数据文件未找到: {e}")
            print("请先运行 data_generation.py 生成数据集")
            return False
        except json.JSONDecodeError as e:
            print(f"JSON文件格式错误: {e}")
            return False
    
    def get_dataset_info(self):
        """
        获取数据集基本信息
        
        返回：
        数据集的元数据信息
        """
        if not self.dataset_info:
            print("数据集未加载，请先调用 load_data()")
            return None
        
        metadata = self.dataset_info['metadata']
        print("\n=== 数据集信息 ===")
        print(f"描述: {metadata['description']}")
        print(f"真实权重: {metadata['true_weight']}")
        print(f"真实偏置: {metadata['true_bias']}")
        print(f"噪声水平: {metadata['noise_level']}")
        print(f"总样本数: {metadata['total_samples']}")
        print(f"训练集样本数: {metadata['train_samples']}")
        print(f"验证集样本数: {metadata['val_samples']}")
        print(f"测试集样本数: {metadata['test_samples']}")
        
        return metadata
    
    def query_data_by_id(self, dataset_type, data_id):
        """
        根据ID查询数据
        
        参数：
        dataset_type: 数据集类型 ('train', 'val', 'test')
        data_id: 数据ID
        
        返回：
        查询到的数据点
        """
        if dataset_type == 'train':
            data = self.train_data
        elif dataset_type == 'val':
            data = self.val_data
        elif dataset_type == 'test':
            data = self.test_data
        else:
            print("无效的数据集类型，请使用 'train', 'val', 或 'test'")
            return None
        
        # 查找指定ID的数据
        for item in data:
            if item['id'] == data_id:
                return item
        
        print(f"在{dataset_type}数据集中未找到ID为{data_id}的数据")
        return None
    
    def query_data_by_range(self, dataset_type, x_min=None, x_max=None, y_min=None, y_max=None):
        """
        根据数值范围查询数据
        
        参数：
        dataset_type: 数据集类型 ('train', 'val', 'test')
        x_min, x_max: X值范围
        y_min, y_max: Y值范围
        
        返回：
        符合条件的数据列表
        """
        if dataset_type == 'train':
            data = self.train_data
        elif dataset_type == 'val':
            data = self.val_data
        elif dataset_type == 'test':
            data = self.test_data
        else:
            print("无效的数据集类型，请使用 'train', 'val', 或 'test'")
            return []
        
        filtered_data = []
        
        for item in data:
            x_val = item['X']
            y_val = item['y']
            
            # 检查X范围
            if x_min is not None and x_val < x_min:
                continue
            if x_max is not None and x_val > x_max:
                continue
            
            # 检查Y范围
            if y_min is not None and y_val < y_min:
                continue
            if y_max is not None and y_val > y_max:
                continue
            
            filtered_data.append(item)
        
        print(f"在{dataset_type}数据集中找到{len(filtered_data)}条符合条件的数据")
        return filtered_data
    
    def get_statistics(self, dataset_type):
        """
        获取数据集的统计信息
        
        参数：
        dataset_type: 数据集类型 ('train', 'val', 'test')
        
        返回：
        统计信息字典
        """
        if dataset_type == 'train':
            data = self.train_data
        elif dataset_type == 'val':
            data = self.val_data
        elif dataset_type == 'test':
            data = self.test_data
        else:
            print("无效的数据集类型，请使用 'train', 'val', 或 'test'")
            return None
        
        # 提取X和Y值
        x_values = [item['X'] for item in data]
        y_values = [item['y'] for item in data]
        
        # 计算统计信息
        stats = {
            'count': len(data),
            'X': {
                'min': min(x_values),
                'max': max(x_values),
                'mean': sum(x_values) / len(x_values),
                'std': np.std(x_values)
            },
            'Y': {
                'min': min(y_values),
                'max': max(y_values),
                'mean': sum(y_values) / len(y_values),
                'std': np.std(y_values)
            }
        }
        
        print(f"\n=== {dataset_type.upper()}数据集统计信息 ===")
        print(f"样本数量: {stats['count']}")
        print(f"X值 - 最小值: {stats['X']['min']:.4f}, 最大值: {stats['X']['max']:.4f}")
        print(f"X值 - 平均值: {stats['X']['mean']:.4f}, 标准差: {stats['X']['std']:.4f}")
        print(f"Y值 - 最小值: {stats['Y']['min']:.4f}, 最大值: {stats['Y']['max']:.4f}")
        print(f"Y值 - 平均值: {stats['Y']['mean']:.4f}, 标准差: {stats['Y']['std']:.4f}")
        
        return stats
    
    def plot_data(self, dataset_type, title=None):
        """
        可视化数据集
        
        参数：
        dataset_type: 数据集类型 ('train', 'val', 'test')
        title: 图表标题
        """
        if dataset_type == 'train':
            data = self.train_data
        elif dataset_type == 'val':
            data = self.val_data
        elif dataset_type == 'test':
            data = self.test_data
        else:
            print("无效的数据集类型，请使用 'train', 'val', 或 'test'")
            return
        
        # 提取X和Y值
        x_values = [item['X'] for item in data]
        y_values = [item['y'] for item in data]
        
        # 绘制散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, y_values, alpha=0.6, s=20)
        
        # 设置图表标题和标签
        if title is None:
            title = f"{dataset_type.upper()}数据集分布"
        plt.title(title)
        plt.xlabel('X (自变量)')
        plt.ylabel('y (因变量)')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_all_datasets(self):
        """
        绘制所有数据集的对比图
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('各数据集分布对比', fontsize=16)
        
        datasets = [
            (self.train_data, '训练集', axes[0]),
            (self.val_data, '验证集', axes[1]),
            (self.test_data, '测试集', axes[2])
        ]
        
        for data, title, ax in datasets:
            x_values = [item['X'] for item in data]
            y_values = [item['y'] for item in data]
            
            ax.scatter(x_values, y_values, alpha=0.6, s=15)
            ax.set_title(title)
            ax.set_xlabel('X (自变量)')
            ax.set_ylabel('y (因变量)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def export_to_csv(self, dataset_type, filename=None):
        """
        将数据导出为CSV格式
        
        参数：
        dataset_type: 数据集类型 ('train', 'val', 'test')
        filename: 输出文件名
        """
        if dataset_type == 'train':
            data = self.train_data
        elif dataset_type == 'val':
            data = self.val_data
        elif dataset_type == 'test':
            data = self.test_data
        else:
            print("无效的数据集类型，请使用 'train', 'val', 或 'test'")
            return
        
        if filename is None:
            filename = f'{dataset_type}_data.csv'
        
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['id', 'X', 'y']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for item in data:
                writer.writerow(item)
        
        print(f"数据已导出到 {filename}")
    
    def generate_report(self):
        """
        生成完整的数据分析报告
        """
        if not self.dataset_info:
            print("数据集未加载，请先调用 load_data()")
            return
        
        print("\n" + "="*60)
        print("线性回归数据集分析报告")
        print("="*60)
        
        # 数据集基本信息
        self.get_dataset_info()
        
        # 各数据集统计信息
        for dataset_type in ['train', 'val', 'test']:
            self.get_statistics(dataset_type)
        
        # 数据可视化
        print("\n生成数据可视化图表...")
        self.plot_all_datasets()
        
        print("\n报告生成完成！")

def main():
    """
    主函数：演示数据查询工具的使用方法
    """
    # 创建数据查询工具实例
    query_tool = DataQueryTool()
    
    # 加载数据
    if not query_tool.load_data():
        return
    
    # 生成完整报告
    query_tool.generate_report()
    
    # 演示查询功能
    print("\n" + "="*40)
    print("数据查询功能演示")
    print("="*40)
    
    # 按ID查询
    print("\n1. 按ID查询数据：")
    sample_data = query_tool.query_data_by_id('train', 0)
    if sample_data:
        print(f"ID 0的数据: {sample_data}")
    
    # 按范围查询
    print("\n2. 按范围查询数据：")
    filtered_data = query_tool.query_data_by_range('train', x_min=0, x_max=5)
    if filtered_data:
        print(f"X值在[0, 5]范围内的前3条数据:")
        for i, item in enumerate(filtered_data[:3]):
            print(f"  {i+1}. ID={item['id']}, X={item['X']:.4f}, y={item['y']:.4f}")
    
    # 导出数据
    print("\n3. 导出数据为CSV格式：")
    query_tool.export_to_csv('train', 'train_sample.csv')

if __name__ == "__main__":
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    main()