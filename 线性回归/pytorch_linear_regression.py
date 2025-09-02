import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

class PyTorchLinearRegression:
    """
    基于PyTorch的线性回归模型类
    
    算法原理：
    使用PyTorch的自动微分和优化器来实现线性回归
    模型结构：y = wX + b
    通过最小化均方误差(MSE)来学习参数w和b
    
    特点：
    - 使用GPU加速计算（如果可用）
    - 支持批量训练
    - 自动梯度计算
    - 灵活的优化器选择
    """
    
    def __init__(self, input_dim=1, learning_rate=0.01, n_iterations=1000, 
                 batch_size=32, optimizer='sgd', device=None):
        """
        初始化PyTorch线性回归模型
        
        参数：
        input_dim: 输入特征维度
        learning_rate: 学习率
        n_iterations: 最大迭代次数
        batch_size: 批量大小
        optimizer: 优化器类型 ('sgd', 'adam', 'rmsprop')
        device: 计算设备 ('cpu', 'cuda')，自动选择最佳设备
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.optimizer_type = optimizer
        
        # 自动选择计算设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 初始化模型
        self.model = nn.Linear(input_dim, 1).to(self.device)
        self.criterion = nn.MSELoss()
        
        # 初始化优化器
        self._init_optimizer()
        
        # 训练历史记录
        self.loss_history = []
        self.train_loss_history = []
        self.val_loss_history = []
        
        print(f"模型初始化完成，使用设备: {self.device}")
    
    def _init_optimizer(self):
        """
        初始化优化器
        """
        if self.optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type.lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"不支持的优化器类型: {self.optimizer_type}")
    
    def _prepare_data(self, X, y):
        """
        准备数据，转换为PyTorch张量
        """
        # 转换为numpy数组（如果输入是列表或其他格式）
        X = np.array(X)
        y = np.array(y)
        
        # 确保X是2D数组
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # 确保y是2D数组
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        return X_tensor, y_tensor
    
    def fit(self, X, y, X_val=None, y_val=None, verbose=True):
        """
        训练模型
        
        参数：
        X: 训练数据特征
        y: 训练数据标签
        X_val: 验证数据特征（可选）
        y_val: 验证数据标签（可选）
        verbose: 是否打印训练信息
        """
        # 准备训练数据
        X_tensor, y_tensor = self._prepare_data(X, y)
        
        # 创建数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 准备验证数据
        val_dataloader = None
        if X_val is not None and y_val is not None:
            X_val_tensor, y_val_tensor = self._prepare_data(X_val, y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 训练循环
        self.model.train()
        
        for epoch in range(self.n_iterations):
            epoch_train_loss = 0.0
            num_batches = 0
            
            # 训练阶段
            for batch_X, batch_y in dataloader:
                # 清零梯度
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # 反向传播
                loss.backward()
                
                # 更新参数
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
                num_batches += 1
            
            # 计算平均训练损失
            avg_train_loss = epoch_train_loss / num_batches
            self.train_loss_history.append(avg_train_loss)
            
            # 验证阶段
            if val_dataloader is not None:
                self.model.eval()
                epoch_val_loss = 0.0
                val_num_batches = 0
                
                with torch.no_grad():
                    for val_X, val_y in val_dataloader:
                        val_outputs = self.model(val_X)
                        val_loss = self.criterion(val_outputs, val_y)
                        epoch_val_loss += val_loss.item()
                        val_num_batches += 1
                
                avg_val_loss = epoch_val_loss / val_num_batches
                self.val_loss_history.append(avg_val_loss)
                self.model.train()
                
                # 记录总损失（训练损失）
                self.loss_history.append(avg_train_loss)
                
                if verbose and (epoch + 1) % 100 == 0:
                    print(f"Epoch {epoch + 1}/{self.n_iterations}, "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {avg_val_loss:.6f}")
            else:
                # 没有验证数据时，只记录训练损失
                self.loss_history.append(avg_train_loss)
                
                if verbose and (epoch + 1) % 100 == 0:
                    print(f"Epoch {epoch + 1}/{self.n_iterations}, "
                          f"Train Loss: {avg_train_loss:.6f}")
    
    def predict(self, X):
        """
        使用训练好的模型进行预测
        
        参数：
        X: 输入特征
        
        返回：
        y_pred: 预测值（numpy数组）
        """
        self.model.eval()
        
        with torch.no_grad():
            X_tensor, _ = self._prepare_data(X, np.zeros(len(X)))
            predictions = self.model(X_tensor)
            
        return predictions.cpu().numpy().flatten()
    
    def calculate_loss(self, X, y):
        """
        计算模型在给定数据上的损失
        
        参数：
        X: 输入特征
        y: 真实标签
        
        返回：
        mse_loss: 均方误差损失
        """
        self.model.eval()
        
        with torch.no_grad():
            X_tensor, y_tensor = self._prepare_data(X, y)
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            
        return loss.item()
    
    def calculate_r2_score(self, X, y):
        """
        计算R²决定系数
        
        参数：
        X: 输入特征
        y: 真实标签
        
        返回：
        r2_score: R²决定系数
        """
        y_pred = self.predict(X)
        y_mean = np.mean(y)
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        
        r2_score = 1 - (ss_res / ss_tot)
        return r2_score
    
    def plot_loss_history(self):
        """
        可视化训练过程中的损失变化
        """
        plt.figure(figsize=(12, 6))
        
        # 绘制训练损失
        plt.plot(self.train_loss_history, label='训练损失', color='blue')
        
        # 如果有验证损失，也绘制出来
        if self.val_loss_history:
            plt.plot(self.val_loss_history, label='验证损失', color='red')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('PyTorch线性回归训练损失变化曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_regression_line(self, X, y, title="PyTorch线性回归拟合结果"):
        """
        可视化回归线和数据点
        
        参数：
        X: 输入特征
        y: 真实标签
        title: 图表标题
        """
        plt.figure(figsize=(10, 6))
        
        # 绘制原始数据点
        plt.scatter(X, y, alpha=0.6, s=20, label='真实数据')
        
        # 绘制回归线
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_line = self.predict(X_line)
        plt.plot(X_line, y_line, color='red', linewidth=2, label='回归线')
        
        plt.xlabel('X (自变量)')
        plt.ylabel('y (因变量)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_parameters(self):
        """
        获取模型参数
        
        返回：
        (weights, bias): 权重和偏置
        """
        weights = self.model.weight.data.cpu().numpy().flatten()
        bias = self.model.bias.data.cpu().numpy().flatten()[0]
        return weights, bias
    
    def print_parameters(self):
        """
        打印模型参数
        """
        weights, bias = self.get_parameters()
        print(f"权重: {weights}")
        print(f"偏置: {bias:.6f}")
    
    def save_model(self, filepath):
        """
        保存模型
        
        参数：
        filepath: 模型保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations,
            'batch_size': self.batch_size,
            'optimizer_type': self.optimizer_type,
            'loss_history': self.loss_history,
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """
        加载模型
        
        参数：
        filepath: 模型文件路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 重新初始化模型和优化器
        self.input_dim = checkpoint['input_dim']
        self.learning_rate = checkpoint['learning_rate']
        self.n_iterations = checkpoint['n_iterations']
        self.batch_size = checkpoint['batch_size']
        self.optimizer_type = checkpoint['optimizer_type']
        
        self.model = nn.Linear(self.input_dim, 1).to(self.device)
        self._init_optimizer()
        
        # 加载状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载训练历史
        self.loss_history = checkpoint.get('loss_history', [])
        self.train_loss_history = checkpoint.get('train_loss_history', [])
        self.val_loss_history = checkpoint.get('val_loss_history', [])
        
        print(f"模型已从 {filepath} 加载")
    
    def summary(self):
        """
        打印模型摘要信息
        """
        print("=" * 50)
        print("PyTorch线性回归模型摘要")
        print("=" * 50)
        print(f"输入维度: {self.input_dim}")
        print(f"学习率: {self.learning_rate}")
        print(f"最大迭代次数: {self.n_iterations}")
        print(f"批量大小: {self.batch_size}")
        print(f"优化器: {self.optimizer_type}")
        print(f"计算设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        
        if self.loss_history:
            print(f"最终训练损失: {self.loss_history[-1]:.6f}")
        
        if self.val_loss_history:
            print(f"最终验证损失: {self.val_loss_history[-1]:.6f}")
        
        print("=" * 50)


# 使用示例和测试函数
def test_pytorch_linear_regression():
    """
    测试PyTorch线性回归模型
    """
    print("开始测试PyTorch线性回归模型...")
    
    # 生成测试数据
    np.random.seed(42)
    X = np.random.randn(100, 1) * 2
    y = 3 * X.flatten() + 2 + np.random.randn(100) * 0.5
    
    # 分割训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建并训练模型
    model = PyTorchLinearRegression(
        input_dim=1,
        learning_rate=0.01,
        n_iterations=1000,
        batch_size=16,
        optimizer='adam'
    )
    
    model.summary()
    
    # 训练模型
    model.fit(X_train, y_train, X_test, y_test, verbose=True)
    
    # 评估模型
    train_loss = model.calculate_loss(X_train, y_train)
    test_loss = model.calculate_loss(X_test, y_test)
    train_r2 = model.calculate_r2_score(X_train, y_train)
    test_r2 = model.calculate_r2_score(X_test, y_test)
    
    print(f"\n训练集损失: {train_loss:.6f}")
    print(f"测试集损失: {test_loss:.6f}")
    print(f"训练集R²: {train_r2:.6f}")
    print(f"测试集R²: {test_r2:.6f}")
    
    # 打印参数
    model.print_parameters()
    
    # 可视化结果
    model.plot_loss_history()
    model.plot_regression_line(X_train, y_train)
    
    print("测试完成！")


if __name__ == "__main__":
    test_pytorch_linear_regression()