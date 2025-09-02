import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import os

class PyTorchShallowNeuralNetwork(nn.Module):
    """基于PyTorch的浅层神经网络类，实现具有一个隐藏层的神经网络进行分类任务"""
    
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        """
        初始化PyTorch浅层神经网络
        
        参数:
        - input_size: 输入特征数量
        - hidden_size: 隐藏层神经元数量
        - output_size: 输出类别数量
        - activation: 激活函数类型 ('relu', 'sigmoid', 'tanh')
        """
        super(PyTorchShallowNeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_type = activation
        
        # 定义网络层
        # 输入层到隐藏层的线性变换
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 隐藏层到输出层的线性变换
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 输出层使用LogSoftmax用于分类任务
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        初始化网络权重
        
        算法原理:
        1. 使用Xavier初始化或He初始化方法
        2. 合适的权重初始化可以避免梯度消失或爆炸问题
        3. 偏置初始化为0
        """
        # 为fc1层初始化权重
        if self.activation_type == 'relu':
            # ReLU使用He初始化
            nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        else:
            # Sigmoid和Tanh使用Xavier初始化
            nn.init.xavier_normal_(self.fc1.weight)
        
        nn.init.zeros_(self.fc1.bias)
        
        # 为fc2层初始化权重
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        """
        前向传播过程
        
        算法原理:
        1. 输入数据通过第一个全连接层: fc1(x)
        2. 应用激活函数: activation(fc1(x))
        3. 通过第二个全连接层: fc2(activation(fc1(x)))
        4. 应用LogSoftmax得到对数概率: log_softmax(fc2(...))
        
        参数:
        - x: 输入张量，形状为(batch_size, input_size)
        
        返回:
        - 输出张量，形状为(batch_size, output_size)
        """
        # 第一层：输入层到隐藏层
        hidden = self.fc1(x)
        hidden = self.activation(hidden)
        
        # 第二层：隐藏层到输出层
        output = self.fc2(hidden)
        output = self.log_softmax(output)
        
        return output

class PyTorchNeuralNetworkTrainer:
    """PyTorch神经网络训练器类，包含训练、评估和可视化功能"""
    
    def __init__(self, model, learning_rate=0.01, weight_decay=0.01, device='cpu'):
        """
        初始化训练器
        
        参数:
        - model: PyTorch神经网络模型
        - learning_rate: 学习率
        - weight_decay: 权重衰减（L2正则化）
        - device: 计算设备 ('cpu' 或 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # 定义损失函数和优化器
        # 使用负对数似然损失函数，配合LogSoftmax输出
        self.criterion = nn.NLLLoss()
        # 使用Adam优化器，自适应学习率
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 记录训练历史
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
    
    def train(self, X_train, y_train, X_val=None, y_val=None, n_epochs=1000, batch_size=32, verbose=True):
        """
        训练神经网络模型
        
        算法原理:
        1. 将数据转换为PyTorch张量并移动到指定设备
        2. 使用小批量梯度下降进行训练
        3. 每个epoch包含前向传播、损失计算、反向传播和参数更新
        4. 定期在验证集上评估模型性能
        
        参数:
        - X_train: 训练集特征
        - y_train: 训练集标签
        - X_val: 验证集特征（可选）
        - y_val: 验证集标签（可选）
        - n_epochs: 训练轮数
        - batch_size: 批量大小
        - verbose: 是否打印训练进度
        """
        # 将数据转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        print(f"开始训练模型，共{n_epochs}个epoch...")
        
        for epoch in range(n_epochs):
            # 训练模式
            self.model.train()
            epoch_train_loss = 0.0
            epoch_train_correct = 0
            epoch_train_total = 0
            
            for batch_X, batch_y in train_loader:
                # 清零梯度
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # 反向传播和参数更新
                loss.backward()
                self.optimizer.step()
                
                # 记录训练损失
                epoch_train_loss += loss.item()
                
                # 计算训练准确率
                _, predicted = torch.max(outputs.data, 1)
                epoch_train_total += batch_y.size(0)
                epoch_train_correct += (predicted == batch_y).sum().item()
            
            # 计算平均训练损失和准确率
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_accuracy = epoch_train_correct / epoch_train_total
            
            self.train_loss_history.append(avg_train_loss)
            self.train_acc_history.append(train_accuracy)
            
            # 在验证集上评估
            if X_val is not None and y_val is not None:
                val_loss, val_accuracy = self.evaluate(X_val_tensor, y_val_tensor)
                self.val_loss_history.append(val_loss)
                self.val_acc_history.append(val_accuracy)
            
            # 打印训练进度
            if verbose and (epoch + 1) % 100 == 0:
                if X_val is not None and y_val is not None:
                    print(f"Epoch {epoch+1}/{n_epochs}, "
                          f"Train Loss: {avg_train_loss:.6f}, Train Acc: {train_accuracy:.4f}, "
                          f"Val Loss: {val_loss:.6f}, Val Acc: {val_accuracy:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{n_epochs}, "
                          f"Train Loss: {avg_train_loss:.6f}, Train Acc: {train_accuracy:.4f}")
        
        print("模型训练完成！")
    
    def evaluate(self, X, y):
        """
        评估模型性能
        
        参数:
        - X: 特征数据（可以是numpy数组或PyTorch张量）
        - y: 标签数据（可以是numpy数组或PyTorch张量）
        
        返回:
        - loss: 平均损失
        - accuracy: 准确率
        """
        # 确保输入是PyTorch张量
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X).to(self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.LongTensor(y).to(self.device)
        
        # 评估模式
        self.model.eval()
        
        with torch.no_grad():
            # 前向传播
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total = y.size(0)
            correct = (predicted == y).sum().item()
            accuracy = correct / total
        
        return loss.item(), accuracy
    
    def predict(self, X):
        """
        预测样本类别
        
        参数:
        - X: 特征数据（可以是numpy数组或PyTorch张量）
        
        返回:
        - 预测的类别标签（numpy数组）
        """
        # 确保输入是PyTorch张量
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X).to(self.device)
        
        # 评估模式
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        """
        预测样本属于各个类别的概率
        
        参数:
        - X: 特征数据（可以是numpy数组或PyTorch张量）
        
        返回:
        - 预测的概率矩阵（numpy数组）
        """
        # 确保输入是PyTorch张量
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X).to(self.device)
        
        # 评估模式
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(X)
            # 将log_softmax转换为概率
            probabilities = torch.exp(outputs)
        
        return probabilities.cpu().numpy()
    
    def get_metrics(self, X, y):
        """
        获取详细的评估指标
        
        参数:
        - X: 特征数据
        - y: 真实标签
        
        返回:
        - 包含各种评估指标的字典
        """
        y_pred = self.predict(X)
        
        # 计算各种评估指标
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def plot_training_history(self, save_path=None):
        """
        绘制训练历史曲线
        
        参数:
        - save_path: 图片保存路径
        """
        plt.figure(figsize=(12, 5))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_history, label='训练损失')
        if self.val_loss_history:
            plt.plot(self.val_loss_history, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title('训练和验证损失')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc_history, label='训练准确率')
        if self.val_acc_history:
            plt.plot(self.val_acc_history, label='验证准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率')
        plt.title('训练和验证准确率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"训练历史曲线已保存到 {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_decision_boundary(self, X, y, save_path=None):
        """
        绘制决策边界（仅适用于2维特征）
        
        参数:
        - X: 特征矩阵
        - y: 真实标签
        - save_path: 图片保存路径
        """
        if X.shape[1] != 2:
            print("只能绘制2维特征的决策边界")
            return
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建网格
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        
        # 预测网格点的类别
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界和数据点
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        
        # 为每个类别绘制散点图
        n_classes = len(np.unique(y))
        for class_idx in range(n_classes):
            mask = y == class_idx
            plt.scatter(X[mask, 0], X[mask, 1], label=f'类别 {class_idx}', alpha=0.8)
        
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.title('PyTorch浅层神经网络决策边界')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"决策边界图已保存到 {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_model(self, save_path):
        """
        保存模型
        
        参数:
        - save_path: 模型保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
            'train_acc_history': self.train_acc_history,
            'val_acc_history': self.val_acc_history,
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_size': self.model.hidden_size,
                'output_size': self.model.output_size,
                'activation': self.model.activation_type
            },
            'training_config': {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay
            }
        }, save_path)
        print(f"模型已保存到 {save_path}")
    
    def load_model(self, load_path):
        """
        加载模型
        
        参数:
        - load_path: 模型加载路径
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # 重建模型
        model_config = checkpoint['model_config']
        self.model = PyTorchShallowNeuralNetwork(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            output_size=model_config['output_size'],
            activation=model_config['activation']
        ).to(self.device)
        
        # 重建优化器
        training_config = checkpoint['training_config']
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        
        # 加载状态字典
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载训练历史
        self.train_loss_history = checkpoint['train_loss_history']
        self.val_loss_history = checkpoint['val_loss_history']
        self.train_acc_history = checkpoint['train_acc_history']
        self.val_acc_history = checkpoint['val_acc_history']
        
        print(f"模型已从 {load_path} 加载")

# 工具函数：检测GPU可用性
def get_device():
    """
    检测并返回可用的计算设备
    
    返回:
    - 'cuda' 如果GPU可用
    - 'cpu' 如果GPU不可用
    """
    if torch.cuda.is_available():
        print("检测到CUDA设备，将使用GPU进行训练")
        return 'cuda'
    else:
        print("未检测到CUDA设备，将使用CPU进行训练")
        return 'cpu'

# 工具函数：创建模型
def create_model(input_size, hidden_size=10, output_size=2, activation='relu'):
    """
    创建PyTorch浅层神经网络模型
    
    参数:
    - input_size: 输入特征数量
    - hidden_size: 隐藏层神经元数量
    - output_size: 输出类别数量
    - activation: 激活函数类型
    
    返回:
    - PyTorchShallowNeuralNetwork实例
    """
    model = PyTorchShallowNeuralNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        activation=activation
    )
    
    return model