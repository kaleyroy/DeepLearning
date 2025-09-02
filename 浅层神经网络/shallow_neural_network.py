import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import os

class ShallowNeuralNetwork:
    """浅层神经网络类，实现具有一个隐藏层的神经网络进行分类任务"""
    
    def __init__(self, hidden_size=10, learning_rate=0.01, n_iterations=1000, 
                 regularization=0.01, random_state=42, activation='relu'):
        """
        初始化浅层神经网络
        
        参数:
        - hidden_size: 隐藏层神经元数量
        - learning_rate: 学习率，控制梯度下降的步长
        - n_iterations: 迭代次数
        - regularization: 正则化系数，防止过拟合
        - random_state: 随机种子
        - activation: 激活函数类型 ('relu', 'sigmoid', 'tanh')
        """
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.random_state = random_state
        self.activation_type = activation
        np.random.seed(random_state)
        
        # 模型参数
        self.W1 = None  # 输入层到隐藏层的权重矩阵
        self.b1 = None  # 隐藏层的偏置向量
        self.W2 = None  # 隐藏层到输出层的权重矩阵
        self.b2 = None  # 输出层的偏置向量
        self.loss_history = []
        
    def _sigmoid(self, z):
        """
        Sigmoid激活函数
        
        算法原理:
        Sigmoid函数将任意实数值映射到(0,1)区间，公式为:
        σ(z) = 1 / (1 + e^(-z))
        
        特点:
        - 输出范围在(0,1)之间
        - 在z=0处导数最大，为0.25
        - 存在梯度消失问题
        
        参数:
        - z: 线性组合的结果
        
        返回:
        - 激活值，范围在(0,1)之间
        """
        # 防止数值溢出，限制z的范围
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _sigmoid_derivative(self, z):
        """
        Sigmoid函数的导数
        
        算法原理:
        Sigmoid函数的导数为: σ'(z) = σ(z) * (1 - σ(z))
        
        这个性质使得在反向传播中计算梯度非常方便
        
        参数:
        - z: Sigmoid函数的输入
        
        返回:
        - Sigmoid函数的导数值
        """
        s = self._sigmoid(z)
        return s * (1 - s)
    
    def _relu(self, z):
        """
        ReLU激活函数（Rectified Linear Unit）
        
        算法原理:
        ReLU函数定义为: ReLU(z) = max(0, z)
        
        特点:
        - 计算简单，效率高
        - 解决了梯度消失问题
        - 存在神经元死亡问题（当z<0时，梯度为0）
        
        参数:
        - z: 线性组合的结果
        
        返回:
        - 激活值，大于等于0
        """
        return np.maximum(0, z)
    
    def _relu_derivative(self, z):
        """
        ReLU函数的导数
        
        算法原理:
        ReLU函数的导数为:
        ReLU'(z) = 1, 如果 z > 0
                 = 0, 如果 z <= 0
        
        参数:
        - z: ReLU函数的输入
        
        返回:
        - ReLU函数的导数值
        """
        return (z > 0).astype(float)
    
    def _tanh(self, z):
        """
        Tanh激活函数（双曲正切函数）
        
        算法原理:
        Tanh函数定义为: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
        
        特点:
        - 输出范围在(-1,1)之间
        - 在原点处导数为1，收敛速度比Sigmoid快
        - 仍然存在梯度消失问题
        
        参数:
        - z: 线性组合的结果
        
        返回:
        - 激活值，范围在(-1,1)之间
        """
        return np.tanh(z)
    
    def _tanh_derivative(self, z):
        """
        Tanh函数的导数
        
        算法原理:
        Tanh函数的导数为: tanh'(z) = 1 - tanh^2(z)
        
        参数:
        - z: Tanh函数的输入
        
        返回:
        - Tanh函数的导数值
        """
        return 1 - np.tanh(z) ** 2
    
    def _activation(self, z):
        """
        根据选择的激活函数类型返回相应的激活函数
        
        参数:
        - z: 线性组合的结果
        
        返回:
        - 激活值
        """
        if self.activation_type == 'sigmoid':
            return self._sigmoid(z)
        elif self.activation_type == 'relu':
            return self._relu(z)
        elif self.activation_type == 'tanh':
            return self._tanh(z)
        else:
            raise ValueError(f"不支持的激活函数: {self.activation_type}")
    
    def _activation_derivative(self, z):
        """
        根据选择的激活函数类型返回相应的导数函数
        
        参数:
        - z: 激活函数的输入
        
        返回:
        - 激活函数的导数值
        """
        if self.activation_type == 'sigmoid':
            return self._sigmoid_derivative(z)
        elif self.activation_type == 'relu':
            return self._relu_derivative(z)
        elif self.activation_type == 'tanh':
            return self._tanh_derivative(z)
        else:
            raise ValueError(f"不支持的激活函数: {self.activation_type}")
    
    def _softmax(self, z):
        """
        Softmax激活函数，用于多分类问题的输出层
        
        算法原理:
        Softmax函数将向量转换为概率分布，公式为:
        softmax(z_i) = e^(z_i) / Σ(e^(z_j))
        
        其中j遍历所有类别，确保所有类别的概率和为1
        
        特点:
        - 输出为概率分布，所有值在(0,1)之间
        - 所有类别的概率和为1
        - 适合多分类问题
        
        参数:
        - z: 线性组合的结果矩阵，形状为(n_samples, n_classes)
        
        返回:
        - 概率分布矩阵，每行和为1
        """
        # 数值稳定性处理：减去每行的最大值
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _initialize_parameters(self, n_features, n_classes):
        """
        初始化神经网络参数
        
        算法原理:
        1. 权重初始化使用Xavier初始化或He初始化
        2. 偏置初始化为0
        3. 合适的初始化可以避免梯度消失或爆炸问题
        
        参数:
        - n_features: 输入特征数量
        - n_classes: 输出类别数量
        """
        # 输入层到隐藏层的权重矩阵
        # 使用Xavier初始化：权重服从均值为0，方差为1/n_in的正态分布
        if self.activation_type == 'relu':
            # ReLU使用He初始化
            self.W1 = np.random.randn(n_features, self.hidden_size) * np.sqrt(2.0 / n_features)
        else:
            # Sigmoid和Tanh使用Xavier初始化
            self.W1 = np.random.randn(n_features, self.hidden_size) * np.sqrt(1.0 / n_features)
        
        self.b1 = np.zeros((1, self.hidden_size))
        
        # 隐藏层到输出层的权重矩阵
        if self.activation_type == 'relu':
            self.W2 = np.random.randn(self.hidden_size, n_classes) * np.sqrt(2.0 / self.hidden_size)
        else:
            self.W2 = np.random.randn(self.hidden_size, n_classes) * np.sqrt(1.0 / self.hidden_size)
        
        self.b2 = np.zeros((1, n_classes))
    
    def _forward_propagation(self, X):
        """
        前向传播过程
        
        算法原理:
        1. 计算隐藏层的线性组合: Z1 = X * W1 + b1
        2. 应用激活函数: A1 = activation(Z1)
        3. 计算输出层的线性组合: Z2 = A1 * W2 + b2
        4. 应用输出层激活函数: A2 = softmax(Z2)
        
        参数:
        - X: 输入特征矩阵，形状为(n_samples, n_features)
        
        返回:
        - A2: 输出层的激活值（预测概率）
        - cache: 缓存的中间结果，用于反向传播
        """
        # 隐藏层的线性组合和激活
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._activation(Z1)
        
        # 输出层的线性组合和激活
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self._softmax(Z2)
        
        # 缓存中间结果用于反向传播
        cache = {
            'Z1': Z1,
            'A1': A1,
            'Z2': Z2,
            'A2': A2
        }
        
        return A2, cache
    
    def _compute_loss(self, X, y, cache):
        """
        计算交叉熵损失函数
        
        算法原理:
        多分类交叉熵损失函数公式为:
        L = -1/n * ΣΣ[y_ij * log(p_ij)] + λ/2 * (||W1||^2_F + ||W2||^2_F)
        
        其中:
        - y_ij: 第i个样本属于第j类的指示器(0或1)
        - p_ij: 第i个样本属于第j类的预测概率
        - ||W||^2_F: 权重矩阵的Frobenius范数平方
        - λ: 正则化系数
        
        参数:
        - X: 特征矩阵
        - y: 真实标签
        - cache: 前向传播的缓存结果
        
        返回:
        - 损失值
        """
        n_samples = X.shape[0]
        A2 = cache['A2']
        
        # 将标签转换为one-hot编码
        n_classes = A2.shape[1]
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y] = 1
        
        # 计算交叉熵损失
        epsilon = 1e-15
        A2 = np.clip(A2, epsilon, 1 - epsilon)
        
        loss = -1/n_samples * np.sum(y_onehot * np.log(A2))
        
        # 添加L2正则化项
        reg_loss = self.regularization / 2 * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        
        return loss + reg_loss
    
    def _backward_propagation(self, X, y, cache):
        """
        反向传播过程
        
        算法原理:
        使用链式法则计算梯度:
        1. 计算输出层的梯度: dZ2 = A2 - Y
        2. 计算W2和b2的梯度: dW2 = A1^T * dZ2, db2 = ΣdZ2
        3. 计算隐藏层的梯度: dZ1 = (dZ2 * W2^T) ⊙ activation'(Z1)
        4. 计算W1和b1的梯度: dW1 = X^T * dZ1, db1 = ΣdZ1
        
        参数:
        - X: 输入特征矩阵
        - y: 真实标签
        - cache: 前向传播的缓存结果
        
        返回:
        - gradients: 包含所有参数梯度的字典
        """
        n_samples = X.shape[0]
        
        # 从缓存中获取前向传播的结果
        A1 = cache['A1']
        A2 = cache['A2']
        Z1 = cache['Z1']
        
        # 将标签转换为one-hot编码
        n_classes = A2.shape[1]
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y] = 1
        
        # 输出层的梯度
        dZ2 = A2 - y_onehot
        
        # W2和b2的梯度
        dW2 = 1/n_samples * np.dot(A1.T, dZ2) + self.regularization * self.W2
        db2 = 1/n_samples * np.sum(dZ2, axis=0, keepdims=True)
        
        # 隐藏层的梯度
        dZ1 = np.dot(dZ2, self.W2.T) * self._activation_derivative(Z1)
        
        # W1和b1的梯度
        dW1 = 1/n_samples * np.dot(X.T, dZ1) + self.regularization * self.W1
        db1 = 1/n_samples * np.sum(dZ1, axis=0, keepdims=True)
        
        gradients = {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }
        
        return gradients
    
    def _update_parameters(self, gradients):
        """
        更新模型参数
        
        算法原理:
        使用梯度下降算法更新参数:
        W = W - learning_rate * dW
        b = b - learning_rate * db
        
        参数:
        - gradients: 包含所有参数梯度的字典
        """
        self.W1 -= self.learning_rate * gradients['dW1']
        self.b1 -= self.learning_rate * gradients['db1']
        self.W2 -= self.learning_rate * gradients['dW2']
        self.b2 -= self.learning_rate * gradients['db2']
    
    def fit(self, X, y, n_classes=None, verbose=True):
        """
        训练浅层神经网络
        
        算法原理:
        使用梯度下降算法优化模型参数:
        1. 初始化网络参数
        2. 对于每次迭代:
           a. 前向传播计算预测值
           b. 计算损失函数
           c. 反向传播计算梯度
           d. 更新网络参数
        3. 重复步骤2直到收敛
        
        参数:
        - X: 训练特征矩阵，形状为(n_samples, n_features)
        - y: 训练标签，形状为(n_samples,)
        - n_classes: 类别数量，如果为None则自动推断
        - verbose: 是否打印训练过程信息
        """
        n_samples, n_features = X.shape
        
        # 确定类别数量
        if n_classes is None:
            n_classes = len(np.unique(y))
        
        # 初始化模型参数
        self._initialize_parameters(n_features, n_classes)
        
        # 训练循环
        for iteration in range(self.n_iterations):
            # 前向传播
            A2, cache = self._forward_propagation(X)
            
            # 计算损失
            loss = self._compute_loss(X, y, cache)
            self.loss_history.append(loss)
            
            # 反向传播
            gradients = self._backward_propagation(X, y, cache)
            
            # 更新参数
            self._update_parameters(gradients)
            
            # 打印训练进度
            if verbose and (iteration + 1) % 100 == 0:
                print(f"迭代 {iteration + 1}/{self.n_iterations}, 损失: {loss:.6f}")
    
    def predict(self, X):
        """
        预测样本的类别
        
        参数:
        - X: 输入特征矩阵
        
        返回:
        - 预测的类别标签
        """
        A2, _ = self._forward_propagation(X)
        return np.argmax(A2, axis=1)
    
    def predict_proba(self, X):
        """
        预测样本属于各个类别的概率
        
        参数:
        - X: 输入特征矩阵
        
        返回:
        - 预测的概率矩阵
        """
        A2, _ = self._forward_propagation(X)
        return A2
    
    def evaluate(self, X, y):
        """
        评估模型性能
        
        参数:
        - X: 特征矩阵
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
    
    def plot_loss_history(self, save_path=None):
        """
        绘制损失函数变化曲线
        
        参数:
        - save_path: 图片保存路径
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.xlabel('迭代次数')
        plt.ylabel('损失值')
        plt.title('训练损失变化曲线')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"损失曲线已保存到 {save_path}")
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
        plt.title('浅层神经网络决策边界')
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
        保存模型参数
        
        参数:
        - save_path: 模型保存路径
        """
        model_params = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'hidden_size': self.hidden_size,
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations,
            'regularization': self.regularization,
            'activation': self.activation_type,
            'loss_history': self.loss_history
        }
        
        np.save(save_path, model_params)
        print(f"模型已保存到 {save_path}")
    
    def load_model(self, load_path):
        """
        加载模型参数
        
        参数:
        - load_path: 模型加载路径
        """
        model_params = np.load(load_path, allow_pickle=True).item()
        
        self.W1 = model_params['W1']
        self.b1 = model_params['b1']
        self.W2 = model_params['W2']
        self.b2 = model_params['b2']
        self.hidden_size = model_params['hidden_size']
        self.learning_rate = model_params['learning_rate']
        self.n_iterations = model_params['n_iterations']
        self.regularization = model_params['regularization']
        self.activation_type = model_params['activation']
        self.loss_history = model_params['loss_history']
        
        print(f"模型已从 {load_path} 加载")