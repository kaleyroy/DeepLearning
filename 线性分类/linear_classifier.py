import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import os

class LinearClassifier:
    """线性分类器类，实现逻辑回归算法进行二分类和多分类任务"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=0.01, random_state=42):
        """
        初始化线性分类器
        
        参数:
        - learning_rate: 学习率，控制梯度下降的步长
        - n_iterations: 迭代次数
        - regularization: 正则化系数，防止过拟合
        - random_state: 随机种子
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.random_state = random_state
        np.random.seed(random_state)
        
        # 模型参数
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def _sigmoid(self, z):
        """
        Sigmoid激活函数
        
        算法原理:
        Sigmoid函数将任意实数值映射到(0,1)区间，公式为:
        σ(z) = 1 / (1 + e^(-z))
        
        在逻辑回归中，Sigmoid函数用于将线性组合的输出转换为概率值
        
        参数:
        - z: 线性组合的结果
        
        返回:
        - 概率值，范围在(0,1)之间
        """
        # 防止数值溢出，限制z的范围
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _softmax(self, z):
        """
        Softmax激活函数，用于多分类问题
        
        算法原理:
        Softmax函数将向量转换为概率分布，公式为:
        softmax(z_i) = e^(z_i) / Σ(e^(z_j))
        
        其中j遍历所有类别，确保所有类别的概率和为1
        
        参数:
        - z: 线性组合的结果矩阵，形状为(n_samples, n_classes)
        
        返回:
        - 概率分布矩阵，每行和为1
        """
        # 数值稳定性处理：减去每行的最大值
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _compute_loss_binary(self, X, y):
        """
        计算二分类问题的交叉熵损失函数
        
        算法原理:
        二分类交叉熵损失函数公式为:
        L = -1/n * Σ[y_i * log(p_i) + (1-y_i) * log(1-p_i)] + λ/2 * ||w||^2
        
        其中:
        - y_i: 真实标签(0或1)
        - p_i: 预测概率
        - λ: 正则化系数
        - ||w||^2: 权重的L2范数平方
        
        参数:
        - X: 特征矩阵
        - y: 真实标签
        
        返回:
        - 损失值
        """
        n_samples = X.shape[0]
        
        # 计算线性组合和预测概率
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_output)
        
        # 计算交叉熵损失
        # 添加小常数防止log(0)的情况
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -1/n_samples * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        
        # 添加L2正则化项
        reg_loss = self.regularization / 2 * np.sum(self.weights ** 2)
        
        return loss + reg_loss
    
    def _compute_loss_multiclass(self, X, y):
        """
        计算多分类问题的交叉熵损失函数
        
        算法原理:
        多分类交叉熵损失函数公式为:
        L = -1/n * ΣΣ[y_ij * log(p_ij)] + λ/2 * ||W||^2_F
        
        其中:
        - y_ij: 第i个样本属于第j类的指示器(0或1)
        - p_ij: 第i个样本属于第j类的预测概率
        - ||W||^2_F: 权重矩阵的Frobenius范数平方
        
        参数:
        - X: 特征矩阵
        - y: 真实标签(需要转换为one-hot编码)
        
        返回:
        - 损失值
        """
        n_samples = X.shape[0]
        
        # 将标签转换为one-hot编码
        n_classes = self.weights.shape[1]
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y] = 1
        
        # 计算线性组合和预测概率
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self._softmax(linear_output)
        
        # 计算交叉熵损失
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -1/n_samples * np.sum(y_onehot * np.log(y_pred))
        
        # 添加L2正则化项
        reg_loss = self.regularization / 2 * np.sum(self.weights ** 2)
        
        return loss + reg_loss
    
    def fit(self, X, y, n_classes=None):
        """
        训练线性分类器
        
        算法原理:
        使用梯度下降算法优化模型参数:
        1. 初始化权重和偏置
        2. 计算预测值和损失
        3. 计算梯度:
           - 对权重w的梯度: ∂L/∂w = 1/n * X^T(y_pred - y) + λw
           - 对偏置b的梯度: ∂L/∂b = 1/n * Σ(y_pred - y)
        4. 更新参数:
           - w = w - learning_rate * ∂L/∂w
           - b = b - learning_rate * ∂L/∂b
        5. 重复步骤2-4直到收敛
        
        参数:
        - X: 训练特征矩阵，形状为(n_samples, n_features)
        - y: 训练标签，形状为(n_samples,)
        - n_classes: 类别数量，如果为None则自动推断
        """
        n_samples, n_features = X.shape
        
        # 确定类别数量
        if n_classes is None:
            n_classes = len(np.unique(y))
        
        # 初始化模型参数
        if n_classes == 2:
            # 二分类：权重向量为(n_features, 1)
            self.weights = np.random.randn(n_features, 1) * 0.01
            self.bias = np.zeros(1)
            
            # 确保y的形状正确
            y = y.reshape(-1, 1)
            
            # 训练循环
            for iteration in range(self.n_iterations):
                # 前向传播
                linear_output = np.dot(X, self.weights) + self.bias
                y_pred = self._sigmoid(linear_output)
                
                # 计算损失
                loss = self._compute_loss_binary(X, y)
                self.loss_history.append(loss)
                
                # 计算梯度
                # 对权重的梯度：∂L/∂w = 1/n * X^T(y_pred - y) + λw
                dw = 1/n_samples * np.dot(X.T, (y_pred - y)) + self.regularization * self.weights
                # 对偏置的梯度：∂L/∂b = 1/n * Σ(y_pred - y)
                db = 1/n_samples * np.sum(y_pred - y)
                
                # 更新参数
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                # 打印训练进度
                if iteration % 100 == 0:
                    print(f"迭代 {iteration}, 损失: {loss:.4f}")
                    
        else:
            # 多分类：权重矩阵为(n_features, n_classes)
            self.weights = np.random.randn(n_features, n_classes) * 0.01
            self.bias = np.zeros(n_classes)
            
            # 训练循环
            for iteration in range(self.n_iterations):
                # 前向传播
                linear_output = np.dot(X, self.weights) + self.bias
                y_pred = self._softmax(linear_output)
                
                # 计算损失
                loss = self._compute_loss_multiclass(X, y)
                self.loss_history.append(loss)
                
                # 计算梯度
                # 将标签转换为one-hot编码
                y_onehot = np.zeros((n_samples, n_classes))
                y_onehot[np.arange(n_samples), y] = 1
                
                # 对权重的梯度
                dw = 1/n_samples * np.dot(X.T, (y_pred - y_onehot)) + self.regularization * self.weights
                # 对偏置的梯度
                db = 1/n_samples * np.sum(y_pred - y_onehot, axis=0)
                
                # 更新参数
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                # 打印训练进度
                if iteration % 100 == 0:
                    print(f"迭代 {iteration}, 损失: {loss:.4f}")
    
    def predict_proba(self, X):
        """
        预测概率
        
        参数:
        - X: 特征矩阵
        
        返回:
        - 预测概率
        """
        linear_output = np.dot(X, self.weights) + self.bias
        
        if self.weights.shape[1] == 1:
            # 二分类
            return self._sigmoid(linear_output)
        else:
            # 多分类
            return self._softmax(linear_output)
    
    def predict(self, X):
        """
        预测类别
        
        参数:
        - X: 特征矩阵
        
        返回:
        - 预测的类别标签
        """
        probabilities = self.predict_proba(X)
        
        if self.weights.shape[1] == 1:
            # 二分类：概率大于0.5为类别1，否则为类别0
            return (probabilities > 0.5).astype(int).flatten()
        else:
            # 多分类：选择概率最大的类别
            return np.argmax(probabilities, axis=1)
    
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
        
        # 计算混淆矩阵
        cm = confusion_matrix(y, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def plot_loss_history(self, save_path=None):
        """
        绘制损失函数变化曲线
        
        参数:
        - save_path: 图片保存路径
        """
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.xlabel('迭代次数')
        plt.ylabel('损失值')
        plt.title('训练过程中损失函数的变化')
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
        
        算法原理:
        决策边界是分类器将不同类别分开的边界。对于线性分类器，
        决策边界是一个超平面。在2维情况下，决策边界是一条直线。
        
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
        
        plt.figure(figsize=(10, 8))
        
        # 创建网格来绘制决策边界
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        
        # 预测网格点的类别
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid_points)
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界和数据点
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        
        # 绘制数据点
        unique_classes = np.unique(y)
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, class_label in enumerate(unique_classes):
            mask = y == class_label
            plt.scatter(X[mask, 0], X[mask, 1], 
                       c=colors[i % len(colors)], 
                       label=f'类别 {class_label}',
                       alpha=0.8, edgecolors='black')
        
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.title('线性分类器决策边界')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"决策边界图已保存到 {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """
        绘制混淆矩阵
        
        参数:
        - cm: 混淆矩阵
        - save_path: 图片保存路径
        """
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[f'预测类别{i}' for i in range(cm.shape[1])],
                   yticklabels=[f'真实类别{i}' for i in range(cm.shape[0])])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        
        if save_path:
            plt.savefig(save_path)
            print(f"混淆矩阵已保存到 {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_model(self, save_path):
        """
        保存模型参数
        
        参数:
        - save_path: 保存路径
        """
        model_params = {
            'weights': self.weights,
            'bias': self.bias,
            'learning_rate': self.learning_rate,
            'regularization': self.regularization,
            'loss_history': self.loss_history
        }
        
        np.save(save_path, model_params)
        print(f"模型已保存到 {save_path}")
    
    def load_model(self, load_path):
        """
        加载模型参数
        
        参数:
        - load_path: 加载路径
        """
        model_params = np.load(load_path, allow_pickle=True).item()
        
        self.weights = model_params['weights']
        self.bias = model_params['bias']
        self.learning_rate = model_params['learning_rate']
        self.regularization = model_params['regularization']
        self.loss_history = model_params['loss_history']
        
        print(f"模型已从 {load_path} 加载")