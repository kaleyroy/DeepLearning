import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """
    线性回归模型类：实现从数据训练到预测的完整流程
    
    算法原理：
    线性回归试图找到最佳的线性关系：y = wX + b
    通过最小化均方误差(MSE)来学习参数w和b
    
    数学推导：
    1. 损失函数：L = (1/2n) * Σ(y_pred - y_true)²
    2. 梯度计算：
       ∂L/∂w = (1/n) * Σ(y_pred - y_true) * X
       ∂L/∂b = (1/n) * Σ(y_pred - y_true)
    3. 参数更新：
       w = w - α * ∂L/∂w
       b = b - α * ∂L/∂b
    其中α是学习率
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        初始化线性回归模型
        
        参数：
        learning_rate: 学习率，控制每次参数更新的步长
        n_iterations: 最大迭代次数
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None  # 权重向量w
        self.bias = None     # 偏置项b
        self.loss_history = []  # 损失历史，用于可视化训练过程
        
    def fit(self, X, y, verbose=True):
        """
        训练模型：使用梯度下降法优化参数
        
        算法步骤：
        1. 初始化参数w和b
        2. 迭代计算预测值和损失
        3. 计算梯度并更新参数
        4. 记录损失变化
        
        参数：
        X: 训练数据特征 (n_samples, n_features)
        y: 训练数据标签 (n_samples,)
        verbose: 是否打印训练信息
        """
        # 获取样本数和特征数
        n_samples, n_features = X.shape
        
        # 1. 初始化参数
        # 使用小随机数初始化权重，避免对称性问题
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        
        # 2. 梯度下降迭代
        for iteration in range(self.n_iterations):
            # 前向传播：计算预测值
            # y_pred = X * w + b (矩阵运算)
            y_pred = np.dot(X, self.weights) + self.bias
            
            # 计算损失：均方误差
            # MSE = (1/2n) * Σ(y_pred - y_true)²
            # 乘以1/2是为了求导时简化计算
            error = y_pred - y
            mse_loss = np.mean(error ** 2) / 2
            self.loss_history.append(mse_loss)
            
            # 3. 计算梯度
            # ∂L/∂w = (1/n) * Σ(y_pred - y_true) * X
            # 使用矩阵运算：X.T * error / n_samples
            dw = np.dot(X.T, error) / n_samples
            
            # ∂L/∂b = (1/n) * Σ(y_pred - y_true)
            db = np.mean(error)
            
            # 4. 更新参数（梯度下降）
            # w = w - α * ∂L/∂w
            # b = b - α * ∂L/∂b
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 打印训练进度
            if verbose and (iteration + 1) % 100 == 0:
                print(f"迭代 {iteration + 1}/{self.n_iterations}, 损失: {mse_loss:.6f}")
    
    def predict(self, X):
        """
        使用训练好的模型进行预测
        
        参数：
        X: 输入特征 (n_samples, n_features)
        
        返回：
        y_pred: 预测值 (n_samples,)
        """
        # 线性变换：y = X * w + b
        return np.dot(X, self.weights) + self.bias
    
    def calculate_loss(self, X, y):
        """
        计算模型在给定数据上的损失
        
        参数：
        X: 输入特征
        y: 真实标签
        
        返回：
        mse_loss: 均方误差损失
        """
        y_pred = self.predict(X)
        error = y_pred - y
        mse_loss = np.mean(error ** 2) / 2
        return mse_loss
    
    def calculate_r2_score(self, X, y):
        """
        计算R²决定系数，评估模型拟合优度
        
        算法原理：
        R² = 1 - (SS_res / SS_tot)
        其中：
        SS_res = Σ(y_true - y_pred)² (残差平方和)
        SS_tot = Σ(y_true - y_mean)² (总平方和)
        
        R²越接近1，说明模型拟合效果越好
        
        参数：
        X: 输入特征
        y: 真实标签
        
        返回：
        r2_score: R²决定系数
        """
        y_pred = self.predict(X)
        y_mean = np.mean(y)
        
        # 计算残差平方和
        ss_res = np.sum((y - y_pred) ** 2)
        
        # 计算总平方和
        ss_tot = np.sum((y - y_mean) ** 2)
        
        # 计算R²
        r2_score = 1 - (ss_res / ss_tot)
        return r2_score
    
    def plot_loss_history(self):
        """
        可视化训练过程中的损失变化
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.xlabel('迭代次数')
        plt.ylabel('损失值')
        plt.title('训练损失变化曲线')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_regression_line(self, X, y, title="线性回归拟合结果"):
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
        X_line = np.linspace(X.min(), X.max(), 100)
        y_line = self.predict(X_line.reshape(-1, 1))
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
        return self.weights, self.bias
    
    def print_parameters(self):
        """
        打印模型参数
        """
        print(f"学习到的权重: {self.weights}")
        print(f"学习到的偏置: {self.bias}")

if __name__ == "__main__":
    # 简单测试
    # 创建一些测试数据
    np.random.seed(42)
    X_test = np.random.uniform(0, 10, (100, 1))
    y_test = 2.5 * X_test.flatten() + 1.0 + np.random.normal(0, 0.1, 100)
    
    # 创建并训练模型
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_test, y_test)
    
    # 打印参数
    model.print_parameters()
    
    # 计算性能指标
    r2_score = model.calculate_r2_score(X_test, y_test)
    print(f"R²决定系数: {r2_score:.4f}")
    
    # 可视化结果
    model.plot_loss_history()
    model.plot_regression_line(X_test, y_test)