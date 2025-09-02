import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

class MiniBatchGD:
    """
    小批量梯度下降线性回归模型
    
    算法原理：
    小批量梯度下降是批量梯度下降和随机梯度下降的折中方案。
    每次迭代使用一小批数据来计算梯度，既减少了计算量，又保持了梯度估计的稳定性。
    
    关键概念：
    1. Epoch（时代）：所有训练数据完整地通过模型一次
    2. Batch（批量）：一个epoch中用于计算梯度的一小部分数据
    3. Batch Size（批量大小）：每个batch包含的样本数量
    4. Iterations（迭代次数）：总迭代次数 = epochs * (n_samples // batch_size)
    
    数学推导：
    1. 损失函数：L = (1/2m) * Σ(y_pred - y_true)²，其中m是batch size
    2. 梯度计算：
       ∂L/∂w = (1/m) * Σ_batch(y_pred - y_true) * X
       ∂L/∂b = (1/m) * Σ_batch(y_pred - y_true)
    3. 参数更新：
       w = w - α * ∂L/∂w
       b = b - α * ∂L/∂b
    其中α是学习率
    """
    
    def __init__(self, learning_rate: float = 0.01, epochs: int = 100, 
                 batch_size: int = 32, random_state: Optional[int] = None):
        """
        初始化小批量梯度下降线性回归模型
        
        参数：
        learning_rate: 学习率，控制每次参数更新的步长
        epochs: 训练轮数（所有数据完整通过模型的次数）
        batch_size: 每个小批量的大小
        random_state: 随机种子，用于数据打乱的可复现性
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        
        # 模型参数
        self.weights = None  # 权重向量w
        self.bias = None     # 偏置项b
        
        # 训练历史记录
        self.epoch_loss_history = []     # 每个epoch的平均损失
        self.batch_loss_history = []     # 每个batch的损失
        self.train_loss_history = []     # 训练集损失历史
        self.val_loss_history = []       # 验证集损失历史
        
        # 训练统计信息
        self.total_iterations = 0
        self.batches_per_epoch = 0
        
    def _create_mini_batches(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        创建小批量数据
        
        算法步骤：
        1. 将数据随机打乱
        2. 按照batch_size分割数据
        3. 返回小批量列表
        
        参数：
        X: 输入特征 (n_samples, n_features)
        y: 输出标签 (n_samples,)
        
        返回：
        mini_batches: 小批量数据列表，每个元素是(X_batch, y_batch)
        """
        n_samples = X.shape[0]
        mini_batches = []
        
        # 设置随机种子
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # 1. 随机打乱数据
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # 2. 创建小批量
        for i in range(0, n_samples, self.batch_size):
            X_batch = X_shuffled[i:i + self.batch_size]
            y_batch = y_shuffled[i:i + self.batch_size]
            mini_batches.append((X_batch, y_batch))
        
        return mini_batches
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
           X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
           verbose: bool = True, early_stopping: bool = False, 
           patience: int = 10) -> 'MiniBatchGD':
        """
        训练模型：使用小批量梯度下降法优化参数
        
        算法步骤：
        1. 初始化模型参数
        2. 对于每个epoch：
           a. 创建小批量数据
           b. 对于每个小批量：
              - 前向传播计算预测值
              - 计算损失和梯度
              - 更新参数
           c. 计算epoch平均损失
           d. 计算验证集损失（如果有）
           e. 早停检查（如果启用）
        
        参数：
        X_train: 训练集特征 (n_samples, n_features)
        y_train: 训练集标签 (n_samples,)
        X_val: 验证集特征 (可选)
        y_val: 验证集标签 (可选)
        verbose: 是否打印训练信息
        early_stopping: 是否启用早停
        patience: 早停耐心值（连续多少个epoch验证损失不下降就停止）
        
        返回：
        self: 返回模型实例，支持链式调用
        """
        # 获取数据维度
        n_samples, n_features = X_train.shape
        
        # 计算训练统计信息
        self.batches_per_epoch = n_samples // self.batch_size
        if n_samples % self.batch_size != 0:
            self.batches_per_epoch += 1
        
        # 1. 初始化参数
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        
        # 早停相关变量
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"开始训练：{self.epochs}个epoch，每个epoch{self.batches_per_epoch}个batch")
        
        # 2. 训练循环
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            # 创建小批量数据
            mini_batches = self._create_mini_batches(X_train, y_train)
            
            # 遍历每个小批量
            for batch_idx, (X_batch, y_batch) in enumerate(mini_batches):
                # 前向传播
                y_pred = np.dot(X_batch, self.weights) + self.bias
                
                # 计算损失
                error = y_pred - y_batch
                batch_loss = np.mean(error ** 2) / 2
                self.batch_loss_history.append(batch_loss)
                epoch_loss += batch_loss
                
                # 计算梯度
                m_batch = X_batch.shape[0]
                dw = np.dot(X_batch.T, error) / m_batch
                db = np.mean(error)
                
                # 更新参数
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                self.total_iterations += 1
            
            # 计算epoch平均损失
            epoch_loss /= len(mini_batches)
            self.epoch_loss_history.append(epoch_loss)
            
            # 计算训练集总损失
            train_loss = self.calculate_loss(X_train, y_train)
            self.train_loss_history.append(train_loss)
            
            # 计算验证集损失（如果有）
            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self.calculate_loss(X_val, y_val)
                self.val_loss_history.append(val_loss)
            
            # 打印训练信息
            if verbose and (epoch + 1) % 10 == 0:
                val_info = f"，验证损失: {val_loss:.6f}" if val_loss is not None else ""
                print(f"Epoch {epoch + 1}/{self.epochs}，训练损失: {train_loss:.6f}{val_info}")
            
            # 早停检查
            if early_stopping and val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"早停触发！在第{epoch + 1}个epoch停止训练")
                    break
        
        print(f"训练完成！总迭代次数: {self.total_iterations}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用训练好的模型进行预测
        
        参数：
        X: 输入特征 (n_samples, n_features)
        
        返回：
        y_pred: 预测值 (n_samples,)
        """
        return np.dot(X, self.weights) + self.bias
    
    def calculate_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算模型在给定数据上的均方误差损失
        
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
    
    def calculate_r2_score(self, X: np.ndarray, y: np.ndarray) -> float:
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
    
    def plot_training_history(self, figsize: Tuple[int, int] = (15, 10)):
        """
        可视化训练历史
        
        显示：
        1. Batch损失变化
        2. Epoch损失变化
        3. 训练集和验证集损失对比
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('小批量梯度下降训练历史', fontsize=16)
        
        # 1. Batch损失变化
        axes[0, 0].plot(self.batch_loss_history, alpha=0.7, linewidth=0.5)
        axes[0, 0].set_title('Batch损失变化')
        axes[0, 0].set_xlabel('Batch迭代次数')
        axes[0, 0].set_ylabel('损失值')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Epoch损失变化
        axes[0, 1].plot(self.epoch_loss_history, 'b-', linewidth=2, label='Epoch平均损失')
        axes[0, 1].set_title('Epoch损失变化')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('损失值')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 训练集和验证集损失对比
        if self.val_loss_history:
            axes[1, 0].plot(self.train_loss_history, 'b-', linewidth=2, label='训练集损失')
            axes[1, 0].plot(self.val_loss_history, 'r-', linewidth=2, label='验证集损失')
            axes[1, 0].set_title('训练集 vs 验证集损失')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('损失值')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].plot(self.train_loss_history, 'b-', linewidth=2, label='训练集损失')
            axes[1, 0].set_title('训练集损失')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('损失值')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 学习曲线（损失分布）
        if len(self.batch_loss_history) > 100:
            # 取最后1000个batch的损失进行统计
            recent_losses = self.batch_loss_history[-1000:]
            axes[1, 1].hist(recent_losses, bins=50, alpha=0.7, density=True)
            axes[1, 1].set_title('最近1000个Batch的损失分布')
            axes[1, 1].set_xlabel('损失值')
            axes[1, 1].set_ylabel('密度')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off')
            axes[1, 1].text(0.5, 0.5, '数据不足\n无法显示损失分布', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.show()
    
    def plot_regression_line(self, X: np.ndarray, y: np.ndarray, 
                           title: str = "小批量梯度下降拟合结果"):
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
    
    def get_parameters(self) -> Tuple[np.ndarray, float]:
        """
        获取模型参数
        
        返回：
        (weights, bias): 权重和偏置
        """
        return self.weights, self.bias
    
    def get_training_stats(self) -> dict:
        """
        获取训练统计信息
        
        返回：
        stats: 包含训练统计信息的字典
        """
        return {
            'total_epochs': len(self.epoch_loss_history),
            'total_iterations': self.total_iterations,
            'batches_per_epoch': self.batches_per_epoch,
            'batch_size': self.batch_size,
            'final_train_loss': self.train_loss_history[-1] if self.train_loss_history else None,
            'final_val_loss': self.val_loss_history[-1] if self.val_loss_history else None,
            'best_train_loss': min(self.train_loss_history) if self.train_loss_history else None,
            'best_val_loss': min(self.val_loss_history) if self.val_loss_history else None
        }
    
    def print_training_summary(self):
        """
        打印训练总结
        """
        stats = self.get_training_stats()
        weights, bias = self.get_parameters()
        
        print("\n" + "="*50)
        print("小批量梯度下降训练总结")
        print("="*50)
        print(f"训练配置:")
        print(f"  学习率: {self.learning_rate}")
        print(f"  Epochs: {stats['total_epochs']}")
        print(f"  Batch Size: {stats['batch_size']}")
        print(f"  每Epoch批次数: {stats['batches_per_epoch']}")
        print(f"  总迭代次数: {stats['total_iterations']}")
        print(f"\n模型参数:")
        print(f"  权重: {weights[0]:.6f}")
        print(f"  偏置: {bias:.6f}")
        print(f"\n训练性能:")
        print(f"  最终训练损失: {stats['final_train_loss']:.6f}")
        print(f"  最佳训练损失: {stats['best_train_loss']:.6f}")
        if stats['final_val_loss'] is not None:
            print(f"  最终验证损失: {stats['final_val_loss']:.6f}")
            print(f"  最佳验证损失: {stats['best_val_loss']:.6f}")
        print("="*50)


def compare_gradient_descent_methods(X_train: np.ndarray, y_train: np.ndarray,
                                    X_val: np.ndarray, y_val: np.ndarray,
                                    learning_rate: float = 0.01) -> dict:
    """
    比较不同梯度下降方法的性能
    
    参数：
    X_train, y_train: 训练数据
    X_val, y_val: 验证数据
    learning_rate: 学习率
    
    返回：
    results: 包含各种方法结果的字典
    """
    from linear_regression import LinearRegression
    
    results = {}
    
    # 1. 批量梯度下降
    print("训练批量梯度下降模型...")
    bgd_model = LinearRegression(learning_rate=learning_rate, n_iterations=1000)
    bgd_model.fit(X_train, y_train, verbose=False)
    
    bgd_train_loss = bgd_model.calculate_loss(X_train, y_train)
    bgd_val_loss = bgd_model.calculate_loss(X_val, y_val)
    bgd_train_r2 = bgd_model.calculate_r2_score(X_train, y_train)
    bgd_val_r2 = bgd_model.calculate_r2_score(X_val, y_val)
    
    results['BatchGD'] = {
        'model': bgd_model,
        'train_loss': bgd_train_loss,
        'val_loss': bgd_val_loss,
        'train_r2': bgd_train_r2,
        'val_r2': bgd_val_r2,
        'iterations': len(bgd_model.loss_history)
    }
    
    # 2. 小批量梯度下降（不同batch size）
    batch_sizes = [16, 32, 64, 128]
    
    for batch_size in batch_sizes:
        print(f"训练小批量梯度下降模型 (batch_size={batch_size})...")
        
        model = MiniBatchGD(learning_rate=learning_rate, epochs=50, batch_size=batch_size)
        model.fit(X_train, y_train, X_val, y_val, verbose=False)
        
        train_loss = model.calculate_loss(X_train, y_train)
        val_loss = model.calculate_loss(X_val, y_val)
        train_r2 = model.calculate_r2_score(X_train, y_train)
        val_r2 = model.calculate_r2_score(X_val, y_val)
        
        results[f'MiniBatch_B{batch_size}'] = {
            'model': model,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'epochs': len(model.epoch_loss_history),
            'total_iterations': model.total_iterations
        }
    
    return results


def plot_comparison_results(results: dict):
    """
    绘制不同梯度下降方法的比较结果
    
    参数：
    results: 比较结果字典
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('梯度下降方法比较', fontsize=16)
    
    methods = list(results.keys())
    train_losses = [results[method]['train_loss'] for method in methods]
    val_losses = [results[method]['val_loss'] for method in methods]
    train_r2_scores = [results[method]['train_r2'] for method in methods]
    val_r2_scores = [results[method]['val_r2'] for method in methods]
    
    # 1. 训练损失对比
    axes[0, 0].bar(methods, train_losses, alpha=0.7)
    axes[0, 0].set_title('训练损失对比')
    axes[0, 0].set_ylabel('损失值')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 验证损失对比
    axes[0, 1].bar(methods, val_losses, alpha=0.7, color='orange')
    axes[0, 1].set_title('验证损失对比')
    axes[0, 1].set_ylabel('损失值')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 训练R²对比
    axes[1, 0].bar(methods, train_r2_scores, alpha=0.7, color='green')
    axes[1, 0].set_title('训练R²对比')
    axes[1, 0].set_ylabel('R²分数')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 验证R²对比
    axes[1, 1].bar(methods, val_r2_scores, alpha=0.7, color='red')
    axes[1, 1].set_title('验证R²对比')
    axes[1, 1].set_ylabel('R²分数')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("小批量梯度下降线性回归算法演示")
    print("="*50)
    
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    X = np.random.rand(n_samples, 1) * 10
    y = 2.5 * X.flatten() + 1.0 + np.random.normal(0, 0.5, n_samples)
    
    # 划分训练集和验证集
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"数据集信息:")
    print(f"  总样本数: {n_samples}")
    print(f"  训练集: {X_train.shape[0]} 样本")
    print(f"  验证集: {X_val.shape[0]} 样本")
    
    # 训练小批量梯度下降模型
    print("\n训练小批量梯度下降模型...")
    model = MiniBatchGD(learning_rate=0.01, epochs=100, batch_size=32, random_state=42)
    model.fit(X_train, y_train, X_val, y_val, verbose=True)
    
    # 打印训练总结
    model.print_training_summary()
    
    # 计算最终性能
    train_r2 = model.calculate_r2_score(X_train, y_train)
    val_r2 = model.calculate_r2_score(X_val, y_val)
    print(f"\n最终性能:")
    print(f"  训练集R²: {train_r2:.4f}")
    print(f"  验证集R²: {val_r2:.4f}")
    
    # 可视化训练历史
    print("\n显示训练历史图表...")
    model.plot_training_history()
    
    # 可视化拟合结果
    print("显示拟合结果...")
    model.plot_regression_line(X_val, y_val, "验证集拟合结果")
    
    # 比较不同方法
    print("\n比较不同梯度下降方法...")
    comparison_results = compare_gradient_descent_methods(X_train, y_train, X_val, y_val)
    plot_comparison_results(comparison_results)
    
    print("\n演示完成！")