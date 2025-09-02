import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
from pytorch_linear_regression import PyTorchLinearRegression

def check_gpu_availability():
    """
    检查GPU可用性并显示相关信息
    """
    print("=" * 60)
    print("GPU可用性检查")
    print("=" * 60)
    
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA是否可用: {cuda_available}")
    
    if cuda_available:
        # 获取GPU数量
        gpu_count = torch.cuda.device_count()
        print(f"可用GPU数量: {gpu_count}")
        
        # 显示每个GPU的信息
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # 转换为GB
            print(f"GPU {i}: {gpu_name}")
            print(f"  - 显存大小: {gpu_memory:.2f} GB")
            
        # 显示当前GPU
        current_device = torch.cuda.current_device()
        print(f"当前使用的GPU: {current_device} ({torch.cuda.get_device_name(current_device)})")
        
        # 显示CUDA版本
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"PyTorch版本: {torch.__version__}")
    else:
        print("未检测到可用的GPU，将使用CPU进行计算")
        print("可能的解决方案:")
        print("1. 确保安装了支持CUDA的PyTorch版本")
        print("2. 检查NVIDIA驱动是否正确安装")
        print("3. 确保有可用的NVIDIA GPU")
    
    print("=" * 60)
    return cuda_available

def generate_large_dataset(n_samples=100000, n_features=10):
    """
    生成大型数据集用于GPU性能测试
    """
    print(f"生成大型数据集: {n_samples}个样本, {n_features}个特征")
    
    # 生成随机数据
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # 生成线性关系 + 噪声
    true_weights = np.random.randn(n_features)
    true_bias = np.random.randn()
    y = X.dot(true_weights) + true_bias + np.random.randn(n_samples) * 0.1
    
    print(f"数据集生成完成，数据形状: X={X.shape}, y={y.shape}")
    return X, y

def benchmark_cpu_vs_gpu(X, y, n_iterations=100):
    """
    对比CPU和GPU的性能
    """
    print("\n" + "=" * 60)
    print("CPU vs GPU 性能对比测试")
    print("=" * 60)
    
    results = {}
    
    # CPU测试
    print("\n--- CPU测试 ---")
    start_time = time.time()
    
    model_cpu = PyTorchLinearRegression(
        input_dim=X.shape[1],
        learning_rate=0.01,
        n_iterations=n_iterations,
        batch_size=1024,
        optimizer='adam',
        device='cpu'
    )
    
    model_cpu.fit(X, y, verbose=False)
    cpu_time = time.time() - start_time
    cpu_loss = model_cpu.calculate_loss(X, y)
    
    results['cpu'] = {
        'time': cpu_time,
        'loss': cpu_loss,
        'device': 'CPU'
    }
    
    print(f"CPU训练时间: {cpu_time:.2f}秒")
    print(f"CPU最终损失: {cpu_loss:.6f}")
    
    # GPU测试（如果可用）
    if torch.cuda.is_available():
        print("\n--- GPU测试 ---")
        
        # 清空GPU缓存
        torch.cuda.empty_cache()
        
        start_time = time.time()
        
        model_gpu = PyTorchLinearRegression(
            input_dim=X.shape[1],
            learning_rate=0.01,
            n_iterations=n_iterations,
            batch_size=1024,
            optimizer='adam',
            device='cuda'
        )
        
        model_gpu.fit(X, y, verbose=False)
        gpu_time = time.time() - start_time
        gpu_loss = model_gpu.calculate_loss(X, y)
        
        results['gpu'] = {
            'time': gpu_time,
            'loss': gpu_loss,
            'device': 'GPU'
        }
        
        print(f"GPU训练时间: {gpu_time:.2f}秒")
        print(f"GPU最终损失: {gpu_loss:.6f}")
        
        # 计算加速比
        speedup = cpu_time / gpu_time
        print(f"\nGPU加速比: {speedup:.2f}x")
        print(f"GPU比CPU快 {speedup:.2f} 倍")
        
        # 显示GPU内存使用情况
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # 转换为GB
        gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3  # 转换为GB
        print(f"GPU内存使用: {gpu_memory_used:.2f} GB")
        print(f"GPU内存缓存: {gpu_memory_cached:.2f} GB")
    else:
        print("\nGPU不可用，跳过GPU测试")
    
    print("=" * 60)
    return results

def plot_performance_comparison(results):
    """
    绘制性能对比图
    """
    if len(results) < 2:
        print("需要至少两个结果才能进行对比")
        return
    
    devices = list(results.keys())
    times = [results[device]['time'] for device in devices]
    losses = [results[device]['loss'] for device in devices]
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 训练时间对比
    bars1 = ax1.bar(devices, times, color=['blue', 'green'])
    ax1.set_ylabel('训练时间 (秒)')
    ax1.set_title('CPU vs GPU 训练时间对比')
    ax1.grid(True, alpha=0.3)
    
    # 在柱状图上显示数值
    for bar, time in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}s', ha='center', va='bottom')
    
    # 损失值对比
    bars2 = ax2.bar(devices, losses, color=['blue', 'green'])
    ax2.set_ylabel('最终损失值')
    ax2.set_title('CPU vs GPU 最终损失对比')
    ax2.grid(True, alpha=0.3)
    
    # 在柱状图上显示数值
    for bar, loss in zip(bars2, losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.6f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def demonstrate_gpu_usage():
    """
    演示GPU使用的各种方式
    """
    print("\n" + "=" * 60)
    print("GPU使用方式演示")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("GPU不可用，跳过演示")
        return
    
    # 生成示例数据
    np.random.seed(42)
    X = np.random.randn(1000, 5)
    y = X.dot(np.array([2, -1, 3, 0.5, -2])) + 1 + np.random.randn(1000) * 0.1
    
    print("\n1. 自动选择设备（推荐方式）")
    model_auto = PyTorchLinearRegression(
        input_dim=5,
        learning_rate=0.01,
        n_iterations=50,
        device=None  # 自动选择
    )
    print(f"自动选择的设备: {model_auto.device}")
    
    print("\n2. 显式指定CPU")
    model_cpu = PyTorchLinearRegression(
        input_dim=5,
        learning_rate=0.01,
        n_iterations=50,
        device='cpu'
    )
    print(f"显式指定的设备: {model_cpu.device}")
    
    print("\n3. 显式指定GPU")
    model_gpu = PyTorchLinearRegression(
        input_dim=5,
        learning_rate=0.01,
        n_iterations=50,
        device='cuda'
    )
    print(f"显式指定的设备: {model_gpu.device}")
    
    print("\n4. 指定特定GPU（如果有多个GPU）")
    if torch.cuda.device_count() > 1:
        model_gpu1 = PyTorchLinearRegression(
            input_dim=5,
            learning_rate=0.01,
            n_iterations=50,
            device='cuda:0'
        )
        model_gpu2 = PyTorchLinearRegression(
            input_dim=5,
            learning_rate=0.01,
            n_iterations=50,
            device='cuda:1'
        )
        print(f"GPU 0: {model_gpu1.device}")
        print(f"GPU 1: {model_gpu2.device}")
    else:
        print("只有一个GPU，跳过多GPU演示")
    
    print("\n5. 手动管理张量设备")
    # 演示手动将张量移动到GPU
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)
    
    print(f"原始张量设备: {X_tensor.device}")
    
    # 移动到GPU
    if torch.cuda.is_available():
        X_gpu = X_tensor.to('cuda')
        y_gpu = y_tensor.to('cuda')
        print(f"移动后张量设备: {X_gpu.device}")
        
        # 移回CPU
        X_cpu = X_gpu.cpu()
        print(f"移回CPU后设备: {X_cpu.device}")
    
    print("=" * 60)

def main():
    """
    主函数：演示PyTorch GPU使用
    """
    print("PyTorch线性回归 - GPU使用演示")
    print("=" * 60)
    
    # 1. 检查GPU可用性
    gpu_available = check_gpu_availability()
    
    # 2. 演示GPU使用方式
    demonstrate_gpu_usage()
    
    # 3. 生成大型数据集进行性能测试
    print("\n生成大型数据集进行性能测试...")
    X_large, y_large = generate_large_dataset(n_samples=50000, n_features=20)
    
    # 4. 进行CPU vs GPU性能对比
    results = benchmark_cpu_vs_gpu(X_large, y_large, n_iterations=100)
    
    # 5. 绘制性能对比图
    if len(results) > 1:
        print("\n绘制性能对比图...")
        plot_performance_comparison(results)
    
    # 6. 显示GPU内存信息
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("GPU内存信息")
        print("=" * 60)
        print(f"当前GPU内存分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"当前GPU内存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"GPU最大内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # 清空GPU缓存
        torch.cuda.empty_cache()
        print("\n已清空GPU缓存")
        print(f"清空后GPU内存分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"清空后GPU内存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    print("\n演示完成！")

if __name__ == "__main__":
    main()