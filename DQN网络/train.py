"""
DQN 训练脚本
包含训练、测试、可视化等完整功能
"""

import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import argparse
import os

from dqn_agent import DQNAgent


class Trainer:
    """
    DQN 训练器
    
    功能：
    - 完整训练循环
    - 训练进度监控
    - 早停判断
    - 模型保存
    """
    def __init__(self, env_name='CartPole-v1', config=None):
        """
        初始化训练器
        
        Args:
            env_name: 环境名称
            config: 训练配置
        """
        self.env_name = env_name
        
        # 创建环境获取状态/动作维度
        temp_env = gym.make(env_name)
        self.state_dim = temp_env.observation_space.shape[0]
        self.action_dim = temp_env.action_space.n
        temp_env.close()
        
        # 默认训练配置
        self.config = {
            'episodes': 500,           # 最大训练集数
            'max_steps': 500,          # 每集最大步数
            'target_reward': 475,      # 目标奖励（达到后早停）
            'eval_every': 10,          # 评估间隔
            'eval_episodes': 5,        # 评估集数
            'save_dir': './models',    # 模型保存目录
            'render': False,           # 是否渲染
        }
        if config:
            self.config.update(config)
        
        # 创建保存目录
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        # 创建智能体
        self.agent = DQNAgent(self.state_dim, self.action_dim)
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
    
    def train(self):
        """
        执行训练
        
        Returns:
            agent: 训练好的智能体
            stats: 训练统计信息
        """
        env = gym.make(self.env_name)
        
        print(f"\n{'='*60}")
        print(f"开始训练 DQN")
        print(f"环境：{self.env_name}")
        print(f"状态维度：{self.state_dim}, 动作维度：{self.action_dim}")
        print(f"{'='*60}\n")
        
        for episode in range(self.config['episodes']):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done and episode_length < self.config['max_steps']:
                # 选择动作
                action = self.agent.select_action(state)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 存储经验
                self.agent.store_transition(state, action, reward, next_state, done)
                
                # 训练
                loss = self.agent.train()
                if loss:
                    self.losses.append(loss)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
            
            # 记录统计
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.agent.episode_count += 1
            
            # 打印进度
            if (episode + 1) % self.config['eval_every'] == 0:
                avg_reward = np.mean(self.episode_rewards[-self.config['eval_every']:])
                avg_length = np.mean(self.episode_lengths[-self.config['eval_every']:])
                avg_loss = np.mean(self.losses[-self.config['eval_every'] * 10:]) if self.losses else 0
                
                print(f"Episode {episode+1}/{self.config['episodes']} | "
                      f"平均奖励：{avg_reward:.2f} | "
                      f"平均长度：{avg_length:.1f} | "
                      f"ε: {self.agent.config['epsilon']:.3f} | "
                      f"损失：{avg_loss:.4f}")
                
                # 保存检查点
                self.agent.save(
                    os.path.join(self.config['save_dir'], 
                                f'dqn_{self.env_name}_ep{episode+1}.pth')
                )
            
            # 早停判断
            if len(self.episode_rewards) >= 100:
                last_100_avg = np.mean(self.episode_rewards[-100:])
                if last_100_avg >= self.config['target_reward']:
                    print(f"\n✅ 环境已解决！用了 {episode+1} 集")
                    print(f"最后 100 集平均奖励：{last_100_avg:.2f}")
                    break
        
        env.close()
        
        # 保存最终模型
        final_path = os.path.join(self.config['save_dir'], f'dqn_{self.env_name}_final.pth')
        self.agent.save(final_path)
        
        print(f"\n{'='*60}")
        print(f"训练完成！")
        print(f"总集数：{len(self.episode_rewards)}")
        print(f"最终模型：{final_path}")
        print(f"{'='*60}\n")
        
        return self.agent, self.get_stats()
    
    def test(self, model_path=None, episodes=5, render=True):
        """
        测试训练好的模型
        
        Args:
            model_path: 模型路径
            episodes: 测试集数
            render: 是否渲染
        """
        render_mode = 'human' if render else None
        env = gym.make(self.env_name, render_mode=render_mode)
        
        # 加载模型
        if model_path:
            self.agent.load(model_path)
        
        # 关闭探索
        self.agent.config['epsilon'] = 0
        
        print(f"\n{'='*60}")
        print(f"测试模型：{model_path}")
        print(f"{'='*60}\n")
        
        test_rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.agent.select_action(state, training=False)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                if render:
                    env.render()
            
            test_rewards.append(episode_reward)
            print(f"测试集 {episode+1}: 奖励 = {episode_reward}")
        
        env.close()
        
        print(f"\n平均测试奖励：{np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
        
        return test_rewards
    
    def get_stats(self):
        """获取训练统计信息"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'final_avg_reward': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
        }
    
    def plot_results(self, save_path='./training_results.png'):
        """
        绘制训练结果图表
        
        Args:
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 奖励曲线
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='原始', color='gray')
        window = 10
        if len(self.episode_rewards) >= window:
            smoothed = np.convolve(self.episode_rewards, np.ones(window)/window, 'valid')
            axes[0, 0].plot(smoothed, label=f'{window}集平均', color='blue', linewidth=2)
        axes[0, 0].axhline(y=self.config['target_reward'], color='r', linestyle='--', label='目标奖励')
        axes[0, 0].set_xlabel('集数')
        axes[0, 0].set_ylabel('奖励')
        axes[0, 0].set_title('训练奖励曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 集长度曲线
        axes[0, 1].plot(self.episode_lengths, alpha=0.5, color='green')
        axes[0, 1].set_xlabel('集数')
        axes[0, 1].set_ylabel('步数')
        axes[0, 1].set_title('每集长度')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 损失曲线
        if self.losses:
            axes[1, 0].plot(self.losses, alpha=0.5, color='orange')
            if len(self.losses) >= 100:
                smoothed_loss = np.convolve(self.losses, np.ones(100)/100, 'valid')
                axes[1, 0].plot(smoothed_loss, color='red', linewidth=2)
            axes[1, 0].set_xlabel('训练步数')
            axes[1, 0].set_ylabel('损失')
            axes[1, 0].set_title('训练损失')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 探索率变化
        epsilon_history = [self.agent.config['epsilon']]
        axes[1, 1].plot(epsilon_history, color='purple')
        axes[1, 1].set_xlabel('集数')
        axes[1, 1].set_ylabel('ε')
        axes[1, 1].set_title('探索率')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"训练结果图已保存到：{save_path}")
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DQN 训练脚本')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='环境名称')
    parser.add_argument('--episodes', type=int, default=500, help='训练集数')
    parser.add_argument('--test', type=str, default=None, help='测试模型路径')
    parser.add_argument('--no-render', action='store_true', help='关闭渲染')
    parser.add_argument('--double-dqn', action='store_true', help='使用 Double DQN')
    parser.add_argument('--dueling', action='store_true', help='使用 Dueling DQN')
    parser.add_argument('--per', action='store_true', help='使用优先经验回放')
    
    args = parser.parse_args()
    
    # 创建训练器
    config = {
        'episodes': args.episodes,
        'use_double_dqn': args.double_dqn,
        'use_dueling': args.dueling,
        'use_per': args.per,
        'render': not args.no_render,
    }
    
    trainer = Trainer(env_name=args.env, config=config)
    
    if args.test:
        # 测试模式
        trainer.test(model_path=args.test, render=not args.no_render)
    else:
        # 训练模式
        agent, stats = trainer.train()
        trainer.plot_results()
        
        # 自动测试
        trainer.test(
            model_path=f"./models/dqn_{args.env}_final.pth",
            episodes=3,
            render=not args.no_render
        )


if __name__ == '__main__':
    main()
