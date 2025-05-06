import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging


class TrainingMonitor:
    """监控训练过程并记录详细指标"""

    def __init__(self, exp_dir, config):
        """初始化训练监视器"""
        self.exp_dir = exp_dir
        self.config = config
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'model_parameters': [],
            'timestamp': []
        }

        # 创建记录目录
        self.log_dir = os.path.join(exp_dir, 'logs', 'training')
        self.csv_dir = os.path.join(exp_dir, 'csv_records', 'training')
        self.checkpoint_dir = os.path.join(exp_dir, 'models', 'checkpoints')
        self.viz_dir = os.path.join(exp_dir, 'visualizations', 'training')

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)

        # 运行ID (使用时间戳避免覆盖) - 在调用_setup_logger()之前初始化
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 设置日志记录器
        self._setup_logger()

    def log_epoch(self, epoch, train_loss, val_loss=None, lr=None, params=None):
        """记录每个训练轮次的信息"""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['learning_rate'].append(lr)
        self.history['model_parameters'].append(params)
        self.history['timestamp'].append(datetime.now().isoformat())

        # 控制台输出
        if self.config['logging']['console_output']:
            log_msg = f"Epoch {epoch + 1}: train_loss={train_loss:.6f}"
            if val_loss is not None:
                log_msg += f", val_loss={val_loss:.6f}"
            if lr is not None:
                log_msg += f", lr={lr:.6f}"
            self.logger.info(log_msg)

        # 定期保存CSV记录和图表
        if (epoch % 10 == 0 or
                epoch == self.config['model']['transformer']['epochs'] - 1):
            self.save_history_csv()

            # 绘制学习曲线
            self.plot_learning_curves()

        # 每个轮次更新单独的历史文件，使用带版本的文件名
        epoch_history = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': lr,
            'timestamp': datetime.now().isoformat()
        }

        epoch_history_path = os.path.join(self.csv_dir, 'epoch_history', f'epoch_{epoch:04d}.json')
        os.makedirs(os.path.dirname(epoch_history_path), exist_ok=True)
        with open(epoch_history_path, 'w') as f:
            json.dump(epoch_history, f, indent=4)

    def save_history_csv(self):
        """保存训练历史到CSV文件"""
        # 基本历史记录
        basic_history = {
            'epoch': self.history['epoch'],
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'learning_rate': self.history['learning_rate'],
            'timestamp': self.history['timestamp']
        }

        history_df = pd.DataFrame(basic_history)

        # 使用运行ID保存历史记录，避免覆盖
        csv_path = os.path.join(self.csv_dir, f'training_history_{self.run_id}.csv')
        history_df.to_csv(csv_path, index=False)
        self.logger.info(f"训练历史已保存到: {csv_path}")

        # 详细参数记录（如果有）
        if any(p is not None for p in self.history['model_parameters']):
            # 尝试提取常见参数路径
            try:
                params_history = {}

                # 找到第一个非空参数字典
                first_params = next(p for p in self.history['model_parameters'] if p is not None)

                if first_params:
                    # 根据参数类型，提取不同的参数路径
                    if 'transformer' in first_params:
                        for param_name in first_params['transformer'].keys():
                            params_history[f'transformer_{param_name}'] = []

                        # 填充数据
                        for epoch_params in self.history['model_parameters']:
                            if epoch_params and 'transformer' in epoch_params:
                                for param_name, value in epoch_params['transformer'].items():
                                    params_history[f'transformer_{param_name}'].append(value)
                            else:
                                for param_name in first_params['transformer'].keys():
                                    params_history[f'transformer_{param_name}'].append(None)

                    elif 'end2end' in first_params:
                        for param_name in first_params['end2end'].keys():
                            params_history[f'end2end_{param_name}'] = []

                        # 填充数据
                        for epoch_params in self.history['model_parameters']:
                            if epoch_params and 'end2end' in epoch_params:
                                for param_name, value in epoch_params['end2end'].items():
                                    params_history[f'end2end_{param_name}'].append(value)
                            else:
                                for param_name in first_params['end2end'].keys():
                                    params_history[f'end2end_{param_name}'].append(None)

                # 创建参数历史DataFrame
                if params_history:
                    params_df = pd.DataFrame(params_history)
                    params_df['epoch'] = self.history['epoch']
                    params_path = os.path.join(self.csv_dir, f'parameter_history_{self.run_id}.csv')
                    params_df.to_csv(params_path, index=False)
                    self.logger.info(f"参数历史已保存到: {params_path}")
            except:
                self.logger.warning("无法提取模型参数历史")

    def plot_learning_curves(self):
        """绘制学习曲线"""
        plt.figure(figsize=(12, 5))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.history['epoch'], self.history['train_loss'], 'b-', label='训练损失')
        if any(v is not None for v in self.history['val_loss']):
            val_losses = [v if v is not None else np.nan for v in self.history['val_loss']]
            plt.plot(self.history['epoch'], val_losses, 'r-', label='验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.title('训练和验证损失')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # 学习率曲线
        if any(v is not None for v in self.history['learning_rate']):
            plt.subplot(1, 2, 2)
            lr_values = [v if v is not None else np.nan for v in self.history['learning_rate']]
            plt.plot(self.history['epoch'], lr_values, 'g-')
            plt.xlabel('轮次')
            plt.ylabel('学习率')
            plt.title('学习率变化')
            plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        # 使用时间戳保存图像，避免覆盖
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存图像
        for fmt in self.config['logging']['visualization_formats']:
            curve_path = os.path.join(self.viz_dir, f'learning_curves_{current_time}.{fmt}')
            plt.savefig(curve_path)
            self.logger.info(f"学习曲线已保存到: {curve_path}")

        plt.close()

    def save_final_history_summary(self):
        """保存最终的训练历史摘要"""
        summary = {
            'total_epochs': len(self.history['epoch']),
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
            'best_val_loss': min([v for v in self.history['val_loss'] if v is not None], default=None),
            'best_epoch': self.history['epoch'][
                np.argmin([v if v is not None else float('inf') for v in self.history['val_loss']])]
            if any(v is not None for v in self.history['val_loss']) else None,
            'training_duration': (datetime.fromisoformat(self.history['timestamp'][-1]) -
                                  datetime.fromisoformat(self.history['timestamp'][0])).total_seconds()
            if len(self.history['timestamp']) > 1 else 0,
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat()
        }

        # 使用运行ID保存摘要，避免覆盖
        summary_path = os.path.join(self.csv_dir, f'training_summary_{self.run_id}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        self.logger.info(f"训练摘要已保存到: {summary_path}")

        # 创建摘要CSV
        summary_df = pd.DataFrame([summary])
        summary_csv_path = os.path.join(self.csv_dir, f'training_summary_{self.run_id}.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        self.logger.info(f"训练摘要CSV已保存到: {summary_csv_path}")

        return summary

    def _setup_logger(self):
        """设置日志记录器"""
        self.logger = logging.getLogger('training_monitor')
        self.logger.setLevel(getattr(logging, self.config['logging']['log_level']))

        # 清除已有的处理器
        if self.logger.handlers:
            self.logger.handlers.clear()

        # 使用运行ID创建日志文件，避免覆盖
        file_handler = logging.FileHandler(os.path.join(self.log_dir, f'training_{self.run_id}.log'))
        file_handler.setLevel(getattr(logging, self.config['logging']['log_level']))

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config['logging']['log_level']))

        # 创建格式器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)