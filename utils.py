"""
工具函数模块
"""
import torch
import numpy as np
import random
import os
import json
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"随机种子已设置: {seed}")


def create_experiment_name():
    """创建实验名称"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"exp_{timestamp}"


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    logger.info(f"检查点已保存: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """加载检查点"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    logger.info(f"检查点已加载: {filepath}")
    return model, optimizer, epoch, loss


def calculate_model_size(model):
    """计算模型大小"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    size_mb = total_params * 4 / 1024 / 1024  # 假设float32

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'size_mb': size_mb
    }


def print_model_summary(model):
    """打印模型摘要"""
    info = calculate_model_size(model)

    print("\n" + "=" * 50)
    print("模型摘要")
    print("=" * 50)
    print(f"总参数量: {info['total_params']:,}")
    print(f"可训练参数量: {info['trainable_params']:,}")
    print(f"模型大小: {info['size_mb']:.2f} MB")
    print("=" * 50 + "\n")


def merge_csv_files(file_list, output_file):
    """合并多个CSV文件"""
    dfs = []
    for file in file_list:
        if os.path.exists(file):
            df = pd.read_csv(file)
            dfs.append(df)

    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(output_file, index=False)
        logger.info(f"CSV文件已合并: {output_file}")
        return merged
    return None


def create_summary_report(config, results, output_path):
    """创建总结报告"""
    report = {
        'experiment_config': config.__dict__,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)

    logger.info(f"总结报告已创建: {output_path}")


class MetricLogger:
    """指标记录器"""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.metrics = []

    def log(self, metrics_dict):
        """记录指标"""
        metrics_dict['timestamp'] = datetime.now().isoformat()
        self.metrics.append(metrics_dict)

    def save(self, filename='metrics.csv'):
        """保存指标"""
        if self.metrics:
            df = pd.DataFrame(self.metrics)
            filepath = os.path.join(self.log_dir, filename)
            df.to_csv(filepath, index=False)
            logger.info(f"指标已保存: {filepath}")
            return df
        return None


def format_time(seconds):
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"