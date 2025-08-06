"""
配置文件 - 包含所有模型参数和路径配置
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Any
import json
from datetime import datetime


@dataclass
class Config:
    # 数据路径
    train_path: str = 'UJIndoorLoc/trainingData.csv'
    test_path: str = 'UJIndoorLoc/validationData.csv'

    # 数据预处理参数
    test_size: float = 0.1  # 验证集比例

    # Transformer参数
    model_dim: int = 64
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1

    # SVR参数
    svr_kernel: str = 'rbf'
    svr_C: float = 100.0
    svr_epsilon: float = 0.1
    svr_gamma: str = 'scale'
    svr_degree: int = 3
    svr_coef0: float = 0.0

    # 训练参数
    epochs: int = 75
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    early_stopping_patience: int = 5
    min_delta_ratio: float = 0.003
    gradient_clip: float = 1.0

    # 优化参数
    n_trials_transformer: int = 100
    n_trials_svr: int = 100
    optuna_n_jobs: int = 1

    # 设备和种子
    device: str = 'cuda'
    seed: int = 42

    # 输出路径
    output_dir: str = field(default_factory=lambda: f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    def __post_init__(self):
        """创建必要的目录结构"""
        self.model_dir = os.path.join(self.output_dir, 'models')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        self.csv_dir = os.path.join(self.output_dir, 'csv')

        for dir_path in [self.model_dir, self.log_dir, self.plot_dir, self.csv_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def save(self, filepath: str = None):
        """保存配置到JSON文件"""
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'config.json')

        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        print(f"配置已保存到: {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """从JSON文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)