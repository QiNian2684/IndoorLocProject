"""
配置文件 - 包含所有模型参数和路径配置（与Optuna配置集成）
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
from datetime import datetime


@dataclass
class Config:
    # ==================== 数据路径配置 ====================
    # 训练数据文件路径 - UJIndoorLoc数据集的训练数据CSV文件
    train_path: str = 'UJIndoorLoc/trainingData.csv'

    # 测试数据文件路径 - UJIndoorLoc数据集的验证/测试数据CSV文件
    test_path: str = 'UJIndoorLoc/validationData.csv'

    # ==================== 数据预处理参数 ====================
    # 验证集比例 - 从训练集中划分出多少比例作为验证集
    test_size: float = 0.1

    # ==================== Transformer默认参数 ====================
    # 这些参数在普通训练时使用，优化时会被Optuna采样的参数覆盖
    model_dim: int = 64          # 模型维度
    num_heads: int = 8           # 注意力头数
    num_layers: int = 6          # Transformer层数
    dropout: float = 0.1         # Dropout率

    # ==================== SVR默认参数 ====================
    # 这些参数在普通训练时使用，优化时会被Optuna采样的参数覆盖
    svr_kernel: str = 'rbf'      # 核函数类型
    svr_C: float = 100.0         # 正则化参数
    svr_epsilon: float = 0.1     # 不敏感损失参数
    svr_gamma: str = 'scale'     # 核函数系数
    svr_degree: int = 3          # 多项式阶数
    svr_coef0: float = 0.0       # 核函数独立项

    # ==================== 训练参数 ====================
    epochs: int = 75             # 训练轮数
    batch_size: int = 32         # 批次大小
    learning_rate: float = 0.001 # 学习率
    weight_decay: float = 1e-5   # L2正则化
    early_stopping_patience: int = 5  # 早停耐心值
    min_delta_ratio: float = 0.003    # 最小改善比例
    gradient_clip: float = 1.0        # 梯度裁剪

    # ==================== 系统参数 ====================
    device: str = 'cuda'         # 计算设备
    seed: int = 42              # 随机种子

    # ==================== 输出路径（自动生成）====================
    output_dir: str = field(default_factory=lambda: f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # ==================== Optuna配置路径（可选）====================
    optuna_config_path: Optional[str] = None  # 自定义Optuna配置文件路径

    def __post_init__(self):
        """初始化后自动执行 - 创建必要的目录结构"""
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

        # 只保存可序列化的属性
        config_dict = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_') and not k.endswith('_dir'):
                config_dict[k] = v

        # 添加目录路径
        config_dict['model_dir'] = self.model_dir
        config_dict['log_dir'] = self.log_dir
        config_dict['plot_dir'] = self.plot_dir
        config_dict['csv_dir'] = self.csv_dir

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False, default=str)
        print(f"配置已保存到: {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """从JSON文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        # 移除自动生成的目录属性，让__post_init__重新创建
        for key in ['model_dir', 'log_dir', 'plot_dir', 'csv_dir']:
            config_dict.pop(key, None)

        return cls(**config_dict)

    def get_training_params(self) -> Dict[str, Any]:
        """获取训练相关参数"""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'early_stopping_patience': self.early_stopping_patience,
            'min_delta_ratio': self.min_delta_ratio,
            'gradient_clip': self.gradient_clip
        }

    def get_transformer_params(self) -> Dict[str, Any]:
        """获取Transformer模型参数"""
        return {
            'model_dim': self.model_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        }

    def get_svr_params(self) -> Dict[str, Any]:
        """获取SVR模型参数"""
        return {
            'svr_kernel': self.svr_kernel,
            'svr_C': self.svr_C,
            'svr_epsilon': self.svr_epsilon,
            'svr_gamma': self.svr_gamma,
            'svr_degree': self.svr_degree,
            'svr_coef0': self.svr_coef0
        }

    def update_from_optuna(self, best_params: Dict[str, Any]):
        """使用Optuna找到的最佳参数更新配置"""
        for key, value in best_params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"更新参数 {key}: {value}")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}