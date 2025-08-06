"""
Optuna超参数优化配置文件
集中管理所有Optuna相关的配置和搜索空间定义
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Union


@dataclass
class OptunaConfig:
    """Optuna优化器的配置"""

    # ==================== 基础优化配置 ====================
    # Transformer优化试验次数
    n_trials_transformer: int = 100

    # SVR优化试验次数
    n_trials_svr: int = 100

    # 并行作业数（1=串行，>1=并行）
    # 注意：并行可能导致GPU内存问题
    n_jobs: int = 1

    # 优化方向（minimize=最小化，maximize=最大化）
    direction: str = 'minimize'

    # 随机种子（确保可重复性）
    seed: int = 42

    # ==================== 采样器配置 ====================
    # 采样器类型：'TPE', 'Random', 'Grid', 'CmaEs'
    sampler_type: str = 'TPE'

    # TPE采样器参数
    tpe_n_startup_trials: int = 10  # 随机采样的初始试验数
    tpe_n_ei_candidates: int = 24  # EI计算的候选点数

    # ==================== 剪枝器配置 ====================
    # 是否启用剪枝（提前停止无望的试验）
    enable_pruning: bool = True

    # 剪枝器类型：'Median', 'Percentile', 'Hyperband'
    pruner_type: str = 'Median'

    # Median剪枝器参数
    pruner_n_startup_trials: int = 5  # 开始剪枝前的试验数
    pruner_n_warmup_steps: int = 10  # 每个试验的预热步数
    pruner_interval_steps: int = 1  # 剪枝检查间隔

    # ==================== 回调和日志配置 ====================
    # 进度报告间隔（每N个试验报告一次）
    report_interval: int = 1

    # 是否保存中间结果
    save_intermediate: bool = True

    # 中间结果保存间隔
    save_interval: int = 1

    # ==================== 提前停止配置 ====================
    # 是否启用提前停止（连续N次没有改善则停止）
    enable_early_stopping: bool = True

    # 提前停止的耐心值
    early_stopping_patience: int = 20

    # 最小改善阈值（相对改善）
    min_improvement_ratio: float = 0.001  # 0.1%


class TransformerSearchSpace:
    """Transformer模型的搜索空间定义"""

    @staticmethod
    def get_search_space():
        """返回Transformer的搜索空间配置"""
        return {
            # 模型架构参数
            'model_dim': {
                'type': 'categorical',
                'choices': [16, 32, 64, 128],
                'default': 64,
                'description': '模型内部维度'
            },

            'num_heads': {
                'type': 'conditional_categorical',  # 条件选择（依赖model_dim）
                'condition': 'model_dim',
                'choices_map': {
                    16: [2, 4, 8, 16],
                    32: [2, 4, 8, 16],
                    64: [2, 4, 8, 16],
                    128: [2, 4, 8, 16]
                },
                'default': 8,
                'description': '注意力头数（必须能整除model_dim）'
            },

            'num_layers': {
                'type': 'int',
                'low': 2,
                'high': 12,
                'step': 1,
                'default': 6,
                'description': 'Transformer层数'
            },

            'dropout': {
                'type': 'float',
                'low': 0.0,
                'high': 0.5,
                'step': 0.05,
                'default': 0.1,
                'description': 'Dropout率'
            },

            # 训练参数
            'learning_rate': {
                'type': 'float',
                'low': 1e-5,
                'high': 1e-2,
                'log': True,  # 对数尺度
                'default': 1e-3,
                'description': '学习率'
            },

            'batch_size': {
                'type': 'categorical',
                'choices': [8, 16, 32, 64, 128],
                'default': 32,
                'description': '批次大小'
            },

            'weight_decay': {
                'type': 'float',
                'low': 1e-6,
                'high': 1e-3,
                'log': True,
                'default': 1e-5,
                'description': 'L2正则化强度'
            },

            # 早停参数
            'early_stopping_patience': {
                'type': 'int',
                'low': 3,
                'high': 15,
                'default': 5,
                'description': '早停耐心值'
            },

            'min_delta_ratio': {
                'type': 'float',
                'low': 0.001,
                'high': 0.01,
                'default': 0.003,
                'description': '最小改善比例'
            },

            # 训练轮数（可选：固定或搜索）
            'epochs': {
                'type': 'fixed',  # 或改为 'int' 来搜索
                'value': 50,  # 优化时用较少轮数加快速度
                # 'low': 30,
                # 'high': 100,
                'description': '训练轮数'
            }
        }


class SVRSearchSpace:
    """SVR模型的搜索空间定义"""

    @staticmethod
    def get_search_space():
        """返回SVR的搜索空间配置"""
        return {
            'svr_kernel': {
                'type': 'categorical',
                'choices': ['rbf', 'poly', 'linear'],
                'default': 'rbf',
                'description': '核函数类型'
            },

            'svr_C': {
                'type': 'float',
                'low': 0.1,
                'high': 1000,
                'log': True,
                'default': 100,
                'description': '正则化参数'
            },

            'svr_epsilon': {
                'type': 'float',
                'low': 0.001,
                'high': 1.0,
                'log': True,
                'default': 0.1,
                'description': '不敏感损失参数'
            },

            'svr_gamma': {
                'type': 'categorical',
                'choices': ['scale', 'auto'],
                # 或者搜索具体数值：
                # 'type': 'float',
                # 'low': 1e-4,
                # 'high': 1,
                # 'log': True,
                'default': 'scale',
                'description': '核函数系数'
            },

            # 条件参数（仅当kernel='poly'时）
            'svr_degree': {
                'type': 'int',
                'low': 2,
                'high': 5,
                'default': 3,
                'condition': "svr_kernel == 'poly'",
                'description': '多项式阶数'
            },

            # 条件参数（仅当kernel in ['poly', 'sigmoid']时）
            'svr_coef0': {
                'type': 'float',
                'low': 0.0,
                'high': 1.0,
                'default': 0.0,
                'condition': "svr_kernel in ['poly', 'sigmoid']",
                'description': '核函数独立项'
            }
        }


class OptimizationStrategy:
    """优化策略配置"""

    # 两阶段优化策略
    TWO_STAGE = {
        'name': '两阶段优化',
        'description': '先优化Transformer，再用最佳Transformer优化SVR',
        'stages': ['transformer', 'svr']
    }

    # 联合优化策略（可选）
    JOINT = {
        'name': '联合优化',
        'description': '同时优化Transformer和SVR参数',
        'stages': ['joint']
    }

    # 粗细搜索策略
    COARSE_TO_FINE = {
        'name': '粗细搜索',
        'description': '先粗搜索找到大致范围，再细搜索',
        'stages': ['coarse', 'fine'],
        'coarse_trials': 50,
        'fine_trials': 50
    }


def get_optuna_config():
    """获取默认的Optuna配置"""
    return OptunaConfig()


def get_transformer_search_space():
    """获取Transformer搜索空间"""
    return TransformerSearchSpace.get_search_space()


def get_svr_search_space():
    """获取SVR搜索空间"""
    return SVRSearchSpace.get_search_space()


# ==================== 使用示例 ====================
if __name__ == '__main__':
    # 获取配置
    optuna_config = get_optuna_config()
    print(f"Transformer试验次数: {optuna_config.n_trials_transformer}")

    # 获取搜索空间
    transformer_space = get_transformer_search_space()
    print(f"Transformer搜索参数: {list(transformer_space.keys())}")

    svr_space = get_svr_search_space()
    print(f"SVR搜索参数: {list(svr_space.keys())}")

    # 修改配置示例
    optuna_config.n_trials_transformer = 200  # 增加试验次数
    optuna_config.sampler_type = 'Random'  # 改用随机搜索
    optuna_config.enable_pruning = True  # 启用剪枝