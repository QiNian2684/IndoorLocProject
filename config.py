"""
配置文件 - 管理整个室内定位系统的参数配置

此模块定义了系统运行所需的所有配置参数，按功能模块分组。提供了获取和更新配置的函数。
配置结构是嵌套的字典，便于组织和访问相关参数。
"""

# 数据相关配置 - 控制数据加载、预处理和特征工程
DATA_CONFIG = {
    'data_dir': './data/raw',                  # 数据文件存储目录的路径
    'download': True,                          # 是否自动下载数据集（如果本地不存在）
    'replace_value': -105,                     # 替换未检测到的WAP值（原始值为100）为实际RSSI最低值（dBm）
    'normalization': 'minmax',                 # 特征归一化方法:
                                               # - 'minmax': 将特征缩放到[0,1]范围
                                               # - 'standard': 均值为0，标准差为1的标准化
                                               # - 'robust': 基于分位数的归一化，对异常值更鲁棒
                                               # - None: 不进行归一化
    'dimension_reduction': 'None',              # 降维方法:
                                               # - 'pca': 主成分分析降维
                                               # - None: 不进行降维，使用原始特征
    'n_components': 50,                        # PCA保留的主成分数量（如果使用PCA）
    'test_size': 0.2,                          # 如果没有单独的验证集，从训练集中分割测试集的比例
}

# 模型相关配置 - 控制所有模型的行为和参数
MODEL_CONFIG = {
    # 通用配置
    'device': 'cuda',                          # 计算设备: 'cuda'使用GPU, 'cpu'使用CPU

    # 早停配置 - 防止过拟合并优化训练时间
    'early_stopping': {
        'enabled': True,                       # 是否启用早停机制
        'patience': 15,                        # 容忍多少轮验证集性能不提升
        'min_delta': 0.0001,                   # 被视为改进的最小变化阈值
        'verbose': True,                       # 是否打印早停相关信息
        'mode': 'min'                          # 监控模式: 'min'表示指标越小越好, 'max'表示越大越好
    },

    # SVR配置 - 支持向量回归模型参数
    'svr': {
        'kernel': 'rbf',                       # 核函数类型:
                                               # - 'linear': 线性核
                                               # - 'poly': 多项式核
                                               # - 'rbf': 径向基函数核（高斯核）
                                               # - 'sigmoid': S型核函数
        'C': 10.0,                             # 正则化参数，控制错误惩罚的强度
        'epsilon': 0.1,                        # ε-不敏感损失函数的ε值，定义无损失区域
        'gamma': 'scale',                      # 核系数:
                                               # - 'scale': 使用1/(n_features*X.var())
                                               # - 'auto': 使用1/n_features
                                               # - float: 直接指定gamma值
        'degree': 3,                           # 多项式核函数的度数（当kernel='poly'时使用）
        'cache_size': 1000                     # SVR计算中的缓存大小（MB）
    },

    # Transformer配置 - 深度学习Transformer模型参数
    'transformer': {
        'input_dim': 520,                      # 输入特征维度（WAP数量或降维后的特征数）
        'd_model': 256,                        # Transformer模型的隐藏维度
        'nhead': 8,                            # 多头注意力机制中的头数（需要能整除d_model）
        'num_layers': 4,                       # Transformer编码器层数
        'dim_feedforward': 512,                # 前馈网络的隐藏层维度
        'dropout': 0.1,                        # Dropout比率，用于防止过拟合
        'output_dim': 2,                       # 输出维度（通常为2：经度和纬度）
        'batch_size': 64,                      # 训练时的批量大小
        'epochs': 100,                         # 训练的最大轮数
        'lr': 0.001                            # 学习率
    },

    # 混合模型配置 - SVR和Transformer的集成方式
    'hybrid': {
        'integration_type': 'feature_extraction',  # 集成类型:
                                                  # - 'feature_extraction': 使用Transformer提取特征，然后用SVR预测
                                                  # - 'ensemble': 独立训练SVR和Transformer，然后加权组合结果
                                                  # - 'end2end': 端到端训练的混合模型
        'weights': {'svr': 0.5, 'transformer': 0.5}  # 集成模式下的加权系数
    },

    # 双层反馈配置 - 自适应优化机制
    'feedback': {
        'enabled': True,                       # 是否启用反馈机制
        'learning_rate': 0.01,                 # 低层反馈（即时调整）的学习率
        'meta_learning_rate': 0.001,           # 高层反馈（元学习）的学习率
        'feedback_window': 100                 # 高层反馈的样本窗口大小
    },

    # 楼层分类器配置 - 用于预测用户所在楼层
    'floor_classifier': {
        'type': 'random_forest',               # 分类器类型:
                                               # - 'random_forest': 随机森林分类器
                                               # - 'svm': 支持向量机分类器
                                               # - 'xgboost': XGBoost分类器（需安装xgboost库）
        'n_estimators': 100,                   # 随机森林或XGBoost中的树数量
        'max_depth': None,                     # 树的最大深度 (None表示无限制)
        'min_samples_split': 2,                # 分裂内部节点所需的最小样本数
        'min_samples_leaf': 1                  # 叶节点所需的最小样本数
    }
}

# 优化配置 - 控制超参数优化过程
OPTIMIZATION_CONFIG = {
    'n_trials': 500,                           # Optuna优化尝试的次数
    'timeout': None,                           # 优化超时时间（秒），None表示无限制
    'n_jobs': 1,                               # 并行运行的作业数
    'direction': 'minimize',                   # 优化方向: 'minimize'表示越小越好, 'maximize'表示越大越好
    'cv': 5,                                   # 交叉验证的折数
    'cv_method': 'spatial',                    # 交叉验证方法:
                                               # - 'spatial': 空间感知的交叉验证，考虑地理位置
                                               # - 'random': 随机分割
                                               # - 'hierarchical': 按建筑和楼层分层的交叉验证
    'search_space': 'default'                  # 搜索空间类型，影响考虑的超参数范围:
                                               # - 'default': 平衡搜索复杂度和性能的默认空间
                                               # - 'light': 轻量级搜索空间，适合快速测试
                                               # - 'comprehensive': 全面的搜索空间，更多参数组合
                                               # - 'feature_extraction': 专注于特征提取
                                               # - 'ensemble': 专注于集成方法
                                               # - 'end2end': 专注于端到端模型
}

# 评估配置 - 控制模型评估和结果分析
EVALUATION_CONFIG = {
    'metrics': [                               # 评估指标列表
        'mean_error',                          # 平均欧氏距离误差
        'median_error',                        # 中位数欧氏距离误差
        'rmse',                                # 均方根误差
        '75th_percentile',                     # 75%分位数误差
        '90th_percentile'                      # 90%分位数误差
    ],
    'visualizations': [                        # 要生成的可视化类型
        'error_cdf',                           # 误差累积分布函数
        'error_distribution',                  # 误差分布直方图
        'error_heatmap',                       # 空间误差热图
        'error_2d_scatter',                    # 2D误差散点图
        'floor_confusion_matrix'               # 楼层预测混淆矩阵
    ],
    'output_dir': './results',                 # 评估结果的输出目录
}

# 日志记录配置 - 控制日志、检查点和结果保存行为
LOGGING_CONFIG = {
    'log_level': 'INFO',                       # 日志级别: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'console_output': True,                    # 是否在控制台输出日志
    'save_training_history': True,             # 是否保存训练历史记录
    'save_predictions': True,                  # 是否保存详细预测结果
    'save_parameters': True,                   # 是否保存模型参数
    'save_optuna_trials': True,                # 是否保存Optuna试验历史
    'save_epoch_checkpoints': True,            # 是否保存每个周期的模型检查点
    'checkpoint_frequency': 10,                # 保存检查点的频率（轮次）
    'csv_format': True,                        # 是否以CSV格式保存结果
    'visualization_formats': ['png', 'pdf'],   # 可视化图表保存格式
    'time_encoded_folders': True,              # 是否使用时间编码的文件夹名
    'compress_results': False,                 # 是否压缩大型结果文件
}

# 合并所有配置到一个主配置字典
CONFIG = {
    'data': DATA_CONFIG,                       # 数据相关配置
    'model': MODEL_CONFIG,                     # 模型相关配置
    'optimization': OPTIMIZATION_CONFIG,       # 优化相关配置
    'evaluation': EVALUATION_CONFIG,           # 评估相关配置
    'logging': LOGGING_CONFIG,                 # 日志记录相关配置
    'random_state': 42                         # 全局随机种子，确保实验可重现性
}

def get_config():
    """
    获取当前系统配置

    此函数返回当前的全局配置字典，包含系统所有模块的配置参数。

    返回:
        dict: 完整的配置字典
    """
    return CONFIG

def update_config(config_updates):
    """
    更新配置字典

    此函数接收一个更新字典，递归地更新全局配置。
    支持更新嵌套字典中的特定值，而不影响其他配置。

    参数:
        config_updates (dict): 包含要更新的配置项的字典

    返回:
        dict: 更新后的配置字典

    示例:
        update_config({
            'model': {
                'hybrid': {
                    'integration_type': 'ensemble'
                }
            }
        })
    """
    global CONFIG

    # 递归更新配置
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = update_dict(d[k], v)
            else:
                d[k] = v
        return d

    CONFIG = update_dict(CONFIG, config_updates)
    return CONFIG