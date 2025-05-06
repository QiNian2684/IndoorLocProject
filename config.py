"""配置文件"""

# 数据相关配置
DATA_CONFIG = {
    'data_dir': './data/raw',
    'download': True,
    'replace_value': -105,  # 替换未检测到的WAP值（原始值为100）
    'normalization': 'minmax',  # 'minmax', 'standard', 'robust', None（提供4种选择的归一化选项）
    'dimension_reduction': 'pca',  # 'pca', None
    'n_components': 50,  # PCA组件数量
    'test_size': 0.2,  # 如果没有验证集，从训练集中分割测试集的比例
}

# 模型相关配置
MODEL_CONFIG = {
    # 通用配置
    'device': 'cuda',  # 'cuda', 'cpu'

    # SVR配置
    'svr': {
        'kernel': 'rbf',  # 'linear', 'poly', 'rbf', 'sigmoid'
        'C': 10.0,
        'epsilon': 0.1,
        'gamma': 'scale',  # 'scale', 'auto', or float
        'degree': 3,
        'cache_size': 1000
    },

    # Transformer配置
    'transformer': {
        'input_dim': 520,  # 或通过降维后的特征数量
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'output_dim': 2,  # 输出维度（经度和纬度）
        'batch_size': 64,
        'epochs': 100,
        'lr': 0.001
    },

    # 混合模型配置
    'hybrid': {
        'integration_type': 'feature_extraction',  # 'feature_extraction', 'ensemble', 'end2end'
        'weights': {'svr': 0.5, 'transformer': 0.5}  # 集成权重
    },

    # 双层反馈配置
    'feedback': {
        'enabled': True,
        'learning_rate': 0.01,
        'meta_learning_rate': 0.001,
        'feedback_window': 100
    },

    # 楼层分类器配置
    'floor_classifier': {
        'type': 'random_forest',  # 'random_forest', 'svm', 'xgboost'
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    }
}

# 优化配置
OPTIMIZATION_CONFIG = {
    'n_trials': 500,  #Optuna调参轮数
    'timeout': None,  # 秒，None表示无限制
    'n_jobs': 1,
    'direction': 'minimize',
    'cv': 5,
    'cv_method': 'spatial',  # 'spatial', 'random', 'hierarchical'
    'search_space': 'default'  # 'default', 'light', 'comprehensive', 'feature_extraction', 'ensemble', 'end2end'
}

# 评估配置
EVALUATION_CONFIG = {
    'metrics': ['mean_error', 'median_error', 'rmse', '75th_percentile', '90th_percentile'],
    'visualizations': ['error_cdf', 'error_distribution', 'error_heatmap', 'error_2d_scatter', 'floor_confusion_matrix'],
    'output_dir': './results',
}

# 日志记录配置
LOGGING_CONFIG = {
    'log_level': 'INFO',  # 日志级别：DEBUG, INFO, WARNING, ERROR
    'console_output': True,  # 是否在控制台输出
    'save_training_history': True,  # 保存训练历史
    'save_predictions': True,  # 保存详细预测结果
    'save_parameters': True,  # 保存模型参数
    'save_optuna_trials': True,  # 保存Optuna试验历史
    'save_epoch_checkpoints': False,  # 是否保存每个轮次的模型检查点
    'checkpoint_frequency': 10,  # 保存检查点的频率（轮次）
    'csv_format': True,  # 是否以CSV格式保存结果
    'visualization_formats': ['png', 'pdf'],  # 可视化格式
    'time_encoded_folders': True,  # 使用时间编码文件夹
    'compress_results': False,  # 是否压缩结果（用于大量数据）
}

# 合并所有配置
CONFIG = {
    'data': DATA_CONFIG,
    'model': MODEL_CONFIG,
    'optimization': OPTIMIZATION_CONFIG,
    'evaluation': EVALUATION_CONFIG,
    'logging': LOGGING_CONFIG,
    'random_state': 42
}

def get_config():
    """获取配置"""
    return CONFIG

def update_config(config_updates):
    """
    更新配置

    参数:
        config_updates (dict): 配置更新
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