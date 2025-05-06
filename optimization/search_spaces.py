"""定义超参数搜索空间的模块"""

# 为所有搜索空间添加早停参数
early_stopping_config = {
    'optimize': True,
    'patience': [5, 30],  # 搜索范围
    'min_delta': [0.0001, 0.01]  # 搜索范围
}

# 默认搜索空间定义
DEFAULT_SEARCH_SPACE = {
    # 集成类型
    'integration_type': ['feature_extraction', 'ensemble', 'end2end'],

    # 早停参数
    'early_stopping': early_stopping_config,

    # Transformer参数
    'd_model': [64, 128, 256, 512],
    'nhead': [2, 4, 8, 16],  # 确保这些值都能被d_model整除
    'num_layers': [2, 6],
    'dim_feedforward': [128, 1024],

    # SVR参数
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 100.0],
    'epsilon': [0.01, 1.0],
    'gamma': ['scale', 'auto'],
    'degree': [2, 5],

    # 楼层分类器参数
    'floor_classifier_type': ['random_forest', 'svm', 'xgboost'],
    'floor_n_estimators': [50, 100, 200],
    'floor_max_depth': [None, 10, 20, 30],
    'floor_min_samples_split': [2, 5, 10],
    'floor_min_samples_leaf': [1, 2, 4]
}

# 轻量级搜索空间 - 用于快速测试
LIGHT_SEARCH_SPACE = {
    'integration_type': ['feature_extraction', 'ensemble'],
    'd_model': [64, 128],
    'nhead': [2, 4, 8],  # 确保这些值都能被d_model整除
    'num_layers': [2, 3],
    'dim_feedforward': [128, 256],
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 10.0],
    'epsilon': [0.1, 0.5],
    'gamma': ['scale'],

    # 楼层分类器参数 - 轻量版
    'floor_classifier_type': ['random_forest', 'svm'],
    'floor_n_estimators': [50, 100],
    'floor_max_depth': [None, 20],
    'floor_min_samples_split': [2, 5],
    'floor_min_samples_leaf': [1, 2]
}

# 完整的搜索空间 - 用于详尽的超参数搜索
COMPREHENSIVE_SEARCH_SPACE = {
    'integration_type': ['feature_extraction', 'ensemble', 'end2end'],
    'd_model': [64, 128, 256, 512, 1024],
    'nhead': [2, 4, 8, 16],  # 确保这些值都能被d_model整除
    'num_layers': [2, 3, 4, 6, 8],
    'dim_feedforward': [128, 256, 512, 1024, 2048],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'epsilon': [0.001, 0.01, 0.1, 0.5, 1.0],
    'gamma': ['scale', 'auto', 0.1, 1.0],
    'degree': [2, 3, 4, 5],

    # 楼层分类器参数 - 完整版
    'floor_classifier_type': ['random_forest', 'svm', 'xgboost'],
    'floor_n_estimators': [50, 100, 150, 200, 300],
    'floor_max_depth': [None, 10, 15, 20, 25, 30],
    'floor_min_samples_split': [2, 3, 5, 8, 10],
    'floor_min_samples_leaf': [1, 2, 3, 4, 5]
}

# 特征提取特定的搜索空间
FEATURE_EXTRACTION_SEARCH_SPACE = {
    'integration_type': ['feature_extraction'],
    'd_model': [128, 256, 512],
    'nhead': [2, 4, 8, 16],  # 确保这些值都能被d_model整除
    'num_layers': [2, 4, 6],
    'dim_feedforward': [256, 512, 1024],
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1.0, 10.0, 100.0],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3],

    # 楼层分类器参数
    'floor_classifier_type': ['random_forest', 'svm', 'xgboost'],
    'floor_n_estimators': [50, 100, 200],
    'floor_max_depth': [None, 15, 25],
    'floor_min_samples_split': [2, 5, 10],
    'floor_min_samples_leaf': [1, 2, 3]
}

# 集成特定的搜索空间
ENSEMBLE_SEARCH_SPACE = {
    'integration_type': ['ensemble'],
    'd_model': [128, 256, 512],
    'nhead': [2, 4, 8, 16],  # 确保这些值都能被d_model整除
    'num_layers': [2, 4, 6],
    'dim_feedforward': [256, 512, 1024],
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1.0, 10.0, 100.0],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3],

    # 楼层分类器参数
    'floor_classifier_type': ['random_forest', 'svm', 'xgboost'],
    'floor_n_estimators': [50, 100, 200],
    'floor_max_depth': [None, 15, 25],
    'floor_min_samples_split': [2, 5, 10],
    'floor_min_samples_leaf': [1, 2, 3]
}

# 端到端特定的搜索空间
END2END_SEARCH_SPACE = {
    'integration_type': ['end2end'],
    'd_model': [128, 256, 512],
    'nhead': [2, 4, 8, 16],  # 确保这些值都能被d_model整除
    'num_layers': [2, 4, 6],
    'dim_feedforward': [256, 512, 1024],
    'C': [0.1, 1.0, 10.0],
    'epsilon': [0.01, 0.1, 0.5],

    # 楼层分类器参数
    'floor_classifier_type': ['random_forest', 'svm', 'xgboost'],
    'floor_n_estimators': [50, 100, 200],
    'floor_max_depth': [None, 15, 25],
    'floor_min_samples_split': [2, 5, 10],
    'floor_min_samples_leaf': [1, 2, 3]
}


def get_search_space(space_type='default'):
    """
    获取指定类型的搜索空间

    参数:
        space_type (str): 搜索空间类型

    返回:
        dict: 搜索空间定义
    """
    space_types = {
        'default': DEFAULT_SEARCH_SPACE,
        'light': LIGHT_SEARCH_SPACE,
        'comprehensive': COMPREHENSIVE_SEARCH_SPACE,
        'feature_extraction': FEATURE_EXTRACTION_SEARCH_SPACE,
        'ensemble': ENSEMBLE_SEARCH_SPACE,
        'end2end': END2END_SEARCH_SPACE
    }

    if space_type not in space_types:
        raise ValueError(f"未知的搜索空间类型: {space_type}")

    return space_types[space_type]

# 确保其他搜索空间也包含早停配置
LIGHT_SEARCH_SPACE['early_stopping'] = early_stopping_config
COMPREHENSIVE_SEARCH_SPACE['early_stopping'] = early_stopping_config
FEATURE_EXTRACTION_SEARCH_SPACE['early_stopping'] = early_stopping_config
ENSEMBLE_SEARCH_SPACE['early_stopping'] = early_stopping_config
END2END_SEARCH_SPACE['early_stopping'] = early_stopping_config