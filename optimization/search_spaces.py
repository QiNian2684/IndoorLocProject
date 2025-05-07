"""
定义超参数搜索空间的模块

本模块定义了Optuna超参数优化过程中使用的各种搜索空间。
搜索空间指定了每个超参数可能的取值范围，Optuna根据这些范围进行采样和优化。
提供了不同类型的搜索空间以适应不同的实验需求和计算资源约束。
"""

# 为所有搜索空间添加早停参数
# 早停机制可以有效减少训练时间并防止过拟合，这里将其参数也纳入优化范围
early_stopping_config = {
    'optimize': True,                # 是否将早停参数纳入优化范围
    'patience': [3, 10],             # 早停耐心值的搜索范围：最小5轮，最大30轮
                                     # 较小的值使模型更快停止训练，较大的值给模型更多改进机会
    'min_delta': [0.0001, 0.01]      # 被视为改进的最小变化阈值搜索范围
                                     # 较小的值对微小改进更敏感，较大的值只在显著改进时继续训练
}

# 默认搜索空间定义 - 适用于一般场景的平衡搜索空间
DEFAULT_SEARCH_SPACE = {
    # 集成类型 - 决定如何组合SVR和Transformer
    'integration_type': ['feature_extraction', 'ensemble', 'end2end'],
                                     # 'feature_extraction': Transformer提取特征，SVR预测位置
                                     # 'ensemble': 独立训练模型后加权组合结果
                                     # 'end2end': 端到端训练的混合架构

    # 早停参数 - 控制训练过程中的停止条件
    'early_stopping': early_stopping_config,

    # Transformer参数 - 深度学习模型的核心参数
    'd_model': [64, 128, 256, 512],  # Transformer模型的隐藏维度
                                     # 较大的值增加模型容量但需要更多计算资源
                                     # 较小的值训练更快但可能限制性能

    'nhead': [2, 4, 8, 16],          # 多头自注意力机制中的头数
                                     # 需确保能被d_model整除
                                     # 多头允许模型关注不同的特征子空间

    'num_layers': [2, 6],            # Transformer编码器的层数
                                     # 更多层可以学习更复杂的模式但更难训练

    'dim_feedforward': [128, 1024],  # 前馈网络隐藏层维度
                                     # 控制每层Transformer的非线性变换能力

    # SVR参数 - 支持向量回归模型的关键参数
    'kernel': ['linear', 'rbf', 'poly'],  # 核函数类型
                                          # 'linear': 线性关系，最简单，计算最快
                                          # 'rbf': 径向基核，适用于非线性复杂关系
                                          # 'poly': 多项式核，适用于特定非线性模式

    'C': [0.1, 100.0],               # 正则化参数，控制误差的惩罚强度
                                     # 较大值：更关注每个样本的拟合，可能过拟合
                                     # 较小值：更关注一般模式，可能欠拟合

    'epsilon': [0.01, 1.0],          # SVR的不敏感区域宽度
                                     # 控制允许的预测误差，影响支持向量的数量
                                     # 较小值要求更精确的拟合，较大值容忍更多误差

    'gamma': ['scale', 'auto'],      # RBF核和多项式核的系数
                                     # 'scale': 使用1/(n_features*X.var())
                                     # 'auto': 使用1/n_features
                                     # 控制核函数的影响范围

    'degree': [2, 5],                # 多项式核函数的度数
                                     # 仅当kernel='poly'时使用
                                     # 更高的度数可以捕获更复杂的非线性关系

    # 楼层分类器参数 - 用于预测用户所在楼层的分类模型参数
    'floor_classifier_type': ['random_forest', 'svm', 'xgboost'],
                                     # 分类器类型选择
                                     # 'random_forest': 集成树模型，鲁棒性好
                                     # 'svm': 支持向量机，适合复杂边界
                                     # 'xgboost': 梯度提升树，通常性能最佳

    'floor_n_estimators': [50, 100, 200],  # 随机森林或XGBoost中的树数量
                                           # 更多树提高模型稳定性和性能，但增加计算成本

    'floor_max_depth': [None, 10, 20, 30],  # 决策树的最大深度
                                            # None表示无限制，让树生长到完全纯净的叶节点
                                            # 较小的深度限制可以防止过拟合

    'floor_min_samples_split': [2, 5, 10],  # 分裂内部节点所需的最小样本数
                                            # 较大值可减少过拟合，较小值可提高拟合能力

    'floor_min_samples_leaf': [1, 2, 4]     # 叶节点所需的最小样本数
                                            # 较大值确保每个叶节点有足够样本，提高泛化能力
}

# 轻量级搜索空间 - 用于快速测试或计算资源有限的场景
# 减少了参数选项和范围，提高搜索效率，牺牲潜在最优性能
LIGHT_SEARCH_SPACE = {
    'integration_type': ['feature_extraction', 'ensemble'],  # 仅包含两种集成方式，排除计算较重的end2end
    'd_model': [64, 128],                    # 较小的Transformer维度，加快训练速度
    'nhead': [2, 4, 8],                      # 较少的注意力头选项
    'num_layers': [2, 3],                    # 较少的Transformer层数
    'dim_feedforward': [128, 256],           # 较小的前馈网络维度
    'kernel': ['linear', 'rbf'],             # 仅包含两种最常用的核函数
    'C': [0.1, 10.0],                        # 缩小正则化参数搜索范围
    'epsilon': [0.1, 0.5],                   # 缩小epsilon搜索范围
    'gamma': ['scale'],                      # 仅使用scale方式计算gamma

    # 楼层分类器参数 - 轻量版
    'floor_classifier_type': ['random_forest', 'svm'],  # 排除计算较重的xgboost
    'floor_n_estimators': [50, 100],         # 较少的树数量
    'floor_max_depth': [None, 20],           # 简化深度选项
    'floor_min_samples_split': [2, 5],       # 较少的分裂参数选项
    'floor_min_samples_leaf': [1, 2]         # 较少的叶节点参数选项
}

# 完整的搜索空间 - 用于详尽的超参数搜索，适合资源充足且追求最佳性能的场景
# 提供了更广泛的参数选择和更精细的搜索范围，但需要更多计算资源
COMPREHENSIVE_SEARCH_SPACE = {
    'integration_type': ['feature_extraction', 'ensemble', 'end2end'],  # 所有集成方式
    'd_model': [64, 128, 256, 512, 1024],    # 更广泛的模型维度选择，包括更大的维度
    'nhead': [2, 4, 8, 16],                  # 完整的注意力头选项
    'num_layers': [2, 3, 4, 6, 8],           # 更多的层数选项，包括深层网络
    'dim_feedforward': [128, 256, 512, 1024, 2048],  # 更广泛的前馈网络维度
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # 包括所有可用的核函数
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],      # 更广泛的正则化参数选择
    'epsilon': [0.001, 0.01, 0.1, 0.5, 1.0], # 更广泛的epsilon选择
    'gamma': ['scale', 'auto', 0.1, 1.0],    # 包括固定值的gamma选项
    'degree': [2, 3, 4, 5],                  # 更多的多项式度数选项

    # 楼层分类器参数 - 完整版
    'floor_classifier_type': ['random_forest', 'svm', 'xgboost'],  # 所有分类器类型
    'floor_n_estimators': [50, 100, 150, 200, 300],  # 更广泛的树数量选择
    'floor_max_depth': [None, 10, 15, 20, 25, 30],   # 更精细的深度选项
    'floor_min_samples_split': [2, 3, 5, 8, 10],     # 更广泛的分裂参数选择
    'floor_min_samples_leaf': [1, 2, 3, 4, 5]        # 更广泛的叶节点参数选择
}

# 特征提取特定的搜索空间 - 专注于优化特征提取集成方式
# 固定集成类型为feature_extraction，重点优化特征提取过程中的参数
FEATURE_EXTRACTION_SEARCH_SPACE = {
    'integration_type': ['feature_extraction'],  # 固定为特征提取方式
    'd_model': [128, 256, 512],                 # 中到大尺寸的Transformer模型维度
    'nhead': [2, 4, 8, 16],                     # 完整的注意力头选项
    'num_layers': [2, 4, 6],                    # 关注中等复杂度的网络
    'dim_feedforward': [256, 512, 1024],        # 中等规模的前馈网络维度
    'kernel': ['linear', 'rbf', 'poly'],        # 主要核函数，用于SVR最终预测
    'C': [0.1, 1.0, 10.0, 100.0],               # 广泛的正则化参数
    'epsilon': [0.01, 0.1, 0.5],                # 中等范围的epsilon选择
    'gamma': ['scale', 'auto'],                 # 自动计算的gamma选项
    'degree': [2, 3],                           # 低阶多项式核

    # 楼层分类器参数
    'floor_classifier_type': ['random_forest', 'svm', 'xgboost'],  # 所有分类器类型
    'floor_n_estimators': [50, 100, 200],       # 适中的树数量选择
    'floor_max_depth': [None, 15, 25],          # 精简但有代表性的深度选项
    'floor_min_samples_split': [2, 5, 10],      # 代表性的分裂参数
    'floor_min_samples_leaf': [1, 2, 3]         # 代表性的叶节点参数
}

# 集成特定的搜索空间 - 专注于优化独立模型集成的加权方式
# 固定集成类型为ensemble，重点优化各模型的超参数以获得最佳组合效果
ENSEMBLE_SEARCH_SPACE = {
    'integration_type': ['ensemble'],            # 固定为集成方式
    'd_model': [128, 256, 512],                 # 中到大尺寸的Transformer模型
    'nhead': [2, 4, 8, 16],                     # 完整的注意力头选项
    'num_layers': [2, 4, 6],                    # 关注中等复杂度的网络
    'dim_feedforward': [256, 512, 1024],        # 中等规模的前馈网络维度
    'kernel': ['linear', 'rbf', 'poly'],        # 主要核函数，影响SVR组件性能
    'C': [0.1, 1.0, 10.0, 100.0],               # 广泛的正则化参数
    'epsilon': [0.01, 0.1, 0.5],                # 中等范围的epsilon选择
    'gamma': ['scale', 'auto'],                 # 自动计算的gamma选项
    'degree': [2, 3],                           # 低阶多项式核

    # 楼层分类器参数
    'floor_classifier_type': ['random_forest', 'svm', 'xgboost'],  # 所有分类器类型
    'floor_n_estimators': [50, 100, 200],       # 适中的树数量选择
    'floor_max_depth': [None, 15, 25],          # 精简但有代表性的深度选项
    'floor_min_samples_split': [2, 5, 10],      # 代表性的分裂参数
    'floor_min_samples_leaf': [1, 2, 3]         # 代表性的叶节点参数
}

# 端到端特定的搜索空间 - 专注于优化端到端混合模型架构
# 固定集成类型为end2end，重点优化深度学习和SVR组件的集成架构参数
END2END_SEARCH_SPACE = {
    'integration_type': ['end2end'],            # 固定为端到端方式
    'd_model': [128, 256, 512],                 # 中到大尺寸的Transformer模型
    'nhead': [2, 4, 8, 16],                     # 完整的注意力头选项
    'num_layers': [2, 4, 6],                    # 探索不同深度的网络
    'dim_feedforward': [256, 512, 1024],        # 中等规模的前馈网络维度
    'C': [0.1, 1.0, 10.0],                      # SVR的正则化参数，端到端模式下更简化
    'epsilon': [0.01, 0.1, 0.5],                # 端到端模式下的epsilon选项

    # 楼层分类器参数
    'floor_classifier_type': ['random_forest', 'svm', 'xgboost'],  # 所有分类器类型
    'floor_n_estimators': [50, 100, 200],       # 适中的树数量选择
    'floor_max_depth': [None, 15, 25],          # 精简但有代表性的深度选项
    'floor_min_samples_split': [2, 5, 10],      # 代表性的分裂参数
    'floor_min_samples_leaf': [1, 2, 3]         # 代表性的叶节点参数
}


def get_search_space(space_type='default'):
    """
    获取指定类型的搜索空间

    根据实验需求和可用计算资源，选择合适的超参数搜索空间。
    不同的搜索空间在参数范围广度和搜索密度上有所区别。

    参数:
        space_type (str): 搜索空间类型
            - 'default': 平衡的默认搜索空间，适合一般用途
            - 'light': 轻量级搜索空间，适合快速测试或计算资源有限
            - 'comprehensive': 全面的搜索空间，适合详尽优化
            - 'feature_extraction': 特征提取特定的搜索空间
            - 'ensemble': 集成特定的搜索空间
            - 'end2end': 端到端特定的搜索空间

    返回:
        dict: 所选搜索空间的定义字典

    异常:
        ValueError: 当指定的搜索空间类型未知时抛出
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
# 为所有搜索空间添加相同的早停参数配置，使早停机制在所有优化场景中都可调整
LIGHT_SEARCH_SPACE['early_stopping'] = early_stopping_config
COMPREHENSIVE_SEARCH_SPACE['early_stopping'] = early_stopping_config
FEATURE_EXTRACTION_SEARCH_SPACE['early_stopping'] = early_stopping_config
ENSEMBLE_SEARCH_SPACE['early_stopping'] = early_stopping_config
END2END_SEARCH_SPACE['early_stopping'] = early_stopping_config