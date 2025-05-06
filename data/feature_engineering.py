import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import torch
import torch.nn as nn


class IdentityExtractor(BaseEstimator, TransformerMixin):
    """不进行任何特征转换的特征提取器"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class PCAFeatureExtractor(BaseEstimator, TransformerMixin):
    """使用PCA进行特征提取"""

    def __init__(self, n_components=50):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.transform(X)

    def explained_variance_ratio(self):
        """返回所保留组件的解释方差比率"""
        return self.pca.explained_variance_ratio_


class SelectKBestFeatureExtractor(BaseEstimator, TransformerMixin):
    """使用SelectKBest进行特征选择"""

    def __init__(self, k=50, score_func=f_regression):
        self.k = k
        self.score_func = score_func
        self.selector = SelectKBest(score_func=score_func, k=k)

    def fit(self, X, y):
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        return self.selector.transform(X)

    def get_support(self):
        """获取所选特征的掩码"""
        return self.selector.get_support()

    def get_feature_indices(self):
        """获取所选特征的索引"""
        return np.where(self.selector.get_support())[0]


class TransformerFeatureExtractor(nn.Module):
    """使用Transformer进行特征提取的神经网络模块"""

    def __init__(self, input_dim=520, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        """
        初始化Transformer特征提取器

        参数:
            input_dim (int): 输入特征维度
            d_model (int): Transformer模型维度
            nhead (int): 多头注意力中的头数
            num_layers (int): Transformer编码器层数
            dim_feedforward (int): 前馈网络的隐藏维度
            dropout (float): Dropout率
        """
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        前向传播

        参数:
            x (Tensor): 形状为 (batch_size, input_dim) 的输入张量

        返回:
            Tensor: 形状为 (batch_size, d_model) 的特征表示
        """
        # 输入形状: (batch_size, input_dim)
        # 重塑为Transformer输入: (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        x = self.input_projection(x)  # (batch_size, 1, d_model)
        x = self.transformer_encoder(x)  # (batch_size, 1, d_model)
        x = x.transpose(1, 2)  # (batch_size, d_model, 1)
        x = self.global_pooling(x)  # (batch_size, d_model, 1)
        return x.squeeze(-1)  # (batch_size, d_model)

    def extract_features(self, X):
        """
        从numpy数组或DataFrame中提取特征

        参数:
            X (ndarray or DataFrame): 输入数据

        返回:
            ndarray: 提取的特征
        """
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            features = self.forward(X_tensor).numpy()
        return features


class TransformerFeatureExtractorWrapper(BaseEstimator, TransformerMixin):
    """将Transformer特征提取器包装为scikit-learn兼容的接口"""

    def __init__(self, input_dim=520, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.extractor = None

    def fit(self, X, y=None):
        """
        初始化特征提取器（不需要拟合，因为我们使用的是未训练的模型）
        """
        self.extractor = TransformerFeatureExtractor(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        )
        return self

    def transform(self, X):
        """
        转换输入数据
        """
        if self.extractor is None:
            self.fit(X)
        return self.extractor.extract_features(X)