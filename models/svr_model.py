import numpy as np
from sklearn.svm import SVR
from .base import PositioningModel


class SVRPositioningModel(PositioningModel):
    """基于SVR的定位模型"""

    def __init__(self, **svr_params):
        """
        初始化SVR定位模型

        参数:
            **svr_params: SVR的参数，如kernel, C, epsilon等
        """
        self.svr_params = svr_params
        self.longitude_model = None
        self.latitude_model = None

    def fit(self, X, y):
        """
        训练模型

        参数:
            X: 特征数据
            y: 目标值 (经度和纬度)

        返回:
            self: 模型实例
        """
        # 确保y是二维的，包含经度和纬度
        if len(y.shape) == 1 or y.shape[1] == 1:
            raise ValueError("目标值y必须包含经度和纬度两列")

        # 分别为经度和纬度创建SVR模型
        self.longitude_model = SVR(**self.svr_params)
        self.latitude_model = SVR(**self.svr_params)

        # 训练模型
        self.longitude_model.fit(X, y[:, 0])
        self.latitude_model.fit(X, y[:, 1])

        return self

    def predict(self, X):
        """
        预测位置

        参数:
            X: 特征数据

        返回:
            ndarray: 预测的位置 (经度和纬度)
        """
        if self.longitude_model is None or self.latitude_model is None:
            raise RuntimeError("模型尚未训练")

        # 预测经度和纬度
        lon_pred = self.longitude_model.predict(X)
        lat_pred = self.latitude_model.predict(X)

        # 合并预测结果
        predictions = np.column_stack((lon_pred, lat_pred))

        return predictions

    def get_params(self):
        """获取模型参数"""
        return {
            'svr_params': self.svr_params,
            'longitude_model_params': self.longitude_model.get_params() if self.longitude_model else None,
            'latitude_model_params': self.latitude_model.get_params() if self.latitude_model else None
        }

    def set_params(self, **params):
        """设置模型参数"""
        if 'svr_params' in params:
            self.svr_params = params['svr_params']
            # 重置模型，因为参数已更改
            self.longitude_model = None
            self.latitude_model = None
        return self


class MultiOutputSVR(PositioningModel):
    """处理多输出回归的SVR包装器"""

    def __init__(self, **svr_params):
        """
        初始化多输出SVR

        参数:
            **svr_params: SVR的参数
        """
        self.svr_params = svr_params
        self.models = []

    def fit(self, X, y):
        """
        训练每个输出维度的独立SVR模型

        参数:
            X: 特征数据
            y: 多维目标值

        返回:
            self: 模型实例
        """
        # 确保y是二维的
        y = np.atleast_2d(y)
        if len(y.shape) != 2:
            raise ValueError("目标值y必须是二维的")

        # 为每个输出维度创建一个SVR
        self.models = []
        for i in range(y.shape[1]):
            model = SVR(**self.svr_params)
            model.fit(X, y[:, i])
            self.models.append(model)

        return self

    def predict(self, X):
        """
        预测所有输出维度

        参数:
            X: 特征数据

        返回:
            ndarray: 多维预测值
        """
        if not self.models:
            raise RuntimeError("模型尚未训练")

        # 预测每个维度
        predictions = np.column_stack([model.predict(X) for model in self.models])

        return predictions