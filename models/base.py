from abc import ABC, abstractmethod
import numpy as np


class PositioningModel(ABC):
    """定位模型的抽象基类"""

    @abstractmethod
    def fit(self, X, y):
        """
        训练模型

        参数:
            X: 特征数据
            y: 目标值
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        使用模型进行预测

        参数:
            X: 要预测的特征数据

        返回:
            预测值
        """
        pass

    def evaluate(self, X, y):
        """
        评估模型性能

        参数:
            X: 测试特征数据
            y: 测试目标值

        返回:
            评估指标
        """
        y_pred = self.predict(X)
        error = np.mean(np.sqrt(np.sum((y - y_pred) ** 2, axis=1)))
        return {
            'mean_error': error
        }


class FeatureExtractor(ABC):
    """特征提取器的抽象基类"""

    @abstractmethod
    def fit(self, X, y=None):
        """拟合特征提取器"""
        pass

    @abstractmethod
    def transform(self, X):
        """转换特征"""
        pass

    def fit_transform(self, X, y=None):
        """拟合并转换"""
        self.fit(X, y)
        return self.transform(X)


class PositioningStrategy(ABC):
    """定位策略的抽象基类"""

    @abstractmethod
    def compute_position(self, signal_data):
        """
        计算位置

        参数:
            signal_data: 信号数据

        返回:
            位置信息
        """
        pass