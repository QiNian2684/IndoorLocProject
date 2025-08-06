"""
SVR回归器模块
"""
import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logger = logging.getLogger(__name__)


class SVRRegressor:
    """SVR位置回归器"""

    def __init__(self, config):
        self.config = config
        self.feature_scaler = StandardScaler()
        self.regression_model = None
        self.is_fitted = False

    def create_model(self, svr_params=None):
        """创建SVR模型"""
        if svr_params is None:
            svr_params = {
                'kernel': self.config.svr_kernel,
                'C': self.config.svr_C,
                'epsilon': self.config.svr_epsilon,
                'gamma': self.config.svr_gamma,
                'degree': self.config.svr_degree,
                'coef0': self.config.svr_coef0
            }

        # 创建基础SVR
        base_svr = SVR(**svr_params)

        # 使用MultiOutputRegressor处理多维输出
        self.regression_model = MultiOutputRegressor(base_svr, n_jobs=-1)

        return self.regression_model

    def fit(self, X_features, y_coords):
        """训练SVR模型"""
        logger.info("训练SVR回归模型...")

        # 特征标准化
        X_scaled = self.feature_scaler.fit_transform(X_features)

        # 确保没有NaN
        if np.isnan(X_scaled).any() or np.isnan(y_coords).any():
            raise ValueError("训练数据包含NaN值")

        # 训练模型
        self.regression_model.fit(X_scaled, y_coords)
        self.is_fitted = True

        logger.info("SVR模型训练完成")

    def predict(self, X_features):
        """预测坐标"""
        if not self.is_fitted:
            raise ValueError("SVR模型尚未训练")

        # 特征标准化
        X_scaled = self.feature_scaler.transform(X_features)

        # 预测
        predictions = self.regression_model.predict(X_scaled)

        return predictions

    def save(self, filepath):
        """保存模型"""
        model_data = {
            'regression_model': self.regression_model,
            'feature_scaler': self.feature_scaler,
            'is_fitted': self.is_fitted,
            'config': self.config
        }
        joblib.dump(model_data, filepath)
        logger.info(f"SVR模型已保存到: {filepath}")

    def load(self, filepath):
        """加载模型"""
        model_data = joblib.load(filepath)
        self.regression_model = model_data['regression_model']
        self.feature_scaler = model_data['feature_scaler']
        self.is_fitted = model_data['is_fitted']
        logger.info(f"SVR模型已加载: {filepath}")