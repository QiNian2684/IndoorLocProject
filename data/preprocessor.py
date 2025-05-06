import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod


class Preprocessor(ABC):
    """数据预处理器的抽象基类"""

    @abstractmethod
    def fit(self, data):
        """拟合预处理器参数"""
        pass

    @abstractmethod
    def transform(self, data):
        """应用预处理转换"""
        pass

    def fit_transform(self, data):
        """拟合并应用预处理转换"""
        self.fit(data)
        return self.transform(data)


class UJIIndoorLocPreprocessor(Preprocessor):
    """UJIIndoorLoc数据集的预处理器"""

    def __init__(self, replace_value=-105, normalization='minmax',
                 dimension_reduction=None, n_components=50):
        """
        初始化UJIIndoorLoc预处理器

        参数:
            replace_value (float): 替换未检测到WAP（值为100）的值
            normalization (str): 标准化方法 ('minmax', 'standard', 'robust', None)
            dimension_reduction (str): 降维方法 ('pca', None)
            n_components (int): 如果使用PCA，保留的组件数量
        """
        self.replace_value = replace_value
        self.normalization = normalization
        self.dimension_reduction = dimension_reduction
        self.n_components = n_components
        self.scaler = None
        self.reducer = None
        self.wap_columns = None

    def fit(self, data):
        """
        拟合预处理器参数

        参数:
            data (DataFrame): 原始数据

        返回:
            self: 预处理器实例
        """
        # 识别WAP列
        if isinstance(data, pd.DataFrame):
            # 如果是DataFrame，取前520列作为WAP列
            self.wap_columns = data.columns[:520].tolist()
            data_processed = data[self.wap_columns].copy()
        else:
            # 如果是numpy数组，假设所有列都是WAP列
            data_processed = data.copy()

        # 替换未检测到的WAP值
        if isinstance(data_processed, pd.DataFrame):
            data_processed.replace(100, self.replace_value, inplace=True)
        else:
            data_processed = np.where(data_processed == 100, self.replace_value, data_processed)

        # 创建并拟合标准化器
        if self.normalization == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.normalization == 'standard':
            self.scaler = StandardScaler()
        elif self.normalization == 'robust':
            self.scaler = RobustScaler()

        if self.scaler:
            self.scaler.fit(data_processed)

        # 创建并拟合降维器
        if self.dimension_reduction == 'pca':
            self.reducer = PCA(n_components=self.n_components)
            if self.scaler:
                normalized_data = self.scaler.transform(data_processed)
                self.reducer.fit(normalized_data)
            else:
                self.reducer.fit(data_processed)

        return self

    def transform(self, data):
        """
        应用预处理转换

        参数:
            data (DataFrame or ndarray): 要转换的数据

        返回:
            ndarray: 预处理后的数据
        """
        # 提取WAP列
        if isinstance(data, pd.DataFrame):
            if self.wap_columns:
                data_processed = data[self.wap_columns].copy()
            else:
                data_processed = data.iloc[:, :520].copy()
        else:
            data_processed = data.copy()

        # 替换未检测到的WAP值
        if isinstance(data_processed, pd.DataFrame):
            data_processed.replace(100, self.replace_value, inplace=True)
        else:
            data_processed = np.where(data_processed == 100, self.replace_value, data_processed)

        # 应用标准化
        if self.scaler:
            data_processed = self.scaler.transform(data_processed)

        # 应用降维
        if self.reducer:
            data_processed = self.reducer.transform(data_processed)

        return data_processed

    def get_feature_names(self):
        """获取处理后的特征名称"""
        if self.dimension_reduction:
            # 如果应用了降维，特征名称将是组件索引
            return [f'PC{i + 1}' for i in range(self.n_components)]
        elif self.wap_columns:
            # 如果没有降维，返回原始WAP列名
            return self.wap_columns
        else:
            # 默认特征名称
            return [f'WAP{i + 1}' for i in range(520)]

    def preprocess_dataset(self, training_data, validation_data):
        """
        预处理完整的训练和验证数据集

        参数:
            training_data (DataFrame): 训练数据
            validation_data (DataFrame): 验证数据

        返回:
            dict: 预处理后的数据集
        """
        # 提取要素和目标变量
        X_train = training_data.iloc[:, :520]  # WAP列
        y_train_building = training_data['BUILDINGID']
        y_train_floor = training_data['FLOOR']
        y_train_coords = training_data[['LONGITUDE', 'LATITUDE']]

        X_val = validation_data.iloc[:, :520]
        y_val_building = validation_data['BUILDINGID']
        y_val_floor = validation_data['FLOOR']
        y_val_coords = validation_data[['LONGITUDE', 'LATITUDE']]

        # 拟合并转换训练数据
        X_train_processed = self.fit_transform(X_train)

        # 转换验证数据
        X_val_processed = self.transform(X_val)

        return {
            'X_train': X_train_processed,
            'y_train_building': y_train_building,
            'y_train_floor': y_train_floor,
            'y_train_coords': y_train_coords,
            'X_val': X_val_processed,
            'y_val_building': y_val_building,
            'y_val_floor': y_val_floor,
            'y_val_coords': y_val_coords
        }