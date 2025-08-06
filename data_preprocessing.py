"""
数据预处理模块 - 基于参考代码优化
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self, config):
        self.config = config
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y_longitude = StandardScaler()
        self.scaler_y_latitude = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.statistics = {}

    def load_and_preprocess_data(self) -> Tuple:
        """
        加载并预处理数据
        返回: (X_train, y_train, y_train_floor, X_val, y_val, y_val_floor,
                X_test, y_test, y_test_floor, filtered_test_indices)
        """
        logger.info("=" * 50)
        logger.info("开始数据预处理")
        logger.info("=" * 50)

        # 加载数据
        train_data = pd.read_csv(self.config.train_path)
        test_data = pd.read_csv(self.config.test_path)

        logger.info(f"原始训练集大小: {train_data.shape}")
        logger.info(f"原始测试集大小: {test_data.shape}")

        # 特征列（WAP信号）
        feature_cols = list(range(0, 520))

        # 替换100为-105（表示极弱信号）
        train_features = train_data.iloc[:, feature_cols].replace(100, -105)
        test_features = test_data.iloc[:, feature_cols].replace(100, -105)

        # 处理缺失值
        train_features = train_features.fillna(-105)
        test_features = test_features.fillna(-105)

        # 不进行测试集筛选，使用所有测试样本
        filtered_test_indices = test_data.index.tolist()

        logger.info(f"使用全部测试集样本: {test_data.shape}")

        # 提取目标变量
        y_train_coords = train_data[['LONGITUDE', 'LATITUDE']].values
        y_test_coords = test_data[['LONGITUDE', 'LATITUDE']].values

        y_train_floor = train_data['FLOOR'].values
        y_test_floor = test_data['FLOOR'].values

        # 楼层编码
        y_train_floor_encoded = self.label_encoder.fit_transform(y_train_floor)
        y_test_floor_encoded = self.label_encoder.transform(y_test_floor)

        # 划分训练集和验证集
        X_train_raw, X_val_raw, y_train_coords_split, y_val_coords_split, \
            y_train_floor_split, y_val_floor_split = train_test_split(
            train_features.values, y_train_coords, y_train_floor_encoded,
            test_size=self.config.test_size, random_state=self.config.seed
        )

        # 特征缩放（直接使用所有520个特征）
        X_train = self.scaler_X.fit_transform(X_train_raw)
        X_val = self.scaler_X.transform(X_val_raw)
        X_test = self.scaler_X.transform(test_features.values)

        logger.info(f"处理后特征维度: {X_train.shape[1]} (保留所有520个WAP特征)")

        # 目标变量标准化
        y_train_longitude = self.scaler_y_longitude.fit_transform(
            y_train_coords_split[:, 0].reshape(-1, 1)
        )
        y_val_longitude = self.scaler_y_longitude.transform(
            y_val_coords_split[:, 0].reshape(-1, 1)
        )
        y_test_longitude = self.scaler_y_longitude.transform(
            y_test_coords[:, 0].reshape(-1, 1)
        )

        y_train_latitude = self.scaler_y_latitude.fit_transform(
            y_train_coords_split[:, 1].reshape(-1, 1)
        )
        y_val_latitude = self.scaler_y_latitude.transform(
            y_val_coords_split[:, 1].reshape(-1, 1)
        )
        y_test_latitude = self.scaler_y_latitude.transform(
            y_test_coords[:, 1].reshape(-1, 1)
        )

        # 合并坐标
        y_train = np.hstack((y_train_longitude, y_train_latitude))
        y_val = np.hstack((y_val_longitude, y_val_latitude))
        y_test = np.hstack((y_test_longitude, y_test_latitude))

        # 检查数据质量
        self._check_data_quality(X_train, X_val, X_test, y_train, y_val, y_test)

        # 打印数据集信息
        self._print_dataset_info(X_train, X_val, X_test, y_train)

        # 保存统计信息
        self.statistics = {
            'train_samples': X_train.shape[0],
            'val_samples': X_val.shape[0],
            'test_samples': X_test.shape[0],
            'feature_dim': X_train.shape[1],
            'floor_classes': len(np.unique(y_train_floor_encoded))
        }

        return (X_train, y_train, y_train_floor_split,
                X_val, y_val, y_val_floor_split,
                X_test, y_test, y_test_floor_encoded,
                filtered_test_indices)

    def inverse_transform_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """反转坐标标准化"""
        longitude = self.scaler_y_longitude.inverse_transform(coords[:, 0].reshape(-1, 1))
        latitude = self.scaler_y_latitude.inverse_transform(coords[:, 1].reshape(-1, 1))
        return np.hstack((longitude, latitude))

    def _check_data_quality(self, *data_arrays):
        """检查数据质量"""
        names = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
        for name, data in zip(names, data_arrays):
            if np.isnan(data).any() or np.isinf(data).any():
                raise ValueError(f"{name} 包含 NaN 或无穷值！")

    def _print_dataset_info(self, X_train, X_val, X_test, y_train):
        """打印数据集信息"""
        print("\n" + "=" * 60)
        print("数据集处理完成后的概况")
        print("-" * 60)
        print("| Dataset   | Samples | Features | Target Dim |")
        print("-" * 60)
        print(f"| X_train   | {X_train.shape[0]:7d} | {X_train.shape[1]:8d} | {y_train.shape[1]:10d} |")
        print(f"| X_val     | {X_val.shape[0]:7d} | {X_val.shape[1]:8d} | {y_train.shape[1]:10d} |")
        print(f"| X_test    | {X_test.shape[0]:7d} | {X_test.shape[1]:8d} | {y_train.shape[1]:10d} |")
        print("-" * 60 + "\n")