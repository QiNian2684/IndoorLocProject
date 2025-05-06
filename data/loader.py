import os
import pandas as pd
import numpy as np
from urllib.request import urlretrieve
import zipfile


class UJIIndoorLocLoader:
    """用于加载UJIIndoorLoc数据集的加载器类"""

    def __init__(self, data_dir='./data/raw', download=True):
        """
        初始化UJIIndoorLoc数据加载器

        参数:
            data_dir (str): 数据文件存储目录
            download (bool): 如果为True且数据不存在，则下载数据
        """
        self.data_dir = data_dir
        self.training_file = os.path.join(data_dir, 'TrainingData.csv')
        self.validation_file = os.path.join(data_dir, 'ValidationData.csv')
        self.download_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00310/UJIndoorLoc.zip"

        # 创建数据目录（如果不存在）
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # 如果需要，下载数据
        if download and (not os.path.exists(self.training_file) or
                         not os.path.exists(self.validation_file)):
            self._download_and_extract()

    def _download_and_extract(self):
        """下载并解压UJIIndoorLoc数据集"""
        print("Downloading UJIIndoorLoc dataset...")
        zip_path = os.path.join(self.data_dir, 'UJIndoorLoc.zip')

        # 下载ZIP文件
        urlretrieve(self.download_url, zip_path)

        # 解压文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)

        # 重命名文件以匹配预期路径
        extracted_dir = os.path.join(self.data_dir, 'UJIndoorLoc')
        if os.path.exists(extracted_dir):
            train_src = os.path.join(extracted_dir, 'trainingData.csv')
            val_src = os.path.join(extracted_dir, 'validationData.csv')

            if os.path.exists(train_src):
                os.rename(train_src, self.training_file)
            if os.path.exists(val_src):
                os.rename(val_src, self.validation_file)

        print("Download and extraction complete.")

    def load_data(self):
        """
        加载UJIIndoorLoc数据集

        返回:
            dict: 包含训练和验证数据的字典
        """
        # 检查文件是否存在
        if not os.path.exists(self.training_file) or not os.path.exists(self.validation_file):
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            self._download_and_extract()

        # 加载数据
        try:
            training_data = pd.read_csv(self.training_file)
            validation_data = pd.read_csv(self.validation_file)

            print(f"Loaded training data: {training_data.shape[0]} samples")
            print(f"Loaded validation data: {validation_data.shape[0]} samples")

            return {
                'training_data': training_data,
                'validation_data': validation_data
            }
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def get_data_summary(self):
        """获取数据集的统计摘要"""
        data = self.load_data()

        training_data = data['training_data']
        validation_data = data['validation_data']

        summary = {
            'training_samples': training_data.shape[0],
            'validation_samples': validation_data.shape[0],
            'features': training_data.shape[1],
            'wap_features': 520,  # 固定的WAP特征数
            'buildings': training_data['BUILDINGID'].nunique(),
            'floors': training_data['FLOOR'].nunique(),
            'buildings_distribution': training_data['BUILDINGID'].value_counts().to_dict(),
            'floors_distribution': training_data['FLOOR'].value_counts().to_dict()
        }

        # 计算缺失值统计
        summary['training_missing_waps'] = (training_data.iloc[:, :520] == 100).sum().sum()
        summary['validation_missing_waps'] = (validation_data.iloc[:, :520] == 100).sum().sum()

        # 平均检测到的WAP
        summary['avg_detected_waps_training'] = 520 - ((training_data.iloc[:, :520] == 100).sum(axis=1).mean())
        summary['avg_detected_waps_validation'] = 520 - ((validation_data.iloc[:, :520] == 100).sum(axis=1).mean())

        return summary