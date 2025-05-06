import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.cluster import KMeans, DBSCAN


class SpatialCrossValidator:
    """空间交叉验证器"""

    def __init__(self, n_splits=5, cluster_method='kmeans', random_state=None):
        """
        初始化空间交叉验证器

        参数:
            n_splits (int): 分割数
            cluster_method (str): 聚类方法 ('kmeans' 或 'dbscan')
            random_state: 随机状态
        """
        self.n_splits = n_splits
        self.cluster_method = cluster_method
        self.random_state = random_state

    def split(self, X, y=None):
        """
        生成空间交叉验证分割

        参数:
            X: 特征数据
            y: 目标值，必须包含坐标信息

        生成:
            train_indices, test_indices 的元组
        """
        if y is None:
            raise ValueError("必须提供目标值（坐标）以进行空间交叉验证")

        # 转换为数组
        X_array = np.asarray(X)
        y_array = np.asarray(y)

        # 提取坐标
        if len(y_array.shape) == 1:
            # 如果y是一维的，假设X包含坐标（例如原始RSSI值）
            if hasattr(X, 'iloc'):
                # 对于DataFrame，尝试查找坐标列
                if 'LONGITUDE' in X.columns and 'LATITUDE' in X.columns:
                    coordinates = X[['LONGITUDE', 'LATITUDE']].values
                else:
                    raise ValueError("无法识别坐标列")
            else:
                raise ValueError("无法提取坐标进行空间分割")
        else:
            # 使用多维y中的前两列作为坐标
            coordinates = y_array[:, :2]

        # 创建空间聚类
        if self.cluster_method == 'kmeans':
            cluster_model = KMeans(
                n_clusters=self.n_splits,
                random_state=self.random_state
            )
        else:  # dbscan
            cluster_model = DBSCAN(eps=10.0, min_samples=5)

        cluster_labels = cluster_model.fit_predict(coordinates)

        # 生成分割
        unique_clusters = np.unique(cluster_labels)

        if len(unique_clusters) < self.n_splits or self.cluster_method == 'dbscan':
            # 如果聚类效果不好，回退到随机分割
            print(f"警告: 检测到{len(unique_clusters)}个簇，少于请求的{self.n_splits}个。回退到随机分割。")
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            for train_idx, test_idx in kf.split(X_array):
                yield train_idx, test_idx
        else:
            # 使用聚类进行分割
            for test_cluster in unique_clusters:
                test_mask = cluster_labels == test_cluster
                test_idx = np.where(test_mask)[0]
                train_idx = np.where(~test_mask)[0]
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """获取分割数"""
        return self.n_splits


class HierarchicalCrossValidator:
    """分层交叉验证器（按建筑和楼层）"""

    def __init__(self, n_splits=5, random_state=None):
        """
        初始化分层交叉验证器

        参数:
            n_splits (int): 分割数
            random_state: 随机状态
        """
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """
        生成分层交叉验证分割

        参数:
            X: 特征数据
            y: 目标值
            groups: 分组信息 [building_id, floor_id]

        生成:
            train_indices, test_indices 的元组
        """
        if groups is None:
            if hasattr(X, 'iloc'):
                # 对于DataFrame，尝试查找建筑和楼层列
                if 'BUILDINGID' in X.columns and 'FLOOR' in X.columns:
                    groups = np.column_stack((
                        X['BUILDINGID'].values,
                        X['FLOOR'].values
                    ))
                else:
                    raise ValueError("无法识别建筑和楼层列")
            else:
                raise ValueError("必须提供分组信息（建筑和楼层）")

        # 创建复合标签进行分层
        combined_groups = groups[:, 0] * 10 + groups[:, 1]

        # 使用StratifiedKFold确保每个分割都有相似的建筑和楼层分布
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, test_idx in skf.split(X, combined_groups):
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """获取分割数"""
        return self.n_splits