from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

try:
    import xgboost

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
from .base import PositioningModel


class FloorClassifier(PositioningModel):
    """楼层分类模型"""

    def __init__(self, classifier_type='random_forest', **params):
        """
        初始化楼层分类器

        参数:
            classifier_type (str): 分类器类型 ('random_forest', 'svm', 'xgboost')
            **params: 分类器参数
        """
        self.classifier_type = classifier_type
        self.params = params
        self.model = None
        self.classes_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        """
        训练楼层分类器

        参数:
            X: 特征数据
            y: 楼层标签

        返回:
            self: 模型实例
        """
        if self.classifier_type == 'random_forest':
            self.model = RandomForestClassifier(**self.params)
        elif self.classifier_type == 'svm':
            self.model = SVC(**self.params)
        elif self.classifier_type == 'xgboost':
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not installed. Please install xgboost package.")
            from xgboost import XGBClassifier
            self.model = XGBClassifier(**self.params)
        else:
            raise ValueError(f"不支持的分类器类型: {self.classifier_type}")

        print(f"训练楼层分类器 ({self.classifier_type})...")
        self.model.fit(X, y)

        # 保存类别和特征重要性（如果可用）
        self.classes_ = self.model.classes_
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_

        return self

    def predict(self, X):
        """
        预测楼层

        参数:
            X: 特征数据

        返回:
            预测的楼层标签
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练")

        return self.model.predict(X)

    def predict_proba(self, X):
        """
        预测楼层概率

        参数:
            X: 特征数据

        返回:
            楼层概率
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练")

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise RuntimeError(f"{self.classifier_type}分类器不支持概率预测")

    def get_feature_importances(self):
        """
        获取特征重要性（如果可用）

        返回:
            ndarray: 特征重要性
        """
        if self.feature_importances_ is None:
            raise RuntimeError("此分类器不提供特征重要性")

        return self.feature_importances_

    def get_confusion_matrix(self, X, y_true):
        """
        计算混淆矩阵

        参数:
            X: 特征数据
            y_true: 真实楼层标签

        返回:
            ndarray: 混淆矩阵
        """
        from sklearn.metrics import confusion_matrix

        y_pred = self.predict(X)
        return confusion_matrix(y_true, y_pred)

    def evaluate(self, X, y_true):
        """
        评估分类器性能

        参数:
            X: 特征数据
            y_true: 真实楼层标签

        返回:
            dict: 评估指标
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        y_pred = self.predict(X)

        try:
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted')
            }
        except:
            # 如果某些指标计算失败（例如某些类没有样本），只返回准确率
            return {
                'accuracy': accuracy_score(y_true, y_pred)
            }

    def save_model(self, path):
        """
        保存模型

        参数:
            path: 保存路径
        """
        import joblib

        if self.model is None:
            raise RuntimeError("模型尚未训练")

        model_data = {
            'classifier_type': self.classifier_type,
            'params': self.params,
            'model': self.model,
            'classes_': self.classes_,
            'feature_importances_': self.feature_importances_
        }

        joblib.dump(model_data, path)

    def load_model(self, path):
        """
        加载模型

        参数:
            path: 模型路径

        返回:
            self: 模型实例
        """
        import joblib

        model_data = joblib.load(path)

        self.classifier_type = model_data['classifier_type']
        self.params = model_data['params']
        self.model = model_data['model']
        self.classes_ = model_data['classes_']
        self.feature_importances_ = model_data['feature_importances_']

        return self