import numpy as np
import pandas as pd
from .base_feedback import FeedbackMechanism, PositioningObserver


class LowerLevelFeedback(PositioningObserver):
    """实现低层反馈循环的观察者"""

    def __init__(self, model, learning_rate=0.01):
        """
        初始化低层反馈机制

        参数:
            model: 要调整的模型
            learning_rate: 学习率
        """
        self.model = model
        self.learning_rate = learning_rate
        self.last_error = None

    def update(self, predicted_position, actual_position, signal_data):
        """
        接收位置更新并进行即时反馈

        参数:
            predicted_position: 预测的位置
            actual_position: 实际位置
            signal_data: 原始信号数据
        """
        # 计算误差
        error = self._compute_error(predicted_position, actual_position)

        # 更新模型权重
        self._update_model(error, signal_data)

        # 记录当前误差
        self.last_error = error

    def _compute_error(self, predicted, actual):
        """计算定位误差"""
        if predicted is None or actual is None:
            return None

        # 欧氏距离误差
        return np.sqrt(np.sum((predicted - actual) ** 2, axis=1))

    def _update_model(self, error, signal_data):
        """更新模型参数"""
        # 具体实现取决于模型类型
        if hasattr(self.model, 'update_weights'):
            self.model.update_weights(error, self.learning_rate)
        elif hasattr(self.model, 'weights') and isinstance(self.model.weights, dict):
            # 对于集成模型，调整权重
            if error is not None and len(error) > 0:
                mean_error = np.mean(error)
                if self.last_error is not None:
                    # 如果误差减小，增强当前权重分配
                    if mean_error < np.mean(self.last_error):
                        self.model.weights = {
                            k: w * (1 + self.learning_rate * 0.1) for k, w in self.model.weights.items()
                        }
                        # 重新归一化权重
                        weight_sum = sum(self.model.weights.values())
                        self.model.weights = {
                            k: w / weight_sum for k, w in self.model.weights.items()
                        }


class HigherLevelFeedback(PositioningObserver):
    """实现高层反馈循环的观察者"""

    def __init__(self, model, feedback_window=100, meta_learning_rate=0.001):
        """
        初始化高层反馈机制

        参数:
            model: 要调整的模型
            feedback_window: 收集多少样本后进行元适应
            meta_learning_rate: 元学习率
        """
        self.model = model
        self.feedback_window = feedback_window
        self.meta_learning_rate = meta_learning_rate
        self.error_history = []

    def update(self, predicted_position, actual_position, signal_data):
        """
        接收位置更新并积累反馈

        参数:
            predicted_position: 预测的位置
            actual_position: 实际位置
            signal_data: 原始信号数据
        """
        # 计算误差
        error = self._compute_error(predicted_position, actual_position)

        # 添加到历史记录
        if error is not None and len(error) > 0:
            self.error_history.append({
                'error': np.mean(error),
                'timestamp': pd.Timestamp.now()
            })

        # 如果收集了足够的样本，执行元适应
        if len(self.error_history) >= self.feedback_window:
            self._perform_meta_adaptation()

    def _compute_error(self, predicted, actual):
        """计算定位误差"""
        if predicted is None or actual is None:
            return None

        # 欧氏距离误差
        return np.sqrt(np.sum((predicted - actual) ** 2, axis=1))

    def _perform_meta_adaptation(self):
        """执行元适应以调整更高级别的参数"""
        if not self.error_history:
            return

        # 创建误差历史的DataFrame
        error_df = pd.DataFrame(self.error_history)

        # 计算误差趋势
        error_df['error_diff'] = error_df['error'].diff()
        mean_diff = error_df['error_diff'].mean()

        # 检查性能趋势
        if mean_diff < 0:  # 性能在改善
            # 可能增加较低层次的学习率
            if hasattr(self.model, 'learning_rate'):
                self.model.learning_rate *= (1 + self.meta_learning_rate)

            # 或者调整其他参数，例如集成权重衰减
            if hasattr(self.model, 'weight_decay'):
                self.model.weight_decay *= (1 - self.meta_learning_rate)
        else:  # 性能在恶化或停滞
            # 减少学习率以稳定
            if hasattr(self.model, 'learning_rate'):
                self.model.learning_rate *= (1 - self.meta_learning_rate)

            # 或者尝试其他策略，例如增加正则化
            if hasattr(self.model, 'weight_decay'):
                self.model.weight_decay *= (1 + self.meta_learning_rate)

        # 重置历史，保留最近一半的数据
        self.error_history = self.error_history[-int(self.feedback_window / 2):]


class DualLayerFeedbackSVRTransformer:
    """双层自适应反馈SVR+Transformer模型"""

    def __init__(self, base_model, learning_rate=0.01, meta_learning_rate=0.001,
                 feedback_window=100):
        """
        初始化双层反馈机制

        参数:
            base_model: 基础SVR+Transformer混合模型
            learning_rate: 低层反馈学习率
            meta_learning_rate: 高层反馈学习率
            feedback_window: 高层反馈窗口大小
        """
        self.base_model = base_model
        self.learning_rate = learning_rate
        self.meta_learning_rate = meta_learning_rate
        self.feedback_window = feedback_window
        self.error_history = []
        self.confidence_weight = 0.5  # 初始相等权重

        # SVR和Transformer组件
        if hasattr(base_model, 'svr_models') and hasattr(base_model, 'transformer_model'):
            self.svr_model = base_model.svr_models
            self.transformer_model = base_model.transformer_model
        elif hasattr(base_model, 'integration_type'):
            if base_model.integration_type == 'ensemble':
                self.svr_model = base_model.svr_models
                self.transformer_model = base_model.transformer_model
            elif base_model.integration_type == 'feature_extraction':
                self.svr_model = base_model.svr_models
                self.transformer_model = base_model.feature_extractor
            else:
                self.svr_model = None
                self.transformer_model = None

    def predict_with_feedback(self, x, y_true=None):
        """
        使用反馈机制进行预测

        参数:
            x: 输入特征
            y_true: 可选的真实标签，用于反馈

        返回:
            预测结果
        """
        # 基本预测
        prediction = self.base_model.predict(x)

        # 如果提供了真实值，使用反馈机制
        if y_true is not None:
            self._apply_feedback(x, prediction, y_true)

        return prediction

    def _apply_feedback(self, x, prediction, y_true):
        """应用双层反馈"""
        # 计算误差
        errors = np.sqrt(np.sum((prediction - y_true) ** 2, axis=1))
        mean_error = np.mean(errors)

        # 更新误差历史
        self.error_history.append({
            'error': mean_error,
            'timestamp': pd.Timestamp.now(),
            'confidence_weight': self.confidence_weight
        })

        # 低层反馈 - 调整集成权重
        if hasattr(self.base_model, 'weights'):
            if self.base_model.integration_type == 'ensemble':
                # 尝试独立获取SVR和Transformer的预测
                svr_pred = np.zeros_like(prediction)
                for i, model in enumerate(self.svr_model):
                    svr_pred[:, i] = model.predict(x)

                import torch
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(x)
                    tf_pred = self.transformer_model(x_tensor).cpu().numpy()

                # 计算单独的错误
                svr_error = np.mean(np.sqrt(np.sum((svr_pred - y_true) ** 2, axis=1)))
                tf_error = np.mean(np.sqrt(np.sum((tf_pred - y_true) ** 2, axis=1)))

                # 更新权重
                total_error = svr_error + tf_error
                if total_error > 0:
                    self.base_model.weights['svr'] = tf_error / total_error
                    self.base_model.weights['transformer'] = svr_error / total_error

                    # 更新内部置信度权重
                    self.confidence_weight = self.base_model.weights['svr']

        # 高层反馈 - 元适应
        if len(self.error_history) >= self.feedback_window:
            self._meta_adaptation()

    def _meta_adaptation(self):
        """执行元适应"""
        # 分析误差历史
        recent_errors = pd.DataFrame(self.error_history[-self.feedback_window:])

        # 提取趋势
        error_trend = recent_errors['error'].diff().mean()
        weight_trend = recent_errors['confidence_weight'].diff().mean()

        # 根据性能趋势调整学习率
        if error_trend < 0:  # 性能在改善
            # 如果权重变化有帮助，稍微增加学习率
            self.learning_rate = min(0.5, self.learning_rate * (1 + self.meta_learning_rate))
        else:  # 性能在恶化
            # 减少学习率以稳定
            self.learning_rate = max(0.001, self.learning_rate * (1 - self.meta_learning_rate))

        # 重置误差历史以避免使用旧数据
        self.error_history = self.error_history[-int(self.feedback_window / 2):]