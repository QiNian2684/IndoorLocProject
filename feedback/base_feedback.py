from abc import ABC, abstractmethod


class FeedbackMechanism(ABC):
    """反馈机制的抽象基类"""

    @abstractmethod
    def update(self, predicted, actual):
        """
        根据预测和实际值更新模型

        参数:
            predicted: 预测值
            actual: 实际值
        """
        pass

    @abstractmethod
    def adjust_model(self):
        """
        基于收集的反馈调整模型
        """
        pass


class PositioningObserver(ABC):
    """定位观察者的抽象基类（使用观察者模式）"""

    @abstractmethod
    def update(self, predicted_position, actual_position, signal_data):
        """
        接收定位结果通知

        参数:
            predicted_position: 预测的位置
            actual_position: 实际位置
            signal_data: 原始信号数据
        """
        pass