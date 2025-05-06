import numpy as np
import pandas as pd
import os


def euclidean_distance(y_true, y_pred):
    """
    计算欧氏距离误差

    参数:
        y_true: 真实坐标 (n_samples, 2)
        y_pred: 预测坐标 (n_samples, 2)

    返回:
        ndarray: 每个样本的欧氏距离误差
    """
    return np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))


def mean_euclidean_error(y_true, y_pred):
    """
    计算平均欧氏距离误差

    参数:
        y_true: 真实坐标 (n_samples, 2)
        y_pred: 预测坐标 (n_samples, 2)

    返回:
        float: 平均欧氏距离误差
    """
    return np.mean(euclidean_distance(y_true, y_pred))


def median_euclidean_error(y_true, y_pred):
    """
    计算中位数欧氏距离误差

    参数:
        y_true: 真实坐标 (n_samples, 2)
        y_pred: 预测坐标 (n_samples, 2)

    返回:
        float: 中位数欧氏距离误差
    """
    return np.median(euclidean_distance(y_true, y_pred))


def percentile_error(y_true, y_pred, percentile=75):
    """
    计算特定百分位的误差

    参数:
        y_true: 真实坐标 (n_samples, 2)
        y_pred: 预测坐标 (n_samples, 2)
        percentile: 要计算的百分位

    返回:
        float: 指定百分位的误差
    """
    return np.percentile(euclidean_distance(y_true, y_pred), percentile)


def root_mean_squared_error(y_true, y_pred):
    """
    计算均方根误差

    参数:
        y_true: 真实坐标 (n_samples, 2)
        y_pred: 预测坐标 (n_samples, 2)

    返回:
        float: 均方根误差
    """
    return np.sqrt(np.mean(np.sum((y_true - y_pred) ** 2, axis=1)))


def floor_accuracy(floor_true, floor_pred):
    """
    计算楼层识别准确率

    参数:
        floor_true: 真实楼层标签
        floor_pred: 预测楼层标签

    返回:
        float: 楼层识别准确率
    """
    return np.mean(floor_true == floor_pred)


def building_accuracy(building_true, building_pred):
    """
    计算建筑识别准确率

    参数:
        building_true: 真实建筑标签
        building_pred: 预测建筑标签

    返回:
        float: 建筑识别准确率
    """
    return np.mean(building_true == building_pred)


def combined_positioning_error(position_true, position_pred, floor_true, floor_pred, floor_penalty=4):
    """
    计算综合定位误差（包括楼层错误的惩罚）

    参数:
        position_true: 真实坐标 (n_samples, 2)
        position_pred: 预测坐标 (n_samples, 2)
        floor_true: 真实楼层标签
        floor_pred: 预测楼层标签
        floor_penalty: 楼层错误的惩罚因子（米）

    返回:
        ndarray: 综合定位误差
    """
    position_errors = euclidean_distance(position_true, position_pred)
    floor_errors = floor_penalty * (floor_true != floor_pred)
    return position_errors + floor_errors


class PositioningEvaluator:
    """定位系统评估器"""

    def __init__(self):
        pass

    def evaluate(self, y_true, y_pred, floor_true=None, floor_pred=None, building_true=None, building_pred=None):
        """
        评估定位系统性能

        参数:
            y_true: 真实坐标 (n_samples, 2)
            y_pred: 预测坐标 (n_samples, 2)
            floor_true: 真实楼层标签（可选）
            floor_pred: 预测楼层标签（可选）
            building_true: 真实建筑标签（可选）
            building_pred: 预测建筑标签（可选）

        返回:
            dict: 评估指标的字典
        """
        results = {}

        # 计算2D定位误差
        position_errors = euclidean_distance(y_true, y_pred)

        # 基本指标
        results['mean_error'] = np.mean(position_errors)
        results['median_error'] = np.median(position_errors)
        results['rmse'] = root_mean_squared_error(y_true, y_pred)

        # 百分位误差
        results['75th_percentile'] = np.percentile(position_errors, 75)
        results['90th_percentile'] = np.percentile(position_errors, 90)
        results['95th_percentile'] = np.percentile(position_errors, 95)

        # 坐标误差分析
        results['x_mean_error'] = np.mean(np.abs(y_true[:, 0] - y_pred[:, 0]))
        results['y_mean_error'] = np.mean(np.abs(y_true[:, 1] - y_pred[:, 1]))
        results['x_median_error'] = np.median(np.abs(y_true[:, 0] - y_pred[:, 0]))
        results['y_median_error'] = np.median(np.abs(y_true[:, 1] - y_pred[:, 1]))

        # 详细误差分布
        results['position_errors'] = position_errors
        results['x_errors'] = np.abs(y_true[:, 0] - y_pred[:, 0])
        results['y_errors'] = np.abs(y_true[:, 1] - y_pred[:, 1])

        # 楼层检测准确率（如果可用）
        if floor_true is not None and floor_pred is not None:
            results['floor_accuracy'] = floor_accuracy(floor_true, floor_pred)

            # 标准UJIIndoorLoc指标，带楼层惩罚
            floor_penalty = 4  # 标准值
            combined_errors = combined_positioning_error(
                y_true, y_pred, floor_true, floor_pred, floor_penalty)
            results['combined_error'] = np.mean(combined_errors)
            results['combined_median_error'] = np.median(combined_errors)
            results['combined_75th_percentile'] = np.percentile(combined_errors, 75)
            results['combined_90th_percentile'] = np.percentile(combined_errors, 90)

            # 按楼层分析误差
            unique_floors = np.unique(floor_true)
            floor_errors = {}
            for floor in unique_floors:
                floor_mask = floor_true == floor
                if np.sum(floor_mask) > 0:
                    floor_errors[int(floor)] = {
                        'mean_error': np.mean(position_errors[floor_mask]),
                        'median_error': np.median(position_errors[floor_mask]),
                        'count': int(np.sum(floor_mask)),
                        'accuracy': float(np.mean(floor_true[floor_mask] == floor_pred[floor_mask]))
                    }
            results['floor_errors'] = floor_errors

        # 建筑检测准确率（如果可用）
        if building_true is not None and building_pred is not None:
            results['building_accuracy'] = building_accuracy(building_true, building_pred)

            # 按建筑分析误差
            unique_buildings = np.unique(building_true)
            building_errors = {}
            for building in unique_buildings:
                building_mask = building_true == building
                if np.sum(building_mask) > 0:
                    building_errors[int(building)] = {
                        'mean_error': np.mean(position_errors[building_mask]),
                        'median_error': np.median(position_errors[building_mask]),
                        'count': int(np.sum(building_mask)),
                        'accuracy': float(np.mean(building_true[building_mask] == building_pred[building_mask]))
                    }
            results['building_errors'] = building_errors

        # 误差CDF数据（用于绘图）
        max_error = max(position_errors)
        error_range = np.linspace(0, max_error, 100)
        cdf_values = [np.mean(position_errors <= e) for e in error_range]
        results['cdf_data'] = {'errors': error_range, 'probabilities': cdf_values}

        return results

    def evaluate_and_save(self, y_true, y_pred, floor_true=None, floor_pred=None,
                          building_true=None, building_pred=None, save_dir=None):
        """
        评估定位系统性能并保存结果

        参数:
            y_true: 真实坐标 (n_samples, 2)
            y_pred: 预测坐标 (n_samples, 2)
            floor_true: 真实楼层标签（可选）
            floor_pred: 预测楼层标签（可选）
            building_true: 真实建筑标签（可选）
            building_pred: 预测建筑标签（可选）
            save_dir: 保存目录

        返回:
            dict: 评估指标的字典
        """
        # 计算评估指标
        metrics = self.evaluate(y_true, y_pred, floor_true, floor_pred,
                                building_true, building_pred)

        if save_dir is not None:
            # 确保目录存在
            os.makedirs(os.path.join(save_dir, 'metrics'), exist_ok=True)
            os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)

            # 保存指标概览
            metrics_df = pd.DataFrame([{
                'metric': k,
                'value': v
            } for k, v in metrics.items()
                if not isinstance(v, (dict, list, np.ndarray)) and k != 'cdf_data'])

            metrics_df.to_csv(os.path.join(save_dir, 'metrics', 'metrics_overview.csv'), index=False)

            # 保存详细预测结果
            predictions_df = pd.DataFrame({
                'true_x': y_true[:, 0],
                'true_y': y_true[:, 1],
                'pred_x': y_pred[:, 0],
                'pred_y': y_pred[:, 1],
                'x_error': np.abs(y_true[:, 0] - y_pred[:, 0]),
                'y_error': np.abs(y_true[:, 1] - y_pred[:, 1]),
                'euclidean_error': np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
            })

            # 如果有楼层信息，添加到DataFrame
            if floor_true is not None:
                predictions_df['true_floor'] = floor_true
            if floor_pred is not None:
                predictions_df['pred_floor'] = floor_pred
                if floor_true is not None:
                    predictions_df['floor_correct'] = floor_true == floor_pred

            # 如果有建筑信息，添加到DataFrame
            if building_true is not None:
                predictions_df['true_building'] = building_true
            if building_pred is not None:
                predictions_df['pred_building'] = building_pred
                if building_true is not None:
                    predictions_df['building_correct'] = building_true == building_pred

            # 保存预测结果
            predictions_df.to_csv(os.path.join(save_dir, 'predictions', 'detailed_predictions.csv'), index=False)

            # 保存误差分布
            error_df = pd.DataFrame({
                'euclidean_error': metrics['position_errors'],
                'x_error': metrics['x_errors'],
                'y_error': metrics['y_errors']
            })
            error_df.to_csv(os.path.join(save_dir, 'metrics', 'error_distribution.csv'), index=False)

            # 保存CDF数据
            cdf_df = pd.DataFrame({
                'error': metrics['cdf_data']['errors'],
                'probability': metrics['cdf_data']['probabilities']
            })
            cdf_df.to_csv(os.path.join(save_dir, 'metrics', 'error_cdf.csv'), index=False)

            # 保存楼层分析（如果可用）
            if 'floor_errors' in metrics:
                floor_errors_df = pd.DataFrame([
                    {
                        'floor': floor,
                        'mean_error': data['mean_error'],
                        'median_error': data['median_error'],
                        'count': data['count'],
                        'accuracy': data['accuracy']
                    }
                    for floor, data in metrics['floor_errors'].items()
                ])
                floor_errors_df.to_csv(os.path.join(save_dir, 'metrics', 'floor_analysis.csv'), index=False)

            # 保存建筑分析（如果可用）
            if 'building_errors' in metrics:
                building_errors_df = pd.DataFrame([
                    {
                        'building': building,
                        'mean_error': data['mean_error'],
                        'median_error': data['median_error'],
                        'count': data['count'],
                        'accuracy': data['accuracy']
                    }
                    for building, data in metrics['building_errors'].items()
                ])
                building_errors_df.to_csv(os.path.join(save_dir, 'metrics', 'building_analysis.csv'), index=False)

        return metrics