"""
评估和可视化模块
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import csv
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# 设置绘图风格
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.dpi'] = 100
sns.set_style("whitegrid")


class Evaluator:
    """评估器"""

    def __init__(self, config):
        self.config = config
        self.results = []

    def compute_error_distances(self, y_true, y_pred):
        """计算欧氏距离误差（米）"""
        distances = np.linalg.norm(y_true - y_pred, axis=1)
        return distances

    def calculate_metrics(self, y_true, y_pred):
        """计算各种评估指标"""
        # 基础指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # 误差距离
        error_distances = self.compute_error_distances(y_true, y_pred)
        mean_error = np.mean(error_distances)
        median_error = np.median(error_distances)
        std_error = np.std(error_distances)
        min_error = np.min(error_distances)
        max_error = np.max(error_distances)
        percentile_50 = np.percentile(error_distances, 50)
        percentile_75 = np.percentile(error_distances, 75)
        percentile_90 = np.percentile(error_distances, 90)
        percentile_95 = np.percentile(error_distances, 95)

        # 分维度误差
        lon_mae = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
        lat_mae = mean_absolute_error(y_true[:, 1], y_pred[:, 1])

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mean_error_distance': mean_error,
            'median_error_distance': median_error,
            'std_error_distance': std_error,
            'min_error_distance': min_error,
            'max_error_distance': max_error,
            'percentile_50': percentile_50,
            'percentile_75': percentile_75,
            'percentile_90': percentile_90,
            'percentile_95': percentile_95,
            'longitude_mae': lon_mae,
            'latitude_mae': lat_mae
        }

        return metrics, error_distances

    def plot_comprehensive_results(self, y_true, y_pred, train_losses, val_losses,
                                   training_params, svr_params, metrics,
                                   save_path=None, show=False):
        """绘制综合结果图"""
        fig = plt.figure(figsize=(16, 12), constrained_layout=True)
        gs = fig.add_gridspec(3, 3)

        # 1. 训练曲线
        ax1 = fig.add_subplot(gs[0, :2])
        if train_losses and val_losses:
            epochs = range(1, len(train_losses) + 1)
            ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
            ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title('Transformer Autoencoder Training Progress', fontsize=14)
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)

        # 2. 参数显示
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')

        params_text = "Training Parameters:\n" + "-" * 30 + "\n"
        if training_params:
            for key, value in training_params.items():
                if isinstance(value, float):
                    params_text += f"{key}: {value:.4f}\n"
                else:
                    params_text += f"{key}: {value}\n"

        params_text += "\nSVR Parameters:\n" + "-" * 30 + "\n"
        if svr_params:
            for key, value in svr_params.items():
                if isinstance(value, float):
                    params_text += f"{key}: {value:.4f}\n"
                else:
                    params_text += f"{key}: {value}\n"

        ax2.text(0.1, 0.9, params_text, transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top')

        # 3. 2D误差分布
        ax3 = fig.add_subplot(gs[1, 0])
        error_x = y_pred[:, 0] - y_true[:, 0]
        error_y = y_pred[:, 1] - y_true[:, 1]
        error_dist = np.sqrt(error_x ** 2 + error_y ** 2)

        scatter = ax3.scatter(error_x, error_y, c=error_dist,
                              cmap='viridis', alpha=0.6, s=10)
        plt.colorbar(scatter, ax=ax3, label='Error Distance (m)')
        ax3.set_xlabel('Longitude Error (m)', fontsize=11)
        ax3.set_ylabel('Latitude Error (m)', fontsize=11)
        ax3.set_title('2D Prediction Errors', fontsize=12)
        ax3.grid(True, alpha=0.3)

        # 4. 误差直方图
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(error_dist, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax4.axvline(metrics['mean_error_distance'], color='r', linestyle='--',
                    label=f'Mean: {metrics["mean_error_distance"]:.2f}m')
        ax4.axvline(metrics['median_error_distance'], color='g', linestyle='--',
                    label=f'Median: {metrics["median_error_distance"]:.2f}m')
        ax4.set_xlabel('Error Distance (m)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title('Error Distribution', fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        # 5. 累积误差分布
        ax5 = fig.add_subplot(gs[1, 2])
        sorted_errors = np.sort(error_dist)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100

        ax5.plot(sorted_errors, cumulative, 'b-', linewidth=2)
        ax5.axhline(50, color='gray', linestyle=':', alpha=0.5)
        ax5.axhline(75, color='gray', linestyle=':', alpha=0.5)
        ax5.axhline(90, color='gray', linestyle=':', alpha=0.5)
        ax5.axhline(95, color='gray', linestyle=':', alpha=0.5)

        ax5.set_xlabel('Error Distance (m)', fontsize=11)
        ax5.set_ylabel('Cumulative Percentage (%)', fontsize=11)
        ax5.set_title('Cumulative Error Distribution', fontsize=12)
        ax5.grid(True, alpha=0.3)

        # 6. 预测vs实际（经度）
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5, s=5)
        min_val = min(y_true[:, 0].min(), y_pred[:, 0].min())
        max_val = max(y_true[:, 0].max(), y_pred[:, 0].max())
        ax6.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax6.set_xlabel('Actual Longitude', fontsize=11)
        ax6.set_ylabel('Predicted Longitude', fontsize=11)
        ax6.set_title('Longitude Predictions', fontsize=12)
        ax6.grid(True, alpha=0.3)

        # 7. 预测vs实际（纬度）
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.5, s=5, color='orange')
        min_val = min(y_true[:, 1].min(), y_pred[:, 1].min())
        max_val = max(y_true[:, 1].max(), y_pred[:, 1].max())
        ax7.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax7.set_xlabel('Actual Latitude', fontsize=11)
        ax7.set_ylabel('Predicted Latitude', fontsize=11)
        ax7.set_title('Latitude Predictions', fontsize=12)
        ax7.grid(True, alpha=0.3)

        # 8. 评估指标
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')

        metrics_text = "Evaluation Metrics:\n" + "=" * 35 + "\n"
        metrics_text += f"MSE: {metrics['mse']:.6f}\n"
        metrics_text += f"RMSE: {metrics['rmse']:.6f}\n"
        metrics_text += f"MAE: {metrics['mae']:.6f}\n"
        metrics_text += f"R² Score: {metrics['r2']:.6f}\n"
        metrics_text += "-" * 35 + "\n"
        metrics_text += f"Mean Error: {metrics['mean_error_distance']:.2f} m\n"
        metrics_text += f"Median Error: {metrics['median_error_distance']:.2f} m\n"
        metrics_text += f"Std Error: {metrics['std_error_distance']:.2f} m\n"
        metrics_text += f"Min Error: {metrics['min_error_distance']:.2f} m\n"
        metrics_text += f"Max Error: {metrics['max_error_distance']:.2f} m\n"
        metrics_text += "-" * 35 + "\n"
        metrics_text += f"50th Percentile: {metrics['percentile_50']:.2f} m\n"
        metrics_text += f"75th Percentile: {metrics['percentile_75']:.2f} m\n"
        metrics_text += f"90th Percentile: {metrics['percentile_90']:.2f} m\n"
        metrics_text += f"95th Percentile: {metrics['percentile_95']:.2f} m\n"

        ax8.text(0.1, 0.9, metrics_text, transform=ax8.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('Indoor Localization Results - Transformer + SVR',
                     fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"结果图已保存: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def save_results_to_csv(self, metrics, params, filepath):
        """保存结果到CSV"""
        row_data = {**metrics, **params}

        # 确定字段名
        fieldnames = list(row_data.keys())

        # 检查文件是否存在
        file_exists = os.path.exists(filepath)

        # 写入CSV
        with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)

        logger.info(f"结果已保存到CSV: {filepath}")

    def save_predictions(self, y_true, y_pred, error_distances, filepath):
        """保存预测结果"""
        df = pd.DataFrame({
            'true_longitude': y_true[:, 0],
            'true_latitude': y_true[:, 1],
            'pred_longitude': y_pred[:, 0],
            'pred_latitude': y_pred[:, 1],
            'error_distance': error_distances
        })

        df.to_csv(filepath, index=False)
        logger.info(f"预测结果已保存: {filepath}")

    def identify_high_error_samples(self, test_data, y_pred, error_distances,
                                    threshold=15.0, filepath=None):
        """识别高误差样本"""
        high_error_indices = np.where(error_distances > threshold)[0]

        if len(high_error_indices) > 0:
            logger.info(f"发现 {len(high_error_indices)} 个误差超过 {threshold}m 的样本")

            if filepath and test_data is not None:
                high_error_data = test_data.iloc[high_error_indices].copy()
                high_error_data['predicted_longitude'] = y_pred[high_error_indices, 0]
                high_error_data['predicted_latitude'] = y_pred[high_error_indices, 1]
                high_error_data['error_distance'] = error_distances[high_error_indices]

                high_error_data.to_csv(filepath, index=False)
                logger.info(f"高误差样本已保存: {filepath}")

                return high_error_data
        else:
            logger.info(f"没有误差超过 {threshold}m 的样本")
            return None