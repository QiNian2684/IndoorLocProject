"""
评估和可视化模块 - 增强版，包含详细记录和单独图表
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
import os
import csv
import json
from typing import Dict, List, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# 设置高质量绘图
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.dpi'] = 300  # 高DPI
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
sns.set_style("whitegrid")


class Evaluator:
    """增强版评估器 - 包含详细记录功能"""

    def __init__(self, config):
        self.config = config
        self.results = []
        self.all_metrics = []

        # 创建详细记录目录
        self.detailed_plot_dir = os.path.join(config.plot_dir, 'detailed')
        self.detailed_csv_dir = os.path.join(config.csv_dir, 'detailed')
        os.makedirs(self.detailed_plot_dir, exist_ok=True)
        os.makedirs(self.detailed_csv_dir, exist_ok=True)

    def compute_error_distances(self, y_true, y_pred):
        """计算欧氏距离误差（米）"""
        distances = np.linalg.norm(y_true - y_pred, axis=1)
        return distances

    def calculate_metrics(self, y_true, y_pred, save_detailed=True):
        """计算各种评估指标并保存详细数据"""
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

        # 更多百分位数
        percentiles = {}
        for p in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]:
            percentiles[f'percentile_{p}'] = np.percentile(error_distances, p)

        # 分维度误差
        lon_mae = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
        lat_mae = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
        lon_mse = mean_squared_error(y_true[:, 0], y_pred[:, 0])
        lat_mse = mean_squared_error(y_true[:, 1], y_pred[:, 1])
        lon_rmse = np.sqrt(lon_mse)
        lat_rmse = np.sqrt(lat_mse)
        lon_r2 = r2_score(y_true[:, 0], y_pred[:, 0])
        lat_r2 = r2_score(y_true[:, 1], y_pred[:, 1])

        # 误差分布统计
        error_bins = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100]
        error_distribution = {}
        for i in range(len(error_bins) - 1):
            count = np.sum((error_distances >= error_bins[i]) &
                          (error_distances < error_bins[i+1]))
            error_distribution[f'errors_{error_bins[i]}_{error_bins[i+1]}m'] = count
            error_distribution[f'errors_{error_bins[i]}_{error_bins[i+1]}m_pct'] = (count / len(error_distances)) * 100

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
            'longitude_mae': lon_mae,
            'latitude_mae': lat_mae,
            'longitude_mse': lon_mse,
            'latitude_mse': lat_mse,
            'longitude_rmse': lon_rmse,
            'latitude_rmse': lat_rmse,
            'longitude_r2': lon_r2,
            'latitude_r2': lat_r2,
            'total_samples': len(error_distances),
            **percentiles,
            **error_distribution
        }

        if save_detailed:
            # 保存详细指标
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metrics_file = os.path.join(self.detailed_csv_dir, f'metrics_{timestamp}.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)

            # 保存误差距离数据
            error_df = pd.DataFrame({
                'error_distance': error_distances,
                'longitude_error': y_pred[:, 0] - y_true[:, 0],
                'latitude_error': y_pred[:, 1] - y_true[:, 1],
                'true_lon': y_true[:, 0],
                'true_lat': y_true[:, 1],
                'pred_lon': y_pred[:, 0],
                'pred_lat': y_pred[:, 1]
            })
            error_csv = os.path.join(self.detailed_csv_dir, f'error_details_{timestamp}.csv')
            error_df.to_csv(error_csv, index=False)

        return metrics, error_distances

    def plot_training_curves(self, train_losses, val_losses, save_path=None):
        """单独绘制训练曲线"""
        if not train_losses or not val_losses:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(train_losses) + 1)

        # 损失曲线
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=3)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=3)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # 损失差异
        loss_diff = [val - train for train, val in zip(train_losses, val_losses)]
        ax2.plot(epochs, loss_diff, 'g-', linewidth=2, marker='^', markersize=3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Validation Loss - Training Loss', fontsize=12)
        ax2.set_title('Overfitting Monitor', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Transformer Autoencoder Training Progress', fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"训练曲线图已保存: {save_path}")

            # 保存数据
            train_data = pd.DataFrame({
                'epoch': epochs,
                'train_loss': train_losses,
                'val_loss': val_losses,
                'loss_diff': loss_diff
            })
            csv_path = save_path.replace('.png', '_data.csv')
            train_data.to_csv(csv_path, index=False)

        plt.close()
        return fig

    def plot_error_distribution(self, error_distances, metrics, save_path=None):
        """单独绘制误差分布图"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # 1. 直方图
        ax = axes[0, 0]
        n, bins, patches = ax.hist(error_distances, bins=50, edgecolor='black',
                                   alpha=0.7, color='skyblue', density=True)

        # 添加核密度估计
        from scipy import stats
        density = stats.gaussian_kde(error_distances)
        xs = np.linspace(error_distances.min(), error_distances.max(), 200)
        ax.plot(xs, density(xs), 'r-', linewidth=2, label='KDE')

        ax.axvline(metrics['mean_error_distance'], color='g', linestyle='--',
                  linewidth=2, label=f'Mean: {metrics["mean_error_distance"]:.2f}m')
        ax.axvline(metrics['median_error_distance'], color='orange', linestyle='--',
                  linewidth=2, label=f'Median: {metrics["median_error_distance"]:.2f}m')
        ax.set_xlabel('Error Distance (m)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Error Distribution with KDE', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 2. 箱线图
        ax = axes[0, 1]
        bp = ax.boxplot(error_distances, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax.set_ylabel('Error Distance (m)', fontsize=11)
        ax.set_title('Error Distance Box Plot', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 添加统计信息
        stats_text = f'Mean: {metrics["mean_error_distance"]:.2f}m\n'
        stats_text += f'Median: {metrics["median_error_distance"]:.2f}m\n'
        stats_text += f'Std: {metrics["std_error_distance"]:.2f}m\n'
        stats_text += f'Min: {metrics["min_error_distance"]:.2f}m\n'
        stats_text += f'Max: {metrics["max_error_distance"]:.2f}m'
        ax.text(1.15, metrics['mean_error_distance'], stats_text,
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat'))

        # 3. 累积分布
        ax = axes[0, 2]
        sorted_errors = np.sort(error_distances)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100

        ax.plot(sorted_errors, cumulative, 'b-', linewidth=2)

        # 添加关键百分位线
        percentiles_to_mark = [50, 75, 90, 95, 99]
        colors = ['green', 'orange', 'red', 'darkred', 'purple']
        for p, c in zip(percentiles_to_mark, colors):
            val = np.percentile(error_distances, p)
            ax.axhline(p, color=c, linestyle=':', alpha=0.5)
            ax.axvline(val, color=c, linestyle=':', alpha=0.5)
            ax.text(val, p, f'{p}%: {val:.1f}m', fontsize=8, color=c)

        ax.set_xlabel('Error Distance (m)', fontsize=11)
        ax.set_ylabel('Cumulative Percentage (%)', fontsize=11)
        ax.set_title('Cumulative Error Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 4. Q-Q图
        ax = axes[1, 0]
        stats.probplot(error_distances, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 5. 误差范围饼图
        ax = axes[1, 1]
        error_ranges = ['0-2m', '2-5m', '5-10m', '10-15m', '15-20m', '>20m']
        range_counts = [
            np.sum(error_distances <= 2),
            np.sum((error_distances > 2) & (error_distances <= 5)),
            np.sum((error_distances > 5) & (error_distances <= 10)),
            np.sum((error_distances > 10) & (error_distances <= 15)),
            np.sum((error_distances > 15) & (error_distances <= 20)),
            np.sum(error_distances > 20)
        ]

        colors_pie = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(error_ranges)))
        wedges, texts, autotexts = ax.pie(range_counts, labels=error_ranges,
                                           colors=colors_pie, autopct='%1.1f%%',
                                           startangle=90)
        ax.set_title('Error Range Distribution', fontsize=12, fontweight='bold')

        # 6. 误差时间序列（按样本索引）
        ax = axes[1, 2]
        ax.plot(error_distances, 'b-', linewidth=0.5, alpha=0.7)
        ax.axhline(metrics['mean_error_distance'], color='r', linestyle='--',
                  alpha=0.7, label='Mean')
        ax.fill_between(range(len(error_distances)),
                        metrics['mean_error_distance'] - metrics['std_error_distance'],
                        metrics['mean_error_distance'] + metrics['std_error_distance'],
                        alpha=0.3, color='gray', label='±1 Std')
        ax.set_xlabel('Sample Index', fontsize=11)
        ax.set_ylabel('Error Distance (m)', fontsize=11)
        ax.set_title('Error by Sample Order', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.suptitle('Comprehensive Error Distribution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"误差分布图已保存: {save_path}")

            # 保存统计数据
            stats_df = pd.DataFrame({
                'statistic': ['mean', 'median', 'std', 'min', 'max'] +
                            [f'percentile_{p}' for p in [10, 25, 50, 75, 90, 95, 99]],
                'value': [
                    metrics['mean_error_distance'],
                    metrics['median_error_distance'],
                    metrics['std_error_distance'],
                    metrics['min_error_distance'],
                    metrics['max_error_distance']
                ] + [metrics[f'percentile_{p}'] for p in [10, 25, 50, 75, 90, 95, 99]]
            })
            csv_path = save_path.replace('.png', '_stats.csv')
            stats_df.to_csv(csv_path, index=False)

        plt.close()
        return fig

    def plot_2d_errors(self, y_true, y_pred, save_path=None):
        """单独绘制2D误差分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        error_x = y_pred[:, 0] - y_true[:, 0]
        error_y = y_pred[:, 1] - y_true[:, 1]
        error_dist = np.sqrt(error_x**2 + error_y**2)

        # 1. 2D误差散点图
        ax = axes[0, 0]
        scatter = ax.scatter(error_x, error_y, c=error_dist,
                           cmap='viridis', alpha=0.6, s=10)
        plt.colorbar(scatter, ax=ax, label='Error Distance (m)')

        # 添加误差椭圆
        from matplotlib.patches import Ellipse
        cov = np.cov(error_x, error_y)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        for n_std in [1, 2, 3]:
            width, height = 2 * n_std * np.sqrt(eigenvalues)
            ellipse = Ellipse(xy=(np.mean(error_x), np.mean(error_y)),
                             width=width, height=height, angle=angle,
                             facecolor='none', edgecolor='red',
                             alpha=0.5, linewidth=2,
                             label=f'{n_std}σ' if n_std == 1 else f'{n_std}σ')
            ax.add_patch(ellipse)

        ax.set_xlabel('Longitude Error (m)', fontsize=11)
        ax.set_ylabel('Latitude Error (m)', fontsize=11)
        ax.set_title('2D Error Distribution with Confidence Ellipses', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # 2. 误差热力图
        ax = axes[0, 1]
        from scipy.stats import gaussian_kde

        # 创建网格
        x_min, x_max = error_x.min() - 1, error_x.max() + 1
        y_min, y_max = error_y.min() - 1, error_y.max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))

        # 计算密度
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([error_x, error_y])
        kernel = gaussian_kde(values)
        density = np.reshape(kernel(positions).T, xx.shape)

        contour = ax.contourf(xx, yy, density, levels=20, cmap='hot_r')
        plt.colorbar(contour, ax=ax, label='Density')
        ax.scatter(error_x, error_y, c='blue', s=1, alpha=0.1)
        ax.set_xlabel('Longitude Error (m)', fontsize=11)
        ax.set_ylabel('Latitude Error (m)', fontsize=11)
        ax.set_title('Error Density Heatmap', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # 3. 误差向量场
        ax = axes[1, 0]

        # 采样点用于向量场
        sample_indices = np.random.choice(len(y_true),
                                        size=min(500, len(y_true)),
                                        replace=False)

        ax.quiver(y_true[sample_indices, 0], y_true[sample_indices, 1],
                 error_x[sample_indices], error_y[sample_indices],
                 error_dist[sample_indices], cmap='coolwarm',
                 scale_units='xy', scale=1, alpha=0.6)

        ax.set_xlabel('True Longitude', fontsize=11)
        ax.set_ylabel('True Latitude', fontsize=11)
        ax.set_title('Error Vector Field', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 4. 误差相关性
        ax = axes[1, 1]
        ax.scatter(error_x, error_y, alpha=0.5, s=5)

        # 添加回归线
        z = np.polyfit(error_x, error_y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(error_x.min(), error_x.max(), 100)
        ax.plot(x_line, p(x_line), "r-", linewidth=2,
               label=f'y={z[0]:.3f}x+{z[1]:.3f}')

        # 计算相关系数
        corr = np.corrcoef(error_x, error_y)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
               transform=ax.transAxes, fontsize=11,
               bbox=dict(boxstyle='round', facecolor='wheat'))

        ax.set_xlabel('Longitude Error (m)', fontsize=11)
        ax.set_ylabel('Latitude Error (m)', fontsize=11)
        ax.set_title('Error Correlation Analysis', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.suptitle('2D Error Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"2D误差分析图已保存: {save_path}")

            # 保存误差数据
            error_df = pd.DataFrame({
                'longitude_error': error_x,
                'latitude_error': error_y,
                'total_error': error_dist,
                'true_longitude': y_true[:, 0],
                'true_latitude': y_true[:, 1],
                'pred_longitude': y_pred[:, 0],
                'pred_latitude': y_pred[:, 1]
            })
            csv_path = save_path.replace('.png', '_data.csv')
            error_df.to_csv(csv_path, index=False)

        plt.close()
        return fig

    def plot_predictions_comparison(self, y_true, y_pred, save_path=None):
        """单独绘制预测对比图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. 经度预测对比
        ax = axes[0, 0]
        ax.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5, s=5, color='blue')
        min_val = min(y_true[:, 0].min(), y_pred[:, 0].min())
        max_val = max(y_true[:, 0].max(), y_pred[:, 0].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        # 添加回归线
        z = np.polyfit(y_true[:, 0], y_pred[:, 0], 1)
        p = np.poly1d(z)
        x_line = np.linspace(min_val, max_val, 100)
        ax.plot(x_line, p(x_line), 'g-', linewidth=2, alpha=0.7,
               label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')

        ax.set_xlabel('Actual Longitude', fontsize=11)
        ax.set_ylabel('Predicted Longitude', fontsize=11)
        ax.set_title('Longitude Predictions', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 2. 纬度预测对比
        ax = axes[0, 1]
        ax.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.5, s=5, color='orange')
        min_val = min(y_true[:, 1].min(), y_pred[:, 1].min())
        max_val = max(y_true[:, 1].max(), y_pred[:, 1].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        z = np.polyfit(y_true[:, 1], y_pred[:, 1], 1)
        p = np.poly1d(z)
        x_line = np.linspace(min_val, max_val, 100)
        ax.plot(x_line, p(x_line), 'g-', linewidth=2, alpha=0.7,
               label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')

        ax.set_xlabel('Actual Latitude', fontsize=11)
        ax.set_ylabel('Predicted Latitude', fontsize=11)
        ax.set_title('Latitude Predictions', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 3. 2D位置对比
        ax = axes[0, 2]
        ax.scatter(y_true[:, 0], y_true[:, 1], alpha=0.5, s=10,
                  color='blue', label='Actual')
        ax.scatter(y_pred[:, 0], y_pred[:, 1], alpha=0.5, s=10,
                  color='red', label='Predicted')

        # 连接对应点
        for i in range(0, len(y_true), max(1, len(y_true)//100)):  # 采样显示
            ax.plot([y_true[i, 0], y_pred[i, 0]],
                   [y_true[i, 1], y_pred[i, 1]],
                   'k-', alpha=0.1, linewidth=0.5)

        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.set_title('2D Position Comparison', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 4. 经度残差图
        ax = axes[1, 0]
        residuals_lon = y_pred[:, 0] - y_true[:, 0]
        ax.scatter(y_true[:, 0], residuals_lon, alpha=0.5, s=5, color='blue')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.axhline(y=residuals_lon.mean(), color='g', linestyle=':',
                  label=f'Mean: {residuals_lon.mean():.3f}')
        ax.fill_between(sorted(y_true[:, 0]),
                        residuals_lon.mean() - residuals_lon.std(),
                        residuals_lon.mean() + residuals_lon.std(),
                        alpha=0.3, color='gray', label='±1 Std')

        ax.set_xlabel('Actual Longitude', fontsize=11)
        ax.set_ylabel('Residual (Pred - Actual)', fontsize=11)
        ax.set_title('Longitude Residuals', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 5. 纬度残差图
        ax = axes[1, 1]
        residuals_lat = y_pred[:, 1] - y_true[:, 1]
        ax.scatter(y_true[:, 1], residuals_lat, alpha=0.5, s=5, color='orange')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.axhline(y=residuals_lat.mean(), color='g', linestyle=':',
                  label=f'Mean: {residuals_lat.mean():.3f}')
        ax.fill_between(sorted(y_true[:, 1]),
                        residuals_lat.mean() - residuals_lat.std(),
                        residuals_lat.mean() + residuals_lat.std(),
                        alpha=0.3, color='gray', label='±1 Std')

        ax.set_xlabel('Actual Latitude', fontsize=11)
        ax.set_ylabel('Residual (Pred - Actual)', fontsize=11)
        ax.set_title('Latitude Residuals', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 6. 误差vs预测值
        ax = axes[1, 2]
        error_dist = np.sqrt((y_pred[:, 0] - y_true[:, 0])**2 +
                           (y_pred[:, 1] - y_true[:, 1])**2)
        pred_dist = np.sqrt(y_pred[:, 0]**2 + y_pred[:, 1]**2)

        ax.scatter(pred_dist, error_dist, alpha=0.5, s=5, c=error_dist, cmap='coolwarm')

        # 添加趋势线
        z = np.polyfit(pred_dist, error_dist, 2)
        p = np.poly1d(z)
        x_line = np.linspace(pred_dist.min(), pred_dist.max(), 100)
        ax.plot(x_line, p(x_line), 'g-', linewidth=2,
               label=f'Trend (2nd order)')

        ax.set_xlabel('Distance from Origin (Predicted)', fontsize=11)
        ax.set_ylabel('Error Distance (m)', fontsize=11)
        ax.set_title('Error vs Predicted Distance', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.suptitle('Prediction Comparison Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"预测对比图已保存: {save_path}")

            # 保存对比数据
            comparison_df = pd.DataFrame({
                'true_longitude': y_true[:, 0],
                'true_latitude': y_true[:, 1],
                'pred_longitude': y_pred[:, 0],
                'pred_latitude': y_pred[:, 1],
                'residual_longitude': residuals_lon,
                'residual_latitude': residuals_lat,
                'error_distance': error_dist
            })
            csv_path = save_path.replace('.png', '_data.csv')
            comparison_df.to_csv(csv_path, index=False)

        plt.close()
        return fig

    def plot_floor_analysis(self, y_true_floor, y_pred_floor, save_path=None):
        """楼层预测分析（如果有楼层预测）"""
        if y_pred_floor is None:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 混淆矩阵
        ax = axes[0, 0]
        cm = confusion_matrix(y_true_floor, y_pred_floor)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Floor', fontsize=11)
        ax.set_ylabel('Actual Floor', fontsize=11)
        ax.set_title('Floor Prediction Confusion Matrix', fontsize=12, fontweight='bold')

        # 2. 楼层准确率
        ax = axes[0, 1]
        unique_floors = np.unique(y_true_floor)
        accuracies = []
        for floor in unique_floors:
            mask = y_true_floor == floor
            if mask.sum() > 0:
                acc = (y_pred_floor[mask] == floor).mean() * 100
                accuracies.append(acc)
            else:
                accuracies.append(0)

        ax.bar(unique_floors, accuracies, color='skyblue', edgecolor='black')
        ax.set_xlabel('Floor', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title('Per-Floor Accuracy', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 3. 楼层误差分布
        ax = axes[1, 0]
        floor_errors = np.abs(y_pred_floor - y_true_floor)
        unique_errors, counts = np.unique(floor_errors, return_counts=True)
        ax.bar(unique_errors, counts, color='coral', edgecolor='black')
        ax.set_xlabel('Floor Error (|Predicted - Actual|)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Floor Error Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 4. 楼层转移概率
        ax = axes[1, 1]
        transition_matrix = cm / cm.sum(axis=1, keepdims=True)
        sns.heatmap(transition_matrix, annot=True, fmt='.2f',
                   cmap='YlOrRd', ax=ax, vmin=0, vmax=1)
        ax.set_xlabel('Predicted Floor', fontsize=11)
        ax.set_ylabel('Actual Floor', fontsize=11)
        ax.set_title('Floor Transition Probability', fontsize=12, fontweight='bold')

        plt.suptitle('Floor Prediction Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"楼层分析图已保存: {save_path}")

            # 保存楼层数据
            floor_df = pd.DataFrame({
                'true_floor': y_true_floor,
                'pred_floor': y_pred_floor,
                'floor_error': floor_errors
            })
            csv_path = save_path.replace('.png', '_data.csv')
            floor_df.to_csv(csv_path, index=False)

        plt.close()
        return fig

    def plot_comprehensive_results(self, y_true, y_pred, train_losses, val_losses,
                                   training_params, svr_params, metrics,
                                   save_path=None, show=False):
        """绘制综合结果图（保持原有功能）"""
        # 原有的综合图代码保持不变
        fig = plt.figure(figsize=(20, 16), constrained_layout=True)
        gs = fig.add_gridspec(4, 3)

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

        # 8. 评估指标表格
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

        # 9. 误差箱线图（按范围）
        ax9 = fig.add_subplot(gs[3, 0])
        error_ranges = []
        range_labels = []
        for i in range(0, len(error_dist), len(error_dist)//10):
            end = min(i + len(error_dist)//10, len(error_dist))
            error_ranges.append(error_dist[i:end])
            range_labels.append(f'{i}-{end}')

        bp = ax9.boxplot(error_ranges, labels=range_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax9.set_xlabel('Sample Range', fontsize=11)
        ax9.set_ylabel('Error Distance (m)', fontsize=11)
        ax9.set_title('Error Distribution by Sample Range', fontsize=12)
        ax9.grid(True, alpha=0.3)
        ax9.tick_params(axis='x', rotation=45)

        # 10. 误差百分位表
        ax10 = fig.add_subplot(gs[3, 1:])
        percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
        percentile_values = [np.percentile(error_dist, p) for p in percentiles]

        ax10.bar(range(len(percentiles)), percentile_values,
                color='coral', edgecolor='black')
        ax10.set_xticks(range(len(percentiles)))
        ax10.set_xticklabels([f'{p}%' for p in percentiles])
        ax10.set_xlabel('Percentile', fontsize=11)
        ax10.set_ylabel('Error Distance (m)', fontsize=11)
        ax10.set_title('Error Percentiles', fontsize=12)
        ax10.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, v in enumerate(percentile_values):
            ax10.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=8)

        plt.suptitle('Indoor Localization Results - Transformer + SVR',
                     fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"综合结果图已保存: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def save_all_detailed_plots(self, y_true, y_pred, train_losses, val_losses,
                               training_params, svr_params, metrics, y_true_floor=None,
                               y_pred_floor=None):
        """保存所有详细的单独图表"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 1. 训练曲线
        self.plot_training_curves(
            train_losses, val_losses,
            os.path.join(self.detailed_plot_dir, f'training_curves_{timestamp}.png')
        )

        # 2. 误差分布
        error_distances = self.compute_error_distances(y_true, y_pred)
        self.plot_error_distribution(
            error_distances, metrics,
            os.path.join(self.detailed_plot_dir, f'error_distribution_{timestamp}.png')
        )

        # 3. 2D误差分析
        self.plot_2d_errors(
            y_true, y_pred,
            os.path.join(self.detailed_plot_dir, f'2d_errors_{timestamp}.png')
        )

        # 4. 预测对比
        self.plot_predictions_comparison(
            y_true, y_pred,
            os.path.join(self.detailed_plot_dir, f'predictions_comparison_{timestamp}.png')
        )

        # 5. 楼层分析（如果有）
        if y_true_floor is not None and y_pred_floor is not None:
            self.plot_floor_analysis(
                y_true_floor, y_pred_floor,
                os.path.join(self.detailed_plot_dir, f'floor_analysis_{timestamp}.png')
            )

        logger.info(f"所有详细图表已保存到: {self.detailed_plot_dir}")

    def save_results_to_csv(self, metrics, params, filepath):
        """保存结果到CSV（增强版）"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row_data = {
            'timestamp': timestamp,
            **metrics,
            **params
        }

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

        # 同时保存详细版本
        detailed_filepath = filepath.replace('.csv', f'_detailed_{timestamp.replace(":", "-").replace(" ", "_")}.csv')
        pd.DataFrame([row_data]).to_csv(detailed_filepath, index=False)

    def save_predictions(self, y_true, y_pred, error_distances, filepath):
        """保存预测结果（增强版）"""
        df = pd.DataFrame({
            'true_longitude': y_true[:, 0],
            'true_latitude': y_true[:, 1],
            'pred_longitude': y_pred[:, 0],
            'pred_latitude': y_pred[:, 1],
            'error_distance': error_distances,
            'longitude_error': y_pred[:, 0] - y_true[:, 0],
            'latitude_error': y_pred[:, 1] - y_true[:, 1],
            'error_percentage': (error_distances / np.linalg.norm(y_true, axis=1)) * 100
        })

        df.to_csv(filepath, index=False)
        logger.info(f"预测结果已保存: {filepath}")

        # 保存统计摘要
        summary_filepath = filepath.replace('.csv', '_summary.txt')
        with open(summary_filepath, 'w') as f:
            f.write("Prediction Summary Statistics\n")
            f.write("=" * 50 + "\n")
            f.write(df.describe().to_string())
            f.write("\n\n" + "=" * 50 + "\n")
            f.write("Correlation Matrix\n")
            f.write("=" * 50 + "\n")
            f.write(df.corr().to_string())

    def identify_high_error_samples(self, test_data, y_pred, error_distances,
                                    threshold=15.0, filepath=None):
        """识别高误差样本（增强版）"""
        # 多个阈值分析
        thresholds = [5, 10, 15, 20, 30, 50]
        analysis_results = {}

        for thresh in thresholds:
            high_error_indices = np.where(error_distances > thresh)[0]
            analysis_results[f'threshold_{thresh}m'] = {
                'count': len(high_error_indices),
                'percentage': (len(high_error_indices) / len(error_distances)) * 100,
                'indices': high_error_indices.tolist()
            }

        # 保存分析结果
        if filepath:
            # 保存主要阈值的详细数据
            high_error_indices = np.where(error_distances > threshold)[0]

            if len(high_error_indices) > 0 and test_data is not None:
                high_error_data = test_data.iloc[high_error_indices].copy()
                high_error_data['predicted_longitude'] = y_pred[high_error_indices, 0]
                high_error_data['predicted_latitude'] = y_pred[high_error_indices, 1]
                high_error_data['error_distance'] = error_distances[high_error_indices]

                high_error_data.to_csv(filepath, index=False)
                logger.info(f"高误差样本已保存: {filepath}")

                # 保存多阈值分析
                analysis_filepath = filepath.replace('.csv', '_analysis.json')
                with open(analysis_filepath, 'w') as f:
                    json.dump(analysis_results, f, indent=4)

                return high_error_data

        return None