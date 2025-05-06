import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# 设置中文字体
try:
    # 尝试设置支持中文的字体
    font_path = 'C:/Windows/Fonts/simhei.ttf'  # Windows系统中的黑体字体
    font_prop = FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    # 或者直接使用：
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
except:
    print("警告：未能设置中文字体，图表中的中文可能无法正确显示")

class PositioningVisualizer:
    """定位系统可视化工具"""

    def __init__(self, floor_plans=None):
        """
        初始化定位可视化器

        参数:
            floor_plans (dict): 楼层平面图字典，键为楼层ID，值为图像数据
        """
        self.floor_plans = floor_plans

    def plot_error_cdf(self, errors, label=None, reference_results=None, figsize=(10, 6), save_path=None):
        """
        绘制定位误差的累积分布函数

        参数:
            errors: 误差列表或数组
            label: 当前结果的标签
            reference_results: 参考结果字典 {name: errors}
            figsize: 图形大小
            save_path: 保存路径

        返回:
            matplotlib图形对象
        """
        plt.figure(figsize=figsize)

        # 排序误差并计算累积概率
        sorted_errors = np.sort(errors)
        cumulative_prob = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

        # 绘制当前结果
        plt.plot(sorted_errors, cumulative_prob, linewidth=2, label=label or '当前模型')

        # 绘制参考结果（如果提供）
        if reference_results:
            for ref_name, ref_errors in reference_results.items():
                ref_sorted = np.sort(ref_errors)
                ref_prob = np.arange(1, len(ref_sorted) + 1) / len(ref_sorted)
                plt.plot(ref_sorted, ref_prob, linestyle='--', label=ref_name)

        # 添加参考线
        plt.axhline(y=0.75, color='r', linestyle=':', label='75%分位')
        plt.axhline(y=0.9, color='g', linestyle=':', label='90%分位')

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('定位误差（米）')
        plt.ylabel('累积概率')
        plt.title('定位误差的累积分布函数')
        plt.legend()
        plt.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(save_path)

        return plt.gcf()

    def plot_error_heatmap(self, true_positions, predicted_positions, floor=0, figsize=(12, 10), save_path=None):
        """
        在楼层平面图上绘制空间误差热图

        参数:
            true_positions: 真实位置 (n_samples, 3)，包含楼层
            predicted_positions: 预测位置 (n_samples, 2)
            floor: 要绘制的楼层
            figsize: 图形大小
            save_path: 保存路径

        返回:
            matplotlib图形对象
        """
        # 确定坐标范围
        min_x = min(np.min(true_positions[:, 0]), np.min(predicted_positions[:, 0]))
        max_x = max(np.max(true_positions[:, 0]), np.max(predicted_positions[:, 0]))
        min_y = min(np.min(true_positions[:, 1]), np.min(predicted_positions[:, 1]))
        max_y = max(np.max(true_positions[:, 1]), np.max(predicted_positions[:, 1]))

        # 创建图形
        if self.floor_plans is None or floor not in self.floor_plans:
            # 如果没有楼层平面图，创建空白背景
            fig, ax = plt.subplots(figsize=figsize)
        else:
            # 加载楼层平面图作为背景
            floor_plan = self.floor_plans[floor]
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(floor_plan, extent=[min_x, max_x, min_y, max_y], alpha=0.5)

        # 为当前楼层筛选位置
        if true_positions.shape[1] > 2:  # 如果包含楼层信息
            floor_mask = true_positions[:, 2] == floor
            floor_true = true_positions[floor_mask, :2]
            floor_pred = predicted_positions[floor_mask, :2] if predicted_positions.shape[1] > 2 else \
            predicted_positions[floor_mask]
        else:
            floor_true = true_positions
            floor_pred = predicted_positions

        # 计算误差
        errors = np.sqrt(np.sum((floor_true - floor_pred) ** 2, axis=1))

        # 创建按误差大小着色的散点图
        scatter = ax.scatter(
            floor_true[:, 0], floor_true[:, 1],
            c=errors, cmap='viridis',
            s=50, alpha=0.8,
            norm=Normalize(vmin=0, vmax=np.percentile(errors, 95))
        )

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('定位误差（米）')

        # 用线连接真实位置和预测位置
        for i in range(len(floor_true)):
            ax.plot(
                [floor_true[i, 0], floor_pred[i, 0]],
                [floor_true[i, 1], floor_pred[i, 1]],
                'k-', alpha=0.3
            )

        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.set_title(f'楼层{floor}的定位误差')
        plt.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(save_path)

        return fig

    def plot_error_distribution(self, errors, bins=30, figsize=(10, 6), save_path=None):
        """
        绘制误差分布直方图

        参数:
            errors: 误差列表或数组
            bins: 直方图的条数
            figsize: 图形大小
            save_path: 保存路径

        返回:
            matplotlib图形对象
        """
        plt.figure(figsize=figsize)

        sns.histplot(errors, bins=bins, kde=True)

        plt.axvline(np.mean(errors), color='r', linestyle='--', label=f'平均值: {np.mean(errors):.2f}m')
        plt.axvline(np.median(errors), color='g', linestyle='--', label=f'中位数: {np.median(errors):.2f}m')

        plt.xlabel('定位误差（米）')
        plt.ylabel('频率')
        plt.title('定位误差分布')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(save_path)

        return plt.gcf()

    def plot_error_2d_scatter(self, true_positions, predicted_positions, figsize=(10, 8),
                              save_path=None):
        """
        绘制预测误差的2D散点图（X误差和Y误差）

        参数:
            true_positions: 真实位置 (n_samples, 2)
            predicted_positions: 预测位置 (n_samples, 2)
            figsize: 图形大小
            save_path: 保存路径

        返回:
            matplotlib图形对象
        """
        # 计算X和Y方向的误差
        x_errors = predicted_positions[:, 0] - true_positions[:, 0]
        y_errors = predicted_positions[:, 1] - true_positions[:, 1]

        # 计算欧氏距离误差（用于颜色编码）
        euclidean_errors = np.sqrt(x_errors ** 2 + y_errors ** 2)

        # 创建散点图
        plt.figure(figsize=figsize)

        # 绘制基准线
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)

        # 绘制误差散点
        scatter = plt.scatter(x_errors, y_errors, c=euclidean_errors,
                              cmap='viridis', alpha=0.8, s=50,
                              norm=matplotlib.colors.Normalize(vmin=0, vmax=np.percentile(euclidean_errors, 95)))

        # 绘制中心点（零误差）
        plt.scatter([0], [0], c='red', s=100, marker='x', label='零误差')

        # 绘制误差椭圆（误差分布的95%置信区间）
        from matplotlib.patches import Ellipse
        cov = np.cov(x_errors, y_errors)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))

        # 95%置信区间对应的比例因子
        scale_factor = 2.4477  # 对应于标准正态分布的95%置信区间
        width, height = 2 * scale_factor * np.sqrt(eigenvals)

        ellipse = Ellipse(xy=(0, 0), width=width, height=height, angle=angle,
                          edgecolor='red', fc='none', lw=2, label='95%置信区间')
        plt.gca().add_patch(ellipse)

        # 添加颜色条
        cbar = plt.colorbar(scatter)
        cbar.set_label('欧氏距离误差 (m)')

        # 设置标签和标题
        plt.xlabel('X方向误差 (m)')
        plt.ylabel('Y方向误差 (m)')
        plt.title('2D定位误差分布')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # 添加误差统计信息
        mean_x_error = np.mean(x_errors)
        mean_y_error = np.mean(y_errors)
        std_x_error = np.std(x_errors)
        std_y_error = np.std(y_errors)

        plt.annotate(f'X误差均值: {mean_x_error:.2f}m\nX误差标准差: {std_x_error:.2f}m\n'
                     f'Y误差均值: {mean_y_error:.2f}m\nY误差标准差: {std_y_error:.2f}m',
                     xy=(0.05, 0.05), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

        plt.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(save_path)

        return plt.gcf()

    def plot_floor_confusion_matrix(self, floor_true, floor_pred, figsize=(8, 6), save_path=None):
        """
        绘制楼层预测的混淆矩阵

        参数:
            floor_true: 真实楼层标签
            floor_pred: 预测楼层标签
            figsize: 图形大小
            save_path: 保存路径

        返回:
            matplotlib图形对象
        """
        from sklearn.metrics import confusion_matrix

        # 计算混淆矩阵
        cm = confusion_matrix(floor_true, floor_pred)

        # 创建图形
        plt.figure(figsize=figsize)

        # 获取唯一楼层标签
        labels = sorted(np.unique(np.concatenate([floor_true, floor_pred])))

        # 绘制热图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels,
                    yticklabels=labels)

        # 设置标签和标题
        plt.xlabel('预测楼层')
        plt.ylabel('真实楼层')
        plt.title('楼层预测混淆矩阵')

        # 计算准确率
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)
        plt.figtext(0.5, 0.01, f'楼层准确率: {accuracy:.2%}',
                    ha='center', fontsize=12)

        plt.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(save_path)

        return plt.gcf()

    def plot_training_history(self, history, figsize=(12, 5), save_path=None):
        """
        绘制训练历史曲线

        参数:
            history: 训练历史字典
            figsize: 图形大小
            save_path: 保存路径

        返回:
            matplotlib图形对象
        """
        plt.figure(figsize=figsize)

        # 提取数据
        epochs = history['epoch']
        train_loss = history['train_loss']
        val_loss = history['val_loss'] if 'val_loss' in history else None
        lr = history['learning_rate'] if 'learning_rate' in history else None

        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, 'b-', label='训练损失')
        if val_loss is not None and any(v is not None for v in val_loss):
            plt.plot(epochs, val_loss, 'r-', label='验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.title('训练和验证损失')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # 绘制学习率曲线
        if lr is not None and any(v is not None for v in lr):
            plt.subplot(1, 2, 2)
            plt.plot(epochs, lr, 'g-')
            plt.xlabel('轮次')
            plt.ylabel('学习率')
            plt.title('学习率变化')
            plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(save_path)

        return plt.gcf()

    def plot_feature_importances(self, importances, feature_names=None, top_n=20, figsize=(10, 8), save_path=None):
        """
        绘制特征重要性图

        参数:
            importances: 特征重要性数组
            feature_names: 特征名称列表（可选）
            top_n: 显示前N个重要特征
            figsize: 图形大小
            save_path: 保存路径

        返回:
            matplotlib图形对象
        """
        if feature_names is None:
            feature_names = [f'特征{i}' for i in range(len(importances))]

        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        # 排序并保留前N个
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)

        # 绘制条形图
        plt.figure(figsize=figsize)
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title(f'Top {top_n} 特征重要性')
        plt.xlabel('重要性')
        plt.ylabel('特征名称')
        plt.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(save_path)

        return plt.gcf()

    def plot_parameter_distribution(self, study, param_name, figsize=(10, 6), save_path=None):
        """
        绘制Optuna优化参数分布

        参数:
            study: Optuna Study对象
            param_name: 参数名称
            figsize: 图形大小
            save_path: 保存路径

        返回:
            matplotlib图形对象
        """
        import optuna

        # 创建图形
        plt.figure(figsize=figsize)

        # 提取参数值
        values = []
        scores = []

        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE and param_name in trial.params:
                values.append(trial.params[param_name])
                scores.append(trial.value)

        # 检查参数类型
        if len(values) == 0:
            plt.title(f'Parameter {param_name} not found in any trial')
            return plt.gcf()

        if isinstance(values[0], (int, float)):
            # 数值参数，绘制散点图和边缘分布
            plt.scatter(values, scores, alpha=0.7)
            plt.xlabel(param_name)
            plt.ylabel('目标值 (误差)')
            plt.title(f'参数 {param_name} 的分布与性能')
            plt.grid(True, linestyle='--', alpha=0.7)
        else:
            # 分类参数，绘制箱线图
            value_score_dict = {}
            for v, s in zip(values, scores):
                if v not in value_score_dict:
                    value_score_dict[v] = []
                value_score_dict[v].append(s)

            labels = list(value_score_dict.keys())
            data = [value_score_dict[label] for label in labels]

            plt.boxplot(data, labels=labels)
            plt.xlabel(param_name)
            plt.ylabel('目标值 (误差)')
            plt.title(f'参数 {param_name} 的分布与性能')
            plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(save_path)

        return plt.gcf()

    def save_visualization_report(self, results, output_dir, config):
        """
        创建并保存完整的可视化报告

        参数:
            results: 评估结果字典
            output_dir: 输出目录
            config: 配置字典

        返回:
            保存的文件列表
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        saved_files = []
        visualization_formats = config['logging']['visualization_formats']

        # 绘制并保存CDF图
        if 'position_errors' in results:
            for fmt in visualization_formats:
                cdf_fig = self.plot_error_cdf(results['position_errors'])
                cdf_path = os.path.join(viz_dir, f'error_cdf.{fmt}')
                cdf_fig.savefig(cdf_path)
                saved_files.append(cdf_path)
                plt.close(cdf_fig)

        # 绘制并保存误差分布图
        if 'position_errors' in results:
            for fmt in visualization_formats:
                dist_fig = self.plot_error_distribution(results['position_errors'])
                dist_path = os.path.join(viz_dir, f'error_distribution.{fmt}')
                dist_fig.savefig(dist_path)
                saved_files.append(dist_path)
                plt.close(dist_fig)

        # 绘制并保存2D误差散点图
        if 'true_positions' in results and 'predicted_positions' in results:
            for fmt in visualization_formats:
                error_2d_fig = self.plot_error_2d_scatter(
                    results['true_positions'],
                    results['predicted_positions']
                )
                error_2d_path = os.path.join(viz_dir, f'error_2d_scatter.{fmt}')
                error_2d_fig.savefig(error_2d_path)
                saved_files.append(error_2d_path)
                plt.close(error_2d_fig)

        # 绘制并保存楼层混淆矩阵
        if 'floor_true' in results and 'floor_pred' in results:
            for fmt in visualization_formats:
                floor_cm_fig = self.plot_floor_confusion_matrix(
                    results['floor_true'],
                    results['floor_pred']
                )
                floor_cm_path = os.path.join(viz_dir, f'floor_confusion_matrix.{fmt}')
                floor_cm_fig.savefig(floor_cm_path)
                saved_files.append(floor_cm_path)
                plt.close(floor_cm_fig)

        # 绘制并保存训练历史
        if 'training_history' in results:
            for fmt in visualization_formats:
                history_fig = self.plot_training_history(results['training_history'])
                history_path = os.path.join(viz_dir, f'training_history.{fmt}')
                history_fig.savefig(history_path)
                saved_files.append(history_path)
                plt.close(history_fig)

        # 绘制并保存特征重要性
        if 'feature_importances' in results and 'feature_names' in results:
            for fmt in visualization_formats:
                fi_fig = self.plot_feature_importances(
                    results['feature_importances'],
                    results['feature_names']
                )
                fi_path = os.path.join(viz_dir, f'feature_importances.{fmt}')
                fi_fig.savefig(fi_path)
                saved_files.append(fi_path)
                plt.close(fi_fig)

        # 创建报告索引HTML
        html_index = os.path.join(viz_dir, 'visualization_index.html')
        with open(html_index, 'w') as f:
            f.write('<html><head><title>可视化报告</title></head><body>\n')
            f.write('<h1>定位系统可视化报告</h1>\n')

            for saved_file in sorted(saved_files):
                if saved_file.endswith('.png'):
                    base_name = os.path.basename(saved_file)
                    f.write(f'<h2>{base_name}</h2>\n')
                    f.write(f'<img src="{base_name}" width="800"><br>\n')

            f.write('</body></html>\n')

        saved_files.append(html_index)

        return saved_files