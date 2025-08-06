"""
主程序入口 - 增强版，包含完整的详细记录系统
"""
import argparse
import logging
import time
import torch
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

from config import Config
from data_preprocessing import DataPreprocessor
from model_definition import WiFiTransformerAutoencoder
from training import Trainer
from svr_regressor import SVRRegressor
from evaluation import Evaluator
from optuna_optimization import OptunaOptimizer
from utils import set_seed, print_model_summary, format_time, create_summary_report

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndoorLocalizationSystem:
    """室内定位系统主类 - 增强版"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")

        # 设置随机种子
        set_seed(config.seed)

        # 保存配置
        config.save()

        # 创建实验记录文件
        self.experiment_log = os.path.join(config.output_dir, 'experiment_log.json')
        self.experiment_data = {
            'start_time': datetime.now().isoformat(),
            'config': config.__dict__,
            'device': str(self.device),
            'stages': {}
        }

    def train_full_pipeline(self):
        """训练完整流程（增强版）"""
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("开始室内定位系统训练")
        logger.info("=" * 60)

        # 记录每个阶段的详细信息
        stage_results = {}

        # 1. 数据预处理
        logger.info("\n步骤1: 数据预处理")
        stage_start = time.time()

        preprocessor = DataPreprocessor(self.config)
        data = preprocessor.load_and_preprocess_data()

        X_train, y_train, y_train_floor = data[0:3]
        X_val, y_val, y_val_floor = data[3:6]
        X_test, y_test, y_test_floor = data[6:9]
        filtered_test_indices = data[9]

        stage_results['data_preprocessing'] = {
            'time_seconds': time.time() - stage_start,
            'train_samples': X_train.shape[0],
            'val_samples': X_val.shape[0],
            'test_samples': X_test.shape[0],
            'feature_dim': X_train.shape[1],
            'statistics': preprocessor.statistics
        }

        # 保存数据统计
        self._save_data_statistics(X_train, X_val, X_test, y_train, y_val, y_test)

        # 2. 训练Transformer自编码器
        logger.info("\n步骤2: 训练Transformer自编码器")
        stage_start = time.time()

        model = WiFiTransformerAutoencoder(
            input_dim=X_train.shape[1],
            model_dim=self.config.model_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.device)

        print_model_summary(model)

        # 保存模型架构
        self._save_model_architecture(model)

        trainer = Trainer(model, self.config, self.device)
        model, train_losses, val_losses = trainer.train_autoencoder(X_train, X_val)

        # 保存Transformer模型
        transformer_path = os.path.join(self.config.model_dir, 'transformer_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config.__dict__,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'architecture': str(model)
        }, transformer_path)
        logger.info(f"Transformer模型已保存: {transformer_path}")

        stage_results['transformer_training'] = {
            'time_seconds': time.time() - stage_start,
            'epochs_trained': len(train_losses),
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'best_val_loss': min(val_losses) if val_losses else None,
            'model_path': transformer_path
        }

        # 3. 提取特征
        logger.info("\n步骤3: 提取特征")
        stage_start = time.time()

        X_train_features = trainer.extract_features(X_train)
        X_val_features = trainer.extract_features(X_val)
        X_test_features = trainer.extract_features(X_test)

        logger.info(f"训练特征维度: {X_train_features.shape}")
        logger.info(f"验证特征维度: {X_val_features.shape}")
        logger.info(f"测试特征维度: {X_test_features.shape}")

        # 保存提取的特征
        self._save_extracted_features(X_train_features, X_val_features, X_test_features)

        stage_results['feature_extraction'] = {
            'time_seconds': time.time() - stage_start,
            'train_feature_shape': X_train_features.shape,
            'val_feature_shape': X_val_features.shape,
            'test_feature_shape': X_test_features.shape
        }

        # 4. 训练SVR回归器
        logger.info("\n步骤4: 训练SVR回归器")
        stage_start = time.time()

        # 反转坐标标准化
        y_train_original = preprocessor.inverse_transform_coordinates(y_train)
        y_val_original = preprocessor.inverse_transform_coordinates(y_val)
        y_test_original = preprocessor.inverse_transform_coordinates(y_test)

        svr = SVRRegressor(self.config)
        svr.create_model()
        svr.fit(X_train_features, y_train_original)

        # 保存SVR模型
        svr_path = os.path.join(self.config.model_dir, 'svr_model.pkl')
        svr.save(svr_path)
        logger.info(f"SVR模型已保存: {svr_path}")

        stage_results['svr_training'] = {
            'time_seconds': time.time() - stage_start,
            'model_path': svr_path,
            'svr_params': {
                'kernel': self.config.svr_kernel,
                'C': self.config.svr_C,
                'epsilon': self.config.svr_epsilon,
                'gamma': self.config.svr_gamma
            }
        }

        # 5. 评估
        logger.info("\n步骤5: 模型评估")
        stage_start = time.time()

        # 在训练集、验证集和测试集上进行评估
        evaluator = Evaluator(self.config)

        # 测试集评估
        y_test_pred = svr.predict(X_test_features)
        test_metrics, test_error_distances = evaluator.calculate_metrics(
            y_test_original, y_test_pred, save_detailed=True
        )

        # 验证集评估
        y_val_pred = svr.predict(X_val_features)
        val_metrics, val_error_distances = evaluator.calculate_metrics(
            y_val_original, y_val_pred, save_detailed=True
        )

        # 训练集评估（用于检查过拟合）
        y_train_pred = svr.predict(X_train_features)
        train_metrics, train_error_distances = evaluator.calculate_metrics(
            y_train_original, y_train_pred, save_detailed=True
        )

        # 打印所有评估结果
        self._print_all_metrics(train_metrics, val_metrics, test_metrics)

        stage_results['evaluation'] = {
            'time_seconds': time.time() - stage_start,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }

        # 6. 生成可视化
        logger.info("\n步骤6: 生成可视化")
        stage_start = time.time()

        # 收集参数
        training_params = {
            'model_dim': self.config.model_dim,
            'num_heads': self.config.num_heads,
            'num_layers': self.config.num_layers,
            'dropout': self.config.dropout,
            'learning_rate': self.config.learning_rate,
            'batch_size': self.config.batch_size,
            'epochs': self.config.epochs,
            'early_stopping_patience': self.config.early_stopping_patience,
            'min_delta_ratio': self.config.min_delta_ratio
        }

        svr_params = {
            'kernel': self.config.svr_kernel,
            'C': self.config.svr_C,
            'epsilon': self.config.svr_epsilon,
            'gamma': self.config.svr_gamma,
            'degree': self.config.svr_degree,
            'coef0': self.config.svr_coef0
        }

        # 绘制综合结果（测试集）
        plot_path = os.path.join(self.config.plot_dir, 'results_comprehensive.png')
        evaluator.plot_comprehensive_results(
            y_test_original, y_test_pred,
            train_losses, val_losses,
            training_params, svr_params,
            test_metrics,
            save_path=plot_path
        )

        # 保存所有详细图表
        evaluator.save_all_detailed_plots(
            y_test_original, y_test_pred,
            train_losses, val_losses,
            training_params, svr_params,
            test_metrics,
            y_test_floor, None  # 如果有楼层预测可以添加
        )

        # 为训练集和验证集也生成详细图表
        self._generate_all_dataset_plots(
            evaluator,
            y_train_original, y_train_pred, train_metrics, 'train',
            y_val_original, y_val_pred, val_metrics, 'val',
            y_test_original, y_test_pred, test_metrics, 'test'
        )

        stage_results['visualization'] = {
            'time_seconds': time.time() - stage_start,
            'plots_generated': True,
            'plot_directory': self.config.plot_dir
        }

        # 7. 保存详细结果
        logger.info("\n步骤7: 保存详细结果")
        stage_start = time.time()

        # 保存所有结果到CSV
        all_params = {**training_params, **svr_params}

        # 测试集结果
        csv_path = os.path.join(self.config.csv_dir, 'test_results.csv')
        evaluator.save_results_to_csv(test_metrics, all_params, csv_path)

        # 验证集结果
        val_csv_path = os.path.join(self.config.csv_dir, 'val_results.csv')
        evaluator.save_results_to_csv(val_metrics, all_params, val_csv_path)

        # 训练集结果
        train_csv_path = os.path.join(self.config.csv_dir, 'train_results.csv')
        evaluator.save_results_to_csv(train_metrics, all_params, train_csv_path)

        # 保存所有预测结果
        test_predictions_path = os.path.join(self.config.csv_dir, 'test_predictions.csv')
        evaluator.save_predictions(y_test_original, y_test_pred,
                                  test_error_distances, test_predictions_path)

        val_predictions_path = os.path.join(self.config.csv_dir, 'val_predictions.csv')
        evaluator.save_predictions(y_val_original, y_val_pred,
                                  val_error_distances, val_predictions_path)

        train_predictions_path = os.path.join(self.config.csv_dir, 'train_predictions.csv')
        evaluator.save_predictions(y_train_original, y_train_pred,
                                  train_error_distances, train_predictions_path)

        # 识别高误差样本
        if hasattr(self.config, 'test_path'):
            test_data = pd.read_csv(self.config.test_path)
            test_data = test_data.iloc[filtered_test_indices].reset_index(drop=True)

            high_error_path = os.path.join(self.config.csv_dir, 'high_error_samples.csv')
            evaluator.identify_high_error_samples(
                test_data, y_test_pred, test_error_distances,
                threshold=15.0, filepath=high_error_path
            )

        stage_results['save_results'] = {
            'time_seconds': time.time() - stage_start,
            'csv_files_saved': True,
            'csv_directory': self.config.csv_dir
        }

        # 8. 生成最终报告
        elapsed_time = time.time() - start_time

        # 更新实验记录
        self.experiment_data['stages'] = stage_results
        self.experiment_data['end_time'] = datetime.now().isoformat()
        self.experiment_data['total_time_seconds'] = elapsed_time

        # 保存实验记录
        with open(self.experiment_log, 'w') as f:
            json.dump(self.experiment_data, f, indent=4)

        # 创建详细的总结报告
        results = {
            'metrics': {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics
            },
            'training_time': format_time(elapsed_time),
            'model_size': trainer.extract_features(X_train[:1]).shape[-1],
            'best_val_loss': min(val_losses) if val_losses else None,
            'stages': stage_results
        }

        report_path = os.path.join(self.config.output_dir, 'summary_report.json')
        create_summary_report(self.config, results, report_path)

        # 创建Markdown报告
        self._create_markdown_report(results, training_params, svr_params)

        logger.info("\n" + "=" * 60)
        logger.info(f"训练完成！总用时: {format_time(elapsed_time)}")
        logger.info(f"所有结果已保存到: {self.config.output_dir}")
        logger.info("=" * 60)

        return results

    def _save_data_statistics(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """保存数据统计信息"""
        stats_dir = os.path.join(self.config.csv_dir, 'data_statistics')
        os.makedirs(stats_dir, exist_ok=True)

        # 特征统计
        feature_stats = {
            'train': {
                'mean': X_train.mean(axis=0).tolist(),
                'std': X_train.std(axis=0).tolist(),
                'min': X_train.min(axis=0).tolist(),
                'max': X_train.max(axis=0).tolist()
            },
            'val': {
                'mean': X_val.mean(axis=0).tolist(),
                'std': X_val.std(axis=0).tolist(),
                'min': X_val.min(axis=0).tolist(),
                'max': X_val.max(axis=0).tolist()
            },
            'test': {
                'mean': X_test.mean(axis=0).tolist(),
                'std': X_test.std(axis=0).tolist(),
                'min': X_test.min(axis=0).tolist(),
                'max': X_test.max(axis=0).tolist()
            }
        }

        with open(os.path.join(stats_dir, 'feature_statistics.json'), 'w') as f:
            json.dump(feature_stats, f, indent=4)

        # 目标变量统计
        target_stats = pd.DataFrame({
            'dataset': ['train', 'val', 'test'],
            'longitude_mean': [y_train[:, 0].mean(), y_val[:, 0].mean(), y_test[:, 0].mean()],
            'longitude_std': [y_train[:, 0].std(), y_val[:, 0].std(), y_test[:, 0].std()],
            'latitude_mean': [y_train[:, 1].mean(), y_val[:, 1].mean(), y_test[:, 1].mean()],
            'latitude_std': [y_train[:, 1].std(), y_val[:, 1].std(), y_test[:, 1].std()]
        })
        target_stats.to_csv(os.path.join(stats_dir, 'target_statistics.csv'), index=False)

    def _save_model_architecture(self, model):
        """保存模型架构"""
        arch_path = os.path.join(self.config.model_dir, 'model_architecture.txt')
        with open(arch_path, 'w') as f:
            f.write(str(model))
            f.write('\n\n' + '=' * 50 + '\n')
            f.write('Model Summary:\n')
            f.write('=' * 50 + '\n')

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            f.write(f'Total parameters: {total_params:,}\n')
            f.write(f'Trainable parameters: {trainable_params:,}\n')
            f.write(f'Model size: {total_params * 4 / 1024 / 1024:.2f} MB\n')

    def _save_extracted_features(self, X_train_features, X_val_features, X_test_features):
        """保存提取的特征"""
        features_dir = os.path.join(self.config.output_dir, 'extracted_features')
        os.makedirs(features_dir, exist_ok=True)

        np.save(os.path.join(features_dir, 'train_features.npy'), X_train_features)
        np.save(os.path.join(features_dir, 'val_features.npy'), X_val_features)
        np.save(os.path.join(features_dir, 'test_features.npy'), X_test_features)

        # 保存特征统计
        feature_info = {
            'train_shape': X_train_features.shape,
            'val_shape': X_val_features.shape,
            'test_shape': X_test_features.shape,
            'feature_dim': X_train_features.shape[1]
        }

        with open(os.path.join(features_dir, 'feature_info.json'), 'w') as f:
            json.dump(feature_info, f, indent=4)

    def _print_all_metrics(self, train_metrics, val_metrics, test_metrics):
        """打印所有数据集的评估结果"""
        logger.info("\n" + "=" * 70)
        logger.info("完整评估结果")
        logger.info("=" * 70)

        # 创建对比表格
        metrics_names = ['mean_error_distance', 'median_error_distance', 'rmse', 'mae', 'r2']

        logger.info("\n主要指标对比:")
        logger.info("-" * 70)
        logger.info(f"{'Metric':<25} {'Train':<15} {'Val':<15} {'Test':<15}")
        logger.info("-" * 70)

        for metric in metrics_names:
            train_val = train_metrics.get(metric, 0)
            val_val = val_metrics.get(metric, 0)
            test_val = test_metrics.get(metric, 0)
            logger.info(f"{metric:<25} {train_val:<15.4f} {val_val:<15.4f} {test_val:<15.4f}")

        logger.info("-" * 70)

        # 计算过拟合指标
        overfit_ratio = (test_metrics['mean_error_distance'] - train_metrics['mean_error_distance']) / train_metrics['mean_error_distance']
        logger.info(f"\n过拟合分析:")
        logger.info(f"训练-测试误差比: {overfit_ratio:.2%}")
        logger.info(f"验证-测试误差差: {test_metrics['mean_error_distance'] - val_metrics['mean_error_distance']:.2f}m")

    def _generate_all_dataset_plots(self, evaluator,
                                   y_train, y_train_pred, train_metrics, train_name,
                                   y_val, y_val_pred, val_metrics, val_name,
                                   y_test, y_test_pred, test_metrics, test_name):
        """为所有数据集生成详细图表"""
        datasets = [
            (y_train, y_train_pred, train_metrics, train_name),
            (y_val, y_val_pred, val_metrics, val_name),
            (y_test, y_test_pred, test_metrics, test_name)
        ]

        for y_true, y_pred, metrics, name in datasets:
            dataset_plot_dir = os.path.join(evaluator.detailed_plot_dir, name)
            os.makedirs(dataset_plot_dir, exist_ok=True)

            # 误差分布
            error_distances = evaluator.compute_error_distances(y_true, y_pred)
            evaluator.plot_error_distribution(
                error_distances, metrics,
                os.path.join(dataset_plot_dir, f'{name}_error_distribution.png')
            )

            # 2D误差
            evaluator.plot_2d_errors(
                y_true, y_pred,
                os.path.join(dataset_plot_dir, f'{name}_2d_errors.png')
            )

            # 预测对比
            evaluator.plot_predictions_comparison(
                y_true, y_pred,
                os.path.join(dataset_plot_dir, f'{name}_predictions.png')
            )

    def _create_markdown_report(self, results, training_params, svr_params):
        """创建Markdown格式的报告"""
        report_path = os.path.join(self.config.output_dir, 'report.md')

        with open(report_path, 'w') as f:
            f.write("# Indoor Localization System - Experimental Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Output Directory**: `{self.config.output_dir}`\n\n")

            f.write("## 1. Configuration\n\n")
            f.write("### Transformer Parameters\n")
            for key, value in training_params.items():
                f.write(f"- **{key}**: {value}\n")

            f.write("\n### SVR Parameters\n")
            for key, value in svr_params.items():
                f.write(f"- **{key}**: {value}\n")

            f.write("\n## 2. Results Summary\n\n")
            f.write("### Performance Metrics\n\n")
            f.write("| Dataset | Mean Error (m) | Median Error (m) | RMSE | MAE | R² |\n")
            f.write("|---------|---------------|-----------------|------|-----|----|\n")

            for dataset in ['train', 'val', 'test']:
                metrics = results['metrics'][dataset]
                f.write(f"| {dataset.capitalize()} | "
                       f"{metrics['mean_error_distance']:.2f} | "
                       f"{metrics['median_error_distance']:.2f} | "
                       f"{metrics['rmse']:.4f} | "
                       f"{metrics['mae']:.4f} | "
                       f"{metrics['r2']:.4f} |\n")

            f.write("\n### Error Percentiles (Test Set)\n\n")
            test_metrics = results['metrics']['test']
            f.write("| Percentile | Error Distance (m) |\n")
            f.write("|------------|-------------------|\n")
            for p in [50, 75, 90, 95, 99]:
                if f'percentile_{p}' in test_metrics:
                    f.write(f"| {p}th | {test_metrics[f'percentile_{p}']:.2f} |\n")

            f.write("\n## 3. Training Progress\n\n")
            f.write(f"- **Total Training Time**: {results['training_time']}\n")
            f.write(f"- **Best Validation Loss**: {results['best_val_loss']:.6f}\n")
            f.write(f"- **Feature Dimension**: {results['model_size']}\n")

            f.write("\n## 4. Stage Timing\n\n")
            f.write("| Stage | Time (seconds) |\n")
            f.write("|-------|---------------|\n")
            for stage, info in results['stages'].items():
                if 'time_seconds' in info:
                    f.write(f"| {stage.replace('_', ' ').title()} | {info['time_seconds']:.2f} |\n")

            f.write("\n## 5. Files Generated\n\n")
            f.write("- Models: `models/`\n")
            f.write("- Plots: `plots/` and `plots/detailed/`\n")
            f.write("- CSV Results: `csv/` and `csv/detailed/`\n")
            f.write("- Training Logs: `logs/training_details/`\n")
            f.write("- Extracted Features: `extracted_features/`\n")

    def run_optimization(self):
        """运行超参数优化（增强版）"""
        logger.info("=" * 60)
        logger.info("开始超参数优化")
        logger.info("=" * 60)

        optimizer = OptunaOptimizer(self.config)
        transformer_study, svr_study = optimizer.run_combined_optimization()

        # 保存优化历史
        self._save_optimization_history(transformer_study, svr_study)

        return transformer_study, svr_study

    def _save_optimization_history(self, transformer_study, svr_study):
        """保存优化历史"""
        optim_dir = os.path.join(self.config.output_dir, 'optimization')
        os.makedirs(optim_dir, exist_ok=True)

        # 保存Transformer优化历史
        transformer_df = transformer_study.trials_dataframe()
        transformer_df.to_csv(os.path.join(optim_dir, 'transformer_trials.csv'), index=False)

        # 保存SVR优化历史
        svr_df = svr_study.trials_dataframe()
        svr_df.to_csv(os.path.join(optim_dir, 'svr_trials.csv'), index=False)

        # 保存最佳参数
        best_params = {
            'transformer': transformer_study.best_params,
            'svr': svr_study.best_params,
            'transformer_best_value': transformer_study.best_value,
            'svr_best_value': svr_study.best_value
        }

        with open(os.path.join(optim_dir, 'best_params.json'), 'w') as f:
            json.dump(best_params, f, indent=4)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='室内定位系统 - Transformer + SVR (增强版)')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'optimize'],
                        help='运行模式: train(训练) 或 optimize(优化)')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='学习率')
    parser.add_argument('--verbose', action='store_true',
                        help='详细输出模式')

    args = parser.parse_args()

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 加载或创建配置
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()

    # 覆盖配置参数
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate

    # 创建系统实例
    system = IndoorLocalizationSystem(config)

    # 运行
    if args.mode == 'train':
        results = system.train_full_pipeline()
    elif args.mode == 'optimize':
        results = system.run_optimization()
    else:
        raise ValueError(f"未知模式: {args.mode}")

    logger.info("\n程序执行完成！")


if __name__ == '__main__':
    main()