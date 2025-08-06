"""
主程序入口
"""
import argparse
import logging
import time
import torch
import pandas as pd
import numpy as np
import os
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
    """室内定位系统主类"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")

        # 设置随机种子
        set_seed(config.seed)

        # 保存配置
        config.save()

    def train_full_pipeline(self):
        """训练完整流程"""
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("开始室内定位系统训练")
        logger.info("=" * 60)

        # 1. 数据预处理
        logger.info("\n步骤1: 数据预处理")
        preprocessor = DataPreprocessor(self.config)
        data = preprocessor.load_and_preprocess_data()

        X_train, y_train, y_train_floor = data[0:3]
        X_val, y_val, y_val_floor = data[3:6]
        X_test, y_test, y_test_floor = data[6:9]
        filtered_test_indices = data[9]

        # 2. 训练Transformer自编码器
        logger.info("\n步骤2: 训练Transformer自编码器")
        model = WiFiTransformerAutoencoder(
            input_dim=X_train.shape[1],
            model_dim=self.config.model_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.device)

        print_model_summary(model)

        trainer = Trainer(model, self.config, self.device)
        model, train_losses, val_losses = trainer.train_autoencoder(X_train, X_val)

        # 保存Transformer模型
        transformer_path = os.path.join(self.config.model_dir, 'transformer_model.pth')
        torch.save(model.state_dict(), transformer_path)
        logger.info(f"Transformer模型已保存: {transformer_path}")

        # 3. 提取特征
        logger.info("\n步骤3: 提取特征")
        X_train_features = trainer.extract_features(X_train)
        X_test_features = trainer.extract_features(X_test)

        logger.info(f"训练特征维度: {X_train_features.shape}")
        logger.info(f"测试特征维度: {X_test_features.shape}")

        # 4. 训练SVR回归器
        logger.info("\n步骤4: 训练SVR回归器")

        # 反转坐标标准化
        y_train_original = preprocessor.inverse_transform_coordinates(y_train)
        y_test_original = preprocessor.inverse_transform_coordinates(y_test)

        svr = SVRRegressor(self.config)
        svr.create_model()
        svr.fit(X_train_features, y_train_original)

        # 保存SVR模型
        svr_path = os.path.join(self.config.model_dir, 'svr_model.pkl')
        svr.save(svr_path)
        logger.info(f"SVR模型已保存: {svr_path}")

        # 5. 评估
        logger.info("\n步骤5: 模型评估")
        y_pred = svr.predict(X_test_features)

        evaluator = Evaluator(self.config)
        metrics, error_distances = evaluator.calculate_metrics(y_test_original, y_pred)

        # 打印评估结果
        logger.info("\n" + "=" * 60)
        logger.info("评估结果")
        logger.info("=" * 60)
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.4f}")

        # 6. 生成可视化
        logger.info("\n步骤6: 生成可视化")

        # 收集参数
        training_params = {
            'model_dim': self.config.model_dim,
            'num_heads': self.config.num_heads,
            'num_layers': self.config.num_layers,
            'dropout': self.config.dropout,
            'learning_rate': self.config.learning_rate,
            'batch_size': self.config.batch_size,
            'epochs': self.config.epochs
        }

        svr_params = {
            'kernel': self.config.svr_kernel,
            'C': self.config.svr_C,
            'epsilon': self.config.svr_epsilon,
            'gamma': self.config.svr_gamma
        }

        # 绘制综合结果
        plot_path = os.path.join(self.config.plot_dir, 'results.png')
        evaluator.plot_comprehensive_results(
            y_test_original, y_pred,
            train_losses, val_losses,
            training_params, svr_params,
            metrics,
            save_path=plot_path
        )

        # 保存结果到CSV
        csv_path = os.path.join(self.config.csv_dir, 'results.csv')
        evaluator.save_results_to_csv(metrics, {**training_params, **svr_params}, csv_path)

        # 保存预测结果
        predictions_path = os.path.join(self.config.csv_dir, 'predictions.csv')
        evaluator.save_predictions(y_test_original, y_pred, error_distances, predictions_path)

        # 识别高误差样本
        if hasattr(self.config, 'test_path'):
            test_data = pd.read_csv(self.config.test_path)
            test_data = test_data.iloc[filtered_test_indices].reset_index(drop=True)

            high_error_path = os.path.join(self.config.csv_dir, 'high_error_samples.csv')
            evaluator.identify_high_error_samples(
                test_data, y_pred, error_distances,
                threshold=15.0, filepath=high_error_path
            )

        # 7. 生成总结报告
        elapsed_time = time.time() - start_time

        results = {
            'metrics': metrics,
            'training_time': format_time(elapsed_time),
            'model_size': trainer.extract_features(X_train[:1]).shape[-1],
            'best_val_loss': min(val_losses) if val_losses else None
        }

        report_path = os.path.join(self.config.output_dir, 'summary_report.json')
        create_summary_report(self.config, results, report_path)

        logger.info("\n" + "=" * 60)
        logger.info(f"训练完成！总用时: {format_time(elapsed_time)}")
        logger.info(f"所有结果已保存到: {self.config.output_dir}")
        logger.info("=" * 60)

        return results

    def run_optimization(self):
        """运行超参数优化"""
        logger.info("=" * 60)
        logger.info("开始超参数优化")
        logger.info("=" * 60)

        optimizer = OptunaOptimizer(self.config)
        transformer_study, svr_study = optimizer.run_combined_optimization()

        return transformer_study, svr_study


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='室内定位系统 - Transformer + SVR')
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

    args = parser.parse_args()

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
