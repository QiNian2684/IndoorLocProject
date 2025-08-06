"""
Optuna超参数优化模块
"""
import optuna
from optuna.exceptions import TrialPruned
import torch
import numpy as np
import json
import os
import logging
from datetime import datetime
from dataclasses import fields

from config import Config
from data_preprocessing import DataPreprocessor
from model_definition import WiFiTransformerAutoencoder
from training import Trainer, NaNLossError
from svr_regressor import SVRRegressor
from evaluation import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """Optuna优化器"""

    def __init__(self, base_config):
        self.base_config = base_config
        self.device = torch.device(base_config.device if torch.cuda.is_available() else 'cpu')

        # 加载数据
        self.preprocessor = DataPreprocessor(base_config)
        self.data = self.preprocessor.load_and_preprocess_data()

        # 解包数据
        (self.X_train, self.y_train, self.y_train_floor,
         self.X_val, self.y_val, self.y_val_floor,
         self.X_test, self.y_test, self.y_test_floor,
         self.filtered_test_indices) = self.data

        # 评估器
        self.evaluator = Evaluator(base_config)

        # 最佳结果
        self.best_transformer_params = None
        self.best_svr_params = None
        self.best_val_loss = float('inf')
        self.best_mean_error = float('inf')

    def _get_config_fields(self):
        """获取Config类的所有字段名"""
        return {f.name for f in fields(Config)}

    def _create_config_from_params(self, params):
        """从参数创建Config对象，只包含Config类定义的字段"""
        config_fields = self._get_config_fields()

        # 从base_config获取基础参数，但只保留Config类定义的字段
        base_params = {k: v for k, v in self.base_config.__dict__.items()
                      if k in config_fields}

        # 更新with试验参数
        base_params.update(params)

        return Config(**base_params)

    def optimize_transformer(self, n_trials=None):
        """优化Transformer参数"""
        if n_trials is None:
            n_trials = self.base_config.n_trials_transformer

        logger.info("=" * 50)
        logger.info("开始Transformer超参数优化")
        logger.info("=" * 50)

        def objective(trial):
            try:
                # 采样超参数
                params = self._sample_transformer_params(trial)

                # 创建配置
                config = self._create_config_from_params(params)

                # 创建模型
                model = WiFiTransformerAutoencoder(
                    input_dim=self.X_train.shape[1],
                    model_dim=params['model_dim'],
                    num_heads=params['num_heads'],
                    num_layers=params['num_layers'],
                    dropout=params['dropout']
                ).to(self.device)

                # 训练
                trainer = Trainer(model, config, self.device)
                model, train_losses, val_losses = trainer.train_autoencoder(
                    self.X_train, self.X_val, epochs=50  # 减少epoch加快搜索
                )

                final_val_loss = val_losses[-1]

                # 检查NaN
                if np.isnan(final_val_loss):
                    raise TrialPruned()

                # 更新最佳参数
                if final_val_loss < self.best_val_loss:
                    self.best_val_loss = final_val_loss
                    self.best_transformer_params = params

                    # 保存模型
                    model_path = os.path.join(
                        self.base_config.model_dir,
                        f'best_transformer_trial_{trial.number}.pth'
                    )
                    torch.save(model.state_dict(), model_path)

                    # 保存参数
                    params_path = os.path.join(
                        self.base_config.output_dir,
                        'best_transformer_params.json'
                    )
                    with open(params_path, 'w') as f:
                        json.dump(params, f, indent=4)

                return final_val_loss

            except NaNLossError:
                raise TrialPruned()
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                raise TrialPruned()

        # 创建study
        study = optuna.create_study(
            direction='minimize',
            study_name='transformer_optimization'
        )

        # 运行优化
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=self.base_config.optuna_n_jobs
        )

        # 打印结果
        logger.info("\n" + "=" * 50)
        logger.info("Transformer优化结果")
        logger.info("=" * 50)

        if len(study.trials) > 0 and study.best_trial is not None:
            logger.info(f"最佳验证损失: {study.best_value:.6f}")
            logger.info("最佳参数:")
            for key, value in study.best_params.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.warning("没有成功完成的试验")

        return study

    def optimize_svr(self, n_trials=None):
        """优化SVR参数（使用已训练的Transformer）"""
        if n_trials is None:
            n_trials = self.base_config.n_trials_svr

        logger.info("=" * 50)
        logger.info("开始SVR超参数优化")
        logger.info("=" * 50)

        # 加载最佳Transformer模型
        if self.best_transformer_params is None:
            raise ValueError("请先运行Transformer优化")

        # 创建并加载Transformer
        model = WiFiTransformerAutoencoder(
            input_dim=self.X_train.shape[1],
            **self.best_transformer_params
        ).to(self.device)

        # 加载权重
        model_files = [f for f in os.listdir(self.base_config.model_dir)
                       if f.startswith('best_transformer')]
        if model_files:
            model_path = os.path.join(self.base_config.model_dir, model_files[-1])
            model.load_state_dict(torch.load(model_path))

        # 提取特征
        trainer = Trainer(model, self.base_config, self.device)
        X_train_features = trainer.extract_features(self.X_train)
        X_test_features = trainer.extract_features(self.X_test)

        # 反转坐标标准化
        y_train_original = self.preprocessor.inverse_transform_coordinates(self.y_train)
        y_test_original = self.preprocessor.inverse_transform_coordinates(self.y_test)

        def objective(trial):
            try:
                # 采样SVR参数
                params = self._sample_svr_params(trial)

                # 创建配置
                config = self._create_config_from_params(params)

                # 创建SVR
                svr = SVRRegressor(config)
                svr.create_model(params)

                # 训练
                svr.fit(X_train_features, y_train_original)

                # 预测
                y_pred = svr.predict(X_test_features)

                # 评估
                metrics, error_distances = self.evaluator.calculate_metrics(
                    y_test_original, y_pred
                )

                mean_error = metrics['mean_error_distance']

                # 更新最佳参数
                if mean_error < self.best_mean_error:
                    self.best_mean_error = mean_error
                    self.best_svr_params = params

                    # 保存模型
                    model_path = os.path.join(
                        self.base_config.model_dir,
                        f'best_svr_trial_{trial.number}.pkl'
                    )
                    svr.save(model_path)

                    # 保存参数
                    params_path = os.path.join(
                        self.base_config.output_dir,
                        'best_svr_params.json'
                    )
                    with open(params_path, 'w') as f:
                        json.dump(params, f, indent=4)

                return mean_error

            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                raise TrialPruned()

        # 创建study
        study = optuna.create_study(
            direction='minimize',
            study_name='svr_optimization'
        )

        # 运行优化
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=self.base_config.optuna_n_jobs
        )

        # 打印结果
        logger.info("\n" + "=" * 50)
        logger.info("SVR优化结果")
        logger.info("=" * 50)

        if len(study.trials) > 0 and study.best_trial is not None:
            logger.info(f"最佳平均误差: {study.best_value:.2f} 米")
            logger.info("最佳参数:")
            for key, value in study.best_params.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.warning("没有成功完成的试验")

        return study

    def _sample_transformer_params(self, trial):
        """采样Transformer参数"""
        model_dim = trial.suggest_categorical('model_dim', [16, 32, 64, 128])

        # 确保num_heads可以整除model_dim
        valid_heads = [h for h in [2, 4, 8, 16] if model_dim % h == 0]
        num_heads = trial.suggest_categorical('num_heads', valid_heads)

        params = {
            'model_dim': model_dim,
            'num_heads': num_heads,
            'num_layers': trial.suggest_int('num_layers', 2, 8),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'early_stopping_patience': trial.suggest_int('early_stopping_patience', 3, 10),
            'min_delta_ratio': trial.suggest_float('min_delta_ratio', 0.001, 0.01)
        }

        return params

    def _sample_svr_params(self, trial):
        """采样SVR参数"""
        kernel = trial.suggest_categorical('svr_kernel', ['rbf', 'poly', 'sigmoid'])

        params = {
            'svr_kernel': kernel,
            'svr_C': trial.suggest_float('svr_C', 1, 1000, log=True),
            'svr_epsilon': trial.suggest_float('svr_epsilon', 0.01, 1.0),
            'svr_gamma': trial.suggest_categorical('svr_gamma', ['scale', 'auto'])
        }

        if kernel == 'poly':
            params['svr_degree'] = trial.suggest_int('svr_degree', 2, 5)
            params['svr_coef0'] = trial.suggest_float('svr_coef0', 0.0, 1.0)

        return params

    def run_combined_optimization(self):
        """运行完整的优化流程"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 阶段1: Transformer优化
        transformer_study = self.optimize_transformer()

        # 检查是否有成功的试验
        if len(transformer_study.trials) == 0 or transformer_study.best_trial is None:
            logger.error("Transformer优化失败，没有成功完成的试验")
            return transformer_study, None

        # 保存Transformer优化结果
        transformer_results_path = os.path.join(
            self.base_config.output_dir,
            f'transformer_optimization_{timestamp}.csv'
        )
        transformer_study.trials_dataframe().to_csv(
            transformer_results_path, index=False
        )

        # 阶段2: SVR优化
        svr_study = self.optimize_svr()

        # 保存SVR优化结果
        if svr_study is not None:
            svr_results_path = os.path.join(
                self.base_config.output_dir,
                f'svr_optimization_{timestamp}.csv'
            )
            svr_study.trials_dataframe().to_csv(
                svr_results_path, index=False
            )

        # 保存最终最佳参数
        final_params = {
            'transformer': self.best_transformer_params,
            'svr': self.best_svr_params,
            'best_val_loss': self.best_val_loss if self.best_val_loss != float('inf') else None,
            'best_mean_error': self.best_mean_error if self.best_mean_error != float('inf') else None
        }

        final_params_path = os.path.join(
            self.base_config.output_dir,
            f'final_best_params_{timestamp}.json'
        )
        with open(final_params_path, 'w') as f:
            json.dump(final_params, f, indent=4)

        logger.info("\n" + "=" * 50)
        logger.info("优化完成！")
        if self.best_val_loss != float('inf'):
            logger.info(f"最佳验证损失: {self.best_val_loss:.6f}")
        if self.best_mean_error != float('inf'):
            logger.info(f"最佳平均误差: {self.best_mean_error:.2f} 米")
        logger.info(f"结果已保存到: {self.base_config.output_dir}")
        logger.info("=" * 50)

        return transformer_study, svr_study