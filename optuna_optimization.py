"""
Optuna超参数优化模块 - 集成版，使用集中配置管理
"""
import optuna
from optuna.exceptions import TrialPruned
import torch
import numpy as np
import json
import os
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from data_preprocessing import DataPreprocessor
from model_definition import WiFiTransformerAutoencoder
from training import Trainer, NaNLossError
from svr_regressor import SVRRegressor
from evaluation import Evaluator
from optuna_config import (
    get_optuna_config,
    get_transformer_search_space,
    get_svr_search_space,
    OptunaConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """集成版Optuna优化器 - 使用集中配置管理"""

    def __init__(self, base_config):
        self.base_config = base_config
        self.device = torch.device(base_config.device if torch.cuda.is_available() else 'cpu')

        # 加载Optuna配置
        self.optuna_config = get_optuna_config()
        self.transformer_search_space = get_transformer_search_space()
        self.svr_search_space = get_svr_search_space()

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

        # 创建优化记录目录
        self.optim_dir = os.path.join(base_config.output_dir, 'optimization')
        self.optim_plots_dir = os.path.join(self.optim_dir, 'plots')
        self.optim_trials_dir = os.path.join(self.optim_dir, 'trials')
        os.makedirs(self.optim_dir, exist_ok=True)
        os.makedirs(self.optim_plots_dir, exist_ok=True)
        os.makedirs(self.optim_trials_dir, exist_ok=True)

        # 试验记录
        self.trial_history = {
            'transformer': [],
            'svr': []
        }

        # 保存配置
        self._save_optimization_config()

    def _save_optimization_config(self):
        """保存优化配置到文件"""
        config_data = {
            'optuna_config': self.optuna_config.__dict__,
            'transformer_search_space': self.transformer_search_space,
            'svr_search_space': self.svr_search_space,
            'timestamp': datetime.now().isoformat()
        }

        config_path = os.path.join(self.optim_dir, 'optimization_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        logger.info(f"优化配置已保存: {config_path}")

    def _create_sampler(self):
        """根据配置创建采样器"""
        if self.optuna_config.sampler_type == 'TPE':
            return optuna.samplers.TPESampler(
                seed=self.optuna_config.seed,
                n_startup_trials=self.optuna_config.tpe_n_startup_trials,
                n_ei_candidates=self.optuna_config.tpe_n_ei_candidates
            )
        elif self.optuna_config.sampler_type == 'Random':
            return optuna.samplers.RandomSampler(seed=self.optuna_config.seed)
        elif self.optuna_config.sampler_type == 'CmaEs':
            return optuna.samplers.CmaEsSampler(seed=self.optuna_config.seed)
        else:
            return optuna.samplers.TPESampler(seed=self.optuna_config.seed)

    def _create_pruner(self):
        """根据配置创建剪枝器"""
        if not self.optuna_config.enable_pruning:
            return None

        if self.optuna_config.pruner_type == 'Median':
            return optuna.pruners.MedianPruner(
                n_startup_trials=self.optuna_config.pruner_n_startup_trials,
                n_warmup_steps=self.optuna_config.pruner_n_warmup_steps,
                interval_steps=self.optuna_config.pruner_interval_steps
            )
        elif self.optuna_config.pruner_type == 'Percentile':
            return optuna.pruners.PercentilePruner(
                percentile=50.0,
                n_startup_trials=self.optuna_config.pruner_n_startup_trials,
                n_warmup_steps=self.optuna_config.pruner_n_warmup_steps
            )
        elif self.optuna_config.pruner_type == 'Hyperband':
            return optuna.pruners.HyperbandPruner()
        else:
            return None

    def _sample_params_from_space(self, trial, search_space, model_type='transformer'):
        """根据搜索空间配置采样参数"""
        params = {}

        for param_name, config in search_space.items():
            # 检查条件参数
            if 'condition' in config:
                if model_type == 'transformer':
                    # 处理条件categorical（如num_heads依赖model_dim）
                    if config['type'] == 'conditional_categorical':
                        condition_param = config['condition']
                        if condition_param in params:
                            choices = config['choices_map'].get(
                                params[condition_param],
                                config['choices_map'][list(config['choices_map'].keys())[0]]
                            )
                            # 确保选择有效的头数
                            valid_choices = [h for h in choices if params[condition_param] % h == 0]
                            params[param_name] = trial.suggest_categorical(
                                param_name, valid_choices
                            )
                        continue
                else:
                    # SVR的条件参数
                    if config['condition'] == "svr_kernel == 'poly'" and params.get('svr_kernel') != 'poly':
                        continue
                    if config['condition'] == "svr_kernel in ['poly', 'sigmoid']" and \
                       params.get('svr_kernel') not in ['poly', 'sigmoid']:
                        continue

            # 根据类型采样
            if config['type'] == 'fixed':
                params[param_name] = config['value']
            elif config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, config['choices']
                )
            elif config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, config['low'], config['high']
                )
            elif config['type'] == 'float':
                if config.get('log', False):
                    params[param_name] = trial.suggest_float(
                        param_name, config['low'], config['high'], log=True
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name, config['low'], config['high']
                    )

        return params

    def optimize_transformer(self, n_trials=None):
        """优化Transformer参数（集成版）"""
        if n_trials is None:
            n_trials = self.optuna_config.n_trials_transformer

        logger.info("=" * 50)
        logger.info("开始Transformer超参数优化")
        logger.info(f"配置: {self.optuna_config.sampler_type}采样器, {n_trials}次试验")
        logger.info("=" * 50)

        # 记录早停状态
        no_improvement_count = 0
        best_trial_number = -1

        def objective(trial):
            nonlocal no_improvement_count, best_trial_number
            trial_start_time = datetime.now()

            try:
                # 采样超参数
                params = self._sample_params_from_space(trial, self.transformer_search_space, 'transformer')

                # 记录试验开始
                logger.info(f"\n试验 {trial.number + 1}/{n_trials}")
                logger.info(f"参数: {params}")

                # 创建配置 - 只更新相关参数
                config_dict = {}
                for key, value in self.base_config.__dict__.items():
                    if not key.endswith('_dir') and not key.startswith('_'):
                        config_dict[key] = value

                # 更新采样的参数
                config_dict.update(params)
                config = Config(**config_dict)

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

                # 获取训练轮数
                epochs = params.get('epochs', 50)
                if isinstance(epochs, dict):
                    epochs = epochs.get('value', 50)

                model, train_losses, val_losses = trainer.train_autoencoder(
                    self.X_train, self.X_val, epochs=epochs
                )

                final_val_loss = val_losses[-1]
                best_epoch_val_loss = min(val_losses)

                # 计算额外指标
                convergence_speed = self._calculate_convergence_speed(val_losses)
                stability = np.std(val_losses[-10:]) if len(val_losses) >= 10 else np.std(val_losses)

                # 检查NaN
                if np.isnan(final_val_loss):
                    raise TrialPruned()

                # 记录试验结果
                trial_result = {
                    'trial_number': trial.number,
                    'params': params,
                    'final_val_loss': final_val_loss,
                    'best_val_loss': best_epoch_val_loss,
                    'convergence_speed': convergence_speed,
                    'stability': stability,
                    'epochs_trained': len(train_losses),
                    'duration_seconds': (datetime.now() - trial_start_time).total_seconds(),
                    'timestamp': datetime.now().isoformat()
                }
                self.trial_history['transformer'].append(trial_result)

                # 更新最佳参数
                improved = False
                if final_val_loss < self.best_val_loss:
                    improvement_ratio = (self.best_val_loss - final_val_loss) / self.best_val_loss if self.best_val_loss != float('inf') else 1.0

                    if improvement_ratio >= self.optuna_config.min_improvement_ratio or self.best_val_loss == float('inf'):
                        improved = True
                        self.best_val_loss = final_val_loss
                        self.best_transformer_params = params
                        best_trial_number = trial.number
                        no_improvement_count = 0

                        # 保存模型和详细信息
                        model_path = os.path.join(
                            self.optim_trials_dir,
                            f'best_transformer_trial_{trial.number}.pth'
                        )
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'params': params,
                            'train_losses': train_losses,
                            'val_losses': val_losses,
                            'trial_number': trial.number
                        }, model_path)

                        # 保存参数
                        params_path = os.path.join(
                            self.optim_dir,
                            'best_transformer_params.json'
                        )
                        with open(params_path, 'w') as f:
                            json.dump(params, f, indent=4)

                        logger.info(f"✓ 新的最佳验证损失: {final_val_loss:.6f} (改善: {improvement_ratio*100:.2f}%)")

                if not improved:
                    no_improvement_count += 1
                    logger.info(f"无改善 ({no_improvement_count}/{self.optuna_config.early_stopping_patience})")

                # 定期保存试验历史
                if (trial.number + 1) % self.optuna_config.save_interval == 0:
                    self._save_trial_history('transformer')

                # 检查早停
                if self.optuna_config.enable_early_stopping and \
                   no_improvement_count >= self.optuna_config.early_stopping_patience:
                    logger.info(f"早停触发: {no_improvement_count}次试验无改善")
                    trial.study.stop()

                return final_val_loss

            except NaNLossError:
                logger.warning(f"试验 {trial.number} 出现NaN损失")
                raise TrialPruned()
            except Exception as e:
                logger.error(f"试验 {trial.number} 失败: {e}")
                raise TrialPruned()

        # 创建study
        study = optuna.create_study(
            direction=self.optuna_config.direction,
            study_name='transformer_optimization',
            sampler=self._create_sampler(),
            pruner=self._create_pruner()
        )

        # 运行优化
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=self.optuna_config.n_jobs,
            callbacks=[self._optimization_callback]
        )

        # 保存最终结果
        self._save_trial_history('transformer')
        self._save_optimization_results(study, 'transformer')
        self._plot_optimization_history(study, 'transformer')

        # 打印结果
        logger.info("\n" + "=" * 50)
        logger.info("Transformer优化结果")
        logger.info("=" * 50)
        logger.info(f"完成试验数: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}/{n_trials}")
        logger.info(f"最佳验证损失: {study.best_value:.6f}")
        logger.info(f"最佳试验: #{study.best_trial.number}")
        logger.info("最佳参数:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")

        return study

    def optimize_svr(self, n_trials=None):
        """优化SVR参数（集成版）"""
        if n_trials is None:
            n_trials = self.optuna_config.n_trials_svr

        logger.info("=" * 50)
        logger.info("开始SVR超参数优化")
        logger.info(f"配置: {self.optuna_config.sampler_type}采样器, {n_trials}次试验")
        logger.info("=" * 50)

        # 加载最佳Transformer模型
        if self.best_transformer_params is None:
            raise ValueError("请先运行Transformer优化")

        # 创建并加载Transformer
        model = WiFiTransformerAutoencoder(
            input_dim=self.X_train.shape[1],
            model_dim=self.best_transformer_params['model_dim'],
            num_heads=self.best_transformer_params['num_heads'],
            num_layers=self.best_transformer_params['num_layers'],
            dropout=self.best_transformer_params['dropout']
        ).to(self.device)

        # 加载权重
        model_files = [f for f in os.listdir(self.optim_trials_dir)
                       if f.startswith('best_transformer')]
        if model_files:
            model_path = os.path.join(self.optim_trials_dir, model_files[-1])
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])

        # 提取特征
        trainer = Trainer(model, self.base_config, self.device)
        X_train_features = trainer.extract_features(self.X_train)
        X_val_features = trainer.extract_features(self.X_val)
        X_test_features = trainer.extract_features(self.X_test)

        # 反转坐标标准化
        y_train_original = self.preprocessor.inverse_transform_coordinates(self.y_train)
        y_val_original = self.preprocessor.inverse_transform_coordinates(self.y_val)
        y_test_original = self.preprocessor.inverse_transform_coordinates(self.y_test)

        # 记录早停状态
        no_improvement_count = 0
        best_trial_number = -1

        def objective(trial):
            nonlocal no_improvement_count, best_trial_number
            trial_start_time = datetime.now()

            try:
                # 采样SVR参数
                params = self._sample_params_from_space(trial, self.svr_search_space, 'svr')

                logger.info(f"\n试验 {trial.number + 1}/{n_trials}")
                logger.info(f"参数: {params}")

                # 创建配置
                config_dict = {}
                for key, value in self.base_config.__dict__.items():
                    if not key.endswith('_dir') and not key.startswith('_'):
                        config_dict[key] = value

                config_dict.update(params)
                config = Config(**config_dict)

                # 创建SVR
                svr = SVRRegressor(config)
                svr.create_model(params)

                # 训练
                svr.fit(X_train_features, y_train_original)

                # 在验证集和测试集上预测
                y_val_pred = svr.predict(X_val_features)
                y_test_pred = svr.predict(X_test_features)

                # 评估
                val_metrics, val_error_distances = self.evaluator.calculate_metrics(
                    y_val_original, y_val_pred, save_detailed=False
                )
                test_metrics, test_error_distances = self.evaluator.calculate_metrics(
                    y_test_original, y_test_pred, save_detailed=False
                )

                mean_error_val = val_metrics['mean_error_distance']
                mean_error_test = test_metrics['mean_error_distance']

                # 记录试验结果
                trial_result = {
                    'trial_number': trial.number,
                    'params': params,
                    'val_mean_error': mean_error_val,
                    'test_mean_error': mean_error_test,
                    'val_median_error': val_metrics['median_error_distance'],
                    'test_median_error': test_metrics['median_error_distance'],
                    'val_rmse': val_metrics['rmse'],
                    'test_rmse': test_metrics['rmse'],
                    'val_r2': val_metrics['r2'],
                    'test_r2': test_metrics['r2'],
                    'duration_seconds': (datetime.now() - trial_start_time).total_seconds(),
                    'timestamp': datetime.now().isoformat()
                }
                self.trial_history['svr'].append(trial_result)

                # 更新最佳参数
                improved = False
                if mean_error_val < self.best_mean_error:
                    improvement_ratio = (self.best_mean_error - mean_error_val) / self.best_mean_error if self.best_mean_error != float('inf') else 1.0

                    if improvement_ratio >= self.optuna_config.min_improvement_ratio or self.best_mean_error == float('inf'):
                        improved = True
                        self.best_mean_error = mean_error_val
                        self.best_svr_params = params
                        best_trial_number = trial.number
                        no_improvement_count = 0

                        # 保存模型
                        model_path = os.path.join(
                            self.optim_trials_dir,
                            f'best_svr_trial_{trial.number}.pkl'
                        )
                        svr.save(model_path)

                        # 保存参数和指标
                        best_info = {
                            'params': params,
                            'val_metrics': val_metrics,
                            'test_metrics': test_metrics,
                            'trial_number': trial.number
                        }
                        params_path = os.path.join(
                            self.optim_dir,
                            'best_svr_params.json'
                        )
                        with open(params_path, 'w') as f:
                            json.dump(best_info, f, indent=4)

                        logger.info(f"✓ 新的最佳验证误差: {mean_error_val:.2f} 米 (改善: {improvement_ratio*100:.2f}%)")
                        logger.info(f"  对应测试误差: {mean_error_test:.2f} 米")

                if not improved:
                    no_improvement_count += 1
                    logger.info(f"无改善 ({no_improvement_count}/{self.optuna_config.early_stopping_patience})")

                # 定期保存试验历史
                if (trial.number + 1) % self.optuna_config.save_interval == 0:
                    self._save_trial_history('svr')

                # 检查早停
                if self.optuna_config.enable_early_stopping and \
                   no_improvement_count >= self.optuna_config.early_stopping_patience:
                    logger.info(f"早停触发: {no_improvement_count}次试验无改善")
                    trial.study.stop()

                return mean_error_val

            except Exception as e:
                logger.error(f"试验 {trial.number} 失败: {e}")
                raise TrialPruned()

        # 创建study
        study = optuna.create_study(
            direction=self.optuna_config.direction,
            study_name='svr_optimization',
            sampler=self._create_sampler(),
            pruner=self._create_pruner()
        )

        # 运行优化
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=self.optuna_config.n_jobs,
            callbacks=[self._optimization_callback]
        )

        # 保存最终结果
        self._save_trial_history('svr')
        self._save_optimization_results(study, 'svr')
        self._plot_optimization_history(study, 'svr')

        # 打印结果
        logger.info("\n" + "=" * 50)
        logger.info("SVR优化结果")
        logger.info("=" * 50)
        logger.info(f"完成试验数: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}/{n_trials}")
        logger.info(f"最佳平均误差: {study.best_value:.2f} 米")
        logger.info(f"最佳试验: #{study.best_trial.number}")
        logger.info("最佳参数:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")

        return study

    def _calculate_convergence_speed(self, losses):
        """计算收敛速度"""
        if len(losses) < 2:
            return 0

        improvements = []
        for i in range(1, len(losses)):
            if losses[i-1] > 0:
                improvement = (losses[i-1] - losses[i]) / losses[i-1]
                improvements.append(improvement)

        return np.mean(improvements) if improvements else 0

    def _optimization_callback(self, study, trial):
        """优化回调函数"""
        if trial.number % self.optuna_config.report_interval == 0:
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed_trials:
                logger.info(f"进度: {len(completed_trials)}次完成, 当前最佳值: {study.best_value:.6f}")
            else:
                logger.info(f"进度: {trial.number}次尝试, 尚无成功完成的试验")

    def _save_trial_history(self, model_type):
        """保存试验历史"""
        if not self.trial_history[model_type]:
            return

        # 保存为CSV
        df = pd.DataFrame(self.trial_history[model_type])
        csv_path = os.path.join(self.optim_dir, f'{model_type}_trials_history.csv')
        df.to_csv(csv_path, index=False)

        # 保存为JSON
        json_path = os.path.join(self.optim_dir, f'{model_type}_trials_history.json')
        with open(json_path, 'w') as f:
            json.dump(self.trial_history[model_type], f, indent=4, default=str)

    def _save_optimization_results(self, study, model_type):
        """保存优化结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存试验数据框
        df = study.trials_dataframe()
        csv_path = os.path.join(self.optim_dir, f'{model_type}_optimization_{timestamp}.csv')
        df.to_csv(csv_path, index=False)

        # 保存重要性分析
        if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) > 10:
            try:
                importance = optuna.importance.get_param_importances(study)
                importance_path = os.path.join(self.optim_dir, f'{model_type}_param_importance.json')
                with open(importance_path, 'w') as f:
                    json.dump(importance, f, indent=4)
                logger.info(f"参数重要性分析已保存: {importance_path}")
            except Exception as e:
                logger.warning(f"无法计算参数重要性: {e}")

        # 保存最佳试验详情
        if study.best_trial:
            best_trial_info = {
                'number': study.best_trial.number,
                'value': study.best_value,
                'params': study.best_params,
                'datetime_start': study.best_trial.datetime_start.isoformat() if study.best_trial.datetime_start else None,
                'datetime_complete': study.best_trial.datetime_complete.isoformat() if study.best_trial.datetime_complete else None,
                'duration': study.best_trial.duration.total_seconds() if study.best_trial.duration else None,
                'state': str(study.best_trial.state)
            }

            best_trial_path = os.path.join(self.optim_dir, f'{model_type}_best_trial.json')
            with open(best_trial_path, 'w') as f:
                json.dump(best_trial_info, f, indent=4)

    def _plot_optimization_history(self, study, model_type):
        """绘制优化历史图表"""
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            logger.warning(f"没有完成的试验，无法绘制{model_type}优化图表")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. 优化历史
        ax = axes[0, 0]
        trial_numbers = [t.number for t in completed_trials]
        trial_values = [t.value for t in completed_trials]

        ax.plot(trial_numbers, trial_values, 'b-', alpha=0.5, label='Trial Values')

        # 添加最佳值线
        best_values = []
        current_best = float('inf')
        for v in trial_values:
            if v < current_best:
                current_best = v
            best_values.append(current_best)
        ax.plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best So Far')

        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Objective Value')
        ax.set_title('Optimization History')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 参数分布
        params_df = study.trials_dataframe()
        param_cols = [col for col in params_df.columns if col.startswith('params_')]

        for i, param_col in enumerate(param_cols[:3]):
            row = (i + 1) // 3
            col = (i + 1) % 3
            ax = axes[row, col]

            param_name = param_col.replace('params_', '')

            # 只绘制完成的试验
            completed_mask = params_df['state'] == 'COMPLETE'
            if completed_mask.sum() > 0:
                ax.scatter(params_df.loc[completed_mask, param_col],
                          params_df.loc[completed_mask, 'value'],
                          alpha=0.5)
                ax.set_xlabel(param_name)
                ax.set_ylabel('Objective Value')
                ax.set_title(f'{param_name} vs Objective')
                ax.grid(True, alpha=0.3)

        # 3. 试验时长分布
        ax = axes[1, 2]
        if model_type in self.trial_history and self.trial_history[model_type]:
            durations = [t['duration_seconds'] for t in self.trial_history[model_type]]
            if durations:
                ax.hist(durations, bins=min(20, len(durations)), edgecolor='black', alpha=0.7)
                ax.set_xlabel('Duration (seconds)')
                ax.set_ylabel('Count')
                ax.set_title('Trial Duration Distribution')
                ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'{model_type.upper()} Optimization Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_path = os.path.join(self.optim_plots_dir, f'{model_type}_optimization_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"优化分析图已保存: {plot_path}")

    def run_combined_optimization(self):
        """运行完整的优化流程（集成版）"""
        optimization_start_time = datetime.now()
        timestamp = optimization_start_time.strftime('%Y%m%d_%H%M%S')

        # 创建优化摘要
        optimization_summary = {
            'start_time': optimization_start_time.isoformat(),
            'config': self.base_config.__dict__,
            'optuna_config': self.optuna_config.__dict__,
            'transformer_search_space': self.transformer_search_space,
            'svr_search_space': self.svr_search_space,
            'stages': {}
        }

        # 阶段1: Transformer优化
        logger.info("\n" + "="*60)
        logger.info("阶段 1/2: Transformer优化")
        logger.info("="*60)
        stage_start = datetime.now()
        transformer_study = self.optimize_transformer()

        optimization_summary['stages']['transformer'] = {
            'duration_seconds': (datetime.now() - stage_start).total_seconds(),
            'n_trials': len(transformer_study.trials),
            'n_completed': len([t for t in transformer_study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'best_value': transformer_study.best_value if transformer_study.best_trial else None,
            'best_params': transformer_study.best_params if transformer_study.best_trial else None,
            'best_trial_number': transformer_study.best_trial.number if transformer_study.best_trial else None
        }

        # 阶段2: SVR优化
        logger.info("\n" + "="*60)
        logger.info("阶段 2/2: SVR优化")
        logger.info("="*60)
        stage_start = datetime.now()
        svr_study = self.optimize_svr()

        optimization_summary['stages']['svr'] = {
            'duration_seconds': (datetime.now() - stage_start).total_seconds(),
            'n_trials': len(svr_study.trials),
            'n_completed': len([t for t in svr_study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'best_value': svr_study.best_value if svr_study.best_trial else None,
            'best_params': svr_study.best_params if svr_study.best_trial else None,
            'best_trial_number': svr_study.best_trial.number if svr_study.best_trial else None
        }

        # 保存最终结果
        optimization_summary['end_time'] = datetime.now().isoformat()
        optimization_summary['total_duration_seconds'] = (datetime.now() - optimization_start_time).total_seconds()
        optimization_summary['final_results'] = {
            'transformer': {
                'params': self.best_transformer_params,
                'best_val_loss': self.best_val_loss
            },
            'svr': {
                'params': self.best_svr_params,
                'best_mean_error': self.best_mean_error
            }
        }

        # 保存优化摘要
        summary_path = os.path.join(self.optim_dir, f'optimization_summary_{timestamp}.json')
        with open(summary_path, 'w') as f:
            json.dump(optimization_summary, f, indent=4, default=str)

        # 创建最终报告
        self._create_optimization_report(transformer_study, svr_study, optimization_summary)

        logger.info("\n" + "=" * 60)
        logger.info("优化完成！")
        logger.info(f"最佳验证损失: {self.best_val_loss:.6f}")
        logger.info(f"最佳平均误差: {self.best_mean_error:.2f} 米")
        logger.info(f"总用时: {optimization_summary['total_duration_seconds']/3600:.2f} 小时")
        logger.info(f"结果已保存到: {self.optim_dir}")
        logger.info("=" * 60)

        return transformer_study, svr_study

    def _create_optimization_report(self, transformer_study, svr_study, summary):
        """创建优化报告"""
        report_path = os.path.join(self.optim_dir, 'optimization_report.md')

        with open(report_path, 'w') as f:
            f.write("# Hyperparameter Optimization Report\n\n")
            f.write(f"**Date**: {summary['start_time'][:10]}\n")
            f.write(f"**Total Duration**: {summary['total_duration_seconds']/3600:.2f} hours\n")
            f.write(f"**Optimization Strategy**: {self.optuna_config.sampler_type} Sampler\n\n")

            f.write("## Configuration\n\n")
            f.write(f"- **Sampler**: {self.optuna_config.sampler_type}\n")
            f.write(f"- **Pruning**: {'Enabled' if self.optuna_config.enable_pruning else 'Disabled'}\n")
            f.write(f"- **Early Stopping**: {'Enabled' if self.optuna_config.enable_early_stopping else 'Disabled'}\n")
            f.write(f"- **Parallel Jobs**: {self.optuna_config.n_jobs}\n\n")

            f.write("## Transformer Optimization\n\n")
            if transformer_study.best_trial:
                f.write(f"- **Trials**: {len(transformer_study.trials)} (Completed: {summary['stages']['transformer']['n_completed']})\n")
                f.write(f"- **Best Validation Loss**: {transformer_study.best_value:.6f}\n")
                f.write(f"- **Best Trial**: #{transformer_study.best_trial.number}\n")
                f.write(f"- **Duration**: {summary['stages']['transformer']['duration_seconds']/60:.1f} minutes\n\n")

                f.write("### Best Parameters:\n")
                for k, v in self.best_transformer_params.items():
                    f.write(f"- {k}: {v}\n")
            else:
                f.write("No successful trials completed.\n")

            f.write("\n## SVR Optimization\n\n")
            if svr_study.best_trial:
                f.write(f"- **Trials**: {len(svr_study.trials)} (Completed: {summary['stages']['svr']['n_completed']})\n")
                f.write(f"- **Best Mean Error**: {svr_study.best_value:.2f} m\n")
                f.write(f"- **Best Trial**: #{svr_study.best_trial.number}\n")
                f.write(f"- **Duration**: {summary['stages']['svr']['duration_seconds']/60:.1f} minutes\n\n")

                f.write("### Best Parameters:\n")
                for k, v in self.best_svr_params.items():
                    f.write(f"- {k}: {v}\n")
            else:
                f.write("No successful trials completed.\n")

            f.write("\n## Files Generated\n\n")
            f.write("- Configuration: `optimization_config.json`\n")
            f.write("- Trial histories: `*_trials_history.csv`\n")
            f.write("- Optimization plots: `plots/`\n")
            f.write("- Best models: `trials/best_*.pth/pkl`\n")
            f.write("- Parameter importance: `*_param_importance.json`\n")
            f.write("- Summary: `optimization_summary_*.json`\n")

        logger.info(f"优化报告已创建: {report_path}")