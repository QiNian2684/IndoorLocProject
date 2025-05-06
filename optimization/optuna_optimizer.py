import numpy as np
import optuna
import os
import json
import pandas as pd
from sklearn.model_selection import KFold
from models.hybrid_model import SVRTransformerHybrid
from evaluation.cross_validation import SpatialCrossValidator
from models.floor_model import FloorClassifier  # 添加这一行在optuna_optimizer.py顶部

class OptunaSVRTransformerOptimizer:
    """使用Optuna优化SVR+Transformer超参数"""

    def __init__(self, search_space, n_trials=100, timeout=None, n_jobs=1,
                 direction='minimize', cv=5, cv_method='spatial'):
        """
        初始化Optuna优化器

        参数:
            search_space (dict): 超参数搜索空间定义
            n_trials (int): 要运行的试验次数
            timeout (int, optional): 优化超时（秒）
            n_jobs (int): 并行作业数
            direction (str): 优化方向 ('minimize' 或 'maximize')
            cv (int): 交叉验证折数
            cv_method (str): 交叉验证方法 ('spatial' 或 'random')
        """
        self.search_space = search_space
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.direction = direction
        self.cv = cv
        self.cv_method = cv_method
        self.study = None
        self.best_trial = None
        self.best_params = None
        self.trial_history = []

    def optimize(self, X, y, X_floor=None, custom_objective=None):
        """
        运行超参数优化

        参数:
            X: 特征数据
            y: 目标值（坐标）
            X_floor: 楼层标签数据（可选）
            custom_objective: 可选的自定义目标函数

        返回:
            dict: 最佳参数
        """
        # 创建目标函数
        if custom_objective:
            objective = custom_objective
        else:
            def objective(trial):
                # 记录开始时间
                import time
                start_time = time.time()

                # 从试验中提取参数
                params = self._suggest_params(trial)

                # 创建早停配置
                early_stopping_config = {
                    'enabled': True,
                    'patience': params.get('patience', 15),
                    'min_delta': params.get('min_delta', 0.0001),
                    'verbose': False,  # 在优化期间保持简洁的输出
                    'mode': 'min'
                }

                # 记录试验信息
                trial_info = {
                    'trial_number': trial.number,
                    'params': params,
                    'datetime': pd.Timestamp.now().isoformat()
                }

                print(f"\n==== 试验 {trial.number + 1}/{self.n_trials} ====")
                print(f"参数: {params}")

                # 创建楼层分类器
                floor_classifier = FloorClassifier(
                    classifier_type=params.get('floor_classifier_type', 'random_forest'),
                    n_estimators=params.get('floor_n_estimators', 100),
                    max_depth=params.get('floor_max_depth', None),
                    min_samples_split=params.get('floor_min_samples_split', 2),
                    min_samples_leaf=params.get('floor_min_samples_leaf', 1)
                )

                # 创建主模型
                model = SVRTransformerHybrid(
                    integration_type=params['integration_type'],
                    transformer_params={k: v for k, v in params.items()
                                        if k in ['d_model', 'nhead', 'num_layers', 'dim_feedforward']},
                    svr_params={k: v for k, v in params.items()
                                if k in ['kernel', 'C', 'epsilon', 'gamma', 'degree']},
                    early_stopping_config=early_stopping_config
                )

                # 设置用于Optuna修剪的回调
                def optuna_pruning_callback(epoch, train_loss, val_loss, lr, params):
                    if val_loss is not None:
                        # 报告中间值以支持修剪
                        trial.report(val_loss, epoch)
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()

                # 执行交叉验证
                if self.cv_method == 'spatial':
                    cv_splitter = SpatialCrossValidator(n_splits=self.cv)
                else:
                    cv_splitter = KFold(n_splits=self.cv, shuffle=True, random_state=42)

                cv_coords_scores = []  # 坐标预测误差
                cv_floor_scores = []  # 楼层预测准确率
                fold_info = []

                print(f"执行{self.cv}折交叉验证...")

                for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    # 获取楼层标签 - 需要传入额外参数
                    if hasattr(X, 'iloc'):
                        y_train_floor = X.iloc[train_idx]['FLOOR'].values
                        y_test_floor = X.iloc[test_idx]['FLOOR'].values
                    else:
                        # 使用传入的楼层数据
                        y_train_floor = X_floor[train_idx] if X_floor is not None else None
                        y_test_floor = X_floor[test_idx] if X_floor is not None else None

                    print(f"  训练楼层分类器 折 {fold_idx + 1}/{self.cv}...")
                    floor_classifier.fit(X_train, y_train_floor)

                    print(f"  训练坐标回归模型 折 {fold_idx + 1}/{self.cv}...")
                    model.fit(X_train, y_train, X_test, y_test, callback=optuna_pruning_callback)

                    print(f"  评估折 {fold_idx + 1}/{self.cv}...")
                    # 预测坐标
                    y_pred = model.predict(X_test)

                    # 预测楼层
                    floor_pred = floor_classifier.predict(X_test)

                    # 计算坐标错误（欧氏距离）
                    coords_errors = np.sqrt(np.sum((y_test - y_pred) ** 2, axis=1))
                    mean_coords_error = np.mean(coords_errors)

                    # 计算楼层准确率
                    floor_accuracy = np.mean(y_test_floor == floor_pred)

                    # 综合分数 - 坐标误差和楼层准确率的加权组合
                    # 这里可以调整权重以平衡两个目标的重要性
                    # 例如：综合分数 = 坐标误差 - 楼层准确率 * 权重
                    # 权重越大，楼层准确率越重要
                    floor_weight = 5.0  # 调整这个值来控制楼层分类的重要性
                    combined_score = mean_coords_error - floor_accuracy * floor_weight

                    cv_coords_scores.append(mean_coords_error)
                    cv_floor_scores.append(floor_accuracy)

                    # 记录每折的信息
                    fold_info.append({
                        'fold': fold_idx,
                        'train_samples': len(X_train),
                        'test_samples': len(X_test),
                        'mean_coords_error': mean_coords_error,
                        'floor_accuracy': floor_accuracy,
                        'combined_score': combined_score
                    })

                    print(
                        f"  折 {fold_idx + 1} 平均坐标误差: {mean_coords_error:.4f}m, 楼层准确率: {floor_accuracy:.4f}")

                    # 报告中间值以支持修剪
                    trial.report(combined_score, fold_idx)
                    if trial.should_prune():
                        print(f"  提前修剪试验 {trial.number}...")
                        raise optuna.exceptions.TrialPruned()

                # 计算平均分数和运行时间
                mean_cv_coords_score = np.mean(cv_coords_scores)
                mean_cv_floor_score = np.mean(cv_floor_scores)
                # 计算综合分数
                mean_combined_score = mean_cv_coords_score - mean_cv_floor_score * floor_weight

                elapsed_time = time.time() - start_time

                # 更新试验信息
                trial_info.update({
                    'mean_cv_coords_score': mean_cv_coords_score,
                    'mean_cv_floor_score': mean_cv_floor_score,
                    'mean_combined_score': mean_combined_score,
                    'fold_scores': fold_info,
                    'elapsed_time': elapsed_time
                })

                self.trial_history.append(trial_info)

                print(f"试验 {trial.number + 1} 完成，平均坐标误差: {mean_cv_coords_score:.4f}m，"
                      f"楼层准确率: {mean_cv_floor_score:.4f}，"
                      f"综合分数: {mean_combined_score:.4f}，"
                      f"用时: {elapsed_time:.2f}秒\n")

                # 返回综合分数作为优化目标
                return mean_combined_score

        # 创建和运行研究
        self.study = optuna.create_study(
            direction=self.direction,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )

        print(f"\n开始Optuna优化 ({self.n_trials}次试验)...\n")

        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs
        )

        # 保存最佳结果
        self.best_trial = self.study.best_trial
        self.best_params = self.best_trial.params

        print(f"\n优化完成！")
        print(f"最佳参数: {self.best_params}")
        print(f"最佳误差: {self.best_trial.value:.4f}m\n")

        return self.best_params

    def _suggest_params(self, trial):
        """根据搜索空间提示参数"""
        params = {}

        # 集成类型
        if 'integration_type' in self.search_space:
            params['integration_type'] = trial.suggest_categorical(
                'integration_type', self.search_space['integration_type'])

        # Transformer参数
        if 'd_model' in self.search_space:
            params['d_model'] = trial.suggest_categorical(
                'd_model', self.search_space['d_model'])

        if 'nhead' in self.search_space:
            # 确保nhead是d_model的约数
            if 'd_model' in params:
                d_model = params['d_model']
                # 从搜索空间中过滤出能整除d_model的nhead值
                valid_nheads = [h for h in self.search_space['nhead'] if d_model % h == 0]
                if valid_nheads:  # 如果存在有效的nhead值
                    params['nhead'] = trial.suggest_categorical('nhead', valid_nheads)
                else:
                    # 如果没有有效的nhead值，使用d_model的一个合适的约数
                    divisors = [i for i in range(1, min(17, d_model + 1)) if d_model % i == 0]
                    if divisors:
                        params['nhead'] = trial.suggest_categorical('nhead', divisors)
                    else:
                        # 确保至少有一个值
                        params['nhead'] = 1
            else:
                params['nhead'] = trial.suggest_categorical(
                    'nhead', self.search_space['nhead'])

        if 'num_layers' in self.search_space:
            params['num_layers'] = trial.suggest_int(
                'num_layers', self.search_space['num_layers'][0], self.search_space['num_layers'][1])

        if 'dim_feedforward' in self.search_space:
            params['dim_feedforward'] = trial.suggest_int(
                'dim_feedforward', self.search_space['dim_feedforward'][0], self.search_space['dim_feedforward'][1])

        # SVR参数
        if 'kernel' in self.search_space:
            params['kernel'] = trial.suggest_categorical(
                'kernel', self.search_space['kernel'])

        if 'C' in self.search_space:
            params['C'] = trial.suggest_float(
                'C', self.search_space['C'][0], self.search_space['C'][1], log=True)

        if 'epsilon' in self.search_space:
            params['epsilon'] = trial.suggest_float(
                'epsilon', self.search_space['epsilon'][0], self.search_space['epsilon'][1], log=True)

        # 条件参数
        if params.get('kernel') in ['rbf', 'poly'] and 'gamma' in self.search_space:
            params['gamma'] = trial.suggest_categorical(
                'gamma', self.search_space['gamma'])

        if params.get('kernel') == 'poly' and 'degree' in self.search_space:
            params['degree'] = trial.suggest_int(
                'degree', self.search_space['degree'][0], self.search_space['degree'][1])

        # 早停参数
        if 'early_stopping' in self.search_space and self.search_space['early_stopping'].get('optimize', False):
            es_config = self.search_space['early_stopping']

            if 'patience' in es_config:
                params['patience'] = trial.suggest_int(
                    'patience', es_config['patience'][0], es_config['patience'][1])

            if 'min_delta' in es_config:
                params['min_delta'] = trial.suggest_float(
                    'min_delta', es_config['min_delta'][0], es_config['min_delta'][1], log=True)

        # 楼层分类器参数
        if 'floor_classifier_type' in self.search_space:
            params['floor_classifier_type'] = trial.suggest_categorical(
                'floor_classifier_type', self.search_space['floor_classifier_type'])

        if 'floor_n_estimators' in self.search_space:
            params['floor_n_estimators'] = trial.suggest_int(
                'floor_n_estimators', self.search_space['floor_n_estimators'][0],
                self.search_space['floor_n_estimators'][1])

        if 'floor_max_depth' in self.search_space:
            # 特殊处理None值
            max_depth_options = self.search_space['floor_max_depth']
            if None in max_depth_options:
                # 先去除None，然后将其作为特殊选项
                non_none_options = [opt for opt in max_depth_options if opt is not None]
                if trial.suggest_categorical('floor_use_max_depth', [True, False]):
                    params['floor_max_depth'] = trial.suggest_int(
                        'floor_max_depth_value', min(non_none_options), max(non_none_options))
                else:
                    params['floor_max_depth'] = None
            else:
                params['floor_max_depth'] = trial.suggest_int(
                    'floor_max_depth', min(max_depth_options), max(max_depth_options))

        if 'floor_min_samples_split' in self.search_space:
            params['floor_min_samples_split'] = trial.suggest_int(
                'floor_min_samples_split', self.search_space['floor_min_samples_split'][0],
                self.search_space['floor_min_samples_split'][1])

        if 'floor_min_samples_leaf' in self.search_space:
            params['floor_min_samples_leaf'] = trial.suggest_int(
                'floor_min_samples_leaf', self.search_space['floor_min_samples_leaf'][0],
                self.search_space['floor_min_samples_leaf'][1])

        return params

    def save_optimization_results(self, save_dir):
        """
        保存优化结果和可视化

        参数:
            save_dir: 保存目录
        """
        if self.study is None:
            raise RuntimeError("必须先运行优化")

        # 创建保存目录
        optuna_dir = os.path.join(save_dir, 'optuna_results')

        # 使用时间戳创建试验结果目录
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trials_dir = os.path.join(optuna_dir, 'trials', f'run_{timestamp}')
        os.makedirs(trials_dir, exist_ok=True)

        # 创建可视化目录
        viz_dir = os.path.join(optuna_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        # 创建最佳参数目录
        best_params_dir = os.path.join(optuna_dir, 'best_params')
        os.makedirs(best_params_dir, exist_ok=True)

        # 保存所有试验数据
        trials_df = self.get_trials_dataframe()
        trials_df.to_csv(os.path.join(trials_dir, f'trials_{timestamp}.csv'), index=False)

        # 保存最佳参数
        with open(os.path.join(best_params_dir, f'best_params_{timestamp}.json'), 'w') as f:
            json.dump(self.best_params, f, indent=4)

        # 保存试验历史
        with open(os.path.join(trials_dir, f'trial_history_{timestamp}.json'), 'w') as f:
            # 使用定制的JSON编码器处理NumPy和日期时间类型
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                        np.int16, np.int32, np.int64, np.uint8, np.uint16,
                                        np.uint32, np.uint64)):
                        return int(obj)
                    if isinstance(obj, (np.float_, np.float16, np.float32,
                                        np.float64)):
                        return float(obj)
                    return super(NumpyEncoder, self).default(obj)

            json.dump(self.trial_history, f, cls=NumpyEncoder, indent=4)

        # 保存可视化，使用时间戳避免覆盖
        try:
            # 优化历史
            history_fig = self.plot_optimization_history()
            history_fig_path = os.path.join(viz_dir, f'optimization_history_{timestamp}.png')
            history_fig.write_image(history_fig_path)

            # 参数重要性
            importance_fig = self.plot_param_importances()
            importance_fig_path = os.path.join(viz_dir, f'param_importances_{timestamp}.png')
            importance_fig.write_image(importance_fig_path)

            # 参数关系
            coord_fig = self.plot_parallel_coordinate()
            coord_fig_path = os.path.join(viz_dir, f'parallel_coordinate_{timestamp}.png')
            coord_fig.write_image(coord_fig_path)

            # 尝试保存参数分布图
            param_dist_dir = os.path.join(viz_dir, f'param_distributions_{timestamp}')
            os.makedirs(param_dist_dir, exist_ok=True)

            for param_name in self.best_params.keys():
                try:
                    dist_fig = self.plot_parameter_distribution(param_name)
                    dist_fig_path = os.path.join(param_dist_dir, f'param_{param_name}_distribution.png')
                    dist_fig.write_image(dist_fig_path)
                except:
                    print(f"绘制参数分布图 {param_name} 失败")
        except Exception as e:
            print(f"生成Optuna可视化时出错: {e}")

        # 保存超参数分布统计
        param_stats = {}
        for param_name in self.best_params.keys():
            param_values = [trial.params.get(param_name) for trial in self.study.trials
                            if param_name in trial.params]
            if param_values:
                if isinstance(param_values[0], (int, float)):
                    param_stats[param_name] = {
                        'mean': float(np.mean(param_values)),
                        'std': float(np.std(param_values)),
                        'min': float(np.min(param_values)),
                        'max': float(np.max(param_values)),
                        'median': float(np.median(param_values))
                    }
                elif isinstance(param_values[0], str):
                    unique, counts = np.unique(param_values, return_counts=True)
                    param_stats[param_name] = {str(value): int(count) for value, count in zip(unique, counts)}

        # 保存参数统计
        with open(os.path.join(best_params_dir, f'param_statistics_{timestamp}.json'), 'w') as f:
            json.dump(param_stats, f, indent=4)

        # 创建参数统计CSV
        param_stats_rows = []
        for param_name, stats in param_stats.items():
            if isinstance(stats, dict) and 'mean' in stats:
                param_stats_rows.append({
                    'parameter': param_name,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'median': stats['median'],
                    'best_value': self.best_params.get(param_name)
                })

        if param_stats_rows:
            param_stats_df = pd.DataFrame(param_stats_rows)
            param_stats_df.to_csv(os.path.join(best_params_dir, f'param_statistics_{timestamp}.csv'), index=False)

        # 创建分类参数统计CSV
        cat_params_data = []
        for param_name, stats in param_stats.items():
            if isinstance(stats, dict) and 'mean' not in stats:
                for value, count in stats.items():
                    cat_params_data.append({
                        'parameter': param_name,
                        'value': value,
                        'count': count,
                        'percentage': count / len(self.study.trials) * 100,
                        'is_best': value == self.best_params.get(param_name)
                    })

        if cat_params_data:
            cat_params_df = pd.DataFrame(cat_params_data)
            cat_params_df.to_csv(os.path.join(best_params_dir, f'categorical_param_statistics_{timestamp}.csv'),
                                 index=False)

        print(f"优化结果已保存到 {optuna_dir}")

    def plot_optimization_history(self):
        """绘制优化历史"""
        if self.study is None:
            raise RuntimeError("必须先运行优化")

        import optuna.visualization as vis
        return vis.plot_optimization_history(self.study)

    def plot_param_importances(self):
        """绘制参数重要性"""
        if self.study is None:
            raise RuntimeError("必须先运行优化")

        import optuna.visualization as vis
        return vis.plot_param_importances(self.study)

    def plot_parallel_coordinate(self):
        """绘制参数关系的平行坐标图"""
        if self.study is None:
            raise RuntimeError("必须先运行优化")

        import optuna.visualization as vis
        return vis.plot_parallel_coordinate(self.study)

    def plot_parameter_distribution(self, param_name):
        """绘制参数分布"""
        if self.study is None:
            raise RuntimeError("必须先运行优化")

        import optuna.visualization as vis
        return vis.plot_param_importances(self.study, target=lambda t: t.params[param_name])

    def get_trials_dataframe(self):
        """获取所有试验的DataFrame表示"""
        if self.study is None:
            raise RuntimeError("必须先运行优化")

        return self.study.trials_dataframe()

    def get_optimization_summary(self):
        """获取优化摘要"""
        if self.study is None:
            raise RuntimeError("必须先运行优化")

        # 收集完成的试验信息
        completed_trials = [trial for trial in self.study.trials
                            if trial.state == optuna.trial.TrialState.COMPLETE]

        # 计算总优化时间
        if self.trial_history:
            total_time = sum(trial.get('elapsed_time', 0) for trial in self.trial_history)
        else:
            total_time = None

        return {
            'n_trials': len(self.study.trials),
            'n_completed_trials': len(completed_trials),
            'n_pruned_trials': len([trial for trial in self.study.trials
                                    if trial.state == optuna.trial.TrialState.PRUNED]),
            'best_value': self.best_trial.value if self.best_trial else None,
            'best_trial_number': self.best_trial.number if self.best_trial else None,
            'best_params': self.best_params,
            'total_optimization_time': total_time,
            'average_trial_time': total_time / len(completed_trials) if total_time and completed_trials else None
        }