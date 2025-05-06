import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import json
import pickle

from data.loader import UJIIndoorLocLoader
from data.preprocessor import UJIIndoorLocPreprocessor
from models.hybrid_model import SVRTransformerHybrid
from models.floor_model import FloorClassifier
from feedback.dual_layer_feedback import DualLayerFeedbackSVRTransformer
from optimization.optuna_optimizer import OptunaSVRTransformerOptimizer
from optimization.search_spaces import get_search_space
from evaluation.metrics import PositioningEvaluator
from evaluation.visualization import PositioningVisualizer
from evaluation.monitor import TrainingMonitor
from config import get_config, update_config
from utils import (
    set_seed, save_results, create_experiment_dir, save_with_versioning,
    get_device, format_time, create_results_dict, setup_logger, save_best_model
)
from datetime import datetime


def train_and_evaluate(config=None):
    """
    训练和评估UJIIndoorLoc定位模型

    参数:
        config (dict): 配置字典

    返回:
        dict: 结果字典
    """
    # 加载配置
    if config is None:
        config = get_config()

    # 设置随机种子
    set_seed(config['random_state'])

    # 检测并打印计算设备 - 新添加的代码
    device = get_device()
    print(f"\n{'='*50}")
    print(f"使用计算设备: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"{'='*50}\n")

    # 创建实验目录
    exp_name = f"{config['model']['hybrid']['integration_type']}_{'with' if config['model']['feedback']['enabled'] else 'without'}_feedback"
    exp_dir = create_experiment_dir(experiment_name=exp_name)
    print(f"实验目录: {exp_dir}")


    # 创建主日志记录器
    logger = setup_logger('main', os.path.join(exp_dir, 'logs', 'training'), level=config['logging']['log_level'])
    logger.info(f"开始新实验: {exp_name}")
    logger.info(f"实验目录: {exp_dir}")

    # 保存配置
    config_path = os.path.join(exp_dir, 'configs', 'original',
                               f'config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    save_results(config, config_path)
    logger.info(f"配置已保存到: {config_path}")

    # 创建训练监视器
    monitor = TrainingMonitor(exp_dir, config)

    # 加载数据
    logger.info("正在加载数据...")
    loader = UJIIndoorLocLoader(
        data_dir=config['data']['data_dir'],
        download=config['data']['download']
    )
    data = loader.load_data()
    logger.info(
        f"数据加载完成，训练集: {data['training_data'].shape[0]}样本，验证集: {data['validation_data'].shape[0]}样本")

    # 创建预处理器
    logger.info("正在预处理数据...")
    preprocessor = UJIIndoorLocPreprocessor(
        replace_value=config['data']['replace_value'],
        normalization=config['data']['normalization'],
        dimension_reduction=config['data']['dimension_reduction'],
        n_components=config['data']['n_components']
    )

    # 预处理数据
    preprocessed_data = preprocessor.preprocess_dataset(
        data['training_data'],
        data['validation_data']
    )

    X_train = preprocessed_data['X_train']
    y_train_coords = preprocessed_data['y_train_coords'].values
    y_train_floor = preprocessed_data['y_train_floor'].values
    y_train_building = preprocessed_data['y_train_building'].values

    X_val = preprocessed_data['X_val']
    y_val_coords = preprocessed_data['y_val_coords'].values
    y_val_floor = preprocessed_data['y_val_floor'].values
    y_val_building = preprocessed_data['y_val_building'].values

    logger.info(f"预处理完成，训练特征: {X_train.shape}，验证特征: {X_val.shape}")

    # 保存预处理后的数据概览
    preprocessed_data_summary = {
        'X_train_shape': list(X_train.shape),
        'X_val_shape': list(X_val.shape),
        'y_train_coords_shape': list(y_train_coords.shape),
        'y_val_coords_shape': list(y_val_coords.shape),
        'unique_floors': np.unique(y_train_floor).tolist(),
        'floor_distribution': {int(k): int(v) for k, v in
                               zip(*np.unique(y_train_floor, return_counts=True))},
        'building_distribution': {int(k): int(v) for k, v in
                                  zip(*np.unique(y_train_building, return_counts=True))}
    }

    # 使用版本控制保存数据概览
    data_summary_path = os.path.join(exp_dir, 'csv_records', 'training', 'data_summary.json')
    os.makedirs(os.path.dirname(data_summary_path), exist_ok=True)
    versioned_path = save_with_versioning(preprocessed_data_summary, data_summary_path)
    logger.info(f"数据概览已保存到: {versioned_path}")

    # 如果启用了优化
    if config['optimization']['n_trials'] > 0:
        logger.info(f"正在使用Optuna进行超参数优化（{config['optimization']['n_trials']}次试验）...")

        # 获取搜索空间
        search_space = get_search_space(config['optimization']['search_space'])

        # 创建优化器
        optimizer = OptunaSVRTransformerOptimizer(
            search_space=search_space,
            n_trials=config['optimization']['n_trials'],
            timeout=config['optimization']['timeout'],
            n_jobs=config['optimization']['n_jobs'],
            direction=config['optimization']['direction'],
            cv=config['optimization']['cv'],
            cv_method=config['optimization']['cv_method']
        )

        # 创建优化日志记录器
        optim_logger = setup_logger('optimization', os.path.join(exp_dir, 'logs', 'optimization'),
                                    level=config['logging']['log_level'])
        optim_logger.info(f"开始超参数优化: {config['optimization']['n_trials']}次试验")

        # 运行优化
        best_params = optimizer.optimize(X_train, y_train_coords)
        logger.info(f"最佳参数: {best_params}")
        optim_logger.info(f"优化完成，最佳参数: {best_params}")

        # 保存优化结果
        optimizer.save_optimization_results(exp_dir)
        logger.info(f"优化结果已保存")

        # 使用版本控制保存最佳参数
        best_params_path = os.path.join(exp_dir, 'optuna_results', 'best_params', 'best_params.json')
        os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
        versioned_best_params_path = save_with_versioning(best_params, best_params_path)
        logger.info(f"最佳参数已保存到: {versioned_best_params_path}")

        # 更新配置
        config_updates = {}

        # 更新集成类型
        if 'integration_type' in best_params:
            config_updates['model'] = config_updates.get('model', {})
            config_updates['model']['hybrid'] = config_updates['model'].get('hybrid', {})
            config_updates['model']['hybrid']['integration_type'] = best_params['integration_type']

        # 更新Transformer参数
        transformer_params = {k: v for k, v in best_params.items()
                              if k in ['d_model', 'nhead', 'num_layers', 'dim_feedforward']}
        if transformer_params:
            config_updates['model'] = config_updates.get('model', {})
            config_updates['model']['transformer'] = config_updates['model'].get('transformer', {})
            config_updates['model']['transformer'].update(transformer_params)

        # 更新SVR参数
        svr_params = {k: v for k, v in best_params.items()
                      if k in ['kernel', 'C', 'epsilon', 'gamma', 'degree']}
        if svr_params:
            config_updates['model'] = config_updates.get('model', {})
            config_updates['model']['svr'] = config_updates['model'].get('svr', {})
            config_updates['model']['svr'].update(svr_params)

        # 应用更新
        config = update_config(config_updates)

        # 保存更新后的配置
        updated_config_path = os.path.join(exp_dir, 'configs', 'optimized',
                                           f'config_updated_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        os.makedirs(os.path.dirname(updated_config_path), exist_ok=True)
        save_results(config, updated_config_path)
        logger.info(f"更新后的配置已保存到: {updated_config_path}")

    # 训练楼层分类器
    logger.info("正在训练楼层分类器...")
    floor_classifier = FloorClassifier(classifier_type=config['model']['floor_classifier']['type'],
                                       n_estimators=config['model']['floor_classifier']['n_estimators'],
                                       random_state=config['random_state'])
    floor_classifier.fit(X_train, y_train_floor)

    # 评估楼层分类器
    floor_pred_train = floor_classifier.predict(X_train)
    floor_pred_val = floor_classifier.predict(X_val)

    floor_train_acc = np.mean(floor_pred_train == y_train_floor)
    floor_val_acc = np.mean(floor_pred_val == y_val_floor)

    logger.info(f"楼层分类器训练完成，训练集准确率: {floor_train_acc:.4f}，验证集准确率: {floor_val_acc:.4f}")

    # 保存楼层分类器评估结果
    floor_eval_results = {
        'train_accuracy': float(floor_train_acc),
        'val_accuracy': float(floor_val_acc),
        'timestamp': datetime.now().isoformat(),
        'classifier_type': config['model']['floor_classifier']['type'],
        'n_estimators': config['model']['floor_classifier']['n_estimators']
    }

    floor_eval_path = os.path.join(exp_dir, 'metrics', 'by_model', 'floor_classifier_evaluation.json')
    os.makedirs(os.path.dirname(floor_eval_path), exist_ok=True)
    versioned_floor_eval_path = save_with_versioning(floor_eval_results, floor_eval_path)
    logger.info(f"楼层分类器评估结果已保存到: {versioned_floor_eval_path}")

    # 如果分类器支持特征重要性，保存特征重要性
    if hasattr(floor_classifier, 'get_feature_importances'):
        try:
            feature_importances = floor_classifier.get_feature_importances()
            feature_names = [f'特征{i}' for i in range(len(feature_importances))]

            # 创建特征重要性DataFrame
            fi_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importances
            }).sort_values('importance', ascending=False)

            # 使用版本控制保存特征重要性
            fi_path = os.path.join(exp_dir, 'csv_records', 'evaluation', 'floor_feature_importances.csv')
            os.makedirs(os.path.dirname(fi_path), exist_ok=True)
            versioned_fi_path = save_with_versioning(fi_df, fi_path)
            logger.info(f"楼层分类器特征重要性已保存到: {versioned_fi_path}")

            # 可视化特征重要性
            visualizer = PositioningVisualizer()
            for fmt in config['logging']['visualization_formats']:
                fi_viz_dir = os.path.join(exp_dir, 'visualizations', 'error_analysis')
                os.makedirs(fi_viz_dir, exist_ok=True)
                fi_viz_path = os.path.join(fi_viz_dir,
                                           f'floor_feature_importances_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{fmt}')
                fi_fig = visualizer.plot_feature_importances(
                    feature_importances, feature_names, top_n=20)
                fi_fig.savefig(fi_viz_path)
                plt.close(fi_fig)
                logger.info(f"楼层分类器特征重要性可视化已保存到: {fi_viz_path}")
        except Exception as e:
            logger.warning(f"保存特征重要性时出错: {e}")

    # 创建和训练坐标回归模型
    logger.info(f"正在创建和训练坐标回归模型 ({config['model']['hybrid']['integration_type']})...")
    model = SVRTransformerHybrid(
        integration_type=config['model']['hybrid']['integration_type'],
        transformer_params=config['model']['transformer'],
        svr_params=config['model']['svr'],
        weights=config['model']['hybrid']['weights'],
        early_stopping_config=config['model']['early_stopping']
    )

    # 记录训练开始时间
    start_time = time.time()

    # 设置训练回调函数，用于记录训练进度
    def training_callback(epoch, train_loss, val_loss=None, lr=None, params=None):
        monitor.log_epoch(epoch, train_loss, val_loss, lr, params)

        # 如果需要保存模型检查点（使用版本控制）
        if (config['logging']['save_epoch_checkpoints'] and
                epoch % config['logging']['checkpoint_frequency'] == 0):
            checkpoint_dir = os.path.join(exp_dir, 'models', 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pkl')
            model.save_model(checkpoint_path)
            logger.info(f"模型检查点已保存到: {checkpoint_path}")

    # 训练模型
    logger.info("开始训练模型...")
    model.fit(X_train, y_train_coords, X_val, y_val_coords, callback=training_callback)

    # 计算训练时间
    train_time = time.time() - start_time
    logger.info(f"训练完成，用时: {format_time(train_time)}")

    # 保存训练历史和学习曲线，使用版本控制
    monitor.save_history_csv()
    monitor.plot_learning_curves()
    training_summary = monitor.save_final_history_summary()
    logger.info(f"训练摘要: {training_summary}")

    # 保存模型训练历史，使用版本控制
    training_history_path = os.path.join(exp_dir, 'csv_records', 'training', 'training_history.csv')
    os.makedirs(os.path.dirname(training_history_path), exist_ok=True)
    versioned_history_path = save_with_versioning(model.get_training_history(), training_history_path)
    logger.info(f"训练历史已保存到: {versioned_history_path}")

    # 如果启用了反馈机制
    if config['model']['feedback']['enabled']:
        logger.info("正在应用双层自适应反馈机制...")
        feedback_model = DualLayerFeedbackSVRTransformer(
            base_model=model,
            learning_rate=config['model']['feedback']['learning_rate'],
            meta_learning_rate=config['model']['feedback']['meta_learning_rate'],
            feedback_window=config['model']['feedback']['feedback_window']
        )

        # 保存原始模型，使用版本控制
        base_model_dir = os.path.join(exp_dir, 'models', 'final')
        os.makedirs(base_model_dir, exist_ok=True)
        base_model_path = os.path.join(base_model_dir, f'base_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
        model.save_model(base_model_path)
        logger.info(f"基础模型已保存到: {base_model_path}")

        # 预测和反馈（使用验证集）
        logger.info("使用反馈机制进行预测...")
        start_pred_time = time.time()
        y_pred_coords = feedback_model.predict_with_feedback(X_val, y_val_coords)
        pred_time = time.time() - start_pred_time
    else:
        # 直接预测
        logger.info("使用模型进行预测...")
        start_pred_time = time.time()
        y_pred_coords = model.predict(X_val)
        pred_time = time.time() - start_pred_time

    # 预测楼层
    logger.info("预测楼层...")
    y_pred_floor = floor_classifier.predict(X_val)

    # 计算并打印推理速度
    samples_per_second = len(X_val) / pred_time
    logger.info(f"推理速度: {samples_per_second:.2f} 样本/秒")

    # 评估模型
    logger.info("正在评估模型...")
    evaluator = PositioningEvaluator()

    # 创建评估保存目录，使用时间戳避免覆盖
    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_save_dir = os.path.join(exp_dir, 'results', 'raw', f'evaluation_{eval_timestamp}')
    os.makedirs(eval_save_dir, exist_ok=True)

    metrics = evaluator.evaluate_and_save(
        y_val_coords, y_pred_coords,
        y_val_floor, y_pred_floor,
        y_val_building, None,
        save_dir=eval_save_dir
    )

    # 打印主要指标
    logger.info(f"平均定位误差: {metrics['mean_error']:.2f} m")
    logger.info(f"中位数定位误差: {metrics['median_error']:.2f} m")
    logger.info(f"均方根误差: {metrics['rmse']:.2f} m")
    logger.info(f"75%分位误差: {metrics['75th_percentile']:.2f} m")
    logger.info(f"90%分位误差: {metrics['90th_percentile']:.2f} m")

    if 'floor_accuracy' in metrics:
        logger.info(f"楼层准确率: {metrics['floor_accuracy']:.2%}")
        logger.info(f"综合误差(含楼层惩罚): {metrics['combined_error']:.2f} m")

    # 可视化结果
    logger.info("正在生成可视化...")
    visualizer = PositioningVisualizer()

    # 为所有可视化创建目录和时间戳，避免覆盖
    viz_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_dir = os.path.join(exp_dir, 'visualizations', 'error_analysis')
    os.makedirs(viz_dir, exist_ok=True)

    # 误差CDF
    for fmt in config['logging']['visualization_formats']:
        logger.info(f"生成误差CDF图 ({fmt})...")
        cdf_fig = visualizer.plot_error_cdf(
            np.sqrt(np.sum((y_val_coords - y_pred_coords) ** 2, axis=1)),
            label=f"{config['model']['hybrid']['integration_type']}"
        )
        cdf_path = os.path.join(viz_dir, f'error_cdf_{viz_timestamp}.{fmt}')
        cdf_fig.savefig(cdf_path)
        plt.close(cdf_fig)
        logger.info(f"误差CDF图已保存到: {cdf_path}")

    # 误差分布直方图
    for fmt in config['logging']['visualization_formats']:
        logger.info(f"生成误差分布直方图 ({fmt})...")
        dist_fig = visualizer.plot_error_distribution(
            np.sqrt(np.sum((y_val_coords - y_pred_coords) ** 2, axis=1))
        )
        dist_path = os.path.join(viz_dir, f'error_distribution_{viz_timestamp}.{fmt}')
        dist_fig.savefig(dist_path)
        plt.close(dist_fig)
        logger.info(f"误差分布直方图已保存到: {dist_path}")

    # 2D误差散点图
    for fmt in config['logging']['visualization_formats']:
        logger.info(f"生成2D误差散点图 ({fmt})...")
        error_2d_fig = visualizer.plot_error_2d_scatter(
            y_val_coords, y_pred_coords
        )
        error_2d_path = os.path.join(viz_dir, f'error_2d_scatter_{viz_timestamp}.{fmt}')
        error_2d_fig.savefig(error_2d_path)
        plt.close(error_2d_fig)
        logger.info(f"2D误差散点图已保存到: {error_2d_path}")

    # 楼层混淆矩阵
    if y_val_floor is not None and y_pred_floor is not None:
        for fmt in config['logging']['visualization_formats']:
            logger.info(f"生成楼层混淆矩阵 ({fmt})...")
            floor_cm_fig = visualizer.plot_floor_confusion_matrix(
                y_val_floor, y_pred_floor
            )
            floor_cm_path = os.path.join(viz_dir, f'floor_confusion_matrix_{viz_timestamp}.{fmt}')
            floor_cm_fig.savefig(floor_cm_path)
            plt.close(floor_cm_fig)
            logger.info(f"楼层混淆矩阵已保存到: {floor_cm_path}")

    # 保存详细预测结果，使用版本控制
    logger.info("保存详细预测结果...")
    predictions_df = pd.DataFrame({
        'true_x': y_val_coords[:, 0],
        'true_y': y_val_coords[:, 1],
        'pred_x': y_pred_coords[:, 0],
        'pred_y': y_pred_coords[:, 1],
        'x_error': np.abs(y_val_coords[:, 0] - y_pred_coords[:, 0]),
        'y_error': np.abs(y_val_coords[:, 1] - y_pred_coords[:, 1]),
        'euclidean_error': np.sqrt(np.sum((y_val_coords - y_pred_coords) ** 2, axis=1)),
        'true_floor': y_val_floor,
        'pred_floor': y_pred_floor,
        'floor_correct': y_val_floor == y_pred_floor,
        'true_building': y_val_building
    })

    predictions_path = os.path.join(exp_dir, 'predictions', 'final', 'detailed_predictions.csv')
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    versioned_predictions_path = save_with_versioning(predictions_df, predictions_path)
    logger.info(f"详细预测结果已保存到: {versioned_predictions_path}")

    # 保存最终模型，使用版本控制
    final_model_dir = os.path.join(exp_dir, 'models', 'final')
    os.makedirs(final_model_dir, exist_ok=True)
    final_model_path = os.path.join(final_model_dir, f'final_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
    model.save_model(final_model_path)
    logger.info(f"最终模型已保存到: {final_model_path}")

    # 检查是否是最佳模型，如果是，保存到best目录
    is_best, best_model_path = save_best_model(model, metrics, exp_dir, metric_name='mean_error', minimize=True)
    if is_best:
        logger.info(f"当前模型是新的最佳模型! 已保存到: {best_model_path}")

    # 保存楼层分类器，使用版本控制
    floor_model_dir = os.path.join(exp_dir, 'models', 'final')
    os.makedirs(floor_model_dir, exist_ok=True)
    floor_model_path = os.path.join(floor_model_dir, f'floor_classifier_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
    floor_classifier.save_model(floor_model_path)
    logger.info(f"楼层分类器已保存到: {floor_model_path}")

    # 保存评估结果，使用版本控制
    results = create_results_dict(
        model=model,
        train_time=train_time,
        eval_metrics=metrics,
        config=config,
        additional_info={
            'visualization_paths': {
                'error_cdf': os.path.join(viz_dir,
                                          f'error_cdf_{viz_timestamp}.{config["logging"]["visualization_formats"][0]}'),
                'error_distribution': os.path.join(viz_dir,
                                                   f'error_distribution_{viz_timestamp}.{config["logging"]["visualization_formats"][0]}'),
                'error_2d_scatter': os.path.join(viz_dir,
                                                 f'error_2d_scatter_{viz_timestamp}.{config["logging"]["visualization_formats"][0]}'),
                'floor_confusion_matrix': os.path.join(viz_dir,
                                                       f'floor_confusion_matrix_{viz_timestamp}.{config["logging"]["visualization_formats"][0]}') if 'floor_accuracy' in metrics else None
            },
            'model_path': final_model_path,
            'floor_model_path': floor_model_path,
            'experiment_dir': exp_dir,
            'inference_speed': samples_per_second,
            'inference_time': pred_time,
            'training_history': model.get_training_history(),
            'floor_accuracy': metrics.get('floor_accuracy'),
            'floor_classification_report': floor_classifier.evaluate(X_val, y_val_floor),
        }
    )

    # 保存评估结果，使用版本控制
    results_dir = os.path.join(exp_dir, 'results', 'processed')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    save_results(results, results_path)
    logger.info(f"评估结果已保存到: {results_path}")

    # 创建结果概览表格
    results_overview = pd.DataFrame([{
        'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Integration Type': config['model']['hybrid']['integration_type'],
        'Feedback': 'Enabled' if config['model']['feedback']['enabled'] else 'Disabled',
        'Mean Error (m)': metrics['mean_error'],
        'Median Error (m)': metrics['median_error'],
        'RMSE (m)': metrics['rmse'],
        '75th Percentile (m)': metrics['75th_percentile'],
        '90th Percentile (m)': metrics['90th_percentile'],
        'Floor Accuracy': metrics.get('floor_accuracy', 'N/A'),
        'Combined Error (m)': metrics.get('combined_error', 'N/A'),
        'Training Time (s)': train_time,
        'Inference Speed (samples/s)': samples_per_second,
        'Experiment Dir': exp_dir,
        'Run ID': datetime.now().strftime('%Y%m%d_%H%M%S')
    }])

    # 追加到主结果表（如果存在）
    main_results_path = os.path.join('experiments', 'all_experiments_results.csv')
    if os.path.exists(main_results_path):
        try:
            main_results = pd.read_csv(main_results_path)
            results_overview = pd.concat([main_results, results_overview], ignore_index=True)
        except Exception as e:
            logger.warning(f"读取主结果表时出错: {e}，将创建新的表格")

    # 确保目录存在
    os.makedirs(os.path.dirname(main_results_path), exist_ok=True)
    results_overview.to_csv(main_results_path, index=False)
    logger.info(f"结果概览已追加到: {main_results_path}")

    # 使用版本控制保存结果概览
    overview_path = os.path.join(exp_dir, 'results', 'processed', 'experiment_overview.csv')
    os.makedirs(os.path.dirname(overview_path), exist_ok=True)
    versioned_overview_path = save_with_versioning(results_overview, overview_path)
    logger.info(f"实验概览已保存到: {versioned_overview_path}")

    # 更新README文件
    readme_path = os.path.join(exp_dir, 'README.md')
    with open(readme_path, 'a') as f:
        f.write("\n\n## 实验结果\n\n")
        f.write(f"- 集成类型: {config['model']['hybrid']['integration_type']}\n")
        f.write(f"- 反馈机制: {'启用' if config['model']['feedback']['enabled'] else '禁用'}\n")
        f.write(f"- 平均定位误差: {metrics['mean_error']:.2f} m\n")
        f.write(f"- 中位数定位误差: {metrics['median_error']:.2f} m\n")
        f.write(f"- 75%分位误差: {metrics['75th_percentile']:.2f} m\n")
        f.write(f"- 90%分位误差: {metrics['90th_percentile']:.2f} m\n")

        if 'floor_accuracy' in metrics:
            f.write(f"- 楼层准确率: {metrics['floor_accuracy']:.2%}\n")
            f.write(f"- 综合误差(含楼层惩罚): {metrics['combined_error']:.2f} m\n")

        f.write(f"- 训练时间: {format_time(train_time)}\n")
        f.write(f"- 推理速度: {samples_per_second:.2f} 样本/秒\n")
        f.write(f"- 完成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 运行ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
        f.write(f"\n### 主要文件路径\n\n")
        f.write(f"- 最终模型: {final_model_path}\n")
        f.write(f"- 楼层分类器: {floor_model_path}\n")
        f.write(f"- 评估结果: {results_path}\n")
        f.write(f"- 详细预测: {versioned_predictions_path}\n")
        f.write(f"- 训练历史: {versioned_history_path}\n")

    logger.info(f"实验完成！所有结果已保存到: {exp_dir}")

    return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='UJIIndoorLoc定位系统')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--no_optimization', action='store_true', help='禁用超参数优化')
    parser.add_argument('--integration_type', type=str,
                        choices=['feature_extraction', 'ensemble', 'end2end'],
                        help='集成类型')
    parser.add_argument('--n_trials', type=int, help='Optuna试验次数')
    parser.add_argument('--random_state', type=int, help='随机种子')
    parser.add_argument('--with_feedback', action='store_true', help='启用反馈机制')
    parser.add_argument('--without_feedback', action='store_true', help='禁用反馈机制')
    parser.add_argument('--experiment_name', type=str, help='实验名称')

    args = parser.parse_args()

    # 加载默认配置
    config = get_config()

    # 如果提供了配置文件，则加载
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config_updates = json.load(f)
        config = update_config(config_updates)

    # 应用命令行参数
    config_updates = {}

    if args.no_optimization:
        config_updates['optimization'] = {'n_trials': 0}

    if args.integration_type:
        config_updates['model'] = config_updates.get('model', {})
        config_updates['model']['hybrid'] = config_updates['model'].get('hybrid', {})
        config_updates['model']['hybrid']['integration_type'] = args.integration_type

    if args.n_trials:
        config_updates['optimization'] = config_updates.get('optimization', {})
        config_updates['optimization']['n_trials'] = args.n_trials

    if args.random_state:
        config_updates['random_state'] = args.random_state

    if args.with_feedback:
        config_updates['model'] = config_updates.get('model', {})
        config_updates['model']['feedback'] = config_updates['model'].get('feedback', {})
        config_updates['model']['feedback']['enabled'] = True

    if args.without_feedback:
        config_updates['model'] = config_updates.get('model', {})
        config_updates['model']['feedback'] = config_updates['model'].get('feedback', {})
        config_updates['model']['feedback']['enabled'] = False

    # 使用实验名称（如果提供）
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        exp_name = None

    # 应用更新
    if config_updates:
        config = update_config(config_updates)

    # 运行训练和评估
    results = train_and_evaluate(config)

    # 打印结果摘要
    print("\n==== 性能摘要 ====")
    print(f"平均定位误差: {results['metrics']['mean_error']:.2f} m")
    print(f"中位数定位误差: {results['metrics']['median_error']:.2f} m")
    if 'floor_accuracy' in results['metrics']:
        print(f"楼层准确率: {results['metrics']['floor_accuracy']:.2%}")
        print(f"综合误差(含楼层惩罚): {results['metrics']['combined_error']:.2f} m")
    print(f"训练时间: {results['train_time_formatted']}")
    print(f"实验目录: {results['additional_info']['experiment_dir']}")
    print(f"运行ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}")


if __name__ == "__main__":
    main()