"""
运行一系列实验以比较不同的模型配置
"""
import os
import sys
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
from datetime import datetime
from config import get_config, update_config
from main import train_and_evaluate
from utils import create_experiment_dir, save_results, setup_logger, save_with_versioning
import argparse

def run_integration_type_experiments():
    """比较不同集成类型的实验"""
    base_config = get_config()

    # 创建主实验目录
    exp_dir = create_experiment_dir(base_dir='./experiments', experiment_name='integration_comparison')
    logger = setup_logger('integration_experiments', os.path.join(exp_dir, 'logs'))
    logger.info(f"开始集成类型比较实验，目录: {exp_dir}")

    # 禁用超参数优化以进行公平比较
    base_config['optimization']['n_trials'] = 0

    integration_types = ['feature_extraction', 'ensemble', 'end2end']
    results = []

    for integration_type in integration_types:
        logger.info(f"\n==== 运行 {integration_type} 实验 ====")

        # 更新配置
        config = update_config({
            'model': {
                'hybrid': {
                    'integration_type': integration_type
                }
            }
        })

        # 确保配置包含楼层分类器参数
        if 'floor_classifier' not in config['model']:
            config['model']['floor_classifier'] = {
                'type': 'random_forest',
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }

        # 使用时间戳，确保每次运行创建新的实验记录
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"实验时间戳: {run_timestamp}")

        # 运行实验
        experiment_results = train_and_evaluate(config)
        results.append(experiment_results)

        # 保存结果，使用时间戳避免覆盖
        result_path = os.path.join(exp_dir, 'results', 'comparisons', f'{integration_type}_results_{run_timestamp}.json')
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        save_results(experiment_results, result_path)
        logger.info(f"实验结果已保存到: {result_path}")

    # 比较结果
    comparison = {
        'integration_types': integration_types,
        'mean_errors': [r['metrics']['mean_error'] for r in results],
        'median_errors': [r['metrics']['median_error'] for r in results],
        'train_times': [r['train_time'] for r in results],
        'floor_accuracies': [r['metrics'].get('floor_accuracy', None) for r in results],
        'combined_errors': [r['metrics'].get('combined_error', None) for r in results],
        'experiment_dirs': [r['additional_info']['experiment_dir'] for r in results],
        'timestamp': datetime.now().isoformat()
    }

    # 保存比较结果，使用时间戳避免覆盖
    comparison_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = os.path.join(exp_dir, 'results', 'comparisons', f'integration_comparison_{comparison_timestamp}.json')
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    save_results(comparison, comparison_path)
    logger.info(f"比较结果已保存到: {comparison_path}")

    # 创建比较表格
    comparison_df = pd.DataFrame({
        'Integration Type': integration_types,
        'Mean Error (m)': comparison['mean_errors'],
        'Median Error (m)': comparison['median_errors'],
        'Floor Accuracy': comparison['floor_accuracies'],
        'Combined Error (m)': comparison['combined_errors'],
        'Training Time (s)': comparison['train_times'],
        'Experiment Dir': comparison['experiment_dirs'],
        'Run Timestamp': comparison_timestamp
    })

    # 保存比较表格，使用时间戳避免覆盖
    comparison_csv_path = os.path.join(exp_dir, 'csv_records', 'comparisons', f'integration_comparison_{comparison_timestamp}.csv')
    os.makedirs(os.path.dirname(comparison_csv_path), exist_ok=True)
    comparison_df.to_csv(comparison_csv_path, index=False)
    logger.info(f"比较表格已保存到: {comparison_csv_path}")

    # 创建比较图表
    plt.figure(figsize=(15, 10))

    # 误差条形图
    ax1 = plt.subplot(2, 2, 1)
    x = np.arange(len(integration_types))
    width = 0.35
    ax1.bar(x - width/2, comparison['mean_errors'], width, label='平均误差')
    ax1.bar(x + width/2, comparison['median_errors'], width, label='中位数误差')
    ax1.set_xticks(x)
    ax1.set_xticklabels(integration_types)
    ax1.set_ylabel('误差（米）')
    ax1.set_title('不同集成类型的定位误差')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # 训练时间条形图
    ax2 = plt.subplot(2, 2, 2)
    ax2.bar(integration_types, comparison['train_times'])
    ax2.set_ylabel('训练时间（秒）')
    ax2.set_title('不同集成类型的训练时间')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 楼层准确率条形图
    if any(acc is not None for acc in comparison['floor_accuracies']):
        ax3 = plt.subplot(2, 2, 3)
        ax3.bar(integration_types, comparison['floor_accuracies'])
        ax3.set_ylabel('楼层准确率')
        ax3.set_title('不同集成类型的楼层准确率')
        ax3.grid(True, linestyle='--', alpha=0.7)

    # 综合误差条形图
    if any(err is not None for err in comparison['combined_errors']):
        ax4 = plt.subplot(2, 2, 4)
        ax4.bar(integration_types, comparison['combined_errors'])
        ax4.set_ylabel('综合误差（米）')
        ax4.set_title('不同集成类型的综合误差（含楼层惩罚）')
        ax4.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # 保存比较图，使用时间戳避免覆盖
    comparison_fig_dir = os.path.join(exp_dir, 'visualizations', 'comparisons')
    os.makedirs(comparison_fig_dir, exist_ok=True)
    comparison_fig_path = os.path.join(comparison_fig_dir, f'integration_comparison_{comparison_timestamp}.png')
    plt.savefig(comparison_fig_path)
    plt.close()
    logger.info(f"比较图表已保存到: {comparison_fig_path}")

    logger.info(f"\n==== 集成类型比较完成 ====")

    return comparison

def run_feedback_experiments():
    """比较有无反馈机制的实验"""
    base_config = get_config()

    # 创建主实验目录
    exp_dir = create_experiment_dir(base_dir='./experiments', experiment_name='feedback_comparison')
    logger = setup_logger('feedback_experiments', os.path.join(exp_dir, 'logs'))
    logger.info(f"开始反馈机制比较实验，目录: {exp_dir}")

    # 禁用超参数优化以进行公平比较
    base_config['optimization']['n_trials'] = 0

    feedback_settings = [True, False]
    feedback_labels = ['有反馈', '无反馈']
    results = []

    for feedback_enabled, label in zip(feedback_settings, feedback_labels):
        logger.info(f"\n==== 运行 {label} 实验 ====")

        # 更新配置
        config = update_config({
            'model': {
                'feedback': {
                    'enabled': feedback_enabled
                }
            }
        })

        # 确保配置包含楼层分类器参数
        if 'floor_classifier' not in config['model']:
            config['model']['floor_classifier'] = {
                'type': 'random_forest',
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }

        # 使用时间戳，确保每次运行创建新的实验记录
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"实验时间戳: {run_timestamp}")

        # 运行实验
        experiment_results = train_and_evaluate(config)
        results.append(experiment_results)

        # 保存结果，使用时间戳避免覆盖
        setting_name = 'with_feedback' if feedback_enabled else 'without_feedback'
        result_path = os.path.join(exp_dir, 'results', 'comparisons', f'{setting_name}_results_{run_timestamp}.json')
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        save_results(experiment_results, result_path)
        logger.info(f"实验结果已保存到: {result_path}")

        # 比较结果
    comparison = {
        'feedback_settings': feedback_labels,
        'mean_errors': [r['metrics']['mean_error'] for r in results],
        'median_errors': [r['metrics']['median_error'] for r in results],
        'p75_errors': [r['metrics']['75th_percentile'] for r in results],
        'p90_errors': [r['metrics']['90th_percentile'] for r in results],
        'floor_accuracies': [r['metrics'].get('floor_accuracy', None) for r in results],
        'combined_errors': [r['metrics'].get('combined_error', None) for r in results],
        'experiment_dirs': [r['additional_info']['experiment_dir'] for r in results],
        'timestamp': datetime.now().isoformat()
    }

    # 保存比较结果，使用时间戳避免覆盖
    comparison_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = os.path.join(exp_dir, 'results', 'comparisons',
                                   f'feedback_comparison_{comparison_timestamp}.json')
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    save_results(comparison, comparison_path)
    logger.info(f"比较结果已保存到: {comparison_path}")

    # 创建比较表格
    comparison_df = pd.DataFrame({
        'Feedback': feedback_labels,
        'Mean Error (m)': comparison['mean_errors'],
        'Median Error (m)': comparison['median_errors'],
        '75th Percentile Error (m)': comparison['p75_errors'],
        '90th Percentile Error (m)': comparison['p90_errors'],
        'Floor Accuracy': comparison['floor_accuracies'],
        'Combined Error (m)': comparison['combined_errors'],
        'Experiment Dir': comparison['experiment_dirs'],
        'Run Timestamp': comparison_timestamp
    })

    # 保存比较表格，使用时间戳避免覆盖
    comparison_csv_path = os.path.join(exp_dir, 'csv_records', 'comparisons',
                                       f'feedback_comparison_{comparison_timestamp}.csv')
    os.makedirs(os.path.dirname(comparison_csv_path), exist_ok=True)
    comparison_df.to_csv(comparison_csv_path, index=False)
    logger.info(f"比较表格已保存到: {comparison_csv_path}")

    # 创建比较图表
    plt.figure(figsize=(12, 8))

    # 误差条形图
    x = np.arange(len(comparison['feedback_settings']))
    width = 0.2

    plt.bar(x - width * 1.5, comparison['mean_errors'], width, label='平均误差')
    plt.bar(x - width / 2, comparison['median_errors'], width, label='中位数误差')
    plt.bar(x + width / 2, comparison['p75_errors'], width, label='75%分位误差')
    plt.bar(x + width * 1.5, comparison['p90_errors'], width, label='90%分位误差')

    plt.xticks(x, comparison['feedback_settings'])
    plt.ylabel('误差（米）')
    plt.title('有无反馈机制的定位误差对比')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()

    # 保存比较图，使用时间戳避免覆盖
    comparison_fig_dir = os.path.join(exp_dir, 'visualizations', 'comparisons')
    os.makedirs(comparison_fig_dir, exist_ok=True)
    comparison_fig_path = os.path.join(comparison_fig_dir, f'feedback_comparison_{comparison_timestamp}.png')
    plt.savefig(comparison_fig_path)
    plt.close()
    logger.info(f"比较图表已保存到: {comparison_fig_path}")

    # 如果有楼层准确率，创建楼层准确率对比图
    if all(acc is not None for acc in comparison['floor_accuracies']):
        plt.figure(figsize=(10, 6))
        plt.bar(comparison['feedback_settings'], comparison['floor_accuracies'])
        plt.xlabel('反馈设置')
        plt.ylabel('楼层准确率')
        plt.title('有无反馈机制的楼层准确率对比')
        plt.grid(True, linestyle='--', alpha=0.7)

        # 保存楼层准确率对比图，使用时间戳避免覆盖
        floor_fig_path = os.path.join(comparison_fig_dir,
                                      f'feedback_floor_accuracy_comparison_{comparison_timestamp}.png')
        plt.savefig(floor_fig_path)
        plt.close()
        logger.info(f"楼层准确率对比图已保存到: {floor_fig_path}")

    logger.info(f"\n==== 反馈机制比较完成 ====")

    return comparison


def run_optimization_experiments():
    """比较不同优化水平的实验"""
    base_config = get_config()

    # 创建主实验目录
    exp_dir = create_experiment_dir(base_dir='./experiments', experiment_name='optimization_comparison')
    logger = setup_logger('optimization_experiments', os.path.join(exp_dir, 'logs'))
    logger.info(f"开始优化水平比较实验，目录: {exp_dir}")

    # 设置不同的优化试验次数
    n_trials_settings = [0, 10, 50, 100]
    results = []

    for n_trials in n_trials_settings:
        logger.info(f"\n==== 运行 {n_trials} 次优化试验 实验 ====")

        # 更新配置
        config = update_config({
            'optimization': {
                'n_trials': n_trials
            }
        })

        # 确保配置包含楼层分类器参数
        if 'floor_classifier' not in config['model']:
            config['model']['floor_classifier'] = {
                'type': 'random_forest',
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }

        # 使用时间戳，确保每次运行创建新的实验记录
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"实验时间戳: {run_timestamp}")

        # 运行实验
        experiment_results = train_and_evaluate(config)
        results.append(experiment_results)

        # 保存结果，使用时间戳避免覆盖
        result_path = os.path.join(exp_dir, 'results', 'comparisons', f'trials_{n_trials}_results_{run_timestamp}.json')
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        save_results(experiment_results, result_path)
        logger.info(f"实验结果已保存到: {result_path}")

    # 比较结果
    comparison = {
        'n_trials_settings': n_trials_settings,
        'mean_errors': [r['metrics']['mean_error'] for r in results],
        'median_errors': [r['metrics']['median_error'] for r in results],
        'train_times': [r['train_time'] for r in results],
        'floor_accuracies': [r['metrics'].get('floor_accuracy', None) for r in results],
        'combined_errors': [r['metrics'].get('combined_error', None) for r in results],
        'experiment_dirs': [r['additional_info']['experiment_dir'] for r in results],
        'timestamp': datetime.now().isoformat()
    }

    # 保存比较结果，使用时间戳避免覆盖
    comparison_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = os.path.join(exp_dir, 'results', 'comparisons',
                                   f'optimization_comparison_{comparison_timestamp}.json')
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    save_results(comparison, comparison_path)
    logger.info(f"比较结果已保存到: {comparison_path}")

    # 创建比较表格
    comparison_df = pd.DataFrame({
        'Optimization Trials': n_trials_settings,
        'Mean Error (m)': comparison['mean_errors'],
        'Median Error (m)': comparison['median_errors'],
        'Floor Accuracy': comparison['floor_accuracies'],
        'Combined Error (m)': comparison['combined_errors'],
        'Training Time (s)': comparison['train_times'],
        'Experiment Dir': comparison['experiment_dirs'],
        'Run Timestamp': comparison_timestamp
    })

    # 保存比较表格，使用时间戳避免覆盖
    comparison_csv_path = os.path.join(exp_dir, 'csv_records', 'comparisons',
                                       f'optimization_comparison_{comparison_timestamp}.csv')
    os.makedirs(os.path.dirname(comparison_csv_path), exist_ok=True)
    comparison_df.to_csv(comparison_csv_path, index=False)
    logger.info(f"比较表格已保存到: {comparison_csv_path}")

    # 创建比较图表
    plt.figure(figsize=(12, 6))

    # 误差曲线图
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(comparison['n_trials_settings'], comparison['mean_errors'], 'o-', label='平均误差')
    ax1.plot(comparison['n_trials_settings'], comparison['median_errors'], 's-', label='中位数误差')
    ax1.set_xlabel('优化试验次数')
    ax1.set_ylabel('误差（米）')
    ax1.set_title('不同优化水平的定位误差')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # 训练时间曲线图
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(comparison['n_trials_settings'], comparison['train_times'], 'o-')
    ax2.set_xlabel('优化试验次数')
    ax2.set_ylabel('总训练时间（秒）')
    ax2.set_title('不同优化水平的训练时间')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # 保存比较图，使用时间戳避免覆盖
    comparison_fig_dir = os.path.join(exp_dir, 'visualizations', 'comparisons')
    os.makedirs(comparison_fig_dir, exist_ok=True)
    comparison_fig_path = os.path.join(comparison_fig_dir, f'optimization_comparison_{comparison_timestamp}.png')
    plt.savefig(comparison_fig_path)
    plt.close()
    logger.info(f"比较图表已保存到: {comparison_fig_path}")

    # 添加楼层准确率和优化轮数的关系图
    if any(acc is not None for acc in comparison['floor_accuracies']):
        plt.figure(figsize=(10, 6))
        plt.plot(comparison['n_trials_settings'], comparison['floor_accuracies'], 'o-')
        plt.xlabel('优化试验次数')
        plt.ylabel('楼层准确率')
        plt.title('不同优化水平的楼层分类准确率')
        plt.grid(True, linestyle='--', alpha=0.7)

        # 保存楼层准确率对比图
        floor_fig_path = os.path.join(comparison_fig_dir, f'optimization_floor_accuracy_{comparison_timestamp}.png')
        plt.savefig(floor_fig_path)
        plt.close()
        logger.info(f"楼层准确率与优化轮数关系图已保存到: {floor_fig_path}")

    logger.info(f"\n==== 优化水平比较完成 ====")

    return comparison


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行一系列实验')
    parser.add_argument('--experiment', type=str, choices=['integration', 'feedback', 'optimization', 'all'],
                        default='all', help='要运行的实验类型')
    parser.add_argument('--sleep', type=int, default=60,
                        help='实验之间的等待时间（秒），以避免资源争用')
    parser.add_argument('--run_id', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"),
                        help='运行ID，用于标识本次运行的所有实验')

    args = parser.parse_args()

    # 创建总体实验运行记录
    main_log_dir = os.path.join('./experiments', 'logs')
    os.makedirs(main_log_dir, exist_ok=True)
    logger = setup_logger('experiment_runner', main_log_dir)
    logger.info(f"开始实验运行: {args.experiment}, 运行ID: {args.run_id}")

    # 确保每次运行都有唯一ID
    run_id = args.run_id
    logger.info(f"运行ID: {run_id}")

    # 创建运行记录目录
    run_record_dir = os.path.join('./experiments', 'run_records', f'run_{run_id}')
    os.makedirs(run_record_dir, exist_ok=True)

    try:
        # 记录运行开始信息
        start_time = datetime.now()
        run_info = {
            'run_id': run_id,
            'experiment_type': args.experiment,
            'start_time': start_time.isoformat(),
            'parameters': vars(args),
            'status': 'running'
        }

        # 保存运行信息
        run_info_path = os.path.join(run_record_dir, 'run_info.json')
        with open(run_info_path, 'w') as f:
            json.dump(run_info, f, indent=4)

        # 记录实验结果
        experiment_results = {}

        if args.experiment == 'integration' or args.experiment == 'all':
            logger.info("开始集成类型比较实验")
            integration_results = run_integration_type_experiments()
            experiment_results['integration'] = integration_results

            # 保存阶段性结果
            stage_results_path = os.path.join(run_record_dir,
                                              f'integration_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(stage_results_path, 'w') as f:
                json.dump(integration_results, f, indent=4)

            if args.experiment == 'all':
                logger.info(f"等待 {args.sleep} 秒后开始下一个实验...")
                time.sleep(args.sleep)

        if args.experiment == 'feedback' or args.experiment == 'all':
            logger.info("开始反馈机制比较实验")
            feedback_results = run_feedback_experiments()
            experiment_results['feedback'] = feedback_results

            # 保存阶段性结果
            stage_results_path = os.path.join(run_record_dir,
                                              f'feedback_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(stage_results_path, 'w') as f:
                json.dump(feedback_results, f, indent=4)

            if args.experiment == 'all':
                logger.info(f"等待 {args.sleep} 秒后开始下一个实验...")
                time.sleep(args.sleep)

        if args.experiment == 'optimization' or args.experiment == 'all':
            logger.info("开始优化水平比较实验")
            optimization_results = run_optimization_experiments()
            experiment_results['optimization'] = optimization_results

            # 保存阶段性结果
            stage_results_path = os.path.join(run_record_dir,
                                              f'optimization_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(stage_results_path, 'w') as f:
                json.dump(optimization_results, f, indent=4)

        # 记录运行完成信息
        end_time = datetime.now()
        run_info.update({
            'end_time': end_time.isoformat(),
            'duration_seconds': (end_time - start_time).total_seconds(),
            'status': 'completed',
            'experiments_run': list(experiment_results.keys())
        })

        # 更新运行信息
        with open(run_info_path, 'w') as f:
            json.dump(run_info, f, indent=4)

        # 保存完整实验结果
        full_results_path = os.path.join(run_record_dir,
                                         f'full_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(full_results_path, 'w') as f:
            json.dump(experiment_results, f, indent=4)

        logger.info("所有实验运行完成")
        logger.info(f"总运行时间: {run_info['duration_seconds']:.2f} 秒")
        logger.info(f"实验结果已保存到: {run_record_dir}")

    except Exception as e:
        # 记录错误信息
        error_time = datetime.now()
        run_info = {
            'run_id': run_id,
            'experiment_type': args.experiment,
            'start_time': start_time.isoformat() if 'start_time' in locals() else None,
            'error_time': error_time.isoformat(),
            'error': str(e),
            'status': 'failed',
            'traceback': traceback.format_exc()
        }

        # 保存错误信息
        error_info_path = os.path.join(run_record_dir, 'error_info.json')
        with open(error_info_path, 'w') as f:
            json.dump(run_info, f, indent=4)

        logger.error(f"实验运行过程中出错: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()