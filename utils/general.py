import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import glob
import shutil
import logging

def set_seed(seed):
    """
    设置随机种子以确保可重现性

    参数:
        seed (int): 随机种子
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 一些额外的设置，以确保在多线程/GPU环境中也是确定性的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """
    获取可用的计算设备

    返回:
        torch.device: 计算设备
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def format_time(seconds):
    """
    格式化时间

    参数:
        seconds (float): 秒数

    返回:
        str: 格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds %= 60
        return f"{int(minutes)}分{seconds:.2f}秒"
    else:
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return f"{int(hours)}时{int(minutes)}分{seconds:.2f}秒"

def save_results(results, filepath, overwrite=False):
    """
    保存结果到文件

    参数:
        results: 要保存的结果
        filepath (str): 文件路径
        overwrite (bool): 是否覆盖现有文件
    """
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(f"文件{filepath}已存在。设置overwrite=True以覆盖。")

    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # 根据文件扩展名确定保存格式
    _, ext = os.path.splitext(filepath)

    if ext.lower() == '.json':
        # 确保结果可以被JSON序列化
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return super(NumpyEncoder, self).default(obj)

        with open(filepath, 'w') as f:
            json.dump(results, f, cls=NumpyEncoder, indent=4)

    elif ext.lower() == '.pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)

    elif ext.lower() == '.csv':
        if isinstance(results, pd.DataFrame):
            results.to_csv(filepath, index=False)
        else:
            pd.DataFrame(results).to_csv(filepath, index=False)

    else:
        raise ValueError(f"不支持的文件格式: {ext}。请使用.json、.pkl或.csv。")

def save_with_versioning(data, base_path, prefix="", extension=None):
    """
    使用版本控制保存文件，防止覆盖

    参数:
        data: 要保存的数据
        base_path (str): 基本文件路径
        prefix (str): 文件名前缀
        extension (str): 文件扩展名（如果为None，则使用base_path的扩展名）

    返回:
        str: 保存的文件路径
    """
    directory = os.path.dirname(base_path)
    basename = os.path.basename(base_path)

    # 处理扩展名
    if extension is None:
        filename, ext = os.path.splitext(basename)
    else:
        filename = os.path.splitext(basename)[0]
        ext = extension if extension.startswith('.') else f'.{extension}'

    # 查找现有版本
    pattern = os.path.join(directory, f"{prefix}{filename}_v*{ext}")
    existing_files = glob.glob(pattern)

    # 确定新版本号
    if not existing_files:
        version = 1
    else:
        # 从现有文件名中提取版本号
        versions = []
        for f in existing_files:
            try:
                v = int(os.path.basename(f).split('_v')[-1].split(ext)[0])
                versions.append(v)
            except ValueError:
                continue
        version = max(versions) + 1 if versions else 1

    # 创建带版本的路径
    versioned_path = os.path.join(directory, f"{prefix}{filename}_v{version}{ext}")

    # 保存结果
    save_results(data, versioned_path)

    return versioned_path

def create_experiment_dir(base_dir='./experiments', experiment_name=None):
    """
    创建带有时间编码的实验目录

    参数:
        base_dir (str): 基础目录
        experiment_name (str, optional): 实验名称

    返回:
        str: 实验目录路径
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if experiment_name:
        exp_dir = os.path.join(base_dir, f'{experiment_name}_{timestamp}')
    else:
        exp_dir = os.path.join(base_dir, f'experiment_{timestamp}')

    # 创建主目录
    os.makedirs(exp_dir, exist_ok=True)

    # 创建详细子目录结构
    detailed_subdirs = {
        'models': {
            'checkpoints': '模型检查点，按轮次保存',
            'final': '最终训练的模型',
            'best': '性能最优的模型'
        },
        'results': {
            'raw': '原始评估结果',
            'processed': '处理后的评估指标',
            'comparisons': '模型比较结果'
        },
        'logs': {
            'training': '训练日志',
            'evaluation': '评估日志',
            'optimization': '超参数优化日志'
        },
        'visualizations': {
            'training': '训练可视化图表',
            'error_analysis': '误差分析图表',
            'distribution': '数据分布图表',
            'comparisons': '模型比较图表'
        },
        'csv_records': {
            'training': '训练记录',
            'evaluation': '评估记录',
            'optimization': '优化记录'
        },
        'metrics': {
            'by_epoch': '按轮次记录的指标',
            'by_model': '按模型记录的指标',
            'by_config': '按配置记录的指标'
        },
        'predictions': {
            'by_epoch': '按轮次记录的预测',
            'final': '最终模型的预测',
            'best': '最佳模型的预测'
        },
        'optuna_results': {
            'trials': '优化试验结果',
            'visualizations': '优化可视化',
            'best_params': '最佳参数记录'
        },
        'configs': {
            'original': '原始配置',
            'optimized': '优化后的配置',
            'by_run': '每次运行的配置'
        }
    }

    # 创建详细目录结构
    for main_dir, subdirs in detailed_subdirs.items():
        for subdir, description in subdirs.items():
            full_path = os.path.join(exp_dir, main_dir, subdir)
            os.makedirs(full_path, exist_ok=True)

            # 在每个目录中创建README，解释其用途
            with open(os.path.join(full_path, "README.md"), "w") as f:
                f.write(f"# {subdir.replace('_', ' ').title()}\n\n")
                f.write(f"{description}\n")

    # 创建README文件
    readme_path = os.path.join(exp_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(f"# 实验：{experiment_name or 'Default'}\n\n")
        f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 目录结构\n\n")
        for main_dir, subdirs in detailed_subdirs.items():
            f.write(f"### `{main_dir}/`\n\n")
            for subdir, description in subdirs.items():
                f.write(f"- `{subdir}/`: {description}\n")
            f.write("\n")

    return exp_dir

def create_results_dict(model, train_time, eval_metrics, config, additional_info=None):
    """
    创建标准结果字典

    参数:
        model: 训练好的模型
        train_time (float): 训练时间（秒）
        eval_metrics (dict): 评估指标
        config (dict): 使用的配置
        additional_info (dict): 附加信息

    返回:
        dict: 结果字典
    """
    results = {
        'model_type': model.__class__.__name__,
        'train_time': train_time,
        'train_time_formatted': format_time(train_time),
        'metrics': eval_metrics,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }

    if additional_info:
        results.update(additional_info)

    return results

def setup_logger(name, log_dir, level='INFO'):
    """
    设置日志记录器

    参数:
        name (str): 日志记录器名称
        log_dir (str): 日志目录
        level (str): 日志级别

    返回:
        logging.Logger: 日志记录器
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # 清除已有的处理器
    if logger.handlers:
        logger.handlers.clear()

    # 创建文件处理器
    # 添加时间戳到日志文件名，确保每次运行创建新文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(os.path.join(log_dir, f'{name}_{timestamp}.log'))
    file_handler.setLevel(getattr(logging, level))

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))

    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def save_best_model(model, metrics, exp_dir, metric_name='mean_error', minimize=True):
    """
    保存最佳模型，基于指定指标，不覆盖先前的模型

    参数:
        model: 要可能保存的模型
        metrics: 评估指标字典
        exp_dir: 实验目录
        metric_name: 用于比较的指标名称
        minimize: 该指标是否越小越好

    返回:
        (bool, str): 是否为新的最佳模型，保存的模型路径
    """
    # 创建最佳模型元数据路径
    best_model_meta_path = os.path.join(exp_dir, 'results', 'raw', 'best_model_meta.json')

    # 检查是否之前保存过"最佳"模型
    current_best_value = float('inf') if minimize else float('-inf')
    if os.path.exists(best_model_meta_path):
        try:
            with open(best_model_meta_path, 'r') as f:
                best_meta = json.load(f)
                current_best_value = best_meta.get('metric_value', current_best_value)
        except (json.JSONDecodeError, FileNotFoundError):
            # 如果文件损坏或不存在，继续使用默认值
            pass

    # 当前模型在指定指标上的值
    current_value = metrics.get(metric_name)

    # 确定当前模型是否更好
    is_better = False
    if current_value is not None:
        is_better = current_value < current_best_value if minimize else current_value > current_best_value

    if is_better:
        # 使用时间戳保存模型，避免任何可能的冲突
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        best_model_dir = os.path.join(exp_dir, 'models', 'best')
        os.makedirs(best_model_dir, exist_ok=True)

        model_path = os.path.join(best_model_dir, f'best_model_{timestamp}.pkl')
        model.save_model(model_path)

        # 更新最佳模型元数据
        best_meta = {
            'model_path': model_path,
            'metric_name': metric_name,
            'metric_value': current_value,
            'timestamp': timestamp,
            'metrics': metrics
        }

        # 保存元数据
        meta_dir = os.path.join(exp_dir, 'results', 'raw')
        os.makedirs(meta_dir, exist_ok=True)

        with open(best_model_meta_path, 'w') as f:
            json.dump(best_meta, f, indent=4)

        # 创建指向"current_best.pkl"的符号链接或副本，以便于引用
        current_best_path = os.path.join(best_model_dir, 'current_best.pkl')
        if os.path.exists(current_best_path):
            os.remove(current_best_path)

        # 在Windows上，复制比创建符号链接更可靠
        shutil.copy2(model_path, current_best_path)

        # 保存简要说明文件
        with open(os.path.join(best_model_dir, f'best_model_{timestamp}_info.txt'), 'w') as f:
            f.write(f"模型保存时间: {timestamp}\n")
            f.write(f"评估指标 ({metric_name}): {current_value}\n")
            f.write(f"先前最佳值: {current_best_value}\n")
            f.write(
                f"改进: {abs(current_best_value - current_value):.6f} ({abs(current_best_value - current_value) / abs(current_best_value) * 100 if current_best_value != 0 else 100:.2f}%)\n")
            f.write("\n详细指标:\n")
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    f.write(f"- {k}: {v}\n")

        return True, model_path

    return False, None