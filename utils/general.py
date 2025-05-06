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
    �������������ȷ����������

    ����:
        seed (int): �������
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

    # һЩ��������ã���ȷ���ڶ��߳�/GPU������Ҳ��ȷ���Ե�
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """
    ��ȡ���õļ����豸

    ����:
        torch.device: �����豸
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def format_time(seconds):
    """
    ��ʽ��ʱ��

    ����:
        seconds (float): ����

    ����:
        str: ��ʽ����ʱ���ַ���
    """
    if seconds < 60:
        return f"{seconds:.2f}��"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds %= 60
        return f"{int(minutes)}��{seconds:.2f}��"
    else:
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return f"{int(hours)}ʱ{int(minutes)}��{seconds:.2f}��"

def save_results(results, filepath, overwrite=False):
    """
    ���������ļ�

    ����:
        results: Ҫ����Ľ��
        filepath (str): �ļ�·��
        overwrite (bool): �Ƿ񸲸������ļ�
    """
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(f"�ļ�{filepath}�Ѵ��ڡ�����overwrite=True�Ը��ǡ�")

    # ����Ŀ¼����������ڣ�
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # �����ļ���չ��ȷ�������ʽ
    _, ext = os.path.splitext(filepath)

    if ext.lower() == '.json':
        # ȷ��������Ա�JSON���л�
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
        raise ValueError(f"��֧�ֵ��ļ���ʽ: {ext}����ʹ��.json��.pkl��.csv��")

def save_with_versioning(data, base_path, prefix="", extension=None):
    """
    ʹ�ð汾���Ʊ����ļ�����ֹ����

    ����:
        data: Ҫ���������
        base_path (str): �����ļ�·��
        prefix (str): �ļ���ǰ׺
        extension (str): �ļ���չ�������ΪNone����ʹ��base_path����չ����

    ����:
        str: ������ļ�·��
    """
    directory = os.path.dirname(base_path)
    basename = os.path.basename(base_path)

    # ������չ��
    if extension is None:
        filename, ext = os.path.splitext(basename)
    else:
        filename = os.path.splitext(basename)[0]
        ext = extension if extension.startswith('.') else f'.{extension}'

    # �������а汾
    pattern = os.path.join(directory, f"{prefix}{filename}_v*{ext}")
    existing_files = glob.glob(pattern)

    # ȷ���°汾��
    if not existing_files:
        version = 1
    else:
        # �������ļ�������ȡ�汾��
        versions = []
        for f in existing_files:
            try:
                v = int(os.path.basename(f).split('_v')[-1].split(ext)[0])
                versions.append(v)
            except ValueError:
                continue
        version = max(versions) + 1 if versions else 1

    # �������汾��·��
    versioned_path = os.path.join(directory, f"{prefix}{filename}_v{version}{ext}")

    # ������
    save_results(data, versioned_path)

    return versioned_path

def create_experiment_dir(base_dir='./experiments', experiment_name=None):
    """
    ��������ʱ������ʵ��Ŀ¼

    ����:
        base_dir (str): ����Ŀ¼
        experiment_name (str, optional): ʵ������

    ����:
        str: ʵ��Ŀ¼·��
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if experiment_name:
        exp_dir = os.path.join(base_dir, f'{experiment_name}_{timestamp}')
    else:
        exp_dir = os.path.join(base_dir, f'experiment_{timestamp}')

    # ������Ŀ¼
    os.makedirs(exp_dir, exist_ok=True)

    # ������ϸ��Ŀ¼�ṹ
    detailed_subdirs = {
        'models': {
            'checkpoints': 'ģ�ͼ��㣬���ִα���',
            'final': '����ѵ����ģ��',
            'best': '�������ŵ�ģ��'
        },
        'results': {
            'raw': 'ԭʼ�������',
            'processed': '����������ָ��',
            'comparisons': 'ģ�ͱȽϽ��'
        },
        'logs': {
            'training': 'ѵ����־',
            'evaluation': '������־',
            'optimization': '�������Ż���־'
        },
        'visualizations': {
            'training': 'ѵ�����ӻ�ͼ��',
            'error_analysis': '������ͼ��',
            'distribution': '���ݷֲ�ͼ��',
            'comparisons': 'ģ�ͱȽ�ͼ��'
        },
        'csv_records': {
            'training': 'ѵ����¼',
            'evaluation': '������¼',
            'optimization': '�Ż���¼'
        },
        'metrics': {
            'by_epoch': '���ִμ�¼��ָ��',
            'by_model': '��ģ�ͼ�¼��ָ��',
            'by_config': '�����ü�¼��ָ��'
        },
        'predictions': {
            'by_epoch': '���ִμ�¼��Ԥ��',
            'final': '����ģ�͵�Ԥ��',
            'best': '���ģ�͵�Ԥ��'
        },
        'optuna_results': {
            'trials': '�Ż�������',
            'visualizations': '�Ż����ӻ�',
            'best_params': '��Ѳ�����¼'
        },
        'configs': {
            'original': 'ԭʼ����',
            'optimized': '�Ż��������',
            'by_run': 'ÿ�����е�����'
        }
    }

    # ������ϸĿ¼�ṹ
    for main_dir, subdirs in detailed_subdirs.items():
        for subdir, description in subdirs.items():
            full_path = os.path.join(exp_dir, main_dir, subdir)
            os.makedirs(full_path, exist_ok=True)

            # ��ÿ��Ŀ¼�д���README����������;
            with open(os.path.join(full_path, "README.md"), "w") as f:
                f.write(f"# {subdir.replace('_', ' ').title()}\n\n")
                f.write(f"{description}\n")

    # ����README�ļ�
    readme_path = os.path.join(exp_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(f"# ʵ�飺{experiment_name or 'Default'}\n\n")
        f.write(f"��ʼʱ��: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Ŀ¼�ṹ\n\n")
        for main_dir, subdirs in detailed_subdirs.items():
            f.write(f"### `{main_dir}/`\n\n")
            for subdir, description in subdirs.items():
                f.write(f"- `{subdir}/`: {description}\n")
            f.write("\n")

    return exp_dir

def create_results_dict(model, train_time, eval_metrics, config, additional_info=None):
    """
    ������׼����ֵ�

    ����:
        model: ѵ���õ�ģ��
        train_time (float): ѵ��ʱ�䣨�룩
        eval_metrics (dict): ����ָ��
        config (dict): ʹ�õ�����
        additional_info (dict): ������Ϣ

    ����:
        dict: ����ֵ�
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
    ������־��¼��

    ����:
        name (str): ��־��¼������
        log_dir (str): ��־Ŀ¼
        level (str): ��־����

    ����:
        logging.Logger: ��־��¼��
    """
    # ������־Ŀ¼
    os.makedirs(log_dir, exist_ok=True)

    # ������־��¼��
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # ������еĴ�����
    if logger.handlers:
        logger.handlers.clear()

    # �����ļ�������
    # ���ʱ�������־�ļ�����ȷ��ÿ�����д������ļ�
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(os.path.join(log_dir, f'{name}_{timestamp}.log'))
    file_handler.setLevel(getattr(logging, level))

    # ��������̨������
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))

    # ������ʽ��
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # ��Ӵ�����
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def save_best_model(model, metrics, exp_dir, metric_name='mean_error', minimize=True):
    """
    �������ģ�ͣ�����ָ��ָ�꣬��������ǰ��ģ��

    ����:
        model: Ҫ���ܱ����ģ��
        metrics: ����ָ���ֵ�
        exp_dir: ʵ��Ŀ¼
        metric_name: ���ڱȽϵ�ָ������
        minimize: ��ָ���Ƿ�ԽСԽ��

    ����:
        (bool, str): �Ƿ�Ϊ�µ����ģ�ͣ������ģ��·��
    """
    # �������ģ��Ԫ����·��
    best_model_meta_path = os.path.join(exp_dir, 'results', 'raw', 'best_model_meta.json')

    # ����Ƿ�֮ǰ�����"���"ģ��
    current_best_value = float('inf') if minimize else float('-inf')
    if os.path.exists(best_model_meta_path):
        try:
            with open(best_model_meta_path, 'r') as f:
                best_meta = json.load(f)
                current_best_value = best_meta.get('metric_value', current_best_value)
        except (json.JSONDecodeError, FileNotFoundError):
            # ����ļ��𻵻򲻴��ڣ�����ʹ��Ĭ��ֵ
            pass

    # ��ǰģ����ָ��ָ���ϵ�ֵ
    current_value = metrics.get(metric_name)

    # ȷ����ǰģ���Ƿ����
    is_better = False
    if current_value is not None:
        is_better = current_value < current_best_value if minimize else current_value > current_best_value

    if is_better:
        # ʹ��ʱ�������ģ�ͣ������κο��ܵĳ�ͻ
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        best_model_dir = os.path.join(exp_dir, 'models', 'best')
        os.makedirs(best_model_dir, exist_ok=True)

        model_path = os.path.join(best_model_dir, f'best_model_{timestamp}.pkl')
        model.save_model(model_path)

        # �������ģ��Ԫ����
        best_meta = {
            'model_path': model_path,
            'metric_name': metric_name,
            'metric_value': current_value,
            'timestamp': timestamp,
            'metrics': metrics
        }

        # ����Ԫ����
        meta_dir = os.path.join(exp_dir, 'results', 'raw')
        os.makedirs(meta_dir, exist_ok=True)

        with open(best_model_meta_path, 'w') as f:
            json.dump(best_meta, f, indent=4)

        # ����ָ��"current_best.pkl"�ķ������ӻ򸱱����Ա�������
        current_best_path = os.path.join(best_model_dir, 'current_best.pkl')
        if os.path.exists(current_best_path):
            os.remove(current_best_path)

        # ��Windows�ϣ����Ʊȴ����������Ӹ��ɿ�
        shutil.copy2(model_path, current_best_path)

        # �����Ҫ˵���ļ�
        with open(os.path.join(best_model_dir, f'best_model_{timestamp}_info.txt'), 'w') as f:
            f.write(f"ģ�ͱ���ʱ��: {timestamp}\n")
            f.write(f"����ָ�� ({metric_name}): {current_value}\n")
            f.write(f"��ǰ���ֵ: {current_best_value}\n")
            f.write(
                f"�Ľ�: {abs(current_best_value - current_value):.6f} ({abs(current_best_value - current_value) / abs(current_best_value) * 100 if current_best_value != 0 else 100:.2f}%)\n")
            f.write("\n��ϸָ��:\n")
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    f.write(f"- {k}: {v}\n")

        return True, model_path

    return False, None