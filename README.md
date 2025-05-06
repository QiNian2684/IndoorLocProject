# 室内定位系统使用指南

本指南详细介绍如何运行室内定位系统代码、调整参数选项以及解读实验结果。

---

## 目录
1. [运行代码](#运行代码)  
2. [参数配置](#参数配置)  
3. [模型调优](#模型调优)  
4. [结果解读](#结果解读)  
5. [常见问题排查](#常见问题排查)  

---

## 运行代码

### 基本运行命令
系统有两个主要运行入口：

```bash
# 单次实验
python main.py

# 比较实验（多组配置）
python run_experiments.py --experiment all
```

### 运行选项

#### `main.py` 运行选项
```bash
# 选择集成策略
python main.py --integration_type feature_extraction  # 特征提取策略
python main.py --integration_type ensemble           # 集成策略
python main.py --integration_type end2end            # 端到端策略

# 启用/禁用反馈机制
python main.py --with_feedback                       # 启用反馈机制
python main.py --without_feedback                    # 禁用反馈机制

# 超参数优化相关
python main.py --no_optimization                     # 跳过超参数优化
python main.py --n_trials 200                        # 设置优化试验次数为200

# 随机种子设置
python main.py --random_state 42                     # 设置随机种子为42

# 自定义实验名称
python main.py --experiment_name my_experiment       # 自定义实验名称

# 组合选项
python main.py --integration_type ensemble --with_feedback --n_trials 50
```

#### `run_experiments.py` 运行选项
```bash
# 运行特定类型的比较实验
python run_experiments.py --experiment integration   # 比较不同集成策略
python run_experiments.py --experiment feedback      # 比较有无反馈机制
python run_experiments.py --experiment optimization  # 比较不同优化水平

# 其他选项
python run_experiments.py --sleep 120               # 设置实验间等待时间（秒）
python run_experiments.py --run_id custom_run_id    # 自定义运行ID
```

### 检查计算设备
在程序开始运行时，会自动打印使用的计算设备信息（GPU 或 CPU）：

```
==================================================
使用计算设备: cuda (NVIDIA RTX 3080) 
==================================================
```

若需强制使用 CPU（即使 GPU 可用），可修改 `config.py` 中的设备参数：

```python
MODEL_CONFIG = {
    'device': 'cpu',  # 将 'cuda' 改为 'cpu' 以强制使用 CPU
    # 其他设置...
}
```

---

## 参数配置

### 配置文件
主要配置位于 `config.py`，分以下几部分。

### 1. 数据预处理配置
```python
DATA_CONFIG = {
    'data_dir': './data/raw',         # 数据目录
    'download': True,                 # 自动下载数据集（如果不存在）
    'replace_value': -105,            # 未检测到的 WAP 值替换值（原始值 100）
    'normalization': 'minmax',        # 归一化方法: 'minmax' | 'standard' | 'robust' | None
    'dimension_reduction': 'pca',     # 降维方法: 'pca' | None
    'n_components': 50,               # 降维后组件数量
}
```

**关键参数说明**

| 参数 | 说明 | 选项 / 建议 |
| --- | --- | --- |
| `normalization` | 数据归一化方法 | `minmax`：缩放至 [0,1]；`standard`：标准化（均值0、方差1）；`robust`：对极端值稳健；`None`：不归一化 |
| `dimension_reduction` | 降维方法 | `pca`：主成分分析；`None`：不降维 |
| `n_components` | PCA 保留组件数 | 数值越大，信息保留越多，计算量越大 |

### 2. 模型配置

#### SVR 模型参数
```python
'svr': {
    'kernel': 'rbf',   # 核函数: 'linear' | 'rbf' | 'poly' | 'sigmoid'
    'C': 10.0,         # 正则化参数
    'epsilon': 0.1,    # ε-不敏感损失函数的 ε
    'gamma': 'scale',  # 核系数: 'scale' | 'auto' | 浮点数
    'degree': 3,       # 多项式核次数（poly）
}
```

| 参数 | 说明 | 建议 |
| --- | --- | --- |
| `kernel` | 核函数类型 | `linear`：线性；`rbf`：径向基；`poly`：多项式；`sigmoid`：S 型 |
| `C` | 正则化参数 | 大值：可能过拟合；小值：可能欠拟合 |
| `epsilon` | 不敏感区域宽度 | 控制支持向量数量 |
| `gamma` | 核系数 | 大值：影响范围小；小值：影响范围大 |

#### Transformer 模型参数
```python
'transformer': {
    'input_dim': 520,      # 输入维度 (WAP 数)
    'd_model': 256,        # 模型维度
    'nhead': 8,            # 注意力头数 (需整除 d_model)
    'num_layers': 4,       # 编码器层数
    'dim_feedforward': 512,# 前馈网络维度
    'dropout': 0.1,        # Dropout 率
    'batch_size': 64,      # 批次大小
    'epochs': 100,         # 训练轮数
    'lr': 0.001            # 学习率
}
```

| 参数 | 说明 | 建议 |
| --- | --- | --- |
| `d_model` | 模型维度 | 64–512：维度大，表示力强，计算量高 |
| `nhead` | 注意力头数 | 必须为 `d_model` 的约数 |
| `num_layers` | Transformer 层数 | 层数多，表达能力强，训练更难 |
| `batch_size` | 批次大小 | 大批次训练快，但占内存 |

#### 混合模型配置
```python
'hybrid': {
    'integration_type': 'feature_extraction',    # 'feature_extraction' | 'ensemble' | 'end2end'
    'weights': {'svr': 0.5, 'transformer': 0.5}  # 仅 'ensemble' 使用
}
```

| 参数 | 说明 |
| --- | --- |
| `integration_type` | `feature_extraction`：Transformer 提特征+SVR；`ensemble`：模型独立训练再加权；`end2end`：端到端统一架构 |
| `weights` | `ensemble` 时两个模型的权重 |

#### 反馈机制配置
```python
'feedback': {
    'enabled': True,          # 启用 / 禁用
    'learning_rate': 0.01,    # 低级反馈学习率
    'meta_learning_rate': 0.001, # 高级反馈学习率
    'feedback_window': 100    # 反馈窗口大小
}
```

### 3. 优化配置
```python
OPTIMIZATION_CONFIG = {
    'n_trials': 100,          # Optuna 试验次数
    'timeout': None,          # 超时时间 (秒)
    'n_jobs': 1,              # 并行作业数
    'cv': 5,                  # 交叉验证折数
    'cv_method': 'spatial',   # 'spatial' | 'random' | 'hierarchical'
    'search_space': 'default' # 搜索空间: 'default' | 'light' | 'comprehensive' | …
}
```

### 4. 日志和评估配置
```python
LOGGING_CONFIG = {
    'log_level': 'INFO',                 # DEBUG | INFO | WARNING | ERROR
    'save_epoch_checkpoints': False,     # 保存模型检查点
    'checkpoint_frequency': 10,          # 检查点频率
    'visualization_formats': ['png', 'pdf'], # 可视化格式
}
```

---

## 模型调优

### 调参策略

#### 1. 使用 Optuna 自动调参
```bash
# 设置较多的优化试验次数
python main.py --n_trials 200
```
自动优化参数：集成类型、Transformer 参数、SVR 参数等。

#### 2. 手动调参
- **SVR**:  
  - `kernel`：`rbf` 通常最好，可用 `linear` 作基准  
  - `C`：0.1、1、10、100  
  - `epsilon`：0.01、0.1、0.5、1  
- **Transformer**:  
  - `d_model`：64、128、256、512  
  - `nhead`：2、4、8、16  
  - `num_layers`：2、4、6、8  
- **集成策略**:
  ```bash
  python main.py --integration_type feature_extraction
  python main.py --integration_type ensemble
  python main.py --integration_type end2end
  ```
- **反馈机制**:
  ```bash
  python main.py --with_feedback
  python main.py --without_feedback
  ```

### 不同集成策略特点
| 策略 | 优点 | 适合场景 | 调参重点 |
| --- | --- | --- | --- |
| `feature_extraction` | Transformer 表示 + SVR 泛化 | 样本多、关系复杂 | `d_model`、`num_layers`、SVR `kernel` & `C` |
| `ensemble` | 模型互补，准确率高 | 追求高准确度 | Ensemble 权重及各自参数 |
| `end2end` | 统一架构，端到端优化 | 高度非线性特征 | 整体架构，尤其 `d_model`、`num_layers` |

### 常用调参组合
```bash
# 快速测试
python main.py --integration_type feature_extraction --no_optimization

# 高精度
python main.py --integration_type ensemble --n_trials 200

# 平衡
python main.py --integration_type feature_extraction --with_feedback --n_trials 50

# 完整比较
python run_experiments.py --experiment all
```

---

## 结果解读

实验结果保存在 `./experiments` 目录（如 `./experiments/feature_extraction_with_feedback_20250506_191730/`）。

### 关键结果文件
| 文件 | 内容 | 查看方式 |
| --- | --- | --- |
| `results/processed/evaluation_results_*.json` | 平均误差、中位数误差、楼层准确率 | 文本编辑器 |
| `predictions/final/detailed_predictions.csv` | 真实位置、预测位置、误差 | Excel / CSV 查看器 |
| `csv_records/training/training_history_*.csv` | 每轮损失、学习率 | Excel / CSV 查看器 |
| `optuna_results/best_params/best_params_*.json` | 最佳超参数 | 文本编辑器 |

### 关键可视化文件
| 图 | 路径 | 解读 |
| --- | --- | --- |
| 学习曲线 | `visualizations/training/learning_curves_*.png` | 验证损失高于训练 → 可能过拟合 |
| 误差分布 | `visualizations/error_analysis/error_distribution_*.png` | 分布越集中到低误差越好 |
| 误差 CDF | `visualizations/error_analysis/error_cdf_*.png` | 曲线越陡性能越好 |
| 楼层混淆矩阵 | `visualizations/error_analysis/floor_confusion_matrix_*.png` | 对角线越亮越好 |
| 优化历史 | `optuna_results/visualizations/optimization_history_*.png` | 误差随试验降低 |

### 实验比较结果
| 比较 | 文件 | 关键指标 |
| --- | --- | --- |
| 集成策略 | `integration_comparison_*/results/comparisons/*.json` | mean & median errors, floor accuracies |
| 反馈机制 | `feedback_comparison_*/results/comparisons/*.json` | mean & median errors, p75 & p90 errors |
| 优化水平 | `optimization_comparison_*/results/comparisons/*.json` | mean errors, train times |
| 可视化 | `*/visualizations/comparisons/*.png` | 条形图、折线图 |

### 性能指标详解
| 指标 | 单位 | 含义 | 好的范围 |
| --- | --- | --- | --- |
| `mean_error` | 米 | 平均欧氏距离误差 | < 5 (理想 2–3) |
| `median_error` | 米 | 误差中位数 | < mean_error，通常 2–4 |
| `75th_percentile` | 米 | 75% 误差阈值 | 5–7 |
| `90th_percentile` | 米 | 90% 误差阈值 | 7–10 |
| `floor_accuracy` | % | 楼层预测准确率 | ≥ 90% |
| `combined_error` | 米 | 2D 误差 + 楼层惩罚 | — |

---

## 常见问题排查

### 1. 运行错误

#### `AssertionError: embed_dim must be divisible by num_heads`
- **问题**：`d_model` 不能被 `nhead` 整除  
- **解决**：调整 `d_model` 或 `nhead` 使可整除

#### CUDA 内存不足
- **问题**：GPU 内存不足  
- **解决**：减小 `batch_size`、`d_model`、`num_layers`，或切换 CPU

#### `FileNotFoundError`（数据集）
- **问题**：找不到 UJIIndoorLoc 数据集  
- **解决**：确认 `DATA_CONFIG['download'] = True`，或手动下载

### 2. 性能问题

#### 训练误差高 / 验证差
- 查看学习曲线与日志  
- 解决：增训练轮数、调学习率、换集成策略、启用优化

#### 模型过拟合
- 表现：训练误差低，验证高  
- 解决：加正则化、减模型复杂度、数据增强

#### 楼层预测准确率低
- 检查楼层混淆矩阵  
- 解决：调分类器参数或更换分类器

### 3. README 文件问题
`experiments` 目录含大量自动生成的 `README.md`。若无法打开：

1. 文件被程序锁定  
2. 缺少 Markdown 查看器  
3. 权限问题  

解决：使用 VS Code、Typora 等支持 Markdown 的编辑器并检查权限。
