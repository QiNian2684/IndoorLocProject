# 实验：feature_extraction_with_feedback

开始时间: 2025-05-06 19:16:14

## 目录结构

### `models/`

- `checkpoints/`: 模型检查点，按轮次保存
- `final/`: 最终训练的模型
- `best/`: 性能最优的模型

### `results/`

- `raw/`: 原始评估结果
- `processed/`: 处理后的评估指标
- `comparisons/`: 模型比较结果

### `logs/`

- `training/`: 训练日志
- `evaluation/`: 评估日志
- `optimization/`: 超参数优化日志

### `visualizations/`

- `training/`: 训练可视化图表
- `error_analysis/`: 误差分析图表
- `distribution/`: 数据分布图表
- `comparisons/`: 模型比较图表

### `csv_records/`

- `training/`: 训练记录
- `evaluation/`: 评估记录
- `optimization/`: 优化记录

### `metrics/`

- `by_epoch/`: 按轮次记录的指标
- `by_model/`: 按模型记录的指标
- `by_config/`: 按配置记录的指标

### `predictions/`

- `by_epoch/`: 按轮次记录的预测
- `final/`: 最终模型的预测
- `best/`: 最佳模型的预测

### `optuna_results/`

- `trials/`: 优化试验结果
- `visualizations/`: 优化可视化
- `best_params/`: 最佳参数记录

### `configs/`

- `original/`: 原始配置
- `optimized/`: 优化后的配置
- `by_run/`: 每次运行的配置

