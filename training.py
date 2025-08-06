"""
训练函数模块 - 增强版，包含详细的epoch记录
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
from typing import Tuple, List, Dict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class NaNLossError(Exception):
    """NaN损失异常"""
    pass


class Trainer:
    """增强版训练器 - 包含详细记录功能"""

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        # 优化器和损失函数
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.criterion = nn.MSELoss()

        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0

        # 详细记录
        self.detailed_history = {
            'epochs': [],
            'train_losses': [],
            'val_losses': [],
            'train_batch_losses': [],
            'val_batch_losses': [],
            'learning_rates': [],
            'gradient_norms': [],
            'weight_norms': [],
            'timestamps': [],
            'early_stopping_info': []
        }

        # 创建详细记录目录
        self.log_dir = os.path.join(config.log_dir, 'training_details')
        os.makedirs(self.log_dir, exist_ok=True)

    def train_autoencoder(self, X_train, X_val, epochs=None):
        """训练Transformer自编码器（增强版）"""

        if epochs is None:
            epochs = self.config.epochs

        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(X_train, dtype=torch.float32)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(X_val, dtype=torch.float32)
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        logger.info("开始训练Transformer自编码器...")
        training_start_time = datetime.now()

        for epoch in range(epochs):
            epoch_start_time = datetime.now()

            # 训练阶段
            self.model.train()
            train_loss, train_batch_losses, grad_norms = self._train_epoch_detailed(train_loader)

            # 验证阶段
            self.model.eval()
            val_loss, val_batch_losses = self._validate_epoch_detailed(val_loader)

            # 计算权重范数
            weight_norm = self._calculate_weight_norm()

            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # 记录详细信息
            self.detailed_history['epochs'].append(epoch + 1)
            self.detailed_history['train_losses'].append(train_loss)
            self.detailed_history['val_losses'].append(val_loss)
            self.detailed_history['train_batch_losses'].append(train_batch_losses)
            self.detailed_history['val_batch_losses'].append(val_batch_losses)
            self.detailed_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            self.detailed_history['gradient_norms'].append(np.mean(grad_norms))
            self.detailed_history['weight_norms'].append(weight_norm)
            self.detailed_history['timestamps'].append(datetime.now().isoformat())

            epoch_time = (datetime.now() - epoch_start_time).total_seconds()

            # 详细日志
            logger.info(f"Epoch [{epoch + 1}/{epochs}] "
                        f"Train Loss: {train_loss:.6f}, "
                        f"Val Loss: {val_loss:.6f}, "
                        f"Time: {epoch_time:.2f}s, "
                        f"Grad Norm: {np.mean(grad_norms):.4f}, "
                        f"Weight Norm: {weight_norm:.4f}")

            # 检查NaN
            if np.isnan(train_loss) or np.isnan(val_loss):
                self._save_error_state(epoch, train_loss, val_loss)
                raise NaNLossError("训练过程中出现NaN损失")

            # 早停检查
            early_stop_info = self._check_early_stopping_detailed(val_loss, epoch)
            self.detailed_history['early_stopping_info'].append(early_stop_info)

            if early_stop_info['should_stop']:
                logger.info(f"早停触发，停止训练 (Epoch {epoch + 1})")
                break

            # 定期保存检查点和详细记录
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, train_loss, val_loss)
                self._save_detailed_history()

        # 训练结束
        training_time = (datetime.now() - training_start_time).total_seconds()
        logger.info(f"训练完成，总用时: {training_time:.2f}秒")

        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("已加载最佳模型参数")

        # 保存最终的详细历史
        self._save_detailed_history()
        self._save_training_summary(training_time)

        return self.model, self.train_losses, self.val_losses

    def _train_epoch_detailed(self, train_loader):
        """训练一个epoch（详细版）"""
        total_loss = 0.0
        total_samples = 0
        batch_losses = []
        gradient_norms = []

        for batch_idx, (X_batch, _) in enumerate(train_loader):
            X_batch = X_batch.to(self.device)
            batch_size = X_batch.size(0)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, X_batch)

            # 反向传播
            loss.backward()

            # 记录梯度范数（裁剪前）
            grad_norm_before = self._calculate_gradient_norm()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.gradient_clip
            )

            # 记录梯度范数（裁剪后）
            grad_norm_after = self._calculate_gradient_norm()
            gradient_norms.append(grad_norm_after)

            self.optimizer.step()

            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            total_loss += batch_loss * batch_size
            total_samples += batch_size

            # 定期记录
            if batch_idx % 10 == 0:
                logger.debug(f"Batch {batch_idx}/{len(train_loader)}, "
                           f"Loss: {batch_loss:.6f}, "
                           f"Grad Norm (before/after): {grad_norm_before:.4f}/{grad_norm_after:.4f}")

        return total_loss / total_samples, batch_losses, gradient_norms

    def _validate_epoch_detailed(self, val_loader):
        """验证一个epoch（详细版）"""
        total_loss = 0.0
        total_samples = 0
        batch_losses = []

        with torch.no_grad():
            for batch_idx, (X_batch, _) in enumerate(val_loader):
                X_batch = X_batch.to(self.device)
                batch_size = X_batch.size(0)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, X_batch)

                batch_loss = loss.item()
                batch_losses.append(batch_loss)
                total_loss += batch_loss * batch_size
                total_samples += batch_size

        return total_loss / total_samples, batch_losses

    def _check_early_stopping_detailed(self, val_loss, epoch):
        """检查早停条件（详细版）"""
        info = {
            'epoch': epoch + 1,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'should_stop': False,
            'improved': False,
            'improvement_ratio': 0.0
        }

        if self.best_val_loss == float('inf'):
            # 第一次验证
            self.best_val_loss = val_loss
            self.best_model_state = self.model.state_dict().copy()
            info['improved'] = True
            info['improvement_ratio'] = 1.0
            return info

        # 计算改进比例
        improvement_ratio = (self.best_val_loss - val_loss) / self.best_val_loss
        info['improvement_ratio'] = improvement_ratio

        if improvement_ratio > self.config.min_delta_ratio:
            # 有改进
            self.best_val_loss = val_loss
            self.best_model_state = self.model.state_dict().copy()
            self.patience_counter = 0
            info['improved'] = True
            logger.info(f"验证损失改进: {improvement_ratio * 100:.2f}%")
        else:
            # 无改进
            self.patience_counter += 1
            info['patience_counter'] = self.patience_counter
            logger.info(f"验证损失无改进 (耐心: {self.patience_counter}/{self.config.early_stopping_patience})")

            if self.patience_counter >= self.config.early_stopping_patience:
                info['should_stop'] = True

        return info

    def _calculate_gradient_norm(self):
        """计算梯度范数"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def _calculate_weight_norm(self):
        """计算权重范数"""
        total_norm = 0.0
        for p in self.model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def _save_checkpoint(self, epoch, train_loss, val_loss):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }

        checkpoint_path = os.path.join(
            self.log_dir,
            f'checkpoint_epoch_{epoch + 1}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"检查点已保存: {checkpoint_path}")

    def _save_error_state(self, epoch, train_loss, val_loss):
        """保存错误状态（用于调试）"""
        error_info = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'detailed_history': self.detailed_history
        }

        error_path = os.path.join(
            self.log_dir,
            f'error_state_epoch_{epoch + 1}.pth'
        )
        torch.save(error_info, error_path)
        logger.error(f"错误状态已保存: {error_path}")

    def _save_detailed_history(self):
        """保存详细训练历史"""
        # 保存为JSON
        json_path = os.path.join(self.log_dir, 'training_history.json')
        with open(json_path, 'w') as f:
            # 转换numpy数组为列表
            history_to_save = {}
            for key, value in self.detailed_history.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], np.ndarray):
                        history_to_save[key] = [v.tolist() for v in value]
                    else:
                        history_to_save[key] = value
                else:
                    history_to_save[key] = value
            json.dump(history_to_save, f, indent=4)

        # 保存为CSV（主要指标）
        csv_path = os.path.join(self.log_dir, 'training_metrics.csv')
        metrics_df = pd.DataFrame({
            'epoch': self.detailed_history['epochs'],
            'train_loss': self.detailed_history['train_losses'],
            'val_loss': self.detailed_history['val_losses'],
            'learning_rate': self.detailed_history['learning_rates'],
            'gradient_norm': self.detailed_history['gradient_norms'],
            'weight_norm': self.detailed_history['weight_norms'],
            'timestamp': self.detailed_history['timestamps']
        })
        metrics_df.to_csv(csv_path, index=False)

        logger.debug(f"详细历史已保存: {json_path}, {csv_path}")

    def _save_training_summary(self, training_time):
        """保存训练总结"""
        summary = {
            'total_epochs': len(self.train_losses),
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'training_time_seconds': training_time,
            'early_stopped': self.patience_counter >= self.config.early_stopping_patience,
            'config': self.config.__dict__,
            'model_architecture': str(self.model),
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }

        summary_path = os.path.join(self.log_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

        logger.info(f"训练总结已保存: {summary_path}")

    def extract_features(self, X_data, batch_size=None):
        """提取特征（增强版，包含进度记录）"""
        if batch_size is None:
            batch_size = self.config.batch_size

        self.model.eval()
        data_loader = torch.utils.data.DataLoader(
            torch.tensor(X_data, dtype=torch.float32),
            batch_size=batch_size,
            shuffle=False
        )

        features = []
        extraction_info = []

        with torch.no_grad():
            for batch_idx, X_batch in enumerate(data_loader):
                X_batch = X_batch.to(self.device)

                if hasattr(self.model, 'extract_features'):
                    encoded = self.model.extract_features(X_batch)
                else:
                    encoded = self.model.encode(X_batch)

                features.append(encoded.cpu().numpy())

                # 记录批次信息
                extraction_info.append({
                    'batch_idx': batch_idx,
                    'batch_size': X_batch.size(0),
                    'feature_dim': encoded.size(-1)
                })

                if batch_idx % 50 == 0:
                    logger.debug(f"特征提取进度: {batch_idx}/{len(data_loader)}")

        # 保存提取信息
        extraction_log_path = os.path.join(
            self.log_dir,
            f'feature_extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(extraction_log_path, 'w') as f:
            json.dump(extraction_info, f, indent=4)

        return np.vstack(features)