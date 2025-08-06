"""
训练函数模块
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class NaNLossError(Exception):
    """NaN损失异常"""
    pass


class Trainer:
    """训练器"""

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

    def train_autoencoder(self, X_train, X_val, epochs=None):
        """训练Transformer自编码器"""

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

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = self._train_epoch(train_loader)

            # 验证阶段
            self.model.eval()
            val_loss = self._validate_epoch(val_loader)

            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            logger.info(f"Epoch [{epoch + 1}/{epochs}] "
                        f"Train Loss: {train_loss:.6f}, "
                        f"Val Loss: {val_loss:.6f}")

            # 检查NaN
            if np.isnan(train_loss) or np.isnan(val_loss):
                raise NaNLossError("训练过程中出现NaN损失")

            # 早停检查
            if self._check_early_stopping(val_loss):
                logger.info(f"早停触发，停止训练 (Epoch {epoch + 1})")
                break

        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("已加载最佳模型参数")

        return self.model, self.train_losses, self.val_losses

    def _train_epoch(self, train_loader):
        """训练一个epoch"""
        total_loss = 0.0
        total_samples = 0

        for X_batch, _ in train_loader:
            X_batch = X_batch.to(self.device)
            batch_size = X_batch.size(0)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, X_batch)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.gradient_clip
            )

            self.optimizer.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / total_samples

    def _validate_epoch(self, val_loader):
        """验证一个epoch"""
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(self.device)
                batch_size = X_batch.size(0)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, X_batch)

                total_loss += loss.item() * batch_size
                total_samples += batch_size

        return total_loss / total_samples

    def _check_early_stopping(self, val_loss):
        """检查早停条件"""
        if self.best_val_loss == float('inf'):
            # 第一次验证
            self.best_val_loss = val_loss
            self.best_model_state = self.model.state_dict().copy()
            return False

        # 计算改进比例
        improvement_ratio = (self.best_val_loss - val_loss) / self.best_val_loss

        if improvement_ratio > self.config.min_delta_ratio:
            # 有改进
            self.best_val_loss = val_loss
            self.best_model_state = self.model.state_dict().copy()
            self.patience_counter = 0
            logger.info(f"验证损失改进: {improvement_ratio * 100:.2f}%")
            return False
        else:
            # 无改进
            self.patience_counter += 1
            logger.info(f"验证损失无改进 (耐心: {self.patience_counter}/{self.config.early_stopping_patience})")
            return self.patience_counter >= self.config.early_stopping_patience

    def extract_features(self, X_data, batch_size=None):
        """提取特征"""
        if batch_size is None:
            batch_size = self.config.batch_size

        self.model.eval()
        data_loader = torch.utils.data.DataLoader(
            torch.tensor(X_data, dtype=torch.float32),
            batch_size=batch_size,
            shuffle=False
        )

        features = []
        with torch.no_grad():
            for X_batch in data_loader:
                X_batch = X_batch.to(self.device)
                if hasattr(self.model, 'extract_features'):
                    encoded = self.model.extract_features(X_batch)
                else:
                    encoded = self.model.encode(X_batch)
                features.append(encoded.cpu().numpy())

        return np.vstack(features)