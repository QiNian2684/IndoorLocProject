import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVR
from .base import PositioningModel
from .transformer_model import TransformerRegressor
from data.feature_engineering import TransformerFeatureExtractor


class SVRTransformerHybrid(PositioningModel):
    """SVR和Transformer的混合模型"""

    def __init__(self, integration_type='feature_extraction',
                 transformer_params=None, svr_params=None, weights=None,
                 early_stopping_config=None):
        """
        初始化SVR+Transformer混合模型

        参数:
            integration_type (str): 集成类型 ('feature_extraction', 'ensemble', 'end2end')
            transformer_params (dict): Transformer模型参数
            svr_params (dict): SVR参数
            weights (dict): 用于ensemble集成的权重 {'svr': 0.5, 'transformer': 0.5}
            early_stopping_config (dict): 早停配置参数
        """
        self.integration_type = integration_type
        self.transformer_params = transformer_params or {}
        self.svr_params = svr_params or {}
        self.weights = weights or {'svr': 0.5, 'transformer': 0.5}
        self.early_stopping_config = early_stopping_config or {}

        # 初始化模型组件
        self.feature_extractor = None
        self.svr_models = None
        self.transformer_model = None
        self.end2end_model = None

        # 训练设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 训练历史
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'timestamp': []
        }

    def fit(self, X, y, X_val=None, y_val=None, callback=None):
        """
        训练混合模型

        参数:
            X: 特征数据
            y: 目标值 (经度和纬度)
            X_val: 验证特征数据 (可选)
            y_val: 验证目标值 (可选)
            callback: 训练回调函数, 接收(epoch, train_loss, val_loss, lr, params) (可选)

        返回:
            self: 模型实例
        """
        if self.integration_type == 'feature_extraction':
            return self._fit_feature_extraction(X, y, X_val, y_val, callback)
        elif self.integration_type == 'ensemble':
            return self._fit_ensemble(X, y, X_val, y_val, callback)
        elif self.integration_type == 'end2end':
            return self._fit_end2end(X, y, X_val, y_val, callback)
        else:
            raise ValueError(f"未知的集成类型: {self.integration_type}")

    def _fit_feature_extraction(self, X, y, X_val=None, y_val=None, callback=None):
        """实现特征提取集成"""
        from datetime import datetime

        # 创建并初始化特征提取器
        self.feature_extractor = TransformerFeatureExtractor(
            input_dim=X.shape[1],
            **self.transformer_params
        ).to(self.device)

        # 如果需要训练Transformer特征提取器
        # 这里略去Transformer训练代码，直接使用预训练模型
        print(f"使用预训练的Transformer特征提取器...")

        # 提取特征
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(X_tensor).cpu().numpy()

        # 训练SVR模型
        print(f"训练SVR模型...")
        self.svr_models = []
        for i in range(y.shape[1]):
            print(f"训练第{i + 1}/{y.shape[1]}个坐标分量的SVR模型...")
            model = SVR(**self.svr_params)
            model.fit(features, y[:, i])
            self.svr_models.append(model)

            # 记录训练信息
            self.training_history['epoch'].append(0)
            self.training_history['train_loss'].append(0.0)  # 无法直接获取SVR的损失
            self.training_history['val_loss'].append(None)
            self.training_history['learning_rate'].append(None)
            self.training_history['timestamp'].append(datetime.now().isoformat())

            # 调用回调函数
            if callback:
                callback(0, 0.0, None, None, None)

        return self

    def _fit_ensemble(self, X, y, X_val=None, y_val=None, callback=None):
        """实现集成方法"""
        from datetime import datetime

        # 导入早停机制
        from utils.early_stopping import EarlyStopping

        # 训练SVR模型
        print(f"训练SVR模型...")
        self.svr_models = []
        for i in range(y.shape[1]):
            print(f"训练第{i + 1}/{y.shape[1]}个坐标分量的SVR模型...")
            model = SVR(**self.svr_params)
            model.fit(X, y[:, i])
            self.svr_models.append(model)

        # 训练Transformer模型
        print(f"训练Transformer模型...")
        self.transformer_model = TransformerRegressor(
            input_dim=X.shape[1],
            output_dim=y.shape[1],
            **self.transformer_params
        ).to(self.device)

        # 准备数据
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # 准备验证数据
        val_dataloader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_dataloader = DataLoader(val_dataset, batch_size=64)

        # 训练Transformer
        optimizer = optim.Adam(self.transformer_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # 初始化早停机制
        early_stopping = None
        if hasattr(self, 'early_stopping_config') and self.early_stopping_config.get('enabled', False):
            early_stopping = EarlyStopping(
                patience=self.early_stopping_config.get('patience', 15),
                min_delta=self.early_stopping_config.get('min_delta', 0.0001),
                verbose=self.early_stopping_config.get('verbose', True),
                mode=self.early_stopping_config.get('mode', 'min'),
            )

        self.transformer_model.train()
        epochs = 100  # 可配置
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self.transformer_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(dataloader)

            # 验证损失
            val_loss = None
            if val_dataloader:
                self.transformer_model.eval()
                val_epoch_loss = 0.0
                with torch.no_grad():
                    for val_inputs, val_targets in val_dataloader:
                        val_outputs = self.transformer_model(val_inputs)
                        val_loss = criterion(val_outputs, val_targets)
                        val_epoch_loss += val_loss.item()
                val_loss = val_epoch_loss / len(val_dataloader)
                self.transformer_model.train()

            # 记录训练信息
            self.training_history['epoch'].append(epoch)
            self.training_history['train_loss'].append(avg_epoch_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(0.001)  # 固定学习率
            self.training_history['timestamp'].append(datetime.now().isoformat())

            # 输出日志
            log_msg = f'Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.6f}'
            if val_loss:
                log_msg += f', Val Loss: {val_loss:.6f}'
            print(log_msg)

            # 调用回调函数
            if callback:
                # 提取当前参数
                current_params = {
                    'transformer': {name: param.data.cpu().numpy().mean()
                                    for name, param in self.transformer_model.named_parameters()}
                }
                callback(epoch, avg_epoch_loss, val_loss, 0.001, current_params)

            # 检查早停条件
            if early_stopping and val_loss is not None:
                if early_stopping(val_loss, self.transformer_model):
                    print(f"早停触发，在第{epoch + 1}轮停止训练")
                    # 加载最佳模型（如果保存了的话）
                    if early_stopping.save_path:
                        self.transformer_model.load_state_dict(torch.load(early_stopping.save_path))
                    break

        return self

    def _fit_end2end(self, X, y, X_val=None, y_val=None, callback=None):
        """实现端到端混合模型"""
        from datetime import datetime

        # 导入早停机制
        from utils.early_stopping import EarlyStopping

        # 创建端到端模型
        self.end2end_model = SVRTransformerEnd2End(
            input_dim=X.shape[1],
            output_dim=y.shape[1],
            epsilon=self.svr_params.get('epsilon', 0.1),
            C=self.svr_params.get('C', 1.0),
            **self.transformer_params
        ).to(self.device)

        # 准备数据
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # 准备验证数据
        val_dataloader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_dataloader = DataLoader(val_dataset, batch_size=64)

        # 训练模型
        optimizer = optim.Adam(self.end2end_model.parameters(), lr=0.001)

        # 初始化早停机制
        early_stopping = None
        if hasattr(self, 'early_stopping_config') and self.early_stopping_config.get('enabled', False):
            early_stopping = EarlyStopping(
                patience=self.early_stopping_config.get('patience', 15),
                min_delta=self.early_stopping_config.get('min_delta', 0.0001),
                verbose=self.early_stopping_config.get('verbose', True),
                mode=self.early_stopping_config.get('mode', 'min'),
            )

        self.end2end_model.train()
        epochs = 100  # 可配置
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self.end2end_model(inputs)
                loss = self.end2end_model.compute_loss(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(dataloader)

            # 验证损失
            val_loss = None
            if val_dataloader:
                self.end2end_model.eval()
                val_epoch_loss = 0.0
                with torch.no_grad():
                    for val_inputs, val_targets in val_dataloader:
                        val_outputs = self.end2end_model(val_inputs)
                        val_loss = self.end2end_model.compute_loss(val_outputs, val_targets)
                        val_epoch_loss += val_loss.item()
                val_loss = val_epoch_loss / len(val_dataloader)
                self.end2end_model.train()

            # 记录训练信息
            self.training_history['epoch'].append(epoch)
            self.training_history['train_loss'].append(avg_epoch_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(0.001)  # 固定学习率
            self.training_history['timestamp'].append(datetime.now().isoformat())

            # 输出日志
            log_msg = f'Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.6f}'
            if val_loss:
                log_msg += f', Val Loss: {val_loss:.6f}'
            print(log_msg)

            # 调用回调函数
            if callback:
                # 提取当前参数
                current_params = {
                    'end2end': {name: param.data.cpu().numpy().mean()
                                for name, param in self.end2end_model.named_parameters()}
                }
                callback(epoch, avg_epoch_loss, val_loss, 0.001, current_params)

            # 检查早停条件
            if early_stopping and val_loss is not None:
                if early_stopping(val_loss, self.end2end_model):
                    print(f"早停触发，在第{epoch + 1}轮停止训练")
                    # 加载最佳模型（如果保存了的话）
                    if early_stopping.save_path:
                        self.end2end_model.load_state_dict(torch.load(early_stopping.save_path))
                    break

        return self

    def predict(self, X):
        """
        预测位置

        参数:
            X: 特征数据

        返回:
            ndarray: 预测的位置 (经度和纬度)
        """
        if self.integration_type == 'feature_extraction':
            return self._predict_feature_extraction(X)
        elif self.integration_type == 'ensemble':
            return self._predict_ensemble(X)
        elif self.integration_type == 'end2end':
            return self._predict_end2end(X)
        else:
            raise ValueError(f"未知的集成类型: {self.integration_type}")

    def _predict_feature_extraction(self, X):
        """使用特征提取方法进行预测"""
        if self.feature_extractor is None or not self.svr_models:
            raise RuntimeError("模型尚未训练")

        # 提取特征
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(X_tensor).cpu().numpy()

        # SVR预测
        predictions = np.column_stack([model.predict(features) for model in self.svr_models])

        return predictions

    def _predict_ensemble(self, X):
        """使用集成方法进行预测"""
        if not self.svr_models or self.transformer_model is None:
            raise RuntimeError("模型尚未训练")

        # SVR预测
        svr_preds = np.column_stack([model.predict(X) for model in self.svr_models])

        # Transformer预测
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.transformer_model.eval()
        with torch.no_grad():
            tf_preds = self.transformer_model(X_tensor).cpu().numpy()

        # 加权组合
        predictions = (self.weights['svr'] * svr_preds +
                       self.weights['transformer'] * tf_preds)

        return predictions

    def _predict_end2end(self, X):
        """使用端到端模型进行预测"""
        if self.end2end_model is None:
            raise RuntimeError("模型尚未训练")

        # 转换为张量并预测
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.end2end_model.eval()
        with torch.no_grad():
            predictions = self.end2end_model(X_tensor).cpu().numpy()

        return predictions

    def save_model(self, path):
        """保存模型到文件"""
        import joblib

        model_data = {
            'integration_type': self.integration_type,
            'transformer_params': self.transformer_params,
            'svr_params': self.svr_params,
            'weights': self.weights,
            'training_history': self.training_history
        }

        if self.integration_type == 'feature_extraction':
            if self.feature_extractor:
                model_data['feature_extractor'] = self.feature_extractor.state_dict()
            model_data['svr_models'] = self.svr_models
        elif self.integration_type == 'ensemble':
            model_data['svr_models'] = self.svr_models
            if self.transformer_model:
                model_data['transformer_model'] = self.transformer_model.state_dict()
        elif self.integration_type == 'end2end':
            if self.end2end_model:
                model_data['end2end_model'] = self.end2end_model.state_dict()

        joblib.dump(model_data, path)

    def load_model(self, path):
        """从文件加载模型"""
        import joblib

        model_data = joblib.load(path)

        self.integration_type = model_data['integration_type']
        self.transformer_params = model_data['transformer_params']
        self.svr_params = model_data['svr_params']
        self.weights = model_data['weights']

        if 'training_history' in model_data:
            self.training_history = model_data['training_history']

        if self.integration_type == 'feature_extraction':
            if 'feature_extractor' in model_data:
                self.feature_extractor = TransformerFeatureExtractor(**self.transformer_params)
                self.feature_extractor.load_state_dict(model_data['feature_extractor'])
            self.svr_models = model_data['svr_models']
        elif self.integration_type == 'ensemble':
            self.svr_models = model_data['svr_models']
            if 'transformer_model' in model_data:
                self.transformer_model = TransformerRegressor(
                    output_dim=2, **self.transformer_params)
                self.transformer_model.load_state_dict(model_data['transformer_model'])
        elif self.integration_type == 'end2end':
            if 'end2end_model' in model_data:
                self.end2end_model = SVRTransformerEnd2End(
                    output_dim=2,
                    epsilon=self.svr_params.get('epsilon', 0.1),
                    C=self.svr_params.get('C', 1.0),
                    **self.transformer_params)
                self.end2end_model.load_state_dict(model_data['end2end_model'])

        return self

    def get_training_history(self):
        """获取训练历史"""
        return self.training_history

    def save_training_history(self, path):
        """保存训练历史到CSV文件"""
        import pandas as pd

        history_df = pd.DataFrame(self.training_history)
        history_df.to_csv(path, index=False)


class SVRLoss(nn.Module):
    """实现SVR的ε-不敏感损失函数"""

    def __init__(self, epsilon=0.1, C=1.0):
        """
        初始化SVR损失

        参数:
            epsilon (float): 不敏感区域的宽度
            C (float): 惩罚参数
        """
        super().__init__()
        self.epsilon = epsilon
        self.C = C

    def forward(self, pred, target):
        """
        计算SVR损失

        参数:
            pred (Tensor): 预测值
            target (Tensor): 目标值

        返回:
            Tensor: 损失值
        """
        # ε-不敏感损失函数
        loss = torch.abs(pred - target) - self.epsilon
        loss = torch.clamp(loss, min=0)
        return self.C * loss.mean()


class SVRTransformerEnd2End(nn.Module):
    """端到端的SVR+Transformer混合模型"""

    def __init__(self, input_dim=520, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=512, output_dim=2, epsilon=0.1, C=1.0, dropout=0.1):
        """
        初始化端到端SVR+Transformer模型

        参数:
            input_dim (int): 输入特征维度
            d_model (int): Transformer模型维度
            nhead (int): 多头注意力中的头数
            num_layers (int): Transformer编码器层数
            dim_feedforward (int): 前馈网络的隐藏维度
            output_dim (int): 输出维度（例如，2表示经度和纬度）
            epsilon (float): SVR损失的不敏感区域宽度
            C (float): SVR损失的惩罚参数
            dropout (float): Dropout率
        """
        super().__init__()

        # 确保d_model能被nhead整除
        if d_model % nhead != 0:
            # 找到最接近的可整除的nhead值
            divisors = [i for i in range(1, min(17, d_model + 1)) if d_model % i == 0]
            if divisors:
                nhead = max([d for d in divisors if d <= nhead]) if any(d <= nhead for d in divisors) else divisors[-1]
            else:
                nhead = 1

        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Linear(d_model, output_dim)
        self.criterion = SVRLoss(epsilon=epsilon, C=C)

    def forward(self, x):
        """
        前向传播

        参数:
            x (Tensor): 形状为 (batch_size, input_dim) 的输入张量

        返回:
            Tensor: 形状为 (batch_size, output_dim) 的输出预测
        """
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        x = self.input_projection(x)  # (batch_size, 1, d_model)
        x = self.transformer_encoder(x)  # (batch_size, 1, d_model)
        x = x.transpose(1, 2)  # (batch_size, d_model, 1)
        x = self.global_pooling(x)  # (batch_size, d_model, 1)
        x = x.squeeze(-1)  # (batch_size, d_model)
        x = self.regressor(x)  # (batch_size, output_dim)
        return x

    def compute_loss(self, pred, target):
        """
        计算损失

        参数:
            pred (Tensor): 预测值
            target (Tensor): 目标值

        返回:
            Tensor: 损失值
        """
        return self.criterion(pred, target)