import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .base import PositioningModel


class TransformerRegressor(nn.Module):
    """基于Transformer的回归模型"""

    def __init__(self, input_dim=520, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=512, output_dim=2, dropout=0.1):
        """
        初始化Transformer回归器

        参数:
            input_dim (int): 输入特征维度
            d_model (int): Transformer模型维度
            nhead (int): 多头注意力中的头数
            num_layers (int): Transformer编码器层数
            dim_feedforward (int): 前馈网络的隐藏维度
            output_dim (int): 输出维度（例如，2表示经度和纬度）
            dropout (float): Dropout率
        """
        super().__init__()
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

    def forward(self, x):
        """
        前向传播

        参数:
            x (Tensor): 形状为 (batch_size, input_dim) 的输入张量

        返回:
            Tensor: 形状为 (batch_size, output_dim) 的输出预测
        """
        # 将输入重塑为序列
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        x = self.input_projection(x)  # (batch_size, 1, d_model)
        x = self.transformer_encoder(x)  # (batch_size, 1, d_model)
        x = x.transpose(1, 2)  # (batch_size, d_model, 1)
        x = self.global_pooling(x)  # (batch_size, d_model, 1)
        x = x.squeeze(-1)  # (batch_size, d_model)
        x = self.regressor(x)  # (batch_size, output_dim)
        return x


class TransformerPositioningModel(PositioningModel):
    """使用Transformer模型进行位置预测的包装器类"""

    def __init__(self, input_dim=520, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=512, output_dim=2, dropout=0.1,
                 batch_size=64, epochs=100, lr=0.001, device=None):
        """
        初始化Transformer定位模型

        参数:
            input_dim (int): 输入特征维度
            d_model (int): Transformer模型维度
            nhead (int): 多头注意力中的头数
            num_layers (int): Transformer编码器层数
            dim_feedforward (int): 前馈网络的隐藏维度
            output_dim (int): 输出维度（例如，2表示经度和纬度）
            dropout (float): Dropout率
            batch_size (int): 训练批次大小
            epochs (int): 训练轮数
            lr (float): 学习率
            device (str): 训练设备 ('cuda' 或 'cpu')
        """
        self.model_params = {
            'input_dim': input_dim,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'output_dim': output_dim,
            'dropout': dropout
        }
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.training_history = []

    def fit(self, X, y):
        """
        训练Transformer模型

        参数:
            X: 特征数据
            y: 目标值 (经度和纬度)

        返回:
            self: 模型实例
        """
        # 创建模型
        self.model = TransformerRegressor(**self.model_params).to(self.device)

        # 准备数据
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # 训练循环
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # 记录每个epoch的损失
            avg_loss = epoch_loss / len(dataloader)
            self.training_history.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}')

        return self

    def predict(self, X):
        """
        预测位置

        参数:
            X: 特征数据

        返回:
            ndarray: 预测的位置 (经度和纬度)
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练")

        # 设置为评估模式
        self.model.eval()

        # 转换为张量
        X_tensor = torch.FloatTensor(X).to(self.device)

        # 预测
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()

        return predictions

    def get_model(self):
        """获取底层的PyTorch模型"""
        return self.model

    def save_model(self, path):
        """保存模型到文件"""
        if self.model is None:
            raise RuntimeError("没有模型可保存")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_params': self.model_params,
            'training_history': self.training_history
        }, path)

    def load_model(self, path):
        """从文件加载模型"""
        checkpoint = torch.load(path, map_location=self.device)

        # 更新参数
        self.model_params = checkpoint['model_params']
        self.training_history = checkpoint['training_history']

        # 创建模型并加载权重
        self.model = TransformerRegressor(**self.model_params).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        return self