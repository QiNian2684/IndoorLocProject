"""
模型定义 - Transformer自编码器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class WiFiTransformerAutoencoder(nn.Module):
    """WiFi信号Transformer自编码器"""

    def __init__(self, input_dim=520, model_dim=128, num_heads=8,
                 num_layers=2, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.model_dim = model_dim

        # 编码器部分
        self.encoder_embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 池化层
        self.encoder_pool = nn.AdaptiveAvgPool1d(1)

        # 解码器部分
        self.decoder_fc = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, input_dim)
        )

        # 特征提取头（用于下游任务）
        self.feature_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, model_dim // 4),
            nn.LayerNorm(model_dim // 4)
        )

        self.activation = nn.ReLU()

    def encode(self, x):
        """编码器"""
        # [batch_size, input_dim] -> [batch_size, model_dim]
        x = self.encoder_embedding(x)
        x = self.activation(x)

        # 添加序列维度
        x = x.unsqueeze(1)  # [batch_size, 1, model_dim]

        # 位置编码
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)

        # Transformer编码
        x = self.transformer_encoder(x)

        # 池化
        x = x.permute(0, 2, 1)  # [batch_size, model_dim, 1]
        x = self.encoder_pool(x).squeeze(-1)  # [batch_size, model_dim]

        return x

    def decode(self, x):
        """解码器"""
        return self.decoder_fc(x)

    def extract_features(self, x):
        """提取用于下游任务的特征"""
        encoded = self.encode(x)
        features = self.feature_head(encoded)
        return features

    def forward(self, x):
        """前向传播 - 自编码器"""
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded