import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class EEGTransformer(nn.Module):
    def __init__(self, n_channels=128, d_model=128, nhead=8, num_layers=4, num_classes=2):
        super().__init__()
        self.embedding = nn.Linear(n_channels, d_model)

        # 空间编码（基于电极坐标）
        self.space_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )

        # 时间位置编码（动态适应任意长度）
        self.time_pos = nn.Parameter(torch.randn(1000, d_model))  # 假设最大序列长度1000

        # Transformer编码器
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model, dropout=0.1)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)

        # 分类头
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, electrode_coords):
        # x: [batch, 128, time]
        batch_size, _, seq_len = x.shape

        # 嵌入层 [batch, time, 128] -> [batch, time, d_model]
        x = x.permute(0, 2, 1)  # [batch, time, 128]
        x = self.embedding(x)  # [batch, time, d_model]

        # 空间编码 [128, d_model]
        space_enc = self.space_encoder(electrode_coords)  # [128, d_model]
        x += space_enc.unsqueeze(0).unsqueeze(0)  # 广播到 [batch, time, d_model]

        # 动态截取时间编码
        time_enc = self.time_pos[:seq_len, :]  # [seq_len, d_model]
        x += time_enc.unsqueeze(0)  # [batch, time, d_model]

        # Transformer输入需为 [seq_len, batch, d_model]
        x = x.permute(1, 0, 2)  # [time, batch, d_model]
        x = self.transformer(x)  # [time, batch, d_model]

        # 全局平均池化
        x = x.mean(dim=0)  # [batch, d_model]
        return self.fc(x)