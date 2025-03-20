import torch.nn as nn
import torch


class EEGTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Linear(128, config.D_MODEL)

        # 空间编码
        self.space_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, config.D_MODEL)
        )

        # 时间位置编码
        self.time_pos = nn.Parameter(torch.randn(1000, config.D_MODEL))

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            config.D_MODEL, config.NHEAD,
            dim_feedforward=4 * config.D_MODEL, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, config.NUM_LAYERS
        )

        # 分类头
        self.fc = nn.Linear(config.D_MODEL, config.NUM_CLASSES)

    def forward(self, x, coords):
        batch_size, _, seq_len = x.shape
        x = x.permute(0, 2, 1)
        x = self.embedding(x)

        # 空间编码
        space_enc = self.space_encoder(coords)
        x += space_enc.unsqueeze(1)

        # 时间编码
        time_enc = self.time_pos[:seq_len]
        x += time_enc.unsqueeze(0)

        # Transformer
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.fc(x)