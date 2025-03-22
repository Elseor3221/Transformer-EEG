# -*- coding: utf-8 -*-
import torch


class Config:
    # 数据路径（只需指定.set文件）
    EEGLAB_SET_FILE = "data/depressed_01.set"

    # 数据参数
    SEGMENT_DURATION = 2.0  # 分段时长（秒）
    TEST_SIZE = 0.2  # 测试集比例

    # 模型参数
    D_MODEL = 129  # Transformer隐藏维度
    NHEAD = 8  # 注意力头数
    NUM_LAYERS = 4  # Transformer层数
    NUM_CLASSES = 2  # 分类类别数

    # 训练参数
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


config = Config()
