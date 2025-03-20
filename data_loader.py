import mne
import numpy as np
import torch
import config
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset



def load_data_from_set(set_file, segment_duration):
    # 加载原始数据
    raw = mne.io.read_raw_eeglab(set_file, preload=True)
    sfreq = raw.info['sfreq']

    # 生成固定时长分段
    events = mne.make_fixed_length_events(raw, id=1, duration=segment_duration)
    epochs = mne.Epochs(
        raw, events,
        tmin=0, tmax=segment_duration - 1 / sfreq,
        baseline=None, preload=True
    )
    eeg_data = epochs.get_data()  # [n_epochs, n_channels, n_time]

    # 直接从raw提取电极坐标（3D位置）
    electrode_coords = np.array([ch['loc'][:3] for ch in raw.info['chs']])
    electrode_coords = torch.from_numpy(electrode_coords).float()

    # 标准化
    n_samples, n_channels, n_time = eeg_data.shape
    scaler = StandardScaler()
    data_flat = eeg_data.reshape(-1, n_time)
    data_normalized = scaler.fit_transform(data_flat.T).T
    X = data_normalized.reshape(n_samples, n_channels, n_time)
    X = torch.from_numpy(X).float()

    return X, electrode_coords


def create_dataloaders(X, y, electrode_coords):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=42
    )

    # 创建数据集（每个样本复制电极坐标）
    train_coords = electrode_coords.unsqueeze(0).repeat(X_train.size(0), 1, 1)
    test_coords = electrode_coords.unsqueeze(0).repeat(X_test.size(0), 1, 1)

    train_dataset = TensorDataset(X_train, y_train, train_coords)
    test_dataset = TensorDataset(X_test, y_test, test_coords)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    return train_loader, test_loader