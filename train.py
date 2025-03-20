import torch.optim as optim
from data_loader import load_data_from_set, create_dataloaders
import torch.nn as nn
import torch
from module import EEGTransformer
from config import config


def main():
    # 加载数据（直接从.set文件）
    X, electrode_coords = load_data_from_set(
        config.EEGLAB_SET_FILE, config.SEGMENT_DURATION
    )

    # 示例标签（需替换为真实标签）
    y = torch.zeros(X.size(0))  # 假设所有样本为健康组

    # 创建DataLoader
    train_loader, test_loader = create_dataloaders(X, y, electrode_coords)

    # 初始化模型
    model = EEGTransformer(config).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 训练循环（与之前相同）
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch, coords_batch in train_loader:
            x_batch = x_batch.to(config.DEVICE)
            y_batch = y_batch.to(config.DEVICE)
            coords_batch = coords_batch.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(x_batch, coords_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_test, y_test, coords_test in test_loader:
                x_test = x_test.to(config.DEVICE)
                y_test = y_test.to(config.DEVICE)
                coords_test = coords_test.to(config.DEVICE)

                outputs = model(x_test, coords_test)
                _, predicted = torch.max(outputs.data, 1)
                total += y_test.size(0)
                correct += (predicted == y_test).sum().item()

        print(f"Epoch {epoch + 1}/{config.EPOCHS} | "
              f"Loss: {train_loss / len(train_loader):.4f} | "
              f"Acc: {correct / total:.4f}")


if __name__ == "__main__":
    main()