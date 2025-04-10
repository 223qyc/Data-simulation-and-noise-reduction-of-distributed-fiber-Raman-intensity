import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from datetime import datetime
import logging


# 1.日志配置
def setup_logger():
    os.makedirs("1DCNN-logs", exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"1DCNN-logs/training_{current_time}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


logger = setup_logger()


# 2. 划分数据集
class RamanDataset(Dataset):
    def __init__(self, clean_signals, noisy_signals):
        self.clean_signals = torch.tensor(clean_signals, dtype=torch.float32).unsqueeze(1)
        self.noisy_signals = torch.tensor(noisy_signals, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.clean_signals)

    def __getitem__(self, idx):
        return self.noisy_signals[idx], self.clean_signals[idx]


def load_datasets(data_dir="../data", train_val_file="train_val.npz", test_file="test.npz", train_ratio=0.9):
    logger.info("Loading datasets...")
    train_val_data = np.load(os.path.join(data_dir, train_val_file))
    test_data = np.load(os.path.join(data_dir, test_file))

    clean_tv = train_val_data["clean_signals"]
    noisy_tv = train_val_data["noisy_signals"]
    clean_test = test_data["clean_signals"]
    noisy_test = test_data["noisy_signals"]

    # 创建训练和验证集
    dataset = RamanDataset(clean_tv, noisy_tv)
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建测试集
    test_dataset = RamanDataset(clean_test, noisy_test)

    logger.info(f"Dataset loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


# 3. 定义模型
class DenoiseCNN(nn.Module):
    def __init__(self):
        super(DenoiseCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            *[nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU()) for _ in range(18)],
            nn.Conv1d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.layers(x)


# 4. 配置超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
LR = 0.0001
BATCH_SIZE = 16
SAVE_PATH = "1DCNN-saved_model/1DCNN_best.pth"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# 记录训练元数据
training_metadata = {
    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "device": str(device),
    "epochs": EPOCHS,
    "learning_rate": LR,
    "batch_size": BATCH_SIZE,
    "model_architecture": str(DenoiseCNN())
}

# 5. 加载数据
train_dataset, val_dataset, test_dataset = load_datasets()
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 6. 初始化模型 ====
model = DenoiseCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
min_val_loss = float('inf')

# 记录损失
train_losses = []
val_losses = []
best_epoch = 0

# 记录训练开始
logger.info("\n" + "=" * 50)
logger.info("Starting Training Session")
logger.info(f"Training metadata:\n{training_metadata}")
logger.info("=" * 50 + "\n")

#  7. 训练
start_time = time.time()
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    train_loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}] Training", leave=False)

    for noisy, clean in train_loop:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        output = model(noisy)
        loss = F.mse_loss(output, clean)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        train_loop.set_postfix(loss=loss.item())

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 验证过程
    model.eval()
    total_val_loss = 0
    val_loop = tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}] Validating", leave=False)
    with torch.no_grad():
        for noisy, clean in val_loop:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            loss = F.mse_loss(output, clean)
            total_val_loss += loss.item()
            val_loop.set_postfix(loss=loss.item())

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # 记录每个epoch的结果
    epoch_log = f"Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
    logger.info(epoch_log)
    print(epoch_log)

    # 保存最优模型
    if avg_val_loss < min_val_loss:
        min_val_loss = avg_val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), SAVE_PATH)
        save_log = f"Model saved to {SAVE_PATH} with Val Loss: {avg_val_loss:.6f}"
        logger.info(save_log)
        print(save_log)

# 8. 训练总结
training_metadata.update({
    "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "training_duration": f"{(time.time() - start_time) / 60:.2f} minutes",
    "best_epoch": best_epoch,
    "best_val_loss": min_val_loss,
    "final_train_loss": train_losses[-1],
    "final_val_loss": val_losses[-1]
})

logger.info("\n" + "=" * 50)
logger.info("Training Summary")
logger.info(f"Best Epoch: {best_epoch}, Best Val Loss: {min_val_loss:.6f}")
logger.info(f"Training Duration: {training_metadata['training_duration']}")
logger.info("=" * 50 + "\n")

# 9. 保存训练结果
os.makedirs("results", exist_ok=True)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 保存损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_curve_path = f"results/loss_curve_{current_time}.png"
plt.savefig(loss_curve_path)
plt.close()

# 保存训练元数据
metadata_path = f"results/training_metadata_{current_time}.txt"
with open(metadata_path, "w") as f:
    for key, value in training_metadata.items():
        f.write(f"{key}: {value}\n")

logger.info(f"Loss curve saved to: {os.path.abspath(loss_curve_path)}")
logger.info(f"Training metadata saved to: {os.path.abspath(metadata_path)}")
logger.info("Training completed successfully!")

print("\nTraining completed. Best model and logs saved.")