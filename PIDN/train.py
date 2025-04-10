import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from datetime import datetime
import logging


# 1.日志配置
def setup_logger():
    os.makedirs("PIDN-logs", exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"PIDN-logs/training_{current_time}.log"

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


# 3. 定义模型和物理约束的损失函数
class PIDN(nn.Module):
    """物理约束去噪网络（基于DSDN改进）"""

    def __init__(self, in_channels=1, num_res_blocks=15):
        super(PIDN, self).__init__()

        # 下采样部分（同DSDN）
        self.down_sampling = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 残差块部分（增加批归一化）
        self.res_blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64)
            ) for _ in range(num_res_blocks)
        ])

        # 输出层（增加Sigmoid保证非负）
        self.conv_out = nn.Sequential(
            nn.Conv1d(64, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出限制在[0,1]
        )

    def forward(self, x):
        x = self.down_sampling(x)
        identity = x
        x = self.res_blocks(x)
        x = x + identity  # 残差连接
        return self.conv_out(x)


class PhysicsAwareLoss(nn.Module):
    """物理约束损失函数"""

    def __init__(self, alpha=0.1, beta=0.05, gamma=0.01):
        super().__init__()
        self.alpha = alpha  # 平滑性约束权重
        self.beta = beta    # 峰保持约束权重
        self.gamma = gamma  # 噪声模型约束权重

    def forward(self, pred, clean, noisy):
        # 基础MSE损失
        mse_loss = F.mse_loss(pred, clean)

        # 平滑性约束（一阶差分惩罚）
        diff1_pred = torch.diff(pred, dim=-1)
        diff1_clean = torch.diff(clean, dim=-1)
        smooth_loss = F.l1_loss(diff1_pred, diff1_clean)

        # 峰保持约束（局部极值保护）—— 需要与二阶差分对齐
        second_order_diff = torch.diff(clean, n=2, dim=-1)       # shape: [B, 1, L-2]
        peak_mask = (second_order_diff < 0).float()              # shape: [B, 1, L-2]

        pred_center = pred[:, :, 1:-1]                            # 裁剪与peak_mask对齐
        clean_center = clean[:, :, 1:-1]
        peak_loss = F.mse_loss(pred_center * peak_mask, clean_center * peak_mask)

        # 噪声分布约束（混合高斯-泊松模型）
        noise = noisy - clean
        var = torch.var(noise, dim=-1, keepdim=True)
        poisson_loss = torch.mean((noise ** 2 - var) ** 2)  # 泊松噪声方差稳定性约束

        # 总损失
        total_loss = (mse_loss +
                      self.alpha * smooth_loss +
                      self.beta * peak_loss +
                      self.gamma * poisson_loss)

        return total_loss


# 4. 配置超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
LR = 0.001
BATCH_SIZE = 16
SAVE_PATH = "PIDN-saved_model/PIDN_best.pth"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# 记录训练元数据
training_metadata = {
    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "device": str(device),
    "epochs": EPOCHS,
    "learning_rate": LR,
    "batch_size": BATCH_SIZE,
    "model_architecture": str(PIDN())
}

# 5. 加载数据
train_dataset, val_dataset, test_dataset = load_datasets()
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 6. 初始化模型
model = PIDN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = PhysicsAwareLoss()
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
        loss = criterion(output, clean, noisy)
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