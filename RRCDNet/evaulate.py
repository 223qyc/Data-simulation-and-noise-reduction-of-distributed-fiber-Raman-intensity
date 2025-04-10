import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from datetime import datetime
import torch
import torch.nn as nn
from train import RRCDNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- 评估指标 ----------
def compute_mse(x, y):
    return np.mean((x - y) ** 2)

def compute_smoothness(x):
    return np.mean(np.abs(np.diff(x)))

def compute_peak_to_peak(x):
    return np.max(x) - np.min(x)


# ---------- 模型评估 ----------
def evaluate(model, noisy_data, clean_data):
    model.eval()
    metrics = {'MSE': [], 'SSIM': [], 'Smoothness': [], 'Peak2Peak': []}

    for noisy, clean in zip(noisy_data, clean_data):
        with torch.no_grad():
            input_tensor = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            denoised = model(input_tensor).cpu().squeeze().numpy()

        metrics['MSE'].append(compute_mse(denoised, clean))
        metrics['SSIM'].append(ssim(clean, denoised, data_range=clean.max() - clean.min()))
        metrics['Smoothness'].append(compute_smoothness(denoised))
        metrics['Peak2Peak'].append(compute_peak_to_peak(denoised))

    return {k: np.mean(v) for k, v in metrics.items()}


# ---------- 可视化 ----------
def plot_sample(noisy, clean, denoised, idx, save_dir):
    plt.figure(figsize=(12, 4))
    plt.plot(clean, label="Clean", linewidth=1)
    plt.plot(noisy, label="Noisy", linestyle="--", alpha=0.6)
    plt.plot(denoised, label="Denoised", linestyle="-.", linewidth=1.5)
    plt.xlabel("Sampling Point")
    plt.ylabel("Amplitude")
    plt.title(f"Sample {idx} Denoising Result")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, f"compare_sample_{idx}.svg")
    plt.savefig(path, format="svg")
    plt.close()
    print(f"[✓] Saved: {path}")


# ---------- 主函数 ----------
def main():
    data = np.load("../data/test.npz")
    noisy_data, clean_data = data["noisy_signals"], data["clean_signals"]
    print(f"[INFO] Loaded {len(noisy_data)} test samples")

    model = RRCDNet().to(DEVICE)
    model.load_state_dict(torch.load("RRCDNet-saved_model/RRCDNet_best.pth", map_location=DEVICE))
    print("[INFO] Model loaded and ready")

    metrics = evaluate(model, noisy_data, clean_data)
    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("eval_results", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")

    for i in [0, 1]:  # 可视化前两个样本
        with torch.no_grad():
            input_tensor = torch.tensor(noisy_data[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            denoised = model(input_tensor).cpu().squeeze().numpy()
        plot_sample(noisy_data[i], clean_data[i], denoised, i, save_dir)


if __name__ == "__main__":
    main()
