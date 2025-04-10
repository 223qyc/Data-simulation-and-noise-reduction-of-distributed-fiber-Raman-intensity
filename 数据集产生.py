import numpy as np
import os


def generate_signals(num_samples, signal_length=10000,
                     snr_range=(20, 37), extreme_noise_prob=0.05,
                     max_repeat=40):
    """
    生成合成拉曼信号（含噪声和干净信号）
    1. 生成具有连续性的拉曼信号（每次连续长度随机0-max_repeat）
    2. 添加固定信噪比的高斯白噪声
    3. 以一定概率添加极端噪声

    参数：
    num_samples: 样本数量
    signal_length: 信号长度
    snr_range: 信噪比范围(dB)
    extreme_noise_prob: 极端噪声概率
    max_repeat: 最大连续重复次数

    返回：
    clean_signals: 干净信号 (num_samples, signal_length)
    noisy_signals: 带噪信号 (num_samples, signal_length)

    """
    clean_signals = np.zeros((num_samples, signal_length))

    for i in range(num_samples):
        pos = 0
        while pos < signal_length:
            segment_length = np.random.randint(1, max_repeat + 1)
            segment_length = min(segment_length, signal_length - pos)
            segment_value = np.random.uniform(0, 1)
            clean_signals[i, pos:pos + segment_length] = segment_value
            pos += segment_length

    # 归一化
    mins = clean_signals.min(axis=1, keepdims=True)
    maxs = clean_signals.max(axis=1, keepdims=True)
    clean_signals = (clean_signals - mins) / (maxs - mins + 1e-8)

    # 添加高斯噪声
    signal_power = np.mean(clean_signals ** 2, axis=1, keepdims=True)
    snrs = np.random.uniform(*snr_range, size=(num_samples, 1))
    noise_std = np.sqrt(signal_power / (10 ** (snrs / 10)))
    gaussian_noise = noise_std * np.random.randn(num_samples, signal_length)
    noisy_signals = clean_signals + gaussian_noise

    # 添加极端噪声
    extreme_mask = np.random.rand(num_samples) < extreme_noise_prob
    extreme_samples = np.where(extreme_mask)[0]

    for sample_idx in extreme_samples:
        num_spikes = np.random.randint(1, 4)
        for _ in range(num_spikes):
            spike_width = np.random.randint(20, 100)
            spike_start = np.random.randint(0, signal_length - spike_width)
            spike_amp = np.random.uniform(5, 15) * noise_std[sample_idx]
            if np.random.rand() > 0.5:
                noisy_signals[sample_idx, spike_start:spike_start + spike_width] += spike_amp
            else:
                noisy_signals[sample_idx, spike_start:spike_start + spike_width] -= spike_amp

    return clean_signals, noisy_signals, snrs, noise_std


def save_dataset(name, num_samples):
    """
    生成并保存一个数据集（train_val/test）
    """
    print(f"正在生成数据集: {name} ({num_samples} samples)")
    clean_signals, noisy_signals, snrs, noise_std = generate_signals(num_samples=num_samples)
    save_path = os.path.join('data', f'{name}.npz')
    np.savez_compressed(save_path,
                        clean_signals=clean_signals,
                        noisy_signals=noisy_signals,
                        snrs=snrs,
                        noise_std=noise_std)
    print(f"{name} 数据已保存到: {os.path.abspath(save_path)}\n")


def generate_all_datasets():
    os.makedirs('data', exist_ok=True)
    save_dataset(name='train_val', num_samples=5000)
    save_dataset(name='test', num_samples=1000)


generate_all_datasets()
