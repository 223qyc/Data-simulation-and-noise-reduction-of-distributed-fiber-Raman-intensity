import numpy as np
import matplotlib.pyplot as plt
import os


plt.rcParams['font.sans-serif'] = ['PingFang HK']  # Windows需要做出调整
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def inspect_and_visualize_data(file_path):
    """
    查看 .npz 文件中的数据形状、数据结构，并对第一组数据进行可视化对比
    """
    # 加载数据
    data = np.load(file_path)

    # 打印文件中的所有数组名称
    print("文件中的数组名称:")
    for key in data.files:
        print(f"  - {key}")

    # 打印每个数组的形状
    print("\n每个数组的形状:")
    for key in data.files:
        print(f"  - {key}: {data[key].shape}")

    # 输出第一组数据
    print("\n输出第一组数据:")
    for key in data.files:
        print(f"\n{key}:")
        print(data[key][0])  # 输出每个数组的第一组数据

    # 提取第一组数据
    first_clean_signal = data['clean_signals'][0]
    first_noisy_signal = data['noisy_signals'][0]
    first_snr = data['snrs'][0]
    first_noise_std = data['noise_std'][0]

    # 打印第一组数据的信噪比和噪声标准差
    print(f"\n第一个样本的信噪比（dB）: {first_snr[0]:.2f}")
    print(f"第一个样本的噪声标准差: {first_noise_std[0]:.6f}")

    # 可视化第一组数据
    plt.figure(figsize=(12, 6))
    plt.plot(first_clean_signal, label='干净信号', color='blue', linewidth=1)
    plt.plot(first_noisy_signal, label='带噪信号', color='red', linewidth=1, alpha=0.7)
    plt.title('第一个样本的拉曼信号')
    plt.xlabel('时间')
    plt.ylabel('信号强度')
    plt.legend()
    plt.grid(True)
    plt.show()

# 指定文件路径
file_path1 = os.path.join('data', 'train_val.npz')
file_path2=os.path.join('data', 'test.npz')

# 调用函数查看数据并进行可视化
inspect_and_visualize_data(file_path1)
inspect_and_visualize_data(file_path2)

