import numpy as np
import matplotlib.pyplot as plt
import os

npy_path = r"D:\毕设\DPZero-main\roberta\result\SST-2-roberta-large-prompt-standard-R2T-SST-2-20260525_021215seed42-bs64-lr5e-6-eps1e-3-wd0-step10000-evalstep10000\512-42\selected_Cs.npy"

# 加载数据
Cs = np.load(npy_path)
steps = np.arange(len(Cs))

# 统计每个 C 值出现的频率
unique_Cs, counts = np.unique(Cs, return_counts=True)

print("C 值统计：")
for c, cnt in zip(unique_Cs, counts):
    print(f"C = {c:3d} : {cnt:5d} 次 ({cnt/len(Cs)*100:.2f}%)")

# 图1：直方图（离散分布）
plt.figure(figsize=(10, 6))
bins = np.arange(min(unique_Cs)-0.5, max(unique_Cs)+1.5, 1)
plt.hist(Cs, bins=bins, rwidth=0.8, align='mid', edgecolor='black')
plt.xlabel('Selected C')
plt.ylabel('Frequency')
plt.title('Distribution of C values selected by R2T during training')
plt.grid(True, linestyle='--', alpha=0.5)
for c, cnt in zip(unique_Cs, counts):
    plt.text(c, cnt + max(counts)*0.01, str(cnt), ha='center', va='bottom')
plt.tight_layout()
plt.savefig("selected_Cs_distribution.png", dpi=150)
plt.show()

# 图2：折线图（C 值随训练步数变化）
plt.figure(figsize=(12, 5))
plt.plot(steps, Cs, marker='.', linestyle='-', markersize=2, alpha=0.7)
plt.xlabel('Training step')
plt.ylabel('Selected C')
plt.title('Selected C value per training step (R2T)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("selected_Cs_over_time.png", dpi=150)
plt.show()

# 图3：滑动平均（平滑趋势）
window = 50
if len(Cs) >= window:
    smoothed = np.convolve(Cs, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(12, 5))
    plt.plot(steps[window-1:], smoothed, 'r-', linewidth=2)
    plt.xlabel('Training step')
    plt.ylabel('Smoothed selected C (window=50)')
    plt.title('Smoothed trend of C values selected by R2T')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("selected_Cs_smoothed.png", dpi=150)
    plt.show()