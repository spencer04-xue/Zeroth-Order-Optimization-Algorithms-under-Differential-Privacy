import numpy as np
import matplotlib.pyplot as plt
import os

# 您提供的正确路径（注意转义反斜杠或使用正斜杠）
npy_path = r"D:\毕设\DPZero-main\roberta\result\SST-2-roberta-large-prompt-standard-k512-roberta-large-dpzero-ft-c100-gradrecordseed42-bs64-lr5e-6-eps1e-3-wd0-step800-evalstep10000\512-42\raw_gradients.npy"

# 加载数据
grads = np.load(npy_path)
steps = np.arange(len(grads))

# 打印前20个值确认
print("First 20 gradients:", grads[:20])
print("Min, Max, Mean:", grads.min(), grads.max(), grads.mean())

# 平滑处理
window = 50
if len(grads) >= window:
    smoothed = np.convolve(grads, np.ones(window)/window, mode='valid')
    steps_smoothed = steps[window-1:]
else:
    smoothed = grads
    steps_smoothed = steps

# 绘图
plt.figure(figsize=(12,5))
plt.plot(steps, grads, alpha=0.3, label='raw gradient')
plt.plot(steps_smoothed, smoothed, 'r', linewidth=2, label=f'smoothed (window={window})')
plt.xlabel('Training step')
plt.ylabel('Projected gradient (mean) before clipping')
plt.title('DPZero Gradient Magnitude over Steps (C=100)')
plt.legend()
plt.grid(True)
plt.yscale('symlog')
plt.tight_layout()

output_dir = os.path.dirname(npy_path)
output_path = os.path.join(output_dir, "gradient_trend.png")
plt.savefig(output_path, dpi=150)
print(f"图片已保存至: {output_path}")