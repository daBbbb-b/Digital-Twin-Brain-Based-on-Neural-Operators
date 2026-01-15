# MLP 模型数据输入输出格式与处理流程（训练/使用/测试）

本文档以仓库当前实现为准，说明如何把 `.pkl` 数据处理为 MLP 的输入、训练时的 shape 约定，以及推理/测试阶段如何保持数据分布一致。

涉及代码：

- 数据加载与切片：[training/data_loader.py](training/data_loader.py)
- 训练脚本：[training/train_mlp.py](training/train_mlp.py)
- 模型定义：[models/mlp.py](models/mlp.py)

---

## 1. MLP 的真实输入/输出是什么？

`MLP.forward(x)` 在 [models/mlp.py](models/mlp.py) 中定义：

- 输入：二维张量 `x`，shape `[batch, input_dim]`
- 输出：二维张量，shape `[batch, output_dim]`

在本项目训练脚本里：

- `input_dim == C`（通道数/脑区数）
- `output_dim == C`

也就是说：MLP 学的是一个**逐时间点的向量映射**：

$$
x(t)\in\mathbb{R}^{C} \;\longrightarrow\; \hat{u}(t)\in\mathbb{R}^{C}
$$

注意：

- 虽然原始数据是时序 `[T, C]`，但 MLP 训练时会把时间维展平成 batch 维。
- 当前 MLP **不使用时间坐标**（不像 DeepONet/FNO 会显式或隐式使用时间网格）。

---

## 2. `.pkl` 数据格式（训练需要 paired 数据）

训练脚本 [training/train_mlp.py](training/train_mlp.py) 通过 `load_data_from_pkl()` 读取一对张量 `(x, u)`：

- `x`：模型输入序列
- `u`：监督目标序列（label）

`load_data_from_pkl()` 支持两类 `.pkl`：

### 2.1 直接配对格式（推荐用于自己制作数据）

`.pkl` 内包含：

- `"x"`
- `"u"`

建议保存成：

- `x`: `[N, T, C]`
- `u`: `[N, T, C]`

其中 `N` 是样本段数。

归一化：当 `normalize=True` 时，loader 只对 `x` 做 Z-score（`u` 不归一化）。

### 2.2 仿真格式（bold/neural + stimulus_config）

当 `.pkl` 内包含：

- `"bold_signal"` 或 `"neural_activity"`（shape `[n_time, C]`）
- `"stimulus_config"`

则 loader 会重建/读取刺激矩阵并输出：

- `x_full = raw_x`（BOLD/神经活动）
- `u_full = stimulus_matrix`（刺激矩阵）

最终返回 `(x, u)`。

> 语义提醒：因此当前默认训练任务是 $x(t)\rightarrow u(t)$。如果你要做 $u(t)\rightarrow x(t)$，只需在训练脚本里把输入/标签交换即可（见第 6 节）。

---

## 3. 训练前的数据处理：切片（T 的含义）

切片逻辑在 [training/data_loader.py](training/data_loader.py)：

- `T is None` 或 `T <= 0`：不切片
	- 返回 `x: [1, n_time, C]`，`u: [1, n_time, C]`
- `T > 0`：按整除切片
	- `num_samples = n_time // T`
	- 使用前 `num_samples*T` 个点 reshape：
		- `x: [num_samples, T, C]`
		- `u: [num_samples, T, C]`
	- 若 `n_time < T`：回退到整段 `[1, n_time, C]`

训练脚本 [training/train_mlp.py](training/train_mlp.py) 的默认策略：

- PDE：默认 `T=None`（用完整序列）
- ODE：默认 `T=1024`

对 MLP 来说，`T` 的主要作用是：

- 控制训练样本的“来源窗口长度”（但最终都会展平成点级别样本）
- 影响归一化统计量（通常每个 `.pkl`/每段序列单独做归一化）

---

## 4. 训练阶段：数据如何喂给 MLP（shape 逐步展开）

训练逻辑在 [training/train_mlp.py](training/train_mlp.py)。核心步骤：

1. 读取每个 `.pkl`，得到：

- `x`: `[num_samples, T, C]`
- `u`: `[num_samples, T, C]`

2. 把所有文件拼接：

- `x_tensor = cat(all_x, dim=0)` → `[N, T, C]`
- `u_tensor = cat(all_u, dim=0)` → `[N, T, C]`

3. 展平时间维（把每个时间点当成一个训练样本）：

- `x_flat = x_tensor.reshape(N*T, C)`
- `u_flat = u_tensor.reshape(N*T, C)`

4. DataLoader batch：

- `x_b`: `[B, C]`
- `u_b`: `[B, C]`

5. 前向 + 损失：

- `pred = model(x_b)` → `[B, C]`
- `loss = MSE(pred, u_b)`

6. 权重保存：

- `results/models/mlp_{sim_type}_norm.pth`（normalize=True）
- `results/models/mlp_{sim_type}.pth`（normalize=False）

---

## 5. 使用/测试阶段：如何把新数据喂给 MLP

MLP 推理时只需要二维输入 `[batch, C]`。

### 5.1 单段序列（Time, C）推理

给定一个新序列 `x_raw`，shape `(Time, C)`：

1. （可选但强烈建议）按训练一致的规则归一化

- 如果训练时 `normalize=True` 且你走的是仿真分支：对每个通道做 Z-score（按时间轴计算 mean/std）。

2. 转成 tensor 并送入模型：

- `x_tensor = torch.tensor(x_raw, float32)` → `[Time, C]`
- 直接 `u_pred = model(x_tensor)` → `[Time, C]`

这里的 “batch” 就是 `Time`，所以不需要再 `unsqueeze(0)`。

### 5.2 长序列的批处理/分块

如果 `Time` 很长，建议分块推理：

- 每块 `chunk` shape `(K, C)`
- `model(chunk_tensor)` 输出 `(K, C)`

最后沿时间拼接。

---

## 6. 常见改法：我想训练/推理 `u -> x` 怎么做？

当前训练是：

- 输入：`x_flat`
- label：`u_flat`

如果你希望学习 $u(t) \rightarrow x(t)$：

- 训练时改为：`pred = model(u_b)`，loss 对齐 `x_b`
- 推理时输入也相应换成 `u`。

数据 loader 不用改（仍返回 `x,u`），只是训练脚本用法交换。

---

## 7. 一个最小可用的 MLP 推理模板

```python
import torch
import numpy as np
from models.mlp import MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 必须与训练一致
C = 246
H = 128
MODEL_PATH = 'results/models/mlp_ode_ec_norm.pth'
normalize = True

# x_raw: (Time, C)
def zscore_per_channel(x: np.ndarray):
		mu = x.mean(axis=0, keepdims=True)
		sd = x.std(axis=0, keepdims=True)
		sd[sd == 0] = 1.0
		return (x - mu) / sd

model = MLP(input_dim=C, output_dim=C, hidden_dims=(H, H)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

if normalize:
		x_in = zscore_per_channel(x_raw)
else:
		x_in = x_raw

with torch.no_grad():
		x_tensor = torch.tensor(x_in, dtype=torch.float32, device=device)  # (Time, C)
		u_pred = model(x_tensor)  # (Time, C)
u_out = u_pred.cpu().numpy()
```
