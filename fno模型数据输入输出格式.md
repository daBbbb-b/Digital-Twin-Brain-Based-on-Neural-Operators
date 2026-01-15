# FNO1d 模型数据输入输出格式与处理流程（训练/使用/测试）

本文档以仓库当前实现为准，说明如何把 `.pkl` 数据处理为 FNO1d 的输入，以及在训练与推理/测试阶段如何保持“采样率、窗口长度、归一化”一致。

涉及代码：

- 数据加载与切片：[training/data_loader.py](training/data_loader.py)
- 训练脚本：[training/train_fno1d.py](training/train_fno1d.py)
- 模型定义：[models/fno.py](models/fno.py) 中 `class FNO1d`
- 真实数据预处理（NIfTI→pkl）：[evaluation/preprocess_bold.py](evaluation/preprocess_bold.py)
- 推理脚本（示例）：[evaluation/run_inference.py](evaluation/run_inference.py)

---

## 1. FNO1d 的真实输入/输出 shape

`FNO1d.forward(x)` 在 [models/fno.py](models/fno.py) 中定义，核心约定是：

- 输入 `x`：`[batch, grid_size, input_size]`
	- 对时间序列来说：`grid_size == T`（序列长度/时间点数）
	- `input_size == C`（通道数/脑区数）
- 输出：`[batch, grid_size, output_size]`
	- 本项目训练时通常 `output_size == C`，即每个时间点输出每个通道一个值。

FNO1d 会在 forward 内部自动构造并拼接 1D 坐标网格：

- `grid = linspace(0,1,T)` → shape `[batch, T, 1]`
- 然后 `torch.cat((x, grid), dim=-1)` → shape `[batch, T, C+1]`

因此：FNO1d **不需要你额外提供时间坐标**，但要求你的输入序列长度 `T` 与训练时一致（至少模型推理时要和模型的“习惯窗口长度”一致）。

---

## 2. `.pkl` 数据格式（训练阶段需要 paired 数据）

训练脚本 [training/train_fno1d.py](training/train_fno1d.py) 调用 `load_data_from_pkl()`，期望能得到一对张量 `(x, u)`：

- `x`：模型输入序列
- `u`：模型监督目标（label）序列

`load_data_from_pkl()` 支持两类 `.pkl`：

### 2.1 直接配对格式（最直接、最推荐）

`.pkl` 内包含键：

- `"x"`
- `"u"`

则直接读出来转成 `torch.float32`。

推荐 shape：

- `x`: `[N, T, C]`
- `u`: `[N, T, C]`

其中：

- `N` 是样本段数（可理解为多段序列）
- `T` 是每段序列长度
- `C` 是通道数

归一化：当 `normalize=True` 时，当前实现只对 `x` 做 Z-score（`u` 不归一化）。

### 2.2 仿真格式（bold/neural + stimulus_config）

如果 `.pkl` 内包含：

- `"bold_signal"` 或 `"neural_activity"`（shape `[n_time, C]`）
- `"stimulus_config"`

则 loader 会：

1. 取 `raw_x = bold_signal | neural_activity`
2. NaN → 0
3. 若 `normalize=True`：按通道（axis=0）做 Z-score
4. 生成/重建 `stimulus_matrix`（shape `[n_time, C]`），并输出：

- `x_full = raw_x`
- `u_full = stimulus_matrix`

最终返回 `(x, u)`，shape 由切片规则决定（见下一节）。

> 语义提醒：这意味着你训练到的是 $x(t)\rightarrow u(t)$（例如 “BOLD/神经活动 → 刺激矩阵”）。如果你希望训练 $u(t)\rightarrow x(t)$，只需在训练脚本里交换输入与 label。

---

## 3. 训练前的数据处理：切片与 T 的含义

切片逻辑在 [training/data_loader.py](training/data_loader.py) 的仿真分支：

- `T is None` 或 `T <= 0`：不切片
	- `x`: `[1, n_time, C]`
	- `u`: `[1, n_time, C]`
- 否则：按整除切片
	- `num_samples = n_time // T`
	- 使用前 `num_samples*T` 个点 reshape：
		- `x`: `[num_samples, T, C]`
		- `u`: `[num_samples, T, C]`
	- 若 `n_time < T`：回退到整段 `[1, n_time, C]`

训练脚本对 `T` 的默认策略在 [training/train_fno1d.py](training/train_fno1d.py)：

- PDE：默认 `T=None`（用完整序列）
- ODE：默认 `T=1024`

所以你推理/测试时必须明确：你的模型到底是用多长的窗口训练的（例如 1024），推理阶段最好也输入同样长度（或用滑窗切片得到同长度窗口）。

---

## 4. 训练阶段：数据如何喂给 FNO1d

训练代码在 [training/train_fno1d.py](training/train_fno1d.py)。流程如下：

1. 读取所有 `.pkl`，得到多段样本并拼接：

- `x_all = torch.cat(all_x, dim=0)` → `[N, T, C]`
- `u_all = torch.cat(all_u, dim=0)` → `[N, T, C]`

2. shuffle 后切分训练/验证集（8:2）。

3. DataLoader 每个 batch：

- `x`: `[B, T, C]`
- `u`: `[B, T, C]`

4. 模型 forward：

- `pred = model(x)` → `[B, T, C]`

5. 损失：

- `loss = MSE(pred, u)`（逐点、逐通道监督）

训练保存权重路径（与 sim_type、normalize 相关）：

- `results/models/fno1d_{sim_type}_norm.pth`（normalize=True）
- `results/models/fno1d_{sim_type}.pth`（normalize=False）

---

## 5. 使用/测试阶段：真实数据如何变成 FNO 输入

### 5.1 真实 BOLD 数据预处理（NIfTI → pkl）

[evaluation/preprocess_bold.py](evaluation/preprocess_bold.py) 会输出到：

- `results/inference_data/*.pkl`

每个 `.pkl` 结构是：

- `"x"`: `(Time, 246)` 的 float32
- `"filename"`
- `"n_regions"`

注意：这个 `.pkl` **只有 `x`，没有 `u`**，因此它不能直接用于训练（训练需要 paired label）。它是为“推理/反演”准备的输入。

### 5.2 推理/测试阶段的输入 shape

对单个样本，FNO1d 需要：

- `x_tensor`: `[1, T, C]`

其中：

- `C` 应与训练时一致（如 246）
- `T` 要与训练窗口一致（常见为 1024 或者你训练时实际用的长度）

如果真实数据的 `Time` 不是你需要的 `T`：

- `Time > T`：可截取前 `T`，或用滑动窗口（更推荐）
- `Time < T`：可用 0 padding 到 `T`

### 5.3 推理脚本里“重采样 + 截取窗口”的含义

[evaluation/run_inference.py](evaluation/run_inference.py) 展示了一个典型推理流程：

1. 从 `.pkl` 读 `x_raw`：shape `(Time, C)`
2. 重采样（插值）到更高频率：

- `ORIGINAL_TR = 2.0s`
- `TARGET_TR = 0.05s`

3. 截取/补齐到固定窗口 `T_WINDOW=256`，形成 `x_input`: `(256, C)`
4. 转 tensor 并加 batch：`x_tensor = [1, 256, C]`
5. 推理：`u_pred = model(x_tensor)` → `[1, 256, C]`
6. 保存 `u_out = (256, C)` 到 `results/inference_results/*.npy`

关键一致性要求：

- `TARGET_TR` 必须与你训练数据的采样间隔匹配（否则模型看到的“时间尺度”不同）
- `T_WINDOW` 必须与你训练时的 `T` 匹配（否则模型虽然能跑，但学习到的算子与推理输入长度分布不一致）

---

## 6. 当前仓库里推理脚本的两个常见坑（建议你使用时对齐）

1) 权重文件名不一致

- 训练脚本保存：`results/models/fno1d_{sim_type}_norm.pth` 或 `results/models/fno1d_{sim_type}.pth`
- 但 [evaluation/run_inference.py](evaluation/run_inference.py) 默认加载：`results/models/best_fno1d.pth`

使用时请把 `MODEL_PATH` 改成你实际训练生成的权重文件。

2) 推理窗口长度可能与训练不一致

- 训练（ODE 默认）通常 `T=1024`
- 推理脚本默认 `T_WINDOW=256`

如果你的模型是用 1024 训练的，推理阶段建议也用 1024（或者用滑窗 256 训练/微调出一个 256 的模型）。

---

## 7. 一个最小可用的“FNO1d 推理模板”（按训练参数对齐）

推理时最关键是保证：`C`、`T`、是否 normalize、以及权重文件一致。

```python
import pickle
import numpy as np
import torch
from models.fno import FNO1d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 必须与训练一致
T = 1024
C = 246
MODES = 32
WIDTH = 64
MODEL_PATH = 'results/models/fno1d_ode_sc_norm.pth'

with open('results/inference_data/sub-01.pkl', 'rb') as f:
		data = pickle.load(f)
x_raw = data['x']  # (Time, C)

# 截取/补齐到 T
if x_raw.shape[0] >= T:
		x_input = x_raw[:T]
else:
		pad = np.zeros((T - x_raw.shape[0], x_raw.shape[1]), dtype=x_raw.dtype)
		x_input = np.concatenate([x_raw, pad], axis=0)

x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, C)

model = FNO1d(input_size=C, output_size=C, modes=MODES, width=WIDTH).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

with torch.no_grad():
		u_pred = model(x_tensor)  # (1, T, C)
u_out = u_pred.cpu().squeeze(0).numpy()  # (T, C)
```

如果你需要“滑动窗口推理”（更稳定），做法是把 `x_raw` 切成多个长度为 `T` 的片段，分别推理后再拼回去（重叠部分可平均）。
