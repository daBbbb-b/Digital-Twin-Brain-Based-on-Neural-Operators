# DeepONet 模型数据输入输出格式与处理流程（训练/推理）

本文档以仓库当前实现为准，解释：

- `.pkl` 原始数据允许的两种结构
- [training/data_loader.py](training/data_loader.py) 中 `load_data_from_pkl()` 如何把原始数据处理成可训练的张量
- [models/deeponet.py](models/deeponet.py) 的真实输入输出 shape
- [training/train_deeponet.py](training/train_deeponet.py) 训练阶段如何拼接/flatten 数据喂给 DeepONet
- 推理阶段如何按同样方式构造输入并还原输出

> 重要：当前 `train_deeponet.py` 的训练目标是“用 `x` 预测 `u`”（MSE 对齐 `u`），也就是学习 $x(t) \rightarrow u(t)$ 的映射（每个时间点独立预测）。这与常见 DeepONet “给定输入函数 $u$，输出 $G(u)(y)$”的语义一致，但在本仓库里 `u`/`x` 的物理含义取决于你的 `.pkl` 数据：
> - 在仿真 `.pkl` 分支中：`x` 是 `bold_signal` 或 `neural_activity`，`u` 是刺激矩阵 `stimulus_matrix`。
> - 如果你希望反过来学习 $u(t) \rightarrow x(t)$，只需要在训练脚本里交换 loss 的目标与分支输入（见文末“常见改法”）。

---

## 1. DeepONet 的“两个输入”是什么？

DeepONet 的 forward 定义在 [models/deeponet.py](models/deeponet.py)：

- `u_input`：分支网络（Branch Net）输入，shape 为 `[batch, num_sensors]`
- `y_input`：主干网络（Trunk Net）输入，shape 为 `[batch, dim_y]`
- 输出：shape 为 `[batch, output_size]`

在本项目中：

- `num_sensors == C`（通道数/脑区数）
- `output_size == C`（同样输出每个通道一个值）
- `y_input` 被构造成“时间坐标的 embedding”（但当前实现里是把同一个 $t$ 值复制成 `dim_y` 维：`repeat(1, dim_y)`）

因此训练时每个样本的时序数据 `[T, C]` 会被 flatten 成 `T` 个点，每个点单独作为一个 DeepONet 样本：

$$
\underbrace{x(t)\in\mathbb{R}^{C}}_{\text{branch输入}}\; + \;\underbrace{y(t)\in\mathbb{R}^{dim\_y}}_{\text{trunk输入}}\;\longrightarrow\;\underbrace{\hat{u}(t)\in\mathbb{R}^{C}}_{\text{输出}}
$$

---

## 2. `.pkl` 数据允许的两种格式

`load_data_from_pkl()` 支持两类 `.pkl`：

### 2.1 直接张量格式（推荐用于自定义数据）

如果 `.pkl` 中包含键：

- `"x"`：输入序列
- `"u"`：输出序列

则会被直接读取并转为 `torch.float32`。

期望 shape（最常用）：

- `x`: `[num_samples, T, C]` 或 `[T, C]`
- `u`: `[num_samples, T, C]` 或 `[T, C]`

注意：当前实现不会对 `[T, C]` 自动补 batch 维；因此如果你走这条分支，建议在生成 `.pkl` 时就存成三维 `[N, T, C]`。

归一化：当 `normalize=True` 时，只对 `x` 做 Z-score（按“除最后一维通道外的所有维度”求均值方差）。`u` 不会被归一化。

### 2.2 仿真输出格式（bold/neural + stimulus_config）

如果 `.pkl` 中包含：

- `"bold_signal"` 或 `"neural_activity"`（二选一，作为原始信号）
- `"stimulus_config"`（刺激任务配置，用于重建刺激矩阵）

则处理流程如下：

1. 读取 `raw_x = bold_signal | neural_activity`，期望 shape 为 `[Time, Channels]` 即 `[n_time, C]`
2. 若 `raw_x` 含 NaN：`np.nan_to_num(raw_x, nan=0.0)`
3. 若 `normalize=True`：按通道做 Z-score（对 axis=0 求均值方差）
4. 生成/重建刺激矩阵 `stimulus_matrix`（shape `[n_time, C]`）：
	 - 若 `.pkl` 已保存 `data["stimulus"]` 且非空，则优先使用，并对齐长度到 `n_time`
	 - 否则依据 `stimulus_config` + `metadata` 重建

输出张量定义：

- `x_full = torch.tensor(raw_x)`
- `u_full = torch.tensor(stimulus_matrix)`

最终 `load_data_from_pkl()` 返回：

- `x`: `[num_samples, T, C]` 或 `[1, n_time, C]`
- `u`: `[num_samples, T, C]` 或 `[1, n_time, C]`

其中 `num_samples` 来自切片规则（下一节）。

---

## 3. `load_data_from_pkl()` 的关键处理规则

函数位置：[training/data_loader.py](training/data_loader.py)

### 3.1 切片（把长序列切成多段样本）

对仿真格式（`raw_x` 是 `[n_time, C]`）的切片逻辑是：

- 若 `T is None` 或 `T <= 0`：不切片
	- `x = x_full.unsqueeze(0)` 得到 `[1, n_time, C]`
	- `u = u_full.unsqueeze(0)` 得到 `[1, n_time, C]`
- 否则：按整除切片
	- `num_samples = n_time // T`
	- 若 `num_samples > 0`：仅使用前 `num_samples*T` 个点
		- `x = x_full[:num_samples*T].view(num_samples, T, C)`
		- `u = u_full[:num_samples*T].view(num_samples, T, C)`
	- 若 `n_time < T`：回退为整段序列 `[1, n_time, C]`

### 3.2 sim_type 与刺激重建（ODE vs PDE）

`load_data_from_pkl(pkl_path, ..., sim_type="ode"|"pde"|"auto")` 用于决定刺激重建策略：

- `sim_type == "ode"`：尝试使用 `simulation.stimulation_generator.StimulationGenerator` 进行更精细的 ODE 刺激重建（含噪声、不同波形 envelope 等）。如果失败，会回退到“简单 boxcar”方式。
- `sim_type == "pde"`：使用简化策略重建 PDE 刺激（如果缺少 surface/vertices，则只能近似地在 seeds 上加 envelope）。
- `sim_type == "auto"`：兼容旧数据，依据 `stimulus_config['type'] == 'mixed_task_ode'` 推断。

> 训练脚本 [training/train_deeponet.py](training/train_deeponet.py) 里 `--sim_type` 实际取值是 `pde | ode_ec | ode_sc | auto`，会把它原样传入 `load_data_from_pkl(..., sim_type=sim_type)`。当前 `data_loader.py` 只显式识别 `"ode"/"pde"/"auto"`；因此 `ode_ec/ode_sc` 会落到 `else` 分支并按 `auto` 的规则推断是否为 ODE。
>
> 如果你希望严格指定 ODE，请在训练参数层把 `ode_ec/ode_sc` 映射为 `ode`（或在 loader 中扩展判断）。这不影响本文档的“shape 和喂入方式”。

---

## 4. 训练阶段：如何把数据喂给 DeepONet

训练脚本：[training/train_deeponet.py](training/train_deeponet.py)

### 4.1 读取并构造 (x, y, u)

对每个 `.pkl`：

1. 调用：

- `x, u = load_data_from_pkl(pkl_path, T=..., normalize=..., sim_type=...)`
- 得到 `x,u` shape 为 `[num_samples, T, C]`（或 `[1, n_time, C]`）

2. 遍历 `num_samples`，把每段序列展开成训练样本：

- `x_sample = x[i]`：`[T, C]`
- `u_sample = u[i]`：`[T, C]`

3. 构造 `y_sample`（时间坐标）：

- `seq_len = T`（或该段真实长度）
- `torch.linspace(0, 1, seq_len)` 得到 `[T]`
- `unsqueeze(1)` 得到 `[T, 1]`
- `repeat(1, dim_y)` 得到 `[T, dim_y]`

最终堆叠得到：

- `x_tensor = stack(all_x)`：`[N, T, C]`
- `y_tensor = stack(all_y)`：`[N, T, dim_y]`
- `u_tensor = stack(all_u)`：`[N, T, C]`

这里 `N` 是“所有 `.pkl` 的所有切片段数之和”。

### 4.2 DataLoader batch 与 flatten

`DataLoader` 产出 batch：

- `x_batch`: `[B, T, C]`
- `y_batch`: `[B, T, dim_y]`
- `u_batch`: `[B, T, C]`

为了喂给 DeepONet（它期望二维输入），训练脚本把时间维展平：

- `x_flat = x_batch.reshape(B*T, C)`
- `y_flat = y_batch.reshape(B*T, dim_y)`
- `u_flat = u_batch.reshape(B*T, C)`

模型输出：

- `preds = model(x_flat, y_flat)` 形状 `[B*T, C]`

损失：

- `loss = MSE(preds, u_flat)`

这等价于对每个时间点独立监督，整体 loss 是所有点的均方误差。

---

## 5. 推理阶段：如何构造输入并还原输出

本仓库的 [evaluation/run_inference.py](evaluation/run_inference.py) 是给 FNO 用的，不是 DeepONet；DeepONet 的推理与训练一致：

### 5.1 单段序列推理（推荐流程）

给定你要推理的 `.pkl`，可复用 loader：

1. 准备与训练一致的参数：

- `sim_type`：与训练一致
- `T`：与训练一致（如果训练用了切片，推理也建议同样切片）
- `normalize`：与训练一致（尤其是你训练时 normalize=True）
- `dim_y`：必须与训练时一致
- `num_sensors=C`：必须与训练时一致

2. 推理步骤：

- `x, u_gt = load_data_from_pkl(..., T=T, normalize=..., sim_type=...)`
	- `x` shape `[N, T, C]`（推理也可能得到多段）
- 对每段样本 `x[i]`：
	- 构造 `y_sample`（同训练：`linspace(0,1,T).repeat(1,dim_y)`）
	- flatten 成 `[T, C]` 和 `[T, dim_y]`
	- `u_pred_flat = model(x_flat, y_flat)` → `[T, C]`
	- reshape/还原成 `u_pred = u_pred_flat.view(T, C)`

3. 输出解释：

- 如果你按当前训练方式：输出 `u_pred[t, c]` 是“第 `t` 个时间点、第 `c` 个通道的刺激值预测”。

### 5.2 一个最小可运行的 DeepONet 推理模板

你可以新建一个脚本（例如 `evaluation/run_inference_deeponet.py`），核心逻辑如下（伪代码/模板）：

```python
import torch
from models.deeponet import DeepONet
from training.data_loader import load_data_from_pkl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 这些超参必须与训练一致
dim_y = 16
hidden_size = 128
num_branch_layers = 2
num_trunk_layers = 2
T = 1024
normalize = True
sim_type = 'ode_sc'

# 先读一次数据确定 C
x, _ = load_data_from_pkl('xxx.pkl', T=T, normalize=normalize, sim_type=sim_type)
N, T_, C = x.shape

model = DeepONet(
		num_sensors=C, dim_y=dim_y,
		num_branch_layers=num_branch_layers,
		num_trunk_layers=num_trunk_layers,
		hidden_size=hidden_size,
		output_size=C,
).to(device)

state = torch.load('results/models/deeponet_ode_sc_norm.pth', map_location=device)
model.load_state_dict(state)
model.eval()

with torch.no_grad():
		x_sample = x[0].to(device)  # [T, C]
		y_sample = torch.linspace(0, 1, x_sample.shape[0], device=device).unsqueeze(1).repeat(1, dim_y)
		u_pred = model(x_sample, y_sample)  # [T, C]
```

说明：这里 `model(x_sample, y_sample)` 之所以可以直接喂三维？是因为 `x_sample` 已经是 `[T, C]`，符合 DeepONet 的 `[batch, C]` 约定（这里的 batch 就是 `T`）。因此推理阶段甚至不需要显式 `reshape(B*T, ...)`。

---

## 6. 常见问题与常见改法

### 6.1 “为什么 y_sample 是 repeat 出来的 dim_y 维？”

当前实现等价于把 1D 时间坐标 $t\in[0,1]$ 复制成 `dim_y` 份。这样做能跑通，但表达能力不如更常见的坐标编码方式（例如把 $t$ 做多频正余弦 positional encoding）。如果你要提升效果，可以在构造 `y_sample` 时换成更合理的 embedding（但要与训练/推理保持一致）。

### 6.2 我想学习 `u -> x`（刺激到信号），怎么改？

当前训练是：

- Branch 输入：`x_flat`
- 监督目标：`u_flat`

要学习 `u -> x`，只需要在训练循环里：

- `preds = model(u_flat, y_flat)`
- `loss = MSE(preds, x_flat)`

同时数据 loader 不用改（仍然返回 `x,u`），只是在训练/推理时交换使用即可。

### 6.3 normalize 应该怎么保持一致？

训练用了 `--normalize True`（默认 True）时：

- 仿真分支：会对 `raw_x` 做 Z-score
- 直接张量分支：只对 `x` 做 Z-score

因此推理也必须用相同 normalize 规则，否则分布不一致会明显影响输出。
