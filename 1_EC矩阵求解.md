# 任务说明：有效连接 (EC) 矩阵求解

## 1. 任务概述
本任务的目标是基于给定的103个任务态fMRI数据，计算大脑区域之间的有效连接（Effective Connectivity, EC）。有效连接描述了一个神经系统对另一个神经系统的因果影响，是有向图。

根据课题要求，我们需要采用 **"Mapping effective connectivity by virtually perturbing a surrogate brain"** (Luo et al., Nature Methods, 2025) 文章中提出的方法进行求解。

## 2. 输入数据与预处理
为了求解EC，首先需要从原始fMRI数据中提取脑区时间序列并计算功能连接（FC）。

### 2.1 核心数据文件
*   **fMRI 原始数据 (MNI空间)**: 
    *   路径: `dataset/T103/func/sub-01_task-training_run-XX_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz`
    *   说明: 数据集包含 **`run-01` 到 `run-12`** 共12个运行的数据。请使用 `space-MNI152NLin2009cAsym` 版本的数据，因为它是体素（Voxel）级别的标准空间数据，可以直接与脑图谱对齐。
*   **脑图谱 (Atlas)**:
    *   路径: `dataset/BN_Atlas_246_1mm.nii.gz`
    *   说明: Brainnetome Atlas (BNA) 246 分区模板，用于定义感兴趣区域 (ROI)。
*   **结构连接 (SC) 矩阵**: 
    *   路径: `dataset/BNA_matrix_binary_246x246.csv`
    *   说明: 二值化的结构连接矩阵，作为EC求解的解剖约束（Mask）。

### 2.2 功能连接 (FC) 计算步骤
在进行EC求解前，必须先计算每个Task的经验FC矩阵。

1.  **脑区时间序列提取 (Parcellation)**:
    *   **工具**: 推荐使用 Python 的 `nilearn` 库 (`NiftiLabelsMasker`)。
    *   **操作**: 将 `BN_Atlas_246_1mm.nii.gz` 重采样并覆盖到 fMRI 数据上。
    *   **计算**: 对每个脑区（ROI）内的所有体素的时间序列取平均值。
    *   **结果**: 获得一个维度为 `(Time_points, 246)` 的时间序列矩阵。

2.  **FC 矩阵计算**:
    *   **方法**: 计算 246 个脑区时间序列两两之间的 **皮尔逊相关系数 (Pearson Correlation Coefficient)**。
    *   **公式**: $FC_{ij} = \frac{cov(x_i, x_j)}{\sigma_{x_i} \sigma_{x_j}}$
    *   **结果**: 获得一个维度为 `(246, 246)` 的对称矩阵，对角线为1。

## 3. 方法原理 (基于参考文献 [6])

该方法的核心思想是通过虚拟扰动一个代理大脑模型来推断有效连接。

1.  **构建代理模型 (Surrogate Brain Construction)**:
    *   建立一个全脑动力学模型（通常是 Hopf 分岔模型或类似模型）。
    *   **拟合目标**: 优化模型参数（如全局耦合强度 $G$），使得模型生成的模拟功能连接 (FC_sim) 与步骤2.2中计算的经验功能连接 (FC_emp) 之间的相似度（通常是 Pearson 相关系数）最大化。

2.  **虚拟扰动 (Virtual Perturbation)**:
    *   在拟合好的模型处于临界状态或工作点时，对每个脑区施加微小的虚拟扰动。
    *   观察该扰动如何传播到其他脑区。

3.  **推导 EC**:
    *   利用线性响应理论 (Linear Response Theory) 或数值扰动法，计算扰动传播的响应矩阵。
    *   该响应矩阵即反映了脑区间的因果相互作用，即有效连接 EC。
    *   **公式参考**: 具体算法细节请严格参照 Luo et al. (2025) 的论文及开源代码（如有）。

## 4. 任务执行步骤

1.  **数据准备**:
    *   加载 SC 矩阵。
    *   遍历 `dataset/T103/func/` 下的所有 `run` (**从 `run-01` 到 `run-12`**，对应不同的 Task)。
    *   对每个 run 进行 **时间序列提取** 和 **FC 计算**。

2.  **模型拟合 (针对每个 Task)**:
    *   使用计算出的经验 FC 拟合全脑动力学模型。

3.  **EC 计算**:
    *   基于拟合好的模型，应用扰动法计算 EC 矩阵。
    *   **注意**: EC 矩阵是非对称的（$EC_{ij} \neq EC_{ji}$）。

4.  **结果保存**:
    *   将计算得到的 EC 矩阵保存为 `.npy` 或 `.csv` 格式。
    *   建议目录结构: `dataset/EC_matrices/task_name_EC.npy`。
    *   矩阵维度应为 $246 \times 246$。

## 5. 输出交付物
*   **代码**: 实现 EC 计算的 Python 脚本（如 `calculate_ec.py`）。
*   **数据**: 103 个任务对应的 EC 矩阵文件。
*   **文档**: 简要说明使用的具体参数设置和收敛情况。

## 6. 参考文献
*   [6] Luo, Z., Peng, K., Liang, Z., Cai, S., Xu, C., Li, D., ... & Liu, Q. (2025). Mapping effective connectivity by virtually perturbing a surrogate brain. Nature Methods, 1-