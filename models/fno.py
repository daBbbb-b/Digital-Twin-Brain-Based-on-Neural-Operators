"""
Fourier Neural Operator (FNO) 模块

功能说明：
    实现Fourier Neural Operator，用于学习PDE和ODE的解算子。

主要类：
    FNO: FNO基类
    FNO1d: 一维FNO（用于时间序列）
    FNO2d: 二维FNO（用于时空数据）
    FNO3d: 三维FNO

输入：
    - 输入函数：连接矩阵、初始条件等
    - 网格坐标：时间或空间网格

输出：
    - 输出函数：刺激函数或解轨迹

参考：
    Li et al. (2021) "Fourier Neural Operator for Parametric Partial Differential Equations"
    ICLR 2021
"""

import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

################################################################
#  1. 一维谱卷积层 (Spectral Conv 1D) - 对应论文 Eq(4) & (5)
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        """
        in_channels: 输入特征通道数
        out_channels: 输出特征通道数
        modes: 保留的傅里叶模态数量 (截断高频，只留低频)
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # 定义可学习的权重参数 (复数张量)
        # 形状: (in_channels, out_channels, modes)
        # scale用于初始化权重分布
        self.scale = (1 / (in_channels * out_channels))
        w_r = torch.randn(in_channels, out_channels, modes) * self.scale
        w_i = torch.randn(in_channels, out_channels, modes) * self.scale
        self.weights = nn.Parameter(torch.stack((w_r, w_i), dim=-1))  # (..., 2)

    # 复数乘法: (batch, in_channel, x) * (in_channel, out_channel, x) -> (batch, out_channel, x)
    def compl_mul1d(self, input, weights):
        # 使用 einsum 进行爱因斯坦求和约定，简洁处理维度乘法
        # "bix,iox->box": 
        # b=batch, i=in_channels, o=out_channels, x=modes
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # 1. 傅里叶变换 (RFFT: 实数到复数)
        # 输入 x 形状: (batch, in_channels, x_grid)
        # 输出 x_ft 形状: (batch, in_channels, x_grid/2 + 1)
        x_ft = torch.fft.rfft(x)

        # 2. 频谱截断与线性变换 (Multiply relevant Fourier modes)
        # 我们只取前 'modes' 个低频系数进行计算
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        
        # 核心操作 R * F(v)
        weights_c = torch.view_as_complex(self.weights.to(x.device))
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], weights_c)

        # 3. 傅里叶逆变换 (IRFFT: 复数到实数)
        # 返回物理空间
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

################################################################
#  2. FNO 1D 模型主架构 - 对应论文 Figure 2(a)
# 针对于一维输入输出函数的学习任务
################################################################
class FNO1d(nn.Module):
    def __init__(self, input_size, output_size, modes, width):
        super(FNO1d, self).__init__()
        """
        modes: 傅里叶层保留的模态数
        width: 提升后的通道维度 (Hidden channels)
        """
        self.input_size = input_size
        self.output_size = output_size
        self.modes = modes
        self.width = width
        self.padding = 2 # 为了处理边界情况，有时会padding (可选)

        # P: Lifting layer (将输入维度映射到高维特征空间)
        # 输入通道 = 原始特征 + 1 维坐标网格
        self.fc0 = nn.Linear(self.input_size + 1, self.width)

        # 4层 Fourier Integral Operators
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes)

        # W: 对应的本地线性变换 (Skip connection / 1x1 Conv)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        # Q: Projection layer (将特征映射回目标输出维度)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.output_size) # 输出维度为 output_size

    def forward(self, x):
        # x shape: (batch, grid_size, self.input_size)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1) # 将坐标信息拼接到输入中
        
        # 1. Lifting
        x = self.fc0(x)
        x = x.permute(0, 2, 1) # 调整为 (batch, channels, grid) 适应 Conv1d

        # 2. Iterative Layers (Eq 2: v_{t+1} = sigma(W v_t + K(v_t)))
        
        # Layer 0
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x) # 论文中多用 ReLU，但在新版本官方代码常改用 GELU

        # Layer 1
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        # Layer 2
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        # Layer 3
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        # 最后一层通常不加激活，或者在投影层加

        # 3. Projection
        x = x.permute(0, 2, 1) # 换回 (batch, grid, channels)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x

    def get_grid(self, shape, device):
        # 生成空间坐标网格，范围 [0, 1]
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(torch.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

################################################################
#  3. 二维谱卷积层 (Spectral Conv 2D) - 适用于 Navier-Stokes
# 针对二维输入输出函数的学习任务
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        """
        2D 版本需要保留两个方向的模态 (modes1, modes2)
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # x轴模态数
        self.modes2 = modes2 # y轴模态数

        self.scale = (1 / (in_channels * out_channels))
        w1r = torch.randn(in_channels, out_channels, self.modes1, self.modes2) * self.scale
        w1i = torch.randn(in_channels, out_channels, self.modes1, self.modes2) * self.scale
        w2r = torch.randn(in_channels, out_channels, self.modes1, self.modes2) * self.scale
        w2i = torch.randn(in_channels, out_channels, self.modes1, self.modes2) * self.scale
        self.weights1 = nn.Parameter(torch.stack((w1r, w1i), dim=-1))
        self.weights2 = nn.Parameter(torch.stack((w2r, w2i), dim=-1))

    def compl_mul2d(self, input, weights):
        # (batch, in, x, y), (in, out, x, y) -> (batch, out, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # x: (batch, in_channels, x_grid, y_grid)
        
        # 2D FFT
        x_ft = torch.fft.rfft2(x)

        # 准备输出容器
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        
        # 2D 频谱截断技巧：
        # rfft2 输出维度中，dim=-1 是 [0, ..., N/2]，dim=-2 是 [0, ..., N-1]
        # 低频主要集中在四个角，由于 rfft 的性质，我们只需要处理两个角：
        # 1. 左上角 (Top-Left): 对应正频率低频
        # 2. 左下角 (Bottom-Left): 对应负频率低频 (wrapped around)
        
        # 处理左上角: indices [0:modes1, 0:modes2]
        w1_c = torch.view_as_complex(self.weights1.to(x.device))
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], w1_c)
            
        # 处理左下角: indices [-modes1:, 0:modes2]
        w2_c = torch.view_as_complex(self.weights2.to(x.device))
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], w2_c)

        # Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, input_size, output_size, modes1, modes2, width):
        super(FNO2d, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # 用于处理非周期边界时的padding，可视情况调整

        # 假设输入包含原始特征 + (x, y) 两个坐标通道
        self.fc0 = nn.Linear(self.input_size + 2, self.width) 

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.output_size)

    def forward(self, x):
        # x: (batch, size_x, size_y, 1) -> 函数值
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(torch.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(torch.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

################################################################
#  4. 简单测试与运行示例
################################################################

if __name__ == "__main__":
    # --- 1D 示例 (Burgers) ---
    print("--- Testing FNO 1D ---")
    # 参数设置
    input_size = 1
    output_size = 1
    modes = 16
    width = 64
    batch_size = 10
    grid_size = 1024 # 分辨率 s=1024

    model = FNO1d(input_size, output_size, modes, width)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # 模拟输入数据: (Batch, Grid, Channels)
    # 输入通常是初始条件 u0(x)，我们将其作为 (Batch, Grid, 1) 的张量
    input_data = torch.randn(batch_size, grid_size, 1)
    
    # 前向传播
    # 模型内部会自动 append 坐标网格，所以 fc0 输入维度是 1(data) + 1(grid) = 2
    output = model(input_data)
    
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}") # 预期 (10, 1024, 1)

    # --- 2D 示例 (Navier-Stokes) ---
    print("\n--- Testing FNO 2D ---")
    modes1 = 12
    modes2 = 12
    width = 32
    grid_size_x = 64
    grid_size_y = 64

    model_2d = FNO2d(input_size, output_size, modes1, modes2, width)
    
    # 模拟输入: (Batch, X, Y, 1)
    input_2d = torch.randn(batch_size, grid_size_x, grid_size_y, 1)
    
    output_2d = model_2d(input_2d)
    print(f"Input 2D shape: {input_2d.shape}")
    print(f"Output 2D shape: {output_2d.shape}") # 预期 (10, 64, 64, 1)