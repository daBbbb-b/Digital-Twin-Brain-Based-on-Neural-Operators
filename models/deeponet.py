"""
DeepONet模块

功能说明：
    实现Deep Operator Network，用于学习算子映射。

主要类：
    DeepONet: DeepONet主类
    BranchNet: 分支网络（编码输入函数）
    TrunkNet: 主干网络（编码输出位置）

输入：
    - 输入函数：在传感器位置采样的函数值
    - 输出位置：需要预测的位置坐标

输出：
    - 输出函数值：在指定位置的函数值

参考：
    Lu et al. (2019) "DeepONet: Learning nonlinear operators for identifying 
    differential equations based on the universal approximation theorem of operators"
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义 DeepONet 模型
# ==========================================
class DeepONet(nn.Module):
    def __init__(self, num_sensors, dim_y, num_branch_layers, num_trunk_layers, hidden_size, output_size):
        super(DeepONet, self).__init__()
        
        # Branch Net: 处理输入函数 u (在 m 个传感器点的值)
        # Input: (Batch, num_sensors) -> Output: (Batch, output_size)
        branch_layers = []
        branch_layers.append(nn.Linear(num_sensors, hidden_size))
        branch_layers.append(nn.Tanh())
        for _ in range(num_branch_layers - 1):
            branch_layers.append(nn.Linear(hidden_size, hidden_size))
            branch_layers.append(nn.Tanh())
        branch_layers.append(nn.Linear(hidden_size, output_size))
        self.branch_net = nn.Sequential(*branch_layers)

        # Trunk Net: 处理查询位置 y
        # Input: (Batch, dim_y) -> Output: (Batch, output_size)
        trunk_layers = []
        trunk_layers.append(nn.Linear(dim_y, hidden_size))
        trunk_layers.append(nn.Tanh())
        for _ in range(num_trunk_layers - 1):
            trunk_layers.append(nn.Linear(hidden_size, hidden_size))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(hidden_size, output_size))
        self.trunk_net = nn.Sequential(*trunk_layers)
        
        # Bias b0
        self.bias = nn.Parameter(torch.tensor([0.0]))

    def forward(self, u_input, y_input):
        # u_input shape: [Batch, m]
        # y_input shape: [Batch, 1] (Assuming 1D output domain)
        
        B = self.branch_net(u_input) # [Batch, p]
        T = self.trunk_net(y_input)  # [Batch, p]
        
        # 输出是点积: sum(B * T) + bias
        # dim=1 表示在特征维度 p 上求和
        output = torch.sum(B * T, dim=1, keepdim=True) + self.bias
        return output

# ==========================================
# 2. 数据生成器 (Antiderivative Operator)
#    学习 G: u(x) -> s(x) = int_0^x u(t) dt
# ==========================================
class DataGenerator:
    def __init__(self, m_sensors):
        self.m = m_sensors
        self.sensor_points = np.linspace(0, 1, m_sensors) # 固定的传感器位置
        
    def generate_data(self, num_samples):
        # 我们使用 Chebyshev 多项式或简单的 Sin/Cos 组合来生成随机函数 u(x)
        # 这里为了简单清晰，生成 u(x) = a*sin(k*x) + b*cos(k*x)
        # 对应的积分 s(x) = -a/k * cos(k*x) + b/k * sin(k*x) + C
        # 使得 s(0) = 0 => C = a/k
        
        u_data = []
        y_data = []
        s_data = []
        
        for _ in range(num_samples):
            # 随机参数
            a = np.random.uniform(-1, 1)
            b = np.random.uniform(-1, 1)
            k = np.random.randint(1, 5) * np.pi 
            
            # 1. 在传感器位置对 u 采样 (Branch Net Input)
            u_sensors = a * np.sin(k * self.sensor_points) + b * np.cos(k * self.sensor_points)
            
            # 2. 随机采样一些 y 点作为查询点 (Trunk Net Input)
            # 每个 u 生成 10 个 y 测试点
            y_points = np.random.uniform(0, 1, 10)
            
            # 3. 计算真实解 s(y) (Label)
            constant_C = a / k
            s_targets = - (a/k) * np.cos(k * y_points) + (b/k) * np.sin(k * y_points) + constant_C
            
            for y, s in zip(y_points, s_targets):
                u_data.append(u_sensors)
                y_data.append([y])
                s_data.append([s])
                
        return np.array(u_data, dtype=np.float32), \
               np.array(y_data, dtype=np.float32), \
               np.array(s_data, dtype=np.float32)

# ==========================================
# 3. 训练流程
# ==========================================
def train():
    # 参数设置
    M_SENSORS = 100   # 论文中使用 100 个传感器点
    P_OUTPUT = 40     # Branch/Trunk 输出特征维度 (p)
    HIDDEN_SIZE = 40  # 隐藏层宽
    LR = 0.001
    EPOCHS = 2000
    NUM_TRAIN_SAMPLES = 1000 # 生成 1000 个不同的函数 u
    
    # 准备数据
    gen = DataGenerator(M_SENSORS)
    u_train, y_train, s_train = gen.generate_data(NUM_TRAIN_SAMPLES)
    
    # 转为 Tensor
    u_train = torch.from_numpy(u_train)
    y_train = torch.from_numpy(y_train)
    s_train = torch.from_numpy(s_train)
    
    # 初始化模型
    model = DeepONet(num_sensors=M_SENSORS, dim_y=1, 
                     num_branch_layers=2, num_trunk_layers=2, 
                     hidden_size=HIDDEN_SIZE, output_size=P_OUTPUT)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    
    print("开始训练 DeepONet...")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Forward
        predictions = model(u_train, y_train)
        
        # Loss
        loss = loss_fn(predictions, s_train)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

    return model, gen

# ==========================================
# 4. 测试与可视化
# ==========================================
def test(model, gen):
    model.eval()
    
    # 生成一个新的测试样本 u_test(x) = sin(2*pi*x)
    # 真实解 s_test(x) = (1 - cos(2*pi*x)) / (2*pi)
    x_grid = np.linspace(0, 1, 100)
    
    # 构造 Branch Input
    # u(x) = sin(2*pi*x)
    u_test_vals = np.sin(2 * np.pi * gen.sensor_points)
    u_input = torch.tensor(u_test_vals, dtype=torch.float32).unsqueeze(0) # [1, m]
    # 需要重复 batch 次，以此来查询 grid 上的每个点
    u_input = u_input.repeat(len(x_grid), 1) # [100, m]
    
    # 构造 Trunk Input
    y_input = torch.tensor(x_grid.reshape(-1, 1), dtype=torch.float32) # [100, 1]
    
    # 预测
    with torch.no_grad():
        s_pred = model(u_input, y_input).numpy()
        
    # 真实值
    s_true = (1 - np.cos(2 * np.pi * x_grid)) / (2 * np.pi)
    
    # 绘图
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(gen.sensor_points, u_test_vals, 'k--', label='Input u(x)')
    plt.title("Input Function u(x)")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x_grid, s_true, 'b-', label='Exact s(x)')
    plt.plot(x_grid, s_pred, 'r--', label='DeepONet Prediction')
    plt.title("Output Function s(x)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    trained_model, generator = train()
    test(trained_model, generator)