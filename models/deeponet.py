import torch
import torch.nn as nn

"""
DeepONet
- 分支网络编码输入函数在传感器处的取值 (u_input)，主干网络编码查询坐标 (y_input)，
  按元素乘得到输出函数在该坐标的预测值。
- 输出形状 [batch, output_size]，支持多通道。
"""

class DeepONet(nn.Module):
    def __init__(
        self,
        num_sensors: int,      # 分支网络输入维度 (= 通道数)
        dim_y: int,            # 主干网络输入维度 (= 查询坐标维度)
        num_branch_layers: int,
        num_trunk_layers: int,
        hidden_size: int,
        output_size: int,      # 输出特征维度 (= 通道数)
    ):
        super().__init__()

        # Branch Net: (batch, num_sensors) -> (batch, output_size)
        branch_layers = [nn.Linear(num_sensors, hidden_size), nn.Tanh()]
        for _ in range(num_branch_layers - 1):
            branch_layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        branch_layers.append(nn.Linear(hidden_size, output_size))
        self.branch_net = nn.Sequential(*branch_layers)

        # Trunk Net: (batch, dim_y) -> (batch, output_size)
        trunk_layers = [nn.Linear(dim_y, hidden_size), nn.Tanh()]
        for _ in range(num_trunk_layers - 1):
            trunk_layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        trunk_layers.append(nn.Linear(hidden_size, output_size))
        self.trunk_net = nn.Sequential(*trunk_layers)

        # 可学习偏置
        self.bias = nn.Parameter(torch.zeros(1, output_size))

    def forward(self, u_input: torch.Tensor, y_input: torch.Tensor) -> torch.Tensor:
        """
        u_input: [batch, num_sensors]
        y_input: [batch, dim_y]
        return:  [batch, output_size]
        """
        B = self.branch_net(u_input)
        T = self.trunk_net(y_input)
        return B * T + self.bias
