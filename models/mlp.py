"""
mlp对照
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union
from abc import ABC, abstractmethod


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int, ...] = (64, 64)):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)