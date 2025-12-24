import os
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import pathlib, sys, pickle
import numpy as np
# 转到项目根目录，以便导入模块
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from models.fno import FNO1d
from data_loader import load_data_from_pkl

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T = 256
    
    # 寻找数据文件: 优先使用 dataset/simulation_data 下的所有 pkl 文件
    data_dir = project_root / "dataset" / "simulation_data"
    pkl_files = list(data_dir.glob("*.pkl"))
    
    # 如果没找到，尝试 dataset/ 目录
    if not pkl_files:
        data_dir = project_root / "dataset"
        pkl_files = list(data_dir.glob("*.pkl"))

    all_x, all_u = [], []
    if not pkl_files:
        print("未找到 .pkl 数据文件，将使用合成数据")
        C = 1
        n_train, n_val = 800, 200
        x_train = torch.randn(n_train, T, C)
        u_train = torch.randn(n_train, T, C)
        x_val   = torch.randn(n_val, T, C)
        u_val   = torch.randn(n_val, T, C)
    else:
        print(f"找到 {len(pkl_files)} 个数据文件，开始加载...")
        for pkl_path in pkl_files:
            x, u = load_data_from_pkl(pkl_path, T=T)
            if x is not None and u is not None:
                all_x.append(x)
                all_u.append(u)
        
        if not all_x:
            print("所有文件加载失败或数据为空，退出")
            return

        x_all = torch.cat(all_x, dim=0)
        u_all = torch.cat(all_u, dim=0)
        
        # 随机打乱
        perm = torch.randperm(x_all.size(0))
        x_all = x_all[perm]
        u_all = u_all[perm]

        # 划分训练集和验证集 (8:2)
        n = x_all.shape[0]
        n_tr = int(n * 0.8)
        x_train, u_train = x_all[:n_tr], u_all[:n_tr]
        x_val, u_val = x_all[n_tr:], u_all[n_tr:]
        
        C = x_train.shape[-1]
        print(f"总样本数: {n}, 训练集: {n_tr}, 验证集: {n - n_tr}, 通道数: {C}")

    train_loader = DataLoader(TensorDataset(x_train, u_train), batch_size=16, shuffle=True)
    if len(x_val) > 0:
        val_loader = DataLoader(TensorDataset(x_val, u_val), batch_size=16)
    else:
        val_loader = None

    model = FNO1d(input_size=C, output_size=C, modes=32, width=64).to(device)
    opt = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best = 1e9
    for epoch in range(1, 41):
        model.train()
        tot = 0
        for x,u in train_loader:
            x,u = x.to(device), u.to(device)
            pred = model(x)
            loss = loss_fn(pred, u)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
        train_loss = tot / len(train_loader.dataset)

        if val_loader:
            model.eval(); tot = 0
            with torch.no_grad():
                for x,u in val_loader:
                    x,u = x.to(device), u.to(device)
                    tot += loss_fn(model(x), u).item() * x.size(0)
            val_loss = tot / len(val_loader.dataset)
        else:
            val_loss = 0.0

        print(f"Epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with val loss: {best:.6f}")


if __name__ == "__main__":
    best_model_path = os.path.join(project_root, "results", "models", "best_fno1d.pth")
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    main()