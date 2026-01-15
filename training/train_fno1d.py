import os
import argparse
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import pathlib, sys
import numpy as np

# 转到项目根目录，以便导入模块
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from models.fno import FNO1d
from data_loader import load_data_from_pkl


def parse_args():
    parser = argparse.ArgumentParser(description="Train 1D FNO on ODE/PDE data")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to directory containing .pkl files")
    parser.add_argument("--sim_type", choices=["pde", "ode_ec", "ode_sc", "auto"], default="pde", help="Simulation type")
    parser.add_argument("--T", type=int, default=None, help="Sequence length; None/<=0 keeps full length")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--normalize", type=bool, default=True, help="Whether to normalize data")
    return parser.parse_args()

def discover_pkl_files(sim_type: str, explicit_dir: pathlib.Path | None):
    if explicit_dir:
        candidates = [explicit_dir]
    elif sim_type == "pde":
        candidates = [project_root / "dataset" / "pde_surface_new_500"]
    elif sim_type == "ode_ec":
        candidates = [project_root / "dataset" / "ode_ec_new_1000"]
    elif sim_type == "ode_sc":
        candidates = [project_root / "dataset" / "ode_sc_new_1000"]

    for d in candidates:
        pkl_files = list(d.glob("*.pkl")) if d.exists() else []
        if pkl_files:
            return d, pkl_files
    # 最后兜底到 dataset 根目录
    fallback = project_root / "dataset"
    pkl_files = list(fallback.glob("*.pkl")) if fallback.exists() else []
    return fallback, pkl_files


def main(args=None):
    args = args or parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 自动判定 sim_type
    if args.sim_type == "auto":
        if args.data_dir and "pde" in args.data_dir.lower():
            sim_type = "pde"
        elif args.data_dir and "ode_ec" in args.data_dir.lower():
            sim_type = "ode_ec"
        elif args.data_dir and "ode_sc" in args.data_dir.lower():
            sim_type = "ode_sc"
    else:
        sim_type = args.sim_type

    # 自动设置 T：PDE 用完整序列，ODE 默认 1024
    T = args.T
    if T is None or T <= 0:
        T = None if sim_type == "pde" else 1024

    data_dir, pkl_files = discover_pkl_files(sim_type, pathlib.Path(args.data_dir) if args.data_dir else None)

    # 根据 sim_type 设置模型保存路径
    if args.normalize:
        best_model_path = os.path.join(project_root, "results", "models", f"fno1d_{sim_type}_norm.pth")
    else:
        best_model_path = os.path.join(project_root, "results", "models", f"fno1d_{sim_type}.pth")
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    all_x, all_u = [], []
    if not pkl_files:
        print("未找到 .pkl 数据文件，将使用合成数据")
        C = 1
        n_train, n_val = 800, 200
        seq_len = T if T is not None else 10
        x_train = torch.randn(n_train, seq_len, C)
        u_train = torch.randn(n_train, seq_len, C)
        x_val   = torch.randn(n_val, seq_len, C)
        u_val   = torch.randn(n_val, seq_len, C)
    else:
        print(f"使用目录 {data_dir}, 找到 {len(pkl_files)} 个数据文件，开始加载...")
        for pkl_path in pkl_files:
            x, u = load_data_from_pkl(pkl_path, T=T, normalize=args.normalize, sim_type=sim_type)
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
        seq_len = x_train.shape[1]
        print(f"总样本数: {n}, 训练集: {n_tr}, 验证集: {n - n_tr}, 通道数: {C}, 序列长度: {seq_len}")

    train_loader = DataLoader(TensorDataset(x_train, u_train), batch_size=args.batch_size, shuffle=True)
    if len(x_val) > 0:
        val_loader = DataLoader(TensorDataset(x_val, u_val), batch_size=args.batch_size)
    else:
        val_loader = None

    model = FNO1d(input_size=C, output_size=C, modes=32, width=64).to(device)
    opt = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best = 1e9
    for epoch in range(1, args.epochs + 1):
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
    main()