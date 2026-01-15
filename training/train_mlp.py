import os, sys, torch
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# 项目根路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.mlp import MLP
from data_loader import load_data_from_pkl


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP on ODE/PDE data")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing .pkl files")
    parser.add_argument("--sim_type", choices=["pde", "ode_ec", "ode_sc", "auto"], default="ode_sc", help="Simulation type")
    parser.add_argument("--T", type=int, default=None, help="Sequence length; None/<=0 keeps full length")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension per layer (2 layers)")
    parser.add_argument("--normalize", type=bool, default=False, help="Whether to normalize data")
    return parser.parse_args()


def discover_pkl_files(sim_type: str, explicit_dir: str | None):
    if explicit_dir:
        candidates = [os.path.abspath(explicit_dir)]
    elif sim_type == "pde":
        candidates = [os.path.join(project_root, "dataset", "pde_surface_new_500")]
    elif sim_type == "ode_ec":
        candidates = [os.path.join(project_root, "dataset", "ode_ec_new_1000")]
    elif sim_type == "ode_sc":
        candidates = [os.path.join(project_root, "dataset", "ode_sc_new_1000")]

    for d in candidates:
        if os.path.isdir(d):
            pkl_files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".pkl")]
            if pkl_files:
                return d, pkl_files

    fallback = os.path.join(project_root, "dataset")
    pkl_files = [os.path.join(fallback, f) for f in os.listdir(fallback) if f.endswith(".pkl")] if os.path.isdir(fallback) else []
    return fallback, pkl_files


def train_mlp(args=None):
    args = args or parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 判定 sim_type
    if args.sim_type == "auto":
        if args.data_dir and "pde" in args.data_dir.lower():
            sim_type = "pde"
        elif args.data_dir and "ode_ec" in args.data_dir.lower():
            sim_type = "ode_ec"
        elif args.data_dir and "ode_sc" in args.data_dir.lower():
            sim_type = "ode_sc"
    else:
        sim_type = args.sim_type

    # 设置 T：PDE 用完整序列，ODE 默认 1024
    T = args.T
    if T is None or T <= 0:
        T = None if sim_type == "pde" else 1024

    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    hidden = args.hidden

    data_dir, pkl_files = discover_pkl_files(sim_type, args.data_dir)
    print(f"加载数据... sim_type={sim_type}, data_dir={data_dir}, files={len(pkl_files)}")

    all_x, all_u = [], []
    for pkl_path in pkl_files:
        x, u = load_data_from_pkl(pkl_path, T=T, normalize=args.normalize, sim_type=sim_type)
        if x is None or u is None:
            continue
        all_x.append(x)  # [num_samples, T, C]
        all_u.append(u)  # [num_samples, T, C]

    if not all_x:
        print("未加载到有效数据，退出")
        return

    x_tensor = torch.cat(all_x, dim=0)  # [N, T, C]
    u_tensor = torch.cat(all_u, dim=0)  # [N, T, C]
    N, T_, C = x_tensor.shape

    # 展平时间维，输入输出均为 (N*T, C)
    x_flat = x_tensor.reshape(N * T_, C)
    u_flat = u_tensor.reshape(N * T_, C)

    # 划分训练/验证
    n_train = int(0.8 * x_flat.size(0))
    train_ds = TensorDataset(x_flat[:n_train], u_flat[:n_train])
    val_ds   = TensorDataset(x_flat[n_train:], u_flat[n_train:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    # 模型、优化器、损失
    model = MLP(input_dim=C, output_dim=C, hidden_dims=(hidden, hidden)).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    if args.normalize:
        save_path = os.path.join(project_root, "results", "models", f"mlp_{sim_type}_norm.pth")
    else:
        save_path = os.path.join(project_root, "results", "models", f"mlp_{sim_type}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_val = float("inf")
    print("开始训练 MLP...")
    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        train_loss = 0.0
        for x_b, u_b in train_loader:
            x_b, u_b = x_b.to(device), u_b.to(device)
            pred = model(x_b)
            loss = loss_fn(pred, u_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * x_b.size(0)
        train_loss /= len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, u_b in val_loader:
                x_b, u_b = x_b.to(device), u_b.to(device)
                pred = model(x_b)
                loss = loss_fn(pred, u_b)
                val_loss += loss.item() * x_b.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch} | train {train_loss:.6f} | val {val_loss:.6f}")

        # 保存最优
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"保存最佳模型，验证损失: {best_val:.6f}")

if __name__ == "__main__":
    train_mlp()