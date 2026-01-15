import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.deeponet import DeepONet
from data_loader import load_data_from_pkl


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepONet on ODE/PDE data")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing .pkl files")
    parser.add_argument("--sim_type", choices=["pde", "ode_ec", "ode_sc", "auto"], default="ode_sc", help="Simulation type")
    parser.add_argument("--T", type=int, default=None, help="Sequence length; None/<=0 keeps full length")
    parser.add_argument("--dim_y", type=int, default=16, help="Query coordinate dimension")
    parser.add_argument("--num_branch_layers", type=int, default=2)
    parser.add_argument("--num_trunk_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--normalize", type=bool, default=True, help="Whether to normalize data")
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


def train_deeponet(args=None):
    args = args or parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 决定 sim_type
    if args.sim_type == "auto":
        if args.data_dir and "pde" in args.data_dir.lower():
            sim_type = "pde"
        elif args.data_dir and "ode_ec" in args.data_dir.lower():
            sim_type = "ode_ec"
        elif args.data_dir and "ode_sc" in args.data_dir.lower():
            sim_type = "ode_sc"
    else:
        sim_type = args.sim_type

    # 决定 T：PDE 默认用完整序列，ODE 默认 1024
    T = args.T
    if T is None or T <= 0:
        T = None if sim_type == "pde" else 1024

    dim_y = args.dim_y
    num_branch_layers = args.num_branch_layers
    num_trunk_layers = args.num_trunk_layers
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epochs
    normalize = args.normalize

    data_dir, pkl_files = discover_pkl_files(sim_type, args.data_dir)

    print(f"加载数据... sim_type={sim_type}, data_dir={data_dir}, files={len(pkl_files)}")

    all_x, all_y, all_u = [], [], []
    for pkl_path in pkl_files:
        x, u = load_data_from_pkl(pkl_path, T=T, normalize=normalize, sim_type=sim_type)
        if x is None or u is None:
            continue
        num_samples, _, C = x.shape
        for i in range(num_samples):
            x_sample = x[i]                               # [T, C]
            u_sample = u[i]                               # [T, C]
            seq_len = x_sample.shape[0]
            y_sample = torch.linspace(0, 1, seq_len).unsqueeze(1).repeat(1, dim_y)  # [T, dim_y]
            all_x.append(x_sample)
            all_u.append(u_sample)
            all_y.append(y_sample)

    if not all_x:
        print("未加载到有效数据，退出")
        return

    x_tensor = torch.stack(all_x)   # [N, T, C]
    y_tensor = torch.stack(all_y)   # [N, T, dim_y]
    u_tensor = torch.stack(all_u)   # [N, T, C]

    num_sensors = x_tensor.shape[-1]
    output_size = num_sensors
    print(f"x_tensor shape: {x_tensor.shape}")
    print(f"y_tensor shape: {y_tensor.shape}")
    print(f"u_tensor shape: {u_tensor.shape}")
    print(f"num_sensors/output_size: {num_sensors}")

    n_train = int(0.8 * len(x_tensor))
    train_dataset = TensorDataset(x_tensor[:n_train], y_tensor[:n_train], u_tensor[:n_train])
    val_dataset   = TensorDataset(x_tensor[n_train:], y_tensor[n_train:], u_tensor[n_train:])
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader    = DataLoader(val_dataset, batch_size=batch_size)

    model = DeepONet(
        num_sensors=num_sensors,
        dim_y=dim_y,
        num_branch_layers=num_branch_layers,
        num_trunk_layers=num_trunk_layers,
        hidden_size=hidden_size,
        output_size=output_size,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    print("开始训练 DeepONet...")
    if args.normalize:
        save_path = os.path.join(project_root, "results", "models", f"deeponet_{sim_type}_norm.pth")
    else:
        save_path = os.path.join(project_root, "results", "models", f"deeponet_{sim_type}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch, u_batch in train_loader:
            x_batch = x_batch.to(device)  # [B, T, C]
            y_batch = y_batch.to(device)  # [B, T, 1]
            u_batch = u_batch.to(device)  # [B, T, C]

            B_, T_, C_ = x_batch.shape
            x_flat = x_batch.reshape(B_ * T_, C_)   # [B*T, C]
            y_flat = y_batch.reshape(B_ * T_, dim_y)    # [B*T, dim_y]
            u_flat = u_batch.reshape(B_ * T_, C_)   # [B*T, C]

            preds = model(x_flat, y_flat)           # [B*T, C]
            loss = loss_fn(preds, u_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch, u_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                u_batch = u_batch.to(device)

                B_, T_, C_ = x_batch.shape
                x_flat = x_batch.reshape(B_ * T_, C_)
                y_flat = y_batch.reshape(B_ * T_, dim_y)
                u_flat = u_batch.reshape(B_ * T_, C_)

                preds = model(x_flat, y_flat)
                loss = loss_fn(preds, u_flat)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch} | train {train_loss:.6f} | val {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"保存最佳模型，验证损失: {best_val:.6f}")

if __name__ == "__main__":
    train_deeponet()