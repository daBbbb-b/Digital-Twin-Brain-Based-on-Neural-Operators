import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.deeponet import DeepONet
from data_loader import load_data_from_pkl

def train_deeponet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_branch_layers = 2
    num_trunk_layers = 2
    hidden_size = 128
    batch_size = 8
    learning_rate = 1e-3
    epochs = 50
    T = 512
    dim_y = 16 # 查询坐标维度，维度越高，表示查询点信息越丰富

    print("加载数据...")
    data_dir = os.path.join(project_root, "dataset", "simulation_data")
    pkl_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pkl")]

    all_x, all_y, all_u = [], [], []
    for pkl_path in pkl_files:
        x, u = load_data_from_pkl(pkl_path, T=T)
        if x is None or u is None:
            continue
        num_samples, _, C = x.shape
        for i in range(num_samples):
            x_sample = x[i]                               # [T, C]
            u_sample = u[i]                               # [T, C]
            y_sample = torch.linspace(0, 1, T).unsqueeze(1).repeat(1, dim_y)  # [T, dim_y]
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
            save_path = os.path.join(project_root, "results", "models", "best_deeponet.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"保存最佳模型，验证损失: {best_val:.6f}")

if __name__ == "__main__":
    train_deeponet()