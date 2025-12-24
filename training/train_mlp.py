import os, sys, torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# 项目根路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.mlp import MLP
from data_loader import load_data_from_pkl

def train_mlp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练参数
    T = 512              # 时间步长，需与数据切分一致
    batch_size = 16
    lr = 1e-3
    epochs = 40

    # 加载数据
    data_dir = os.path.join(project_root, "dataset", "simulation_data")
    pkl_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pkl")]

    all_x, all_u = [], []
    for pkl_path in pkl_files:
        x, u = load_data_from_pkl(pkl_path, T=T)
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
    model = MLP(input_dim=C, output_dim=C, hidden_dims=(128, 128)).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

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
            save_path = os.path.join(project_root, "results", "models", "best_mlp.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"保存最佳模型，验证损失: {best_val:.6f}")

if __name__ == "__main__":
    train_mlp()