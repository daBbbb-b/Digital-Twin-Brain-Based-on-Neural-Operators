import os
import sys
import pathlib
import pickle
import numpy as np
import torch
import scipy.interpolate

# ================= 1. 路径环境配置 =================
# 获取当前脚本所在目录
current_dir = pathlib.Path(__file__).resolve().parent
# 将项目根目录加入 Python 搜索路径，确保能 import models
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# 尝试导入模型定义
try:
    from models.fno import FNO1d
    from models.deeponet import DeepONet
except ImportError as e:
    print("❌ 导入错误: 无法找到模型定义。")
    print("请确保 'models' 文件夹在当前目录下，且包含 fno.py 和 deeponet.py")
    raise e

# ================= 2. 全局参数配置 =================
# 路径配置
MODEL_DIR = current_dir / "results" / "models"
DATA_DIR = current_dir / "inference_data"
RESULT_ROOT = current_dir / "inference_results"

# --- 数据处理参数 ---
ORIGINAL_TR = 2.0   # 原始 BOLD 采样间隔 (s)
TARGET_TR = 0.05    # 模型训练用的采样间隔 (s)
T_WINDOW = 1024     # 推理窗口长度 (对应 51.2s)

# --- FNO 模型参数 (需与训练一致) ---
FNO_CONFIG = {
    "modes": 32,
    "width": 64
}

# --- DeepONet 模型参数 (需与训练一致) ---
DEEPONET_CONFIG = {
    "dim_y": 16,             # 查询坐标维度
    "num_branch_layers": 2,
    "num_trunk_layers": 2,
    "hidden_size": 128
}

# ================= 3. 核心工具函数 =================

def resample_signal(signal, original_tr, target_tr):
    """
    重采样：将低频信号 (2s) 插值到高频 (0.05s)
    """
    n_time, n_channels = signal.shape
    original_time = np.arange(n_time) * original_tr
    max_time = original_time[-1]
    target_time = np.arange(0, max_time, target_tr)
    
    f = scipy.interpolate.interp1d(original_time, signal, kind='linear', axis=0, fill_value="extrapolate")
    return f(target_time)

def predict_chunk(model, chunk, device):
    """
    对单个时间窗口进行推理，自动适配 FNO 和 DeepONet 的输入格式
    chunk shape: (T_WINDOW, Channels)
    """
    # 1. 如果是 FNO 模型
    if isinstance(model, FNO1d):
        # FNO 需要 (Batch, Time, Channels) -> (1, T, C)
        x_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            u_pred = model(x_tensor)
        return u_pred.cpu().squeeze(0).numpy() # (T, C)

    # 2. 如果是 DeepONet 模型
    elif isinstance(model, DeepONet):
        # DeepONet 逻辑 (参考 train_deeponet.py):
        # Branch Input (x): 传感器读数 -> 这里直接把时刻 t 的 BOLD 值作为传感器输入
        # Trunk Input (y): 查询坐标 -> 时刻 t 的归一化时间
        
        T, C = chunk.shape
        
        # 构造 Branch 输入: (Batch_Size, Num_Sensors) -> 这里 Batch_Size = T
        x_flat = torch.tensor(chunk, dtype=torch.float32).to(device) # (T, C)
        
        # 构造 Trunk 输入: (Batch_Size, Dim_Y)
        # 生成 [0, 1] 的线性时间轴
        dim_y = DEEPONET_CONFIG["dim_y"]
        grid = torch.linspace(0, 1, T).to(device) # (T,)
        y_flat = grid.unsqueeze(1).repeat(1, dim_y) # (T, dim_y)
        
        with torch.no_grad():
            # DeepONet 前向传播: B(x) * T(y) + bias
            u_pred = model(x_flat, y_flat) # (T, Output_Size)
            
        return u_pred.cpu().numpy()

    else:
        raise ValueError(f"未知的模型类型: {type(model)}")

def predict_full_sequence(model, long_signal, window_size, device):
    """
    全序列推理逻辑 (Look-back Strategy)
    """
    total_len = long_signal.shape[0]
    output_list = []
    
    # 边界情况：总长度不足一个窗口
    if total_len < window_size:
        pad_len = window_size - total_len
        chunk_padded = np.pad(long_signal, ((0, pad_len), (0, 0)), mode='constant')
        u_out = predict_chunk(model, chunk_padded, device)
        return u_out[:total_len]

    # --- 1. 处理所有完整的窗口 ---
    num_full_windows = total_len // window_size
    
    for i in range(num_full_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        chunk = long_signal[start_idx : end_idx, :]
        
        u_out = predict_chunk(model, chunk, device)
        output_list.append(u_out)

    # --- 2. 处理剩余尾部 (Look-back) ---
    remainder = total_len % window_size
    if remainder > 0:
        # 回退指针，截取最后 window_size 长度
        start_idx = total_len - window_size
        end_idx = total_len
        chunk = long_signal[start_idx : end_idx, :]
        
        u_out = predict_chunk(model, chunk, device)
        
        # 只取最后 remainder 部分
        u_tail = u_out[-remainder:, :]
        output_list.append(u_tail)

    # 3. 拼接
    return np.concatenate(output_list, axis=0)

def load_model_instance(model_path, n_regions, device):
    """
    根据文件名自动选择并实例化模型
    """
    name = model_path.name.lower()
    
    # --- 实例化逻辑 ---
    if "fno" in name:
        print(f"   Detected FNO model. Config: {FNO_CONFIG}")
        model = FNO1d(
            input_size=n_regions, 
            output_size=n_regions, 
            modes=FNO_CONFIG["modes"], 
            width=FNO_CONFIG["width"]
        )
        
    elif "deeponet" in name:
        print(f"   Detected DeepONet model. Config: {DEEPONET_CONFIG}")
        model = DeepONet(
            num_sensors=n_regions,     # Branch 输入维度 = 脑区数
            output_size=n_regions,     # 输出维度 = 脑区数
            dim_y=DEEPONET_CONFIG["dim_y"],
            num_branch_layers=DEEPONET_CONFIG["num_branch_layers"],
            num_trunk_layers=DEEPONET_CONFIG["num_trunk_layers"],
            hidden_size=DEEPONET_CONFIG["hidden_size"]
        )
    else:
        print(f"⚠️ 无法从文件名 '{name}' 识别模型类型，跳过。")
        return None

    # --- 加载权重 ---
    model = model.to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # 处理可能的 state_dict 嵌套
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.eval()
        print("   ✅ 权重加载成功")
        return model
        
    except Exception as e:
        print(f"   ❌ 权重加载失败: {e}")
        return None

# ================= 4. 主程序 =================

def run_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 运行设备: {device}")
    
    # 1. 扫描模型
    if not MODEL_DIR.exists():
        print(f"❌ 模型目录不存在: {MODEL_DIR}")
        return
    model_files = list(MODEL_DIR.glob("*.pth"))
    if not model_files:
        print(f"❌ 未找到模型文件")
        return
    print(f"📂 发现 {len(model_files)} 个模型: {[m.name for m in model_files]}")

    # 2. 准备数据
    pkl_files = list(DATA_DIR.glob("*.pkl"))
    if not pkl_files:
        print(f"❌ 未找到数据文件")
        return
    
    # 获取通道数
    with open(pkl_files[0], "rb") as f:
        meta = pickle.load(f)
        sample_data = meta.get("x", meta.get("bold_signal"))
        n_regions = sample_data.shape[1]
    print(f"🧠 数据通道数: {n_regions}")

    # ================= 循环开始 =================
    for i, model_path in enumerate(model_files):
        model_stem = model_path.stem
        print(f"\n[{i+1}/{len(model_files)}] 处理模型: {model_stem} ...")
        
        # 1. 创建结果目录
        current_save_dir = RESULT_ROOT / model_stem
        os.makedirs(current_save_dir, exist_ok=True)
        
        # 2. 加载模型
        model = load_model_instance(model_path, n_regions, device)
        if model is None: continue

        # 3. 遍历数据
        print(f"   🚀 开始推理 {len(pkl_files)} 个数据文件...")
        success_count = 0
        
        for pkl_file in pkl_files:
            try:
                # 检查是否存在
                save_path = current_save_dir / (pkl_file.stem + "_u.npy")
                # if save_path.exists(): continue # 可选：跳过已存在的

                # 加载与重采样
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)
                x_raw = data.get("x", data.get("bold_signal"))
                if x_raw is None: continue

                x_resampled = resample_signal(x_raw, ORIGINAL_TR, TARGET_TR)
                
                # 执行推理 (自动适配模型类型)
                u_out = predict_full_sequence(model, x_resampled, T_WINDOW, device)

                # 保存
                np.save(save_path, u_out)
                success_count += 1
                
            except Exception as e:
                print(f"   ⚠️ 数据错误 {pkl_file.stem}: {e}")

        print(f"   ✅ 模型 {model_stem} 完成。结果保存在: {current_save_dir}")

    print("\n🎉 全部任务结束！")

if __name__ == "__main__":
    run_inference()