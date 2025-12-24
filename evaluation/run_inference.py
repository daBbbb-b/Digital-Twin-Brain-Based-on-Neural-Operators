import os
import sys
import pathlib
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.interpolate  #用于重采样

# ================= 1. 路径修复与导入 =================
# 获取当前文件所在目录 (即项目根目录 d:/python/Projecet/brain/)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
model_path = PROJECT_ROOT / "results" / "models"
# 将根目录加入 Python 搜索路径，解决 "ModuleNotFoundError"
if str(model_path) not in sys.path:
    sys.path.append(str(model_path))

try:
    from models.fno import FNO1d
except ImportError as e:
    print("❌ 导入错误: 无法找到 models.fno。")
    print(f"当前 sys.path: {sys.path}")
    print("请检查 models 文件夹下是否有 fno.py 和 __init__.py")
    raise e

# ================= 2. 参数配置 =================
# 模型权重路径
MODEL_PATH = PROJECT_ROOT / "results" / "models" / "best_fno1d.pth"
# 输入数据路径 (预处理后的 .pkl 文件夹)
DATA_DIR = PROJECT_ROOT / "results" / "inference_data"
# 结果保存路径
SAVE_DIR = PROJECT_ROOT / "results" / "inference_results"
# 关键参数 (必须与训练时一致)
T_WINDOW = 256  # 模型输入的时间窗口长度 (对应 12.8s)
MODES = 32
WIDTH = 64

# 重采样参数
ORIGINAL_TR = 2.0   # 真实数据的采样间隔 (秒)
TARGET_TR = 0.05    # 模型训练时的采样间隔 (秒)

def resample_signal(signal, original_tr, target_tr):
    """
    将低频信号 (2s) 插值为高频信号 (0.05s) 以匹配 FNO 模型
    :param signal: (Time, Channels) numpy array
    :return: (New_Time, Channels) numpy array
    """
    n_time, n_channels = signal.shape
    # 原始时间轴: [0, 2, 4, ...]
    original_time = np.arange(n_time) * original_tr
    
    # 目标时间轴: [0, 0.05, 0.1, ..., max_time]
    max_time = original_time[-1]
    target_time = np.arange(0, max_time, target_tr)
    
    # 线性插值
    # axis=0 表示沿时间轴插值
    f = scipy.interpolate.interp1d(original_time, signal, kind='linear', axis=0, fill_value="extrapolate")
    new_signal = f(target_time)
    
    return new_signal

def run_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"运行设备: {device}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. 扫描数据文件
    pkl_files = list(DATA_DIR.glob("*.pkl"))
    if not pkl_files:
        print(f"❌ 未在 {DATA_DIR} 找到 .pkl 文件。请先运行 preprocess_bold.py。")
        return

    # 2. 自动检测脑区数 (Channels)
    with open(pkl_files[0], "rb") as f:
        meta = pickle.load(f)
        # 兼容不同的键名 ("x" 或 "bold_signal")
        if "x" in meta:
            sample_data = meta["x"]
        elif "bold_signal" in meta:
            sample_data = meta["bold_signal"]
        else:
            print("❌ 数据格式错误: 找不到 'x' 或 'bold_signal' 键")
            return
            
        n_regions = sample_data.shape[1] # 应该是 246
    
    print(f"检测到脑区数 (Channels): {n_regions}")
    print(f"重采样策略: {ORIGINAL_TR}s -> {TARGET_TR}s (倍率: {ORIGINAL_TR/TARGET_TR}x)")

    # 3. 初始化模型并加载权重
    model = FNO1d(input_size=n_regions, output_size=n_regions, 
                  modes=MODES, width=WIDTH).to(device)
    
    if not MODEL_PATH.exists():
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        return

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        # 处理可能的 state_dict 嵌套
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print("✅ 模型权重加载成功")
    except Exception as e:
        print(f"❌ 加载模型权重失败: {e}")
        return

    # 4. 批量推理
    success_count = 0
    with torch.no_grad():
        for pkl_file in pkl_files:
            try:
                # --- 加载数据 ---
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)
                
                # 获取原始 BOLD 信号
                x_raw = data.get("x", data.get("bold_signal"))
                if x_raw is None: continue

                # --- 关键步骤：重采样 (Upsampling) ---
                # 从 2s 插值到 0.05s
                x_resampled = resample_signal(x_raw, ORIGINAL_TR, TARGET_TR)
                
                # --- 数据切片/处理 ---
                # 这里的策略：我们使用重采样后的数据进行推理
                # 由于模型输入固定为 T_WINDOW (256)，即 12.8秒
                # 我们可以截取前 12.8秒，或者做滑动窗口。
                # 演示：仅截取第一段 T=256 (如果不够长则补零)
                
                total_len = x_resampled.shape[0]
                if total_len >= T_WINDOW:
                    x_input = x_resampled[:T_WINDOW, :]
                else:
                    # 补零
                    pad_len = T_WINDOW - total_len
                    x_input = np.pad(x_resampled, ((0, pad_len), (0, 0)), mode='constant')
                
                # 转为 Tensor (Batch, Time, Channels)
                x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(device)

                # --- 模型推理 ---
                # Input: BOLD -> Output: u (Neural Activity)
                u_pred = model(x_tensor)
                
                # 转回 Numpy
                u_out = u_pred.cpu().squeeze(0).numpy() # (256, 246)

                # --- 保存结果 ---
                save_name = pkl_file.stem + "_u.npy"
                np.save(SAVE_DIR / save_name, u_out)
                
                # --- 可视化验证 (保存第一张图) ---
                if success_count == 0:
                    plt.figure(figsize=(12, 6))
                    # 归一化以便对比形状
                    region_idx = 0 # 观察第0个脑区
                    b_plot = x_input[:, region_idx]
                    u_plot = u_out[:, region_idx]
                    
                    # 简单 MinMax 归一化用于绘图
                    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-6)
                    
                    plt.plot(norm(b_plot), label='Input BOLD (Resampled)', alpha=0.7)
                    plt.plot(norm(u_plot), label='Inferred u', alpha=0.7)
                    plt.title(f"Inference Check: {pkl_file.name} (Region {region_idx})")
                    plt.legend()
                    plt.savefig(SAVE_DIR / "check_inference.png")
                    plt.close()
                    print(f"已保存可视化检查图: {SAVE_DIR / 'check_inference.png'}")

                print(f"[{success_count+1}] 反演完成: {pkl_file.name} -> {save_name} (Input Shape: {x_input.shape})")
                success_count += 1
                
            except Exception as e:
                print(f"处理文件 {pkl_file.name} 时出错: {e}")

    print(f"\n🎉 全部完成！共处理 {success_count} 个文件。")
    print(f"结果保存在: {SAVE_DIR}")

if __name__ == "__main__":
    run_inference()