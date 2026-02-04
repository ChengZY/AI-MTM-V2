import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from scipy.interpolate import interp1d

# ==========================================
# 0. GPU 设置
# ==========================================
def get_device():
    if torch.cuda.is_available():
        print(f"🚀 GPU Ready: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')
    return torch.device('cpu')

DEVICE = get_device()

# ==========================================
# 1. 可微 KLA 物理层 (核心)
# ==========================================
class DifferentiableKLA(nn.Module):
    def __init__(self, wavelengths):
        super(DifferentiableKLA, self).__init__()
        self.wavelengths = wavelengths
        # 注册波长和k0为 buffer (不更新梯度，但随模型移动)
        self.register_buffer('lam_tensor', torch.tensor(wavelengths, dtype=torch.float32))
        self.register_buffer('k0', 2 * np.pi / self.lam_tensor)
        
        # 加载 KLA 数据并转为 Tensor
        self._load_kla_materials()
        
    def _load_kla_materials(self):
        """ 读取 txt -> 插值 -> Tensor (n + ik) """
        materials = {}
        
        # 定义加载逻辑
        def load_mat(fname, default_val):
            # 兼容路径
            paths = [fname, os.path.join('data', fname)]
            valid_path = next((p for p in paths if os.path.exists(p)), None)
            
            if not valid_path:
                print(f"[警告] 缺失 {fname}，使用默认值 {default_val}")
                return torch.full((len(self.wavelengths),), default_val, dtype=torch.cfloat)
                
            # 读取与插值
            try:
                # 读取 (跳过非数字行)
                data = []
                with open(valid_path, 'r') as f:
                    for line in f:
                        parts = line.split()
                        if parts and parts[0][0].isdigit():
                            data.append([float(p) for p in parts[:3]])
                data = np.array(data)
                data = data[data[:,0].argsort()] # 按波长排序
                
                # 插值
                f_n = interp1d(data[:,0], data[:,1], fill_value="extrapolate")
                f_k = interp1d(data[:,0], data[:,2], fill_value="extrapolate")
                n_interp = f_n(self.wavelengths)
                k_interp = f_k(self.wavelengths)
                
                # === 符号修正: N = n + ik (适配 PyTorch exp(-i*phi)) ===
                n_complex = n_interp + 1j * k_interp
                return torch.from_numpy(n_complex).cfloat()
                
            except Exception as e:
                print(f"[错误] 解析 {fname} 失败: {e}")
                return torch.full((len(self.wavelengths),), default_val, dtype=torch.cfloat)

        # 加载三种材料
        print("[物理层] 正在加载 KLA 材料库...")
        n_si_t = load_mat('./data/si.txt', 4.0+0j)
        n_sio2_t = load_mat('./data/sio2.txt', 1.46+0j)
        n_si3n4_t = load_mat('./data/si3n4.txt', 2.0+0j)
        n_air_t = torch.ones_like(n_si_t)
        
        # 注册为 buffer
        self.register_buffer('n_Si', n_si_t)
        self.register_buffer('n_SiO2', n_sio2_t)
        self.register_buffer('n_Si3N4', n_si3n4_t)
        self.register_buffer('n_Air', n_air_t)

    def forward(self, d_phys):
        """
        正演计算: d_phys (Batch, 7) -> Spectrum (Batch, 801)
        结构: Air | SiO2 | Si3N4 | SiO2 | Si3N4 | SiO2 | Si3N4 | SiO2 | Si
        """
        batch_size = d_phys.shape[0]
        
        # 1. 准备材料栈
        # 每一层的折射率需要扩展到 (Batch, 801)
        # 结构: L1(SiO2), L2(Si3N4), L3(SiO2), L4(Si3N4), L5(SiO2), L6(Si3N4), L7(SiO2)
        layers_n = [
            self.n_Air,   # 0
            self.n_SiO2,  # 1
            self.n_Si3N4, # 2
            self.n_SiO2,  # 3
            self.n_Si3N4, # 4
            self.n_SiO2,  # 5
            self.n_Si3N4, # 6
            self.n_SiO2,  # 7
            self.n_Si     # 8
        ]
        
        # 扩展维度: (801) -> (1, 801) -> (Batch, 801)
        layers_n = [n.view(1, -1).expand(batch_size, -1) for n in layers_n]
        
        # 准备厚度: d_phys is (Batch, 7)
        # 需要扩展 Air 和 Substrate 的厚度 (0)
        zeros = torch.zeros(batch_size, 1, device=d_phys.device)
        # d_list: [0, d1, d2, d3, d4, d5, d6, d7, 0]
        d_list = [zeros] + [d_phys[:, i:i+1] for i in range(7)] + [zeros]
        
        # 2. TMM 矩阵计算
        M00 = torch.ones(batch_size, len(self.wavelengths), dtype=torch.cfloat, device=d_phys.device)
        M11 = torch.ones_like(M00)
        M01 = torch.zeros_like(M00)
        M10 = torch.zeros_like(M00)
        
        N_curr = layers_n[0]
        
        for i in range(1, len(layers_n)):
            N_next = layers_n[i]
            d_next = d_list[i] # (Batch, 1)
            
            # 界面
            sum_n = N_curr + N_next
            sub_n = N_curr - N_next
            r = sub_n / sum_n
            inv_t = sum_n / (2 * N_curr)
            
            m00 = (M00 + M01*r) * inv_t
            m01 = (M00*r + M01) * inv_t
            m10 = (M10 + M11*r) * inv_t
            m11 = (M10*r + M11) * inv_t
            M00, M01, M10, M11 = m00, m01, m10, m11
            
            # 传播 (最后基底不传播)
            if i < len(layers_n) - 1:
                # phi = k0 * n * d
                phi = self.k0.view(1, -1) * N_next * d_next
                # exp(-i * phi)
                P = torch.exp(-1j * phi)
                Pi = torch.exp(1j * phi)
                
                M00 = M00 * P
                M01 = M01 * Pi
                M10 = M10 * P
                M11 = M11 * Pi
                
            N_curr = N_next
            
        # 3. 反射率 R = |M10/M00|^2
        r_final = M10 / M00
        R = torch.abs(r_final)**2
        return R

# ==========================================
# 2. 神经网络 (7层输出版)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class Net7Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 256, 2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.regressor = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 7) # 输出 7 个厚度
        )
    def forward(self, x):
        x = x.unsqueeze(1) # (B, 1, 801)
        feat = self.encoder(x).view(x.size(0), -1)
        return self.regressor(feat)

# ==========================================
# 3. 数据加载
# ==========================================
class SimulationDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        # 前7列是厚度 (Label)
        self.d = df.iloc[:, :7].values.astype(np.float32)
        # 第8列往后是光谱 (Input)
        self.R = df.iloc[:, 7:].values.astype(np.float32)
        print(f"Dataset Loaded: {len(df)} samples")
        
    def __len__(self):
        return len(self.d)
    
    def __getitem__(self, idx):
        return self.R[idx], self.d[idx]

# ==========================================
# 4. 主程序
# ==========================================
def train():
    csv_file = './data/sobol_7layer_16384_chunk_0000.csv'
    if not os.path.exists(csv_file):
        print(f"❌ 找不到 {csv_file}")
        return

    # 数据集
    dataset = SimulationDataset(csv_file)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    
    # 归一化统计量 (用于恢复物理数值)
    d_all = dataset.d
    d_mean = torch.tensor(d_all.mean(axis=0), device=DEVICE)
    d_std = torch.tensor(d_all.std(axis=0), device=DEVICE)
    
    # 模型与物理层
    model = Net7Layer().to(DEVICE)
    tmm_layer = DifferentiableKLA(np.arange(200, 1001)).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    print(">>> 开始训练 7层 KLA-PINN 模型...")
    loss_history = []
    
    start_time = time.time()
    
    # 训练循环
    for epoch in range(100): # 演示用 100 epoch，实际建议 500+
        model.train()
        total_loss = 0
        
        for R_batch, d_batch in loader:
            R_batch, d_batch = R_batch.to(DEVICE), d_batch.to(DEVICE)
            
            # 1. 归一化厚度 (让网络好学)
            d_target_norm = (d_batch - d_mean) / d_std
            
            optimizer.zero_grad()
            
            # 2. 前向传播
            d_pred_norm = model(R_batch)
            
            # 3. 计算数据 Loss (Supervised)
            loss_data = criterion(d_pred_norm, d_target_norm)
            
            # 4. 物理 Loss (Physics-Informed)
            # 反归一化得到物理厚度
            d_pred_phys = d_pred_norm * d_std + d_mean
            d_pred_phys = torch.abs(d_pred_phys) # 物理约束: 厚度非负
            
            # 通过可微 KLA 物理层计算光谱
            R_recon = tmm_layer(d_pred_phys)
            loss_phy = criterion(R_recon, R_batch)
            
            # 5. 总 Loss
            loss = loss_data + 5.0 * loss_phy # 物理权重
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)
        
        if (epoch+1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/100] | Loss: {avg_loss:.6f} | LR: {lr:.6f}")
            
    print(f"\n✅ 训练完成! 耗时: {time.time() - start_time:.2f}s")
    
    # ==========================================
    # 验证与绘图
    # ==========================================
    model.eval()
    # 随机取一个样本
    sample_idx = 0
    test_R = torch.from_numpy(dataset.R[sample_idx:sample_idx+1]).to(DEVICE)
    true_d = dataset.d[sample_idx]
    
    with torch.no_grad():
        pred_norm = model(test_R)
        pred_phys = pred_norm * d_std + d_mean
        rec_R = tmm_layer(pred_phys)
        
    pred_d_np = pred_phys.cpu().numpy()[0]
    rec_R_np = rec_R.cpu().numpy()[0]
    true_R_np = dataset.R[sample_idx]
    
    # 绘图
    plt.figure(figsize=(15, 5))
    
    # 1. Loss
    plt.subplot(1, 3, 1)
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.yscale('log')
    plt.grid(True)
    
    # 2. Spectrum
    plt.subplot(1, 3, 2)
    wl = np.arange(200, 1001)
    plt.plot(wl, true_R_np, 'k-', linewidth=3, alpha=0.3, label='True')
    plt.plot(wl, rec_R_np, 'r--', label='AI Pred')
    plt.title('KLA-Physics Reconstruction')
    plt.legend()
    plt.grid(True)
    
    # 3. Thickness Error (7层)
    plt.subplot(1, 3, 3)
    layers_idx = np.arange(7)
    width = 0.35
    plt.bar(layers_idx - width/2, true_d, width, label='True', color='gray', alpha=0.6)
    plt.bar(layers_idx + width/2, pred_d_np, width, label='Pred', color='#1f77b4', alpha=0.9)
    
    # 标注误差
    err = np.abs(true_d - pred_d_np)
    for i in range(7):
        plt.text(i, max(true_d[i], pred_d_np[i]), f'{err[i]:.1f}', 
                 ha='center', va='bottom', color='red', fontsize=8)
                 
    plt.title('7-Layer Thickness Prediction')
    plt.xticks(layers_idx, [f'L{i+1}' for i in range(7)])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 保存模型
    torch.save(model.state_dict(), 'pinn_7layer_kla.pth')
    print("模型已保存: pinn_7layer_kla.pth")

if __name__ == "__main__":
    train()