import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import time
from scipy.interpolate import interp1d

# ==============================================================================
# 模块一：KLA 材料加载器 (保持不变，支持符号修正)
# ==============================================================================
class KLA_Material_Loader:
    def __init__(self, wavelengths):
        """
        初始化加载器
        :param wavelengths: 我们仿真需要的目标波长数组 (例如 200-1000nm)
        """
        self.target_wl = wavelengths
        self.materials = {} 
        
    def load_from_file(self, filename, mat_name):
        """ 
        读取 KLA 格式的 txt 文件，并插值到目标波长 
        核心修正：此处采用了 n + ik 的符号约定，以匹配 TMM 算法中的 exp(-iφ)
        """
        # 兼容性：尝试在当前目录或 data 子目录查找
        if not os.path.exists(filename):
            if os.path.exists(os.path.join('data', filename)):
                filename = os.path.join('data', filename)
            else:
                print(f"[警告] 找不到文件: {filename}，{mat_name} 将无法正确加载！")
                return False
            
        try:
            data = []
            with open(filename, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    if parts[0].replace('.', '', 1).isdigit():
                        if len(parts) >= 3:
                            data.append([float(parts[0]), float(parts[1]), float(parts[2])])
            
            data = np.array(data)
            data = data[data[:, 0].argsort()]
            
            f_n = interp1d(data[:,0], data[:,1], kind='linear', fill_value="extrapolate")
            f_k = interp1d(data[:,0], data[:,2], kind='linear', fill_value="extrapolate")
            
            n_interp = f_n(self.target_wl)
            k_interp = f_k(self.target_wl)
            
            # [核心物理修正] N = n + ik
            self.materials[mat_name] = n_interp + 1j * k_interp
            
            return True
        except Exception as e:
            print(f"[错误] 读取 {filename} 失败: {e}")
            return False

    def get_refractive_index(self, name):
        return self.materials.get(name)

# ==============================================================================
# 模块二：7层专用 TMM 仿真引擎
# ==============================================================================
class TMM_Simulator_7Layer:
    def __init__(self):
        """ 初始化仿真参数 """
        self.wavelengths = np.arange(200, 1001, 1.0)
        self.k0_list = 2 * np.pi / self.wavelengths
        
        # 初始化材料加载器
        self.loader = KLA_Material_Loader(self.wavelengths)
        
        # 加载 KLA 文件
        print("-" * 50)
        print("[系统] 正在加载 KLA 材料库...")
        self.has_si = self.loader.load_from_file('.\data\si.txt', 'Si')
        self.has_sio2 = self.loader.load_from_file('.\data\sio2.txt', 'SiO2')
        self.has_si3n4 = self.loader.load_from_file('.\data\si3n4.txt', 'Si3N4')
        
        # 预获取折射率数组 (缓存)
        # 如果文件缺失，提供默认值防止崩溃
        self.n_Air = np.ones_like(self.wavelengths, dtype=np.complex128)
        self.n_SiO2 = self.loader.get_refractive_index('SiO2') if self.has_sio2 else (1.46 + 0j)
        self.n_Si3N4 = self.loader.get_refractive_index('Si3N4') if self.has_si3n4 else (2.02 + 0j)
        self.n_Si = self.loader.get_refractive_index('Si') if self.has_si else (4.0 + 0j)
        print("-" * 50)

    def calculate_spectrum(self, d_list):
        """
        计算 7 层薄膜的反射率光谱
        :param d_list: 包含 7 个厚度的列表 [d1, d2, ..., d7]
                       对应结构: SiO2 / Si3N4 / SiO2 / Si3N4 / SiO2 / Si3N4 / SiO2
        """
        if len(d_list) != 7:
            raise ValueError(f"需要 7 个厚度参数，实际收到 {len(d_list)} 个")

        # 1. 定义膜层结构 (Stack Structure)
        # 结构顺序: Air -> 7层交替 -> Si基底
        layers = [
            (self.n_Air, 0),        # 0. 入射介质
            (self.n_SiO2, d_list[0]),  # 1. SiO2
            (self.n_Si3N4, d_list[1]), # 2. Si3N4
            (self.n_SiO2, d_list[2]),  # 3. SiO2
            (self.n_Si3N4, d_list[3]), # 4. Si3N4
            (self.n_SiO2, d_list[4]),  # 5. SiO2
            (self.n_Si3N4, d_list[5]), # 6. Si3N4
            (self.n_SiO2, d_list[6]),  # 7. SiO2
            (self.n_Si, 0)          # 8. 基底
        ]
        
        # 2. 初始化传输矩阵 M (单位矩阵)
        M00 = np.ones_like(self.wavelengths, dtype=np.complex128)
        M11 = np.ones_like(self.wavelengths, dtype=np.complex128)
        M01 = np.zeros_like(self.wavelengths, dtype=np.complex128)
        M10 = np.zeros_like(self.wavelengths, dtype=np.complex128)
        
        N_curr = layers[0][0]
        
        # 3. 循环计算每一层
        for i in range(1, len(layers)):
            N_next = layers[i][0]
            d_next = layers[i][1]
            
            # --- 界面反射 (Interface) ---
            sum_n = N_curr + N_next
            sub_n = N_curr - N_next
            r = sub_n / sum_n
            inv_t = sum_n / (2 * N_curr)
            
            m00 = (M00 + M01*r) * inv_t
            m01 = (M00*r + M01) * inv_t
            m10 = (M10 + M11*r) * inv_t
            m11 = (M10*r + M11) * inv_t
            M00, M01, M10, M11 = m00, m01, m10, m11
            
            # --- 相位传播 (Propagation) ---
            # 最后一层(基底)不需要传播
            if i < len(layers) - 1:
                phi = self.k0_list * N_next * d_next
                # 注意: 配合 N=n+ik，此处用 exp(-j*phi)
                P = np.exp(-1j * phi)
                Pi = np.exp(1j * phi)
                M00 *= P; M01 *= Pi; M10 *= P; M11 *= Pi
            
            N_curr = N_next
            
        # 4. 提取结果 R = |r|^2
        r_final = M10 / M00
        return np.abs(r_final)**2

# ==========================================
# 模块三：主程序
# ==========================================
def main():
    print("="*60)
    print("      7层 TMM 光学正演模拟器 (KLA 内核)      ")
    print("="*60)

    # 1. 初始化
    start_init = time.time()
    simulator = TMM_Simulator_7Layer()
    print(f"[系统] 初始化完成。")
    
    # 2. 读取 7层 CSV 数据 (用于获取真实的厚度进行测试)
    csv_file = './data/sobol_7layer_16384_chunk_0000.csv'
    if not os.path.exists(csv_file):
        print(f"[错误] 找不到 {csv_file}，请确保文件已上传。")
        return

    print(f"\n[系统] 正在读取 {csv_file} ...")
    df = pd.read_csv(csv_file)
    print(f"[系统] 读取成功，共 {len(df)} 条数据。")

    # 3. 选择测试样本
    try:
        idx_input = input(f"\n>>> 请输入要测试的行号 (0 - {len(df)-1}, 默认0): ").strip()
        target_idx = int(idx_input) if idx_input else 0
    except:
        target_idx = 0

    # 提取该行的 7 个厚度
    # 假设前 7 列是厚度: SiO2_1, Si3N4_1, ..., SiO2_4
    row_data = df.iloc[target_idx]
    target_d = row_data.iloc[:7].values.tolist()
    
    # 提取源光谱 (假设从第 8 列开始是光谱 R_200nm...)
    # CSV通常格式: d1...d7, R_200, R_201...
    source_spectrum = row_data.iloc[7:].values.astype(float)
    
    print(f"\n[测试样本 Index {target_idx}]")
    print(f"结构 (7层): {', '.join([f'{d:.1f}' for d in target_d])} nm")
    
    # 4. 运行仿真
    print("\n[系统] 正在进行 TMM 计算...")
    start_calc = time.time()
    
    spec_sim = simulator.calculate_spectrum(target_d)
    
    print(f"[系统] 计算完成！耗时: {(time.time() - start_calc)*1000:.2f} ms")

    # 5. 误差分析
    mse = np.mean((spec_sim - source_spectrum)**2)
    print(f"\n[结果分析]")
    print(f"MSE 误差: {mse:.8f}")
    if mse < 1e-4:
        print("🎉 完美吻合！7层 KLA 模型验证通过。")
    elif mse < 1e-3:
        print("✅ 高度吻合。")
    else:
        print("⚠️ 存在偏差，请检查材料文件或层序是否对应。")

    # 6. 绘图对比
    print("\n[系统] 正在生成对比图...")
    plt.figure(figsize=(12, 6))
    
    wl = simulator.wavelengths
    plt.plot(wl, source_spectrum, 'k-', linewidth=3, alpha=0.3, label='Benchmark (Source 7-Layer)')
    plt.plot(wl, spec_sim, 'r--', linewidth=1.5, label=f'Simulated (KLA 7-Layer) MSE={mse:.2e}')
    
    plt.title(f'7-Layer Simulation Check\nSample {target_idx}', fontsize=12)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 7. 保存本次结果
    save_name = "sim_result_7layer.csv"
    if input(f"\n是否保存本次仿真数据到 {save_name}? (y/n): ").lower() == 'y':
        header = ['d1','d2','d3','d4','d5','d6','d7'] + [f"R_{int(w)}nm" for w in wl]
        data = target_d + list(spec_sim)
        with open(save_name, 'a', newline='') as f:
            writer = csv.writer(f)
            if not os.path.exists(save_name): writer.writerow(header)
            writer.writerow(data)
        print(f"✅ 已保存。")

if __name__ == "__main__":
    main()