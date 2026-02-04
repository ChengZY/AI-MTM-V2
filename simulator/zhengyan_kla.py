import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import time
from scipy.interpolate import interp1d

# ==============================================================================
# 模块一：KLA 材料加载器 (负责读取 txt 数据并进行物理插值)
# ==============================================================================
class KLA_Material_Loader:
    def __init__(self, wavelengths):
        """
        初始化加载器
        :param wavelengths: 我们仿真需要的目标波长数组 (例如 200-1000nm)
        """
        self.target_wl = wavelengths
        self.materials = {} # 用于存储加载好的折射率数据 (name -> complex_index)
        
    def load_from_file(self, filename, mat_name):
        """ 
        读取 KLA 格式的 txt 文件，并插值到目标波长 
        核心修正：此处采用了 n + ik 的符号约定，以匹配 TMM 算法中的 exp(-iφ)
        """
        # 1. 检查文件是否存在
        if not os.path.exists(filename):
            print(f"[警告] 找不到文件: {filename}，{mat_name} 将无法正确加载！")
            return False
            
        try:
            data = []
            print(f"[系统] 正在读取材料文件: {filename} ...")
            
            # 2. 逐行读取并解析数据
            with open(filename, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    # 检查第一列是否为数字 (跳过标题行)
                    if parts[0].replace('.', '', 1).isdigit():
                        if len(parts) >= 3:
                            # 格式通常为: 波长(nm)  n  k
                            data.append([float(parts[0]), float(parts[1]), float(parts[2])])
            
            # 转为 numpy 数组并按波长排序 (防止数据乱序导致插值错误)
            data = np.array(data)
            data = data[data[:, 0].argsort()]
            
            # 3. 创建插值函数 (Linear Interpolation)
            # fill_value="extrapolate" 允许处理略微超出范围的数据
            f_n = interp1d(data[:,0], data[:,1], kind='linear', fill_value="extrapolate")
            f_k = interp1d(data[:,0], data[:,2], kind='linear', fill_value="extrapolate")
            
            # 4. 映射到我们仿真需要的 801 个波长点
            n_interp = f_n(self.target_wl)
            k_interp = f_k(self.target_wl)
            
            # ======================================================
            # [核心物理修正] 符号约定 (Sign Convention)
            # 之前测试发现，配合 exp(-iwt) 的 TMM 算法，
            # 这里的 k 必须取正号进入复数 (N = n + ik)，
            # 否则会导致吸收变成增益（能量爆炸）。
            # ======================================================
            self.materials[mat_name] = n_interp + 1j * k_interp
            
            print(f"[成功] {mat_name} 加载完毕。")
            return True
            
        except Exception as e:
            print(f"[错误] 读取 {filename} 失败: {e}")
            return False

    def get_refractive_index(self, name):
        """ 获取指定材料的复折射率数组 """
        return self.materials.get(name)

# ==============================================================================
# 模块二：传输矩阵法 (TMM) 仿真引擎
# ==============================================================================
class TMM_Simulator:
    def __init__(self):
        """ 初始化仿真参数 """
        # 波长范围: 200nm 到 1000nm，步长 1nm
        self.wavelengths = np.arange(200, 1001, 1.0)
        # 真空波矢量 k0 = 2*pi / lambda
        self.k0_list = 2 * np.pi / self.wavelengths
        
        # 初始化材料加载器
        self.loader = KLA_Material_Loader(self.wavelengths)
        
        # 预加载所有 KLA 文件 (Si, SiO2, Si3N4)
        print("-" * 50)
        self.has_si = self.loader.load_from_file('./data/si.txt', 'Si')
        self.has_sio2 = self.loader.load_from_file('./data/sio2.txt', 'SiO2')
        self.has_si3n4 = self.loader.load_from_file('./data/si3n4.txt', 'Si3N4')
        print("-" * 50)

    def calculate_spectrum(self, d1, d2, d3):
        """
        核心正演函数：计算三层薄膜的反射率光谱
        :param d1: 顶层 SiO2 厚度 (nm)
        :param d2: 中间 Si3N4 厚度 (nm)
        :param d3: 底层 SiO2 厚度 (nm)
        :return: 反射率数组 (R)
        """
        
        # 1. 准备材料折射率 N (Complex Refractive Index)
        # 空气折射率默认为 1
        n_Air = np.ones_like(self.wavelengths, dtype=np.complex128)
        
        # 获取材料数据 (如果加载失败，给一个默认值防止程序崩溃)
        n_SiO2 = self.loader.get_refractive_index('SiO2') if self.has_sio2 else (1.46 + 0j)
        n_Si3N4 = self.loader.get_refractive_index('Si3N4') if self.has_si3n4 else (2.02 + 0j)
        n_Si = self.loader.get_refractive_index('Si') if self.has_si else (4.0 + 0j)

        # 2. 定义膜层结构 (Stack Structure)
        # 结构顺序: 空气 -> SiO2 -> Si3N4 -> SiO2 -> Si基底
        # 格式: (折射率, 厚度)
        layers = [
            (n_Air, 0),      # 层0: 入射介质 (厚度无所谓)
            (n_SiO2, d1),    # 层1: Top Oxide
            (n_Si3N4, d2),   # 层2: Nitride
            (n_SiO2, d3),    # 层3: Bottom Oxide
            (n_Si, 0)        # 层4: Substrate (视为半无限厚)
        ]
        
        # 3. 初始化传输矩阵 (Transfer Matrix)
        # M = [M00 M01]
        #     [M10 M11]
        # 初始矩阵为单位矩阵 I
        M00 = np.ones_like(self.wavelengths, dtype=np.complex128)
        M11 = np.ones_like(self.wavelengths, dtype=np.complex128)
        M01 = np.zeros_like(self.wavelengths, dtype=np.complex128)
        M10 = np.zeros_like(self.wavelengths, dtype=np.complex128)
        
        # 当前层折射率 (从空气开始)
        N_curr = layers[0][0]
        
        # 4. 循环计算每一层 (The Loop)
        for i in range(1, len(layers)):
            N_next = layers[i][0] # 下一层折射率
            d_next = layers[i][1] # 下一层厚度
            
            # --- 步骤 A: 计算界面反射与透射 (Interface) ---
            # 菲涅尔反射系数 r = (n1 - n2) / (n1 + n2)
            sum_n = N_curr + N_next
            sub_n = N_curr - N_next
            r = sub_n / sum_n
            # 透射系数倒数 1/t = (n1 + n2) / 2n1
            inv_t = sum_n / (2 * N_curr)
            
            # 更新全局矩阵 M_new = M_old * M_interface
            m00 = (M00 + M01*r) * inv_t
            m01 = (M00*r + M01) * inv_t
            m10 = (M10 + M11*r) * inv_t
            m11 = (M10*r + M11) * inv_t
            M00, M01, M10, M11 = m00, m01, m10, m11
            
            # --- 步骤 B: 计算相位传播 (Propagation) ---
            # 最后一层(基底)不需要再传播，光被吸收或透射走了
            if i < len(layers) - 1:
                # 相位延迟 phi = k0 * n * d
                phi = self.k0_list * N_next * d_next
                # 传播因子 P = exp(-i * phi)
                P = np.exp(-1j * phi)   # 正向波传播
                Pi = np.exp(1j * phi)   # 反向波传播
                
                # 更新全局矩阵 M_new = M_old * M_propagation
                M00 *= P
                M01 *= Pi
                M10 *= P
                M11 *= Pi
            
            # 更新当前折射率，准备进入下一层循环
            N_curr = N_next
            
        # 5. 提取最终结果
        # 总反射系数 r_total = M10 / M00
        r_final = M10 / M00
        # 反射率 R = |r|^2
        R = np.abs(r_final)**2
        
        return R

# ==========================================
# 模块三：主程序 (交互、绘图与保存)
# ==========================================
def main():
    print("="*60)
    print("      TMM 光学正演模拟器 (KLA 物理模型版)      ")
    print("="*60)
    print("说明: 本程序基于 KLA 官方折射率数据 (SiO2/Si3N4/Si) 进行仿真。")
    print("      支持自定义厚度，并与原始 CSV 数据进行自动对比。")
    print("-" * 60)

    # =========================================================
    # 1. 初始化仿真引擎
    # =========================================================
    start_init = time.time()
    simulator = TMM_Simulator()
    print(f"[系统] 初始化耗时: {time.time()-start_init:.4f} 秒")
    
    # =========================================================
    # 2. 获取用户输入
    # =========================================================
    print("\n>>> 请输入薄膜物理厚度 (单位: nm, 直接回车使用默认值)")
    
    # try:
    #     d1_in = input("    1. 顶层 SiO2 厚度 (默认 1000): ").strip()
    #     d1 = float(d1_in) if d1_in else 1000.0
    #
    #     d2_in = input("    2. 中间 Si3N4 厚度 (默认 100): ").strip()
    #     d2 = float(d2_in) if d2_in else 100.0
    #
    #     d3_in = input("    3. 底层 SiO2 厚度 (默认 1000): ").strip()
    #     d3 = float(d3_in) if d3_in else 1000.0
    #
    # except ValueError:
    #     print("[错误] 输入无效，将使用默认参数 (1000/100/1000)。")
    #     d1, d2, d3 = 1000.0, 100.0, 1000.0

    d1, d2, d3 = 1000.0, 100.0, 1000.0
    print(f"\n[系统] 当前仿真结构: Air | SiO2({d1}nm) | Si3N4({d2}nm) | SiO2({d3}nm) | Si")
    
    # =========================================================
    # 3. 运行 KLA 仿真计算
    # =========================================================
    print("[系统] 正在进行 TMM 矩阵运算...")
    start_calc = time.time()
    spectrum_sim = simulator.calculate_spectrum(d1, d2, d3)
    end_calc = time.time()
    print(f"[系统] 计算完成！耗时: {(end_calc - start_calc)*1000:.2f} ms")

    # =========================================================
    # 4. 加载源数据并寻找最接近的参考光谱
    # =========================================================
    source_file = '/home/zhongyao/dl/data/IPSR-AI-MTM/tmm3_smooth_1000_chunk_0000.csv'
    spectrum_ref = None
    found_d = None
    
    if os.path.exists(source_file):
        print(f"\n[系统] 正在从 {source_file} 中寻找参考光谱...")
        try:
            df = pd.read_csv(source_file)
            # 假设前3列是厚度
            thickness_data = df.iloc[:, :3].values
            target_d = np.array([d1, d2, d3])
            
            # 计算欧氏距离寻找最邻近
            distances = np.linalg.norm(thickness_data - target_d, axis=1)
            best_idx = np.argmin(distances)
            min_dist = distances[best_idx]
            
            found_row = df.iloc[best_idx]
            found_d = found_row.iloc[:3].values
            # 假设光谱数据从第4列开始 (index 3:)
            spectrum_ref = found_row.iloc[3:].values.astype(float)
            
            print(f"    -> 找到最接近厚度: {found_d} (Index {best_idx})")
            print(f"    -> 厚度差异: {min_dist:.2f} nm")
            
        except Exception as e:
            print(f"[警告] 源数据读取失败: {e}")
            spectrum_ref = None
    else:
        print(f"[提示] 未找到 {source_file}，将不显示参考光谱。")

    # =========================================================
    # 5. 保存数据
    # =========================================================
    csv_filename = f"simulation_result_kla_{d1}_{d2}_{d3}.csv"
    print(f"\n>>> 正在保存本次仿真结果至 {csv_filename} ...")
    
    header = ['d1', 'd2', 'd3'] + [f"R_{int(wl)}nm" for wl in simulator.wavelengths]
    row_data = [d1, d2, d3] + list(spectrum_sim)
    
    try:
        file_exists = os.path.exists(csv_filename)
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row_data)
        print(f"✅ 保存成功！文件路径: {os.path.abspath(csv_filename)}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")

    # =========================================================
    # 6. 绘图展示 (双线对比版)
    # =========================================================
    print("\n[系统] 正在生成光谱对比图...")
    plt.figure(figsize=(12, 6))
    
    # 1. 绘制源数据 (如果存在)
    mse_label = ""
    if spectrum_ref is not None:
        plt.plot(simulator.wavelengths, spectrum_ref, 'k-', linewidth=3, alpha=0.3, label='Benchmark (Source Data)')
        # 计算 MSE 误差
        mse = np.mean((spectrum_sim - spectrum_ref)**2)
        mse_label = f" (MSE={mse:.2e})"
        print(f"[分析] 仿真 vs 源数据 MSE: {mse:.6f}")
        if mse < 1e-4:
            print("    -> 结果完美吻合！")
    
    # 2. 绘制本次仿真数据
    plt.plot(simulator.wavelengths, spectrum_sim, 'r-', linewidth=1.5, label=f'Simulated (KLA Model){mse_label}')
    
    # 图表装饰
    title_str = f'KLA Model vs Benchmark\nInput: {d1:.0f}/{d2:.0f}/{d3:.0f} nm'
    if found_d is not None:
        title_str += f' (Ref: {found_d[0]:.0f}/{found_d[1]:.0f}/{found_d[2]:.0f})'
        
    plt.title(title_str, fontsize=12, fontweight='bold')
    plt.xlabel('Wavelength (nm)', fontsize=10)
    plt.ylabel('Reflectance', fontsize=10)
    plt.xlim(200, 1000)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    print("[系统] 程序结束。")

if __name__ == "__main__":
    main()