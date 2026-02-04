import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, least_squares
from scipy.interpolate import interp1d
import time
import os

# ==============================================================================
# 模块一：KLA 材料加载器 (物理核心，保持 n+ik 符号约定)
# ==============================================================================
class KLA_Material_Loader:
    def __init__(self, wavelengths):
        self.target_wl = wavelengths
        self.materials = {} 
        
    def load_from_file(self, filename, mat_name):
        """ 读取 KLA 格式 txt 并插值，采用 n + ik 符号约定 """
        if not os.path.exists(filename):
            # 尝试在 data 子目录查找
            alt_path = os.path.join('data', filename)
            if os.path.exists(alt_path):
                filename = alt_path
            else:
                print(f"[警告] 找不到 {filename}，反演可能不准确！")
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
            
            n_val = f_n(self.target_wl)
            k_val = f_k(self.target_wl)
            
            # === 核心修正：N = n + ik (适配 TMM 算法) ===
            self.materials[mat_name] = n_val + 1j * k_val
            return True
            
        except Exception as e:
            print(f"[错误] 读取 {filename} 失败: {e}")
            return False

    def get_refractive_index(self, name):
        return self.materials.get(name)

# ==============================================================================
# 模块二：7层专用 TMM 引擎 (KLA 内核)
# ==============================================================================
class TMM_Engine_7Layer:
    def __init__(self):
        self.wavelengths = np.arange(200, 1001, 1.0)
        self.k0_list = 2 * np.pi / self.wavelengths
        
        # 初始化并加载材料
        self.loader = KLA_Material_Loader(self.wavelengths)
        self.loader.load_from_file('./data/si.txt', 'Si')
        self.loader.load_from_file('./data/sio2.txt', 'SiO2')
        self.loader.load_from_file('./data/si3n4.txt', 'Si3N4')
        
        # 预存折射率数组
        self.n_Air = np.ones_like(self.wavelengths, dtype=np.complex128)
        self.n_SiO2 = self.loader.get_refractive_index('SiO2')
        self.n_Si3N4 = self.loader.get_refractive_index('Si3N4')
        self.n_Si = self.loader.get_refractive_index('Si')
        
        # 兜底默认值
        if self.n_SiO2 is None: self.n_SiO2 = 1.46 + 0j
        if self.n_Si3N4 is None: self.n_Si3N4 = 2.02 + 0j
        if self.n_Si is None: self.n_Si = 4.0 + 0j

    def calculate_spectrum(self, thicknesses):
        """
        计算 7 层薄膜的光谱
        结构: Air | SiO2 | Si3N4 | SiO2 | Si3N4 | SiO2 | Si3N4 | SiO2 | Si
        """
        # 确保输入是 7 个厚度
        if len(thicknesses) != 7:
            raise ValueError("需要 7 个厚度参数")
            
        d = thicknesses # d[0]..d[6]
        
        layers = [
            (self.n_Air, 0),
            (self.n_SiO2, d[0]),   # Layer 1
            (self.n_Si3N4, d[1]),  # Layer 2
            (self.n_SiO2, d[2]),   # Layer 3
            (self.n_Si3N4, d[3]),  # Layer 4
            (self.n_SiO2, d[4]),   # Layer 5
            (self.n_Si3N4, d[5]),  # Layer 6
            (self.n_SiO2, d[6]),   # Layer 7
            (self.n_Si, 0)         # Substrate
        ]
        
        M00 = np.ones_like(self.wavelengths, dtype=np.complex128)
        M11 = np.ones_like(self.wavelengths, dtype=np.complex128)
        M01 = np.zeros_like(self.wavelengths, dtype=np.complex128)
        M10 = np.zeros_like(self.wavelengths, dtype=np.complex128)
        
        N_curr = layers[0][0]
        
        for i in range(1, len(layers)):
            N_next = layers[i][0]
            d_next = layers[i][1]
            
            sum_n = N_curr + N_next
            sub_n = N_curr - N_next
            r = sub_n / sum_n
            inv_t = sum_n / (2 * N_curr)
            
            m00 = (M00 + M01*r) * inv_t
            m01 = (M00*r + M01) * inv_t
            m10 = (M10 + M11*r) * inv_t
            m11 = (M10*r + M11) * inv_t
            M00, M01, M10, M11 = m00, m01, m10, m11
            
            if i < len(layers) - 1:
                # 传播因子：exp(-i * phi)
                phi = self.k0_list * N_next * d_next
                P = np.exp(-1j * phi)
                Pi = np.exp(1j * phi)
                M00 *= P; M01 *= Pi; M10 *= P; M11 *= Pi
            
            N_curr = N_next
            
        return np.abs(M10 / M00)**2

# ==============================================================================
# 模块三：7层混合反演求解器
# ==============================================================================
class IndustrialSolver_7Layer:
    def __init__(self):
        self.engine = TMM_Engine_7Layer()
        # 定义 7 层膜的搜索范围 (Bounds)
        # SiO2 一般较厚 (10-2000nm), Si3N4 一般较薄 (10-500nm)
        # 这里给一个比较宽泛的范围，你可以根据 Sobol 数据的实际生成范围调整
        b_sio2 = (10, 2000)
        b_si3n4 = (10, 500)
        
        self.bounds = [
            b_sio2,   # 1. SiO2
            b_si3n4,  # 2. Si3N4
            b_sio2,   # 3. SiO2
            b_si3n4,  # 4. Si3N4
            b_sio2,   # 5. SiO2
            b_si3n4,  # 6. Si3N4
            b_sio2    # 7. SiO2
        ]
        
    def objective_function(self, params, target_spectrum):
        sim_spectrum = self.engine.calculate_spectrum(params)
        return sim_spectrum - target_spectrum

    def global_objective(self, params, target_spectrum):
        res = self.objective_function(params, target_spectrum)
        return np.sum(res**2)

    def solve(self, target_spectrum):
        print(">>> 启动 7层 KLA 反演算法...")
        start_time = time.time()

        # Phase 1: 差分进化 (Global Search)
        print("    [Phase 1] 全局搜索 (Differential Evolution)...")
        # 变量多了，增加 popsize 和 maxiter
        result_global = differential_evolution(
            self.global_objective, 
            bounds=self.bounds, 
            args=(target_spectrum,),
            strategy='best1bin',
            maxiter=30,      # 7层比较难搜，增加迭代次数
            popsize=20,      # 增加种群密度
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            workers=-1       # 并行计算加速
        )
        x0 = result_global.x
        print(f"    -> 粗略解 (Global): {[f'{x:.1f}' for x in x0]}")

        # Phase 2: 局部精修 (Local Refinement)
        print("    [Phase 2] 局部精修 (Trust Region)...")
        ls_bounds = ([b[0] for b in self.bounds], [b[1] for b in self.bounds])
        
        res_local = least_squares(
            self.objective_function, 
            x0=x0, 
            args=(target_spectrum,),
            bounds=ls_bounds,
            ftol=1e-12, xtol=1e-12, gtol=1e-12,
            loss='soft_l1'
        )
        
        print(f"    -> 最终解 (Local): {[f'{x:.1f}' for x in res_local.x]}")
        print(f"    -> 总耗时: {time.time() - start_time:.2f}s")
        
        return res_local.x, res_local.cost

# ==============================================================================
# 模块四：主程序
# ==============================================================================
def main():
    # 1. 读取 7层 CSV 数据
    filename = './data/sobol_7layer_16384_chunk_0000.csv'
    if not os.path.exists(filename):
        print(f"错误: 找不到 {filename}")
        return

    df = pd.read_csv(filename)
    
    # 2. 选择样本
    try:
        idx_str = input(f"请输入样本行号 (0-{len(df)-1}, 默认0): ").strip()
        target_idx = int(idx_str) if idx_str else 0
    except:
        target_idx = 0

    row = df.iloc[target_idx]
    # 假设前7列是厚度，第8列开始是光谱
    true_thickness = row.iloc[:7].values
    target_spectrum = row.iloc[7:].values.astype(float)
    
    print(f"\n--- 目标样本 (Index {target_idx}) ---")
    print(f"真实厚度: {[f'{x:.1f}' for x in true_thickness]}")
    
    # 3. 反演
    solver = IndustrialSolver_7Layer()
    pred_thickness, final_cost = solver.solve(target_spectrum)
    
    # 4. 结果展示
    print("\n" + "="*60)
    print("             7层膜反演结果对比 (Result)             ")
    print("="*60)
    print(f"{'Layer':<15} | {'True (nm)':<12} | {'Pred (nm)':<12} | {'Error':<10}")
    print("-" * 60)
    
    layer_names = [
        'L1 (SiO2)', 'L2 (Si3N4)', 'L3 (SiO2)', 'L4 (Si3N4)', 
        'L5 (SiO2)', 'L6 (Si3N4)', 'L7 (SiO2)'
    ]
    
    for i, name in enumerate(layer_names):
        t_val = true_thickness[i]
        p_val = pred_thickness[i]
        err = abs(t_val - p_val)
        print(f"{name:<15} | {t_val:<12.2f} | {p_val:<12.2f} | {err:<10.2f}")
    print("-" * 60)
    print(f"Final MSE Cost: {final_cost:.2e}")

    # 5. 绘图
    engine = TMM_Engine_7Layer()
    fitted_spectrum = engine.calculate_spectrum(pred_thickness)
    wl = engine.wavelengths
    
    plt.figure(figsize=(15, 6))

    # 左图: 光谱
    plt.subplot(1, 2, 1)
    plt.plot(wl, target_spectrum, 'k-', linewidth=3, alpha=0.4, label='Ground Truth')
    plt.plot(wl, fitted_spectrum, 'r--', linewidth=1.5, label='Inversion Result')
    plt.title(f'7-Layer Spectrum Fit (MSE: {final_cost:.2e})', fontsize=12)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 右图: 误差棒图
    plt.subplot(1, 2, 2)
    x = np.arange(7)
    width = 0.35
    
    plt.bar(x - width/2, true_thickness, width, label='True', color='gray', alpha=0.6)
    plt.bar(x + width/2, pred_thickness, width, label='Pred', color='#1f77b4', alpha=0.9)
    
    # 标注误差
    abs_err = np.abs(true_thickness - pred_thickness)
    max_h = max(true_thickness.max(), pred_thickness.max())
    
    for i in range(7):
        plt.text(x[i], max(true_thickness[i], pred_thickness[i]) + max_h*0.02, 
                 f'{abs_err[i]:.1f}', 
                 ha='center', va='bottom', color='red', fontsize=9, fontweight='bold')

    plt.title('Thickness Prediction Error (nm)', fontsize=12)
    plt.xticks(x, [f'L{i+1}' for i in range(7)])
    plt.ylabel('Thickness (nm)')
    plt.legend()
    plt.ylim(0, max_h * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()