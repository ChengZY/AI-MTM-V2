import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, least_squares
from scipy.interpolate import interp1d
import time
import os

# ==============================================================================
# 模块一：KLA 材料加载器 (物理核心)
# ==============================================================================
class KLA_Material_Loader:
    def __init__(self, wavelengths):
        self.target_wl = wavelengths
        self.materials = {} 
        
    def load_from_file(self, filename, mat_name):
        """ 读取 KLA 格式 txt 并插值，采用 n + ik 符号约定 """
        # 兼容性检查：尝试当前目录和 data 目录
        if not os.path.exists(filename):
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
            
            # === 核心修正：N = n + ik ===
            self.materials[mat_name] = n_val + 1j * k_val
            return True
            
        except Exception as e:
            print(f"[错误] 读取 {filename} 失败: {e}")
            return False

    def get_refractive_index(self, name):
        return self.materials.get(name)

# ==============================================================================
# 模块二：基于 KLA 的 TMM 引擎 (供反演算法调用)
# ==============================================================================
class TMM_Engine_KLA:
    def __init__(self):
        self.wavelengths = np.arange(200, 1001, 1.0)
        self.k0_list = 2 * np.pi / self.wavelengths
        
        # 初始化并加载材料
        self.loader = KLA_Material_Loader(self.wavelengths)
        self.loader.load_from_file('.\data\si.txt', 'Si')
        self.loader.load_from_file('.\data\sio2.txt', 'SiO2')
        self.loader.load_from_file('.\data\si3n4.txt', 'Si3N4')
        
        # 预存折射率数组以加速计算
        self.n_Air = np.ones_like(self.wavelengths, dtype=np.complex128)
        self.n_SiO2 = self.loader.get_refractive_index('SiO2')
        self.n_Si3N4 = self.loader.get_refractive_index('Si3N4')
        self.n_Si = self.loader.get_refractive_index('Si')
        
        # 兜底：如果文件没加载到，给个默认值防止报错
        if self.n_SiO2 is None: self.n_SiO2 = 1.46 + 0j
        if self.n_Si3N4 is None: self.n_Si3N4 = 2.02 + 0j
        if self.n_Si is None: self.n_Si = 4.0 + 0j

    def calculate_spectrum(self, thicknesses):
        """
        计算光谱 (被 Solver 调用)
        :param thicknesses: [d1, d2, d3]
        """
        d1, d2, d3 = thicknesses
        
        layers = [
            (self.n_Air, 0),
            (self.n_SiO2, d1),
            (self.n_Si3N4, d2),
            (self.n_SiO2, d3),
            (self.n_Si, 0)
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
# 模块三：混合反演求解器 (Global + Local)
# ==============================================================================
class IndustrialSolver:
    def __init__(self):
        # 使用新的 KLA 引擎
        self.engine = TMM_Engine_KLA()
        # 搜索范围 (nm): [d1范围, d2范围, d3范围]
        self.bounds = [(500, 1500), (50, 200), (500, 1500)] 
        
    def objective_function(self, params, target_spectrum):
        """ 残差函数 (Least Squares 用) """
        sim_spectrum = self.engine.calculate_spectrum(params)
        return sim_spectrum - target_spectrum

    def global_objective(self, params, target_spectrum):
        """ 标量损失函数 (差分进化用) """
        res = self.objective_function(params, target_spectrum)
        return np.sum(res**2)

    def solve(self, target_spectrum):
        print(">>> 启动 KLA 物理模型反演...")
        start_time = time.time()

        # Phase 1: 差分进化 (全局粗搜)
        print("    [Phase 1] 全局搜索 (Differential Evolution)...")
        result_global = differential_evolution(
            self.global_objective, 
            bounds=self.bounds, 
            args=(target_spectrum,),
            strategy='best1bin',
            maxiter=20,     # 可根据需要调大
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7
        )
        x0 = result_global.x
        print(f"    -> 粗略解: {x0}")

        # Phase 2: 最小二乘法 (局部精修)
        print("    [Phase 2] 局部精修 (Trust Region Reflective)...")
        ls_bounds = ([b[0] for b in self.bounds], [b[1] for b in self.bounds])
        
        res_local = least_squares(
            self.objective_function, 
            x0=x0, 
            args=(target_spectrum,),
            bounds=ls_bounds,
            ftol=1e-12, 
            xtol=1e-12, 
            gtol=1e-12,
            loss='soft_l1'
        )
        
        print(f"    -> 最终解: {res_local.x}")
        print(f"    -> 总耗时: {time.time() - start_time:.2f}s")
        
        return res_local.x, res_local.cost

# ==============================================================================
# 模块四：主程序 (测试与绘图)
# ==============================================================================
def main():
    # 1. 读取源数据
    filename = '.\data/tmm3_smooth_1000_chunk_0000.csv'
    if not os.path.exists(filename):
        print(f"错误: 找不到 {filename}，请先生成或下载数据。")
        return

    df = pd.read_csv(filename)
    
    # 2. 交互式选择样本
    try:
        idx_str = input(f"请输入要反演的样本行号 (0-{len(df)-1}, 默认0): ").strip()
        target_row_idx = int(idx_str) if idx_str else 0
    except:
        target_row_idx = 0

    row_data = df.iloc[target_row_idx]
    true_thickness = row_data.iloc[:3].values
    target_spectrum = row_data.iloc[3:].values.astype(float)
    
    print(f"\n--- 目标样本 (Index {target_row_idx}) ---")
    print(f"真实厚度: {true_thickness}")
    
    # 3. 执行反演
    solver = IndustrialSolver()
    pred_thickness, final_cost = solver.solve(target_spectrum)
    
    # 4. 打印结果表
    print("\n" + "="*45)
    print("           反演结果 (Inversion Result)           ")
    print("="*45)
    print(f"{'Layer':<10} | {'True (nm)':<12} | {'Pred (nm)':<12} | {'Error':<10}")
    print("-" * 55)
    layers = ['SiO2_1', 'Si3N4', 'SiO2_2']
    for i, name in enumerate(layers):
        err = abs(pred_thickness[i] - true_thickness[i])
        print(f"{name:<10} | {true_thickness[i]:<12.4f} | {pred_thickness[i]:<12.4f} | {err:<10.4f}")
    print("-" * 55)
    print(f"Final Cost (MSE): {final_cost:.2e}")

    # 5. 绘图验证
    engine = TMM_Engine_KLA()
    fitted_spectrum = engine.calculate_spectrum(pred_thickness)
    wavelengths = engine.wavelengths
    abs_error = np.abs(pred_thickness - true_thickness)
    
    plt.figure(figsize=(14, 6))

    # 左图: 光谱拟合
    plt.subplot(1, 2, 1)
    plt.plot(wavelengths, target_spectrum, 'k-', linewidth=3, alpha=0.4, label='Ground Truth (Measured)')
    plt.plot(wavelengths, fitted_spectrum, 'r--', linewidth=1.5, label='Fitted (KLA Inversion)')
    plt.title(f'Spectrum Fitting (MSE: {final_cost:.2e})', fontsize=12, fontweight='bold')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 右图: 厚度误差
    plt.subplot(1, 2, 2)
    labels = ['SiO2 (Top)', 'Si3N4', 'SiO2 (Bot)']
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = plt.bar(x - width/2, true_thickness, width, label='True', color='gray', alpha=0.6)
    rects2 = plt.bar(x + width/2, pred_thickness, width, label='Pred', color='#1f77b4', alpha=0.9)
    
    # 标注误差
    max_h = max(true_thickness.max(), pred_thickness.max())
    for i in range(len(labels)):
        plt.text(x[i], max(true_thickness[i], pred_thickness[i]) + max_h*0.02, 
                 f'Err:\n{abs_error[i]:.2f}', 
                 ha='center', va='bottom', color='red', fontweight='bold')

    plt.title('Thickness Prediction Accuracy', fontsize=12, fontweight='bold')
    plt.ylabel('Thickness (nm)')
    plt.xticks(x, labels)
    plt.ylim(0, max_h * 1.15)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()