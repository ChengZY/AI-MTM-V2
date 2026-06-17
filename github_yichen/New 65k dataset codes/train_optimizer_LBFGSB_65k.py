"""
train_optimizer_LBFGSB_65k.py

Single-start L-BFGS-B optimizer for 65k dataset
CNN-LSTM-Attention, 200-1000nm, 1nm resolution
"""

import pandas as pd
import numpy as np
import argparse
import os
import time
from scipy.optimize import minimize
from scipy.interpolate import interp1d


# ==============================================================================
# TMM Simulator
# ==============================================================================
class KLA_Material_Loader:
    def __init__(self, wavelengths):
        self.target_wl = wavelengths
        self.materials = {}

    def load_from_file(self, filename, mat_name):
        if not os.path.exists(filename):
            print(f"[WARNING] {filename} not found — using default for {mat_name}")
            return False
        try:
            data = []
            with open(filename, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    if parts[0].replace('.', '', 1).isdigit() and len(parts) >= 3:
                        data.append([float(p) for p in parts[:3]])
            data = np.array(data); data = data[data[:, 0].argsort()]
            fn = interp1d(data[:, 0], data[:, 1], kind='linear', fill_value='extrapolate')
            fk = interp1d(data[:, 0], data[:, 2], kind='linear', fill_value='extrapolate')
            self.materials[mat_name] = fn(self.target_wl) + 1j * fk(self.target_wl)
            return True
        except Exception as e:
            print(f"[ERROR] {filename}: {e}"); return False

    def get(self, name): return self.materials.get(name)


class TMM_Numpy:
    def __init__(self, data_dir):
        self.wl  = np.arange(200, 1001, 1.0)
        self.k0  = 2 * np.pi / self.wl
        loader   = KLA_Material_Loader(self.wl)
        print("Loading TMM material files...")
        loader.load_from_file(os.path.join(data_dir, 'si.txt'),    'Si')
        loader.load_from_file(os.path.join(data_dir, 'sio2.txt'),  'SiO2')
        loader.load_from_file(os.path.join(data_dir, 'si3n4.txt'), 'Si3N4')

        n = loader.get('SiO2')
        self.n_sio2  = n if n is not None else np.full_like(self.wl, 1.46+0j, dtype=np.complex128)
        n = loader.get('Si3N4')
        self.n_si3n4 = n if n is not None else np.full_like(self.wl, 2.02+0j, dtype=np.complex128)
        n = loader.get('Si')
        self.n_si    = n if n is not None else np.full_like(self.wl, 4.00+0j, dtype=np.complex128)
        self.n_air   = np.ones_like(self.wl, dtype=np.complex128)
        print("TMM materials loaded.")

    def spectrum(self, d):
        layers = [
            (self.n_air,   0),
            (self.n_sio2,  d[0]), (self.n_si3n4, d[1]),
            (self.n_sio2,  d[2]), (self.n_si3n4, d[3]),
            (self.n_sio2,  d[4]), (self.n_si3n4, d[5]),
            (self.n_sio2,  d[6]), (self.n_si,    0),
        ]
        M00 = np.ones_like(self.wl,  dtype=np.complex128)
        M11 = np.ones_like(self.wl,  dtype=np.complex128)
        M01 = np.zeros_like(self.wl, dtype=np.complex128)
        M10 = np.zeros_like(self.wl, dtype=np.complex128)
        Nc  = layers[0][0]
        for i in range(1, len(layers)):
            Nn, dn = layers[i]
            r  = (Nc - Nn) / (Nc + Nn)
            it = (Nc + Nn) / (2 * Nc)
            m00=(M00+M01*r)*it; m01=(M00*r+M01)*it
            m10=(M10+M11*r)*it; m11=(M10*r+M11)*it
            M00,M01,M10,M11 = m00,m01,m10,m11
            if i < len(layers)-1:
                phi = self.k0 * Nn * dn
                M00*=np.exp(-1j*phi); M01*=np.exp(1j*phi)
                M10*=np.exp(-1j*phi); M11*=np.exp(1j*phi)
            Nc = Nn
        return np.abs(M10/M00)**2


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='Single-start L-BFGS-B optimizer — 65k dataset')
    parser.add_argument('--pred_csv',     type=str,
                        default='/home/yichenstu/projects/AI_MTM/AI-MTM-main/DL/'
                                'results_cnn_lstm_attention_65k/CNN_LSTM_Attention/'
                                'hidden256_cnnch128_stride4_lr0.001/pred_200_1000_1nm.csv')
    parser.add_argument('--file_path',    type=str,
                        default='~/projects/dataset_65k_sio2_100_2000_si3n4_100_1000/')
    parser.add_argument('--data_dir',     type=str, default='../simulator/data')
    parser.add_argument('--input_layers', type=int, default=7)
    parser.add_argument('--method',       type=str, default='L-BFGS-B')
    parser.add_argument('--max_iter',     type=int, default=500)
    parser.add_argument('--tol',          type=float, default=1e-12)
    args = parser.parse_args()

    # ── Load pred CSV ──────────────────────────────────────────────────────────
    pred_path = os.path.expanduser(args.pred_csv)
    print(f"Loading DL predictions from: {pred_path}")
    df_pred = pd.read_csv(pred_path)

    layer_names = ['thk_SiO2_1', 'thk_Si3N4_1', 'thk_SiO2_2',
                   'thk_Si3N4_2', 'thk_SiO2_3',  'thk_Si3N4_3', 'thk_SiO2_4']
    gt_cols   = [f"{n}_gt"   for n in layer_names]
    pred_cols = [f"{n}_pred" for n in layer_names]

    Y_true   = df_pred[gt_cols].values.astype(float)
    dl_preds = df_pred[pred_cols].values.astype(float)
    n_samples = len(Y_true)

    # Verify DL MAPE
    dl_mape_per_layer = np.mean(np.abs((dl_preds - Y_true) / Y_true), axis=0)
    print(f"\nDL MAPE verification:")
    for i, name in enumerate(layer_names):
        print(f"  {name}: {dl_mape_per_layer[i]:.12f}")
    print(f"  Average: {dl_mape_per_layer.mean():.12f}")

    # ── Load actual spectra — full range 200-1000nm 1nm ───────────────────────
    file_path = os.path.expanduser(args.file_path)
    print(f"\nLoading spectra from: {file_path}")
    dfs = [pd.read_csv(os.path.join(file_path, f"sobol_7layer_65536_chunk_000{i}.csv"))
           for i in range(7)]
    df    = pd.concat(dfs, axis=0, ignore_index=True)
    X_raw = df.iloc[:, 7:].values.astype(float)   # all 801 points
    print(f"Spectra shape: {X_raw.shape}")

    assert len(X_raw) == n_samples, \
        f"Mismatch: pred CSV has {n_samples} rows but data has {len(X_raw)} rows"

    # ── Thickness bounds ───────────────────────────────────────────────────────
    Y_raw  = df.iloc[:, :args.input_layers].values.astype(float)
    bounds = [(Y_raw[:, i].min() * 0.8, Y_raw[:, i].max() * 1.2)
              for i in range(args.input_layers)]
    print(f"Bounds: {[(f'{b[0]:.0f}', f'{b[1]:.0f}') for b in bounds]}")

    # ── TMM simulator ─────────────────────────────────────────────────────────
    tmm = TMM_Numpy(data_dir=os.path.expanduser(args.data_dir))

    # ── Run single-start optimizer ─────────────────────────────────────────────
    print(f"\nRunning {args.method} single-start optimizer on {n_samples} samples...")
    print(f"Max iterations: {args.max_iter}, Tolerance: {args.tol}\n")

    opt_preds   = np.zeros_like(dl_preds)
    n_converged = 0
    t0 = time.time()

    for i in range(n_samples):
        target_spectrum = X_raw[i]
        dl_init         = dl_preds[i]

        def objective(d):
            d_clipped     = np.clip(d, 1.0, None)
            pred_spectrum = tmm.spectrum(d_clipped)   # full range 801 points
            return np.mean((pred_spectrum - target_spectrum) ** 2)

        result = minimize(
            objective,
            x0      = dl_init,
            method  = args.method,
            bounds  = bounds,
            options = {'maxiter': args.max_iter, 'ftol': args.tol}
        )

        opt_preds[i] = result.x
        if result.success or result.fun < 1e-6:
            n_converged += 1

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n_samples} done | {elapsed:.0f}s elapsed | "
                  f"avg {elapsed/(i+1)*1000:.0f}ms/sample")

    t_total = time.time() - t0

    # ── Final evaluation ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)
    print(f"{'Layer':<16} {'DL MAPE':>16} {'DL+Opt MAPE':>18} {'Improvement':>12}")
    print("-" * 65)

    dl_mapes = []; opt_mapes = []
    for i, name in enumerate(layer_names):
        dl_m  = np.mean(np.abs((dl_preds[:, i]  - Y_true[:, i]) / Y_true[:, i]))
        opt_m = np.mean(np.abs((opt_preds[:, i] - Y_true[:, i]) / Y_true[:, i]))
        impr  = (dl_m - opt_m) / dl_m * 100
        dl_mapes.append(dl_m); opt_mapes.append(opt_m)
        print(f"{name:<16} {dl_m:>16.12f} {opt_m:>18.12f} {impr:>+11.2f}%")

    print("-" * 65)
    avg_dl  = np.mean(dl_mapes)
    avg_opt = np.mean(opt_mapes)
    avg_imp = (avg_dl - avg_opt) / avg_dl * 100
    print(f"{'AVERAGE':<16} {avg_dl:>16.12f} {avg_opt:>18.12f} {avg_imp:>+11.2f}%")
    print("=" * 65)
    print(f"\nTotal optimizer time:          {t_total:.1f}s")
    print(f"Avg optimizer time per sample: {t_total/n_samples*1000:.1f}ms")
    print(f"Converged: {n_converged}/{n_samples}")

    # ── Save results ───────────────────────────────────────────────────────────
    out_dir = os.path.dirname(pred_path)
    columns = ([f"{n}_gt"    for n in layer_names] +
               [f"{n}_dl"    for n in layer_names] +
               [f"{n}_dlopt" for n in layer_names])
    df_out  = pd.DataFrame(
        np.concatenate([Y_true, dl_preds, opt_preds], axis=1),
        columns=columns
    )
    out_path = os.path.join(out_dir, 'pred_dl_plus_optimizer_singlestart_65k.csv')
    df_out.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
