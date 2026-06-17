"""
run_optimizer_multistart_parallel.py

Two-stage inference pipeline for CNN-LSTM-Attention (200-1000nm, 1nm) — 65k dataset
MULTI-START + PARALLEL version using joblib.

Key improvement over sequential multistart:
- All 5 starts per sample run in parallel across CPU cores
- Samples themselves also processed in parallel batches
- With 24 cores: ~20x faster than sequential multistart

Usage:
    python run_optimizer_multistart_parallel.py
"""

import pandas as pd
import numpy as np
import argparse
import os
import time
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from joblib import Parallel, delayed


# ==============================================================================
# TMM Simulator (numpy — used inside scipy optimizer)
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
    """Numpy TMM simulator — full range 200-1000nm (801 points)."""
    def __init__(self, data_dir):
        self.wl  = np.arange(200, 1001, 1.0)
        self.k0  = 2 * np.pi / self.wl
        loader   = KLA_Material_Loader(self.wl)
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
# Single-start optimizer — called in parallel for each start point
# ==============================================================================
def run_single_start(x0, target_spectrum, bounds, method, max_iter, tol,
                     data_dir):
    """
    Runs one optimizer start. Defined at module level so joblib can pickle it.
    Creates its own TMM instance since each worker process needs its own copy.
    """
    tmm = TMM_Numpy(data_dir=data_dir)

    def objective(d):
        d_clipped     = np.clip(d, 1.0, None)
        pred_spectrum = tmm.spectrum(d_clipped)
        return np.mean((pred_spectrum - target_spectrum) ** 2)

    result = minimize(
        objective,
        x0      = x0,
        method  = method,
        bounds  = bounds,
        options = {'maxiter': max_iter, 'ftol': tol}
    )
    return result


# ==============================================================================
# Process one sample — runs all starts in parallel
# ==============================================================================
def process_sample(i, dl_init, target_spectrum, bounds, n_starts, method,
                   max_iter, tol, data_dir, seed):
    """Process one sample with parallel multi-start optimization."""
    rng = np.random.default_rng(seed + i)   # unique seed per sample

    # Generate starting points
    starts = [dl_init]   # Start 1: DL prediction
    for _ in range(n_starts - 1):
        noise        = rng.normal(0, 0.02, size=7)
        x0_perturbed = dl_init * (1 + noise)
        x0_perturbed = np.array([
            np.clip(x0_perturbed[j], bounds[j][0], bounds[j][1])
            for j in range(7)
        ])
        starts.append(x0_perturbed)

    # Run all starts in parallel
    results = Parallel(n_jobs=n_starts, prefer='threads')(
        delayed(run_single_start)(
            x0, target_spectrum, bounds, method, max_iter, tol, data_dir
        )
        for x0 in starts
    )

    # Pick best result
    best_result = min(results, key=lambda r: r.fun)
    converged   = best_result.success or best_result.fun < 1e-6

    return i, best_result.x, converged


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Multistart optimizer — parallel version using joblib'
    )
# -- full range, 1nm res ---------------------------------------------------------------------------------------------------------
    parser.add_argument('--pred_csv',    type=str,
                        default='/home/yichenstu/projects/AI_MTM/AI-MTM-main/DL/results_cnn_lstm_attention_65k/'
                                'CNN_LSTM_Attention/hidden256_cnnch128_stride4_lr0.001/pred_200_1000_1nm.csv',
                        help='Path to pred CSV from 65k CNN-LSTM-Attention training')
    parser.add_argument('--file_path',    type=str,
                        default='~/projects/dataset_65k_sio2_100_2000_si3n4_100_1000/')
    parser.add_argument('--data_dir',     type=str, default='../simulator/data')
    parser.add_argument('--input_layers', type=int, default=7)
    parser.add_argument('--method',       type=str, default='L-BFGS-B')
    parser.add_argument('--max_iter',     type=int, default=500)
    parser.add_argument('--tol',          type=float, default=1e-12)
    parser.add_argument('--n_starts',     type=int, default=5,
                        help='Number of random restarts per sample')
    parser.add_argument('--n_jobs',       type=int, default=20,
                        help='Number of parallel workers for sample-level parallelism')
    parser.add_argument('--seed',         type=int, default=42)
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
    # ── Load actual spectra ────────────────────────────────────────────────────
    file_path = os.path.expanduser(args.file_path)
    print(f"\nLoading spectra from: {file_path}")
    dfs   = [pd.read_csv(os.path.join(file_path, f"sobol_7layer_65536_chunk_000{i}.csv"))
             for i in range(7)]
    df    = pd.concat(dfs, axis=0, ignore_index=True)
    X_raw = df.iloc[:, 7:].values.astype(float)   # (N, 801) full range
    print(f"Spectra shape: {X_raw.shape}")

    assert len(X_raw) == n_samples

    # ── Thickness bounds ───────────────────────────────────────────────────────
    Y_raw  = df.iloc[:, :args.input_layers].values.astype(float)
    bounds = [(Y_raw[:, i].min() * 0.8, Y_raw[:, i].max() * 1.2)
              for i in range(args.input_layers)]
    print(f"Bounds: {[(f'{b[0]:.0f}', f'{b[1]:.0f}') for b in bounds]}")

    data_dir = os.path.expanduser(args.data_dir)

    # ── Run parallel multistart optimizer ─────────────────────────────────────
    print(f"\nRunning parallel multistart optimizer:")
    print(f"  Samples:        {n_samples}")
    print(f"  Starts/sample:  {args.n_starts}")
    print(f"  Sample workers: {args.n_jobs}")
    print(f"  Method:         {args.method}")
    print(f"  Max iter:       {args.max_iter}")
    print(f"  Tolerance:      {args.tol}\n")

    t0 = time.time()

    # Process all samples in parallel batches
    batch_size = args.n_jobs
    opt_preds  = np.zeros_like(dl_preds)
    n_converged = 0

    for batch_start in range(0, n_samples, batch_size):
        batch_end     = min(batch_start + batch_size, n_samples)
        batch_indices = list(range(batch_start, batch_end))

        batch_results = Parallel(n_jobs=len(batch_indices), prefer='processes')(
            delayed(process_sample)(
                i, dl_preds[i], X_raw[i], bounds,
                args.n_starts, args.method, args.max_iter, args.tol,
                data_dir, args.seed
            )
            for i in batch_indices
        )

        for idx, opt_x, converged in batch_results:
            opt_preds[idx] = opt_x
            if converged:
                n_converged += 1

        if batch_end % 1000 < batch_size or batch_end == n_samples:
            elapsed = time.time() - t0
            print(f"  {batch_end}/{n_samples} done | {elapsed:.0f}s elapsed | "
                  f"avg {elapsed/batch_end*1000:.0f}ms/sample")

    t_total = time.time() - t0

    # ── Final evaluation ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)
    print(f"{'Layer':<16} {'DL MAPE':>12} {'DL+Opt MAPE':>14} {'Improvement':>12}")
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
    out_path = os.path.join(out_dir, 'pred_dl_plus_optimizer_multistart_parallel_65k.csv')
    df_out.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
