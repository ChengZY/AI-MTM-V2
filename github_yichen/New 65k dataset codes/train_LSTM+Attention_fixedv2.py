import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import os
import random

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from model import CNN_LSTM_Attention

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN-LSTM-Attention for spectral thickness prediction')
#    parser.add_argument('--file_path',      type=str,   default="~/projects/AI_MTM/AI-MTM-main/data/TMM_Data/")
    parser.add_argument('--file_path',      type=str,   default="~/projects/dataset_65k_sio2_100_2000_si3n4_100_1000/")
    parser.add_argument('--epochs',         type=int,   default=2000)
    parser.add_argument('--lr',             type=float, default=0.001)
    parser.add_argument('--hidden_size',    type=int,   default=256,  help='LSTM hidden size')
    parser.add_argument('--num_layers',     type=int,   default=3,    help='Number of LSTM layers')
    parser.add_argument('--input_layers',   type=int,   default=7,    help='Number of output targets')
    parser.add_argument('--weight_decay',   type=float, default=1e-4)
#    parser.add_argument('--model_path',     type=str,   default='./results_cnn_lstm_attention/')
    parser.add_argument('--model_path',     type=str,   default='./results_cnn_lstm_attention_65k/')
    parser.add_argument('--batch_size',     type=int,   default=64)
    parser.add_argument('--val_step',       type=int,   default=5)
    parser.add_argument('--dropout',        type=float, default=0.1)
    parser.add_argument('--num_heads',      type=int,   default=8,
                        help='Attention heads — hidden_size must be divisible by this')

    # ── CNN-specific parameters ───────────────────────────────────────────────
    # cnn_channels   : number of feature maps the CNN produces
    #                  recommended: 64 or 128
    # cnn_kernel_size: how many wavelength points each CNN filter covers
    #                  recommended: 7 or 11
    #                  larger = captures wider fringe patterns
    # cnn_stride     : compression factor — stride=4 on 801 points gives ~200
    #                  timesteps for LSTM instead of 801
    #                  recommended: 4 (good compression without losing too much)
    # cnn_layers     : number of stacked CNN layers
    #                  recommended: 2
    # ────────────────────────────────────────────────────────────────────────
    parser.add_argument('--cnn_channels',    type=int, default=128)
    parser.add_argument('--cnn_kernel_size', type=int, default=11)
    parser.add_argument('--cnn_stride',      type=int, default=4)
    parser.add_argument('--cnn_layers',      type=int, default=2)

    args = parser.parse_args()

    assert args.hidden_size % args.num_heads == 0, \
        f"hidden_size ({args.hidden_size}) must be divisible by num_heads ({args.num_heads})"

    # ── Reproducibility ──────────────────────────────────────────────────────
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    # ── Data loading — full range 200-1000nm 1nm resolution ──────────────────
#    dfs = [pd.read_csv(f"{args.file_path}/sobol_7layer_16384_chunk_000{i}.csv") for i in range(4)]
    dfs = [pd.read_csv(os.path.join(os.path.expanduser(args.file_path), f"sobol_7layer_65536_chunk_000{i}.csv")) for i in range(7)]
    df  = pd.concat(dfs, axis=0, ignore_index=True)

    start_nm = 200
    end_nm = 1000
    resolution = 1

    input_layers = args.input_layers
    df_out = df.iloc[:, :input_layers]
#    df_in_full = df.iloc[:, input_layers:]
#    df_in     = df_in_full.iloc[:, ::2]   # 2nm resolution (every other column)
    df_in  = df.iloc[:, input_layers:]      # full 200-1000nm 1nm = 801 points
#    df_out = df.iloc[:, :args.input_layers]
#    wanted_cols = [f"R_{w}nm" for w in range(250, 901, 1)]
#    df_in = df[wanted_cols]
    X = df_in.values.astype(float)
    Y = df_out.values.astype(float)
    print(f"Input shape: {X.shape}, Output shape: {Y.shape}")

    sc_in  = StandardScaler(); X_std = sc_in.fit_transform(X)
    sc_out = StandardScaler(); Y_std = sc_out.fit_transform(Y)

    # Print what compression the CNN will apply
    compressed_len = X_std.shape[1] // args.cnn_stride
    print(f"CNN will compress {X_std.shape[1]} wavelength points → ~{compressed_len} LSTM timesteps")

    # ── Output paths ─────────────────────────────────────────────────────────
    model_name = 'CNN_LSTM_Attention'
    model_path = os.path.join(
        args.model_path,
        f"{model_name}/hidden{args.hidden_size}_cnnch{args.cnn_channels}_stride{args.cnn_stride}_lr{args.lr}"
    )
    os.makedirs(model_path, exist_ok=True)
    tb_writer = SummaryWriter(model_path)
    print("TensorBoard logs:", model_path)

    # ── Dataset ───────────────────────────────────────────────────────────────
    class SpectraDataset(torch.utils.data.Dataset):
        def __init__(self, X, Y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.Y = torch.tensor(Y, dtype=torch.float32)
        def __len__(self): return len(self.Y)
        def __getitem__(self, idx): return idx, self.X[idx], self.Y[idx]

    base_dataset = SpectraDataset(X_std, Y_std)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    preds_np = np.zeros_like(Y)
    gts_np   = np.zeros_like(Y)

    # ── K-Fold ───────────────────────────────────────────────────────────────
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_std), 1):
        train_loader = DataLoader(Subset(base_dataset, train_idx),
                                  batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(Subset(base_dataset, val_idx),
                                  batch_size=args.batch_size, shuffle=False)

        model = CNN_LSTM_Attention(
            input_size      = X_std.shape[1],
            cnn_channels    = args.cnn_channels,
            cnn_kernel_size = args.cnn_kernel_size,
            cnn_stride      = args.cnn_stride,
            cnn_layers      = args.cnn_layers,
            hidden_size     = args.hidden_size,
            num_layers      = args.num_layers,
            output_size     = input_layers,
            num_heads       = args.num_heads,
            dropout         = args.dropout,
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=20, factor=0.5
        )
        criterion  = nn.MSELoss()
        best_loss  = float("inf")
        best_epoch = -1

        for epoch in range(1, args.epochs + 1):
            # ── Train ─────────────────────────────────────────────────────────
            model.train()
            total_loss = 0.0
            for _, bX, bY in train_loader:
                bX, bY = bX.to(device), bY.to(device)
                preds = model(bX)
                loss  = criterion(preds, bY)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * bX.size(0)

            avg_train = total_loss / len(train_loader.dataset)
            tb_writer.add_scalar(f'Fold-{fold}/train_loss', avg_train, epoch)

            # ── Validate ──────────────────────────────────────────────────────
            if epoch % args.val_step == 0:
                print(f"Fold {fold} | Epoch {epoch}/{args.epochs} | Train Loss: {avg_train:.6f}")
                model.eval()
                with torch.no_grad():
                    total     = 0.0
                    preds_val = []
                    gt_val    = []

                    for _, vX, vY in val_loader:
                        vX, vY = vX.to(device), vY.to(device)
                        out    = model(vX)
                        total += criterion(out, vY).item() * vX.size(0)
                        preds_val.append(sc_out.inverse_transform(out.cpu().numpy()))
                        gt_val.append(sc_out.inverse_transform(vY.cpu().numpy()))

                    avg_val   = total / len(val_loader.dataset)
                    preds_val = np.concatenate(preds_val)
                    gt_val    = np.concatenate(gt_val)

                    scheduler.step(avg_val)
                    print(f"  Val MSE: {avg_val:.6f}")
                    tb_writer.add_scalar(f'Fold-{fold}/val_loss', avg_val, epoch)

                    for k in range(preds_val.shape[1]):
                        mape = np.mean(np.abs((preds_val[:, k] - gt_val[:, k]) / gt_val[:, k]))
                        tb_writer.add_scalar(f'Fold-{fold}/MAPE_{df_out.columns[k]}', mape, epoch)

                    if avg_val < best_loss:
                        best_loss  = avg_val
                        best_epoch = epoch
                        torch.save(model.state_dict(), f"{model_path}/model_fold{fold}.pth")
                        preds_np[val_idx] = preds_val
                        gts_np[val_idx]   = gt_val
                        print(f"  → Saved best (epoch {epoch}, val {avg_val:.6f})")

        print(f"Fold {fold} done | best epoch: {best_epoch} | best val: {best_loss:.6f}\n")

    # ── Final evaluation ──────────────────────────────────────────────────────
    columns  = [c + '_gt' for c in df_out.columns] + [c + '_pred' for c in df_out.columns]
    df_final = pd.DataFrame(np.concatenate([gts_np, preds_np], axis=1), columns=columns)
#    df_final.to_csv(f"{model_path}/pred.csv", index=False)
    df_final.to_csv(f"{model_path}/pred_{start_nm}_{end_nm}_{resolution}nm.csv",index=False)

    MAPEs = []
    for i in range(input_layers):
        Y_true = df_final.iloc[:, i].values
        Y_pred = df_final.iloc[:, i + input_layers].values
        mape   = np.mean(np.abs((Y_pred - Y_true) / Y_true))
        MAPEs.append(mape)
        print(f"Final MAPE  {df_out.columns[i]}: {mape}")

    pd.DataFrame(MAPEs, index=df_out.columns, columns=["MAPE"]).to_csv(f"{model_path}/eval.csv")
    print("\nTraining complete.")
