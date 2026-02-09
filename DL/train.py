import pandas as pd
import numpy as np
import argparse

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, DotProduct
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from xgboost import XGBRegressor
import os
import random

import torch
from torch import optim
from torch.optim.swa_utils import AveragedModel, SWALR
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from model import SimpleLSTM, SimpleMLP, Spectrum1DCNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM model for prediction')
    parser.add_argument('--file_path', type=str, default="/home/zhongyao/dl/data/IPSR-AI-MTM/tmm3_smooth_1000_chunk_0000.csv", help='Number of training epochs')
    # parser.add_argument('--file_path', type=str,
    #                     default="/home/zhongyao/dl/data/IPSR-AI-MTM/TMM-data-2026-01-14-2026/sobol_7layer_16384_chunk_0000.csv",
    #                     help='Number of training epochs')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=8, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=3, help='num layers')
    parser.add_argument('--input_layers', type=int, default=3, help='num layers') #7
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizer')
    parser.add_argument('--model_path', type=str, default='./1000_simdata_seed42_7layers/',help='Path to save models')
    parser.add_argument('--model_name', type=str, default='Spectrum1DCNN', help='Spectrum1DCNN')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--val_step', type=int, default=10, help='val')
    # parser.add_argument('--target', type=str, default='Thickness', help='Thickness or Doping')
    # # parser.add_argument('--excel_path', type=str, default='/home/zhongyao/dl/data/doe_data/multimodel_data/SiC_Exp_Runs/SiC_8inch_After_PM_Exp_Runs_Data_20250905.xlsx', help='excel path')
    # parser.add_argument('--excel_path', type=str, default='/home/zhongyao/dl/data/doe_data/multimodel_data/SiC_Exp_Runs/SiC_8inch_Exp_Runs_Data_20250826_denoised.xlsx', help='excel path')
    # parser.add_argument("--include_bw", action="store_true")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    torch.manual_seed(42)  # Add this line for PyTorch reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    df = pd.read_csv(args.file_path)

    input_layers = args.input_layers
    df_out = df.iloc[:, :input_layers]
    df_in = df.iloc[:, input_layers:]
    X = df_in.values.astype(float)
    Y = df_out.values.astype(float)

    sc_in = StandardScaler()
    X_std = sc_in.fit_transform(X)
    sc_out = StandardScaler()
    Y_std = sc_out.fit_transform(Y)

    nr_cv = 10
    kfold = KFold(n_splits=nr_cv, shuffle=True, random_state=42)

    # nr_cv = 10
    # n_samples = X_std.shape[0]
    #
    # #自定义平均取的分割逻辑
    # indices = np.arange(n_samples)
    # folds = [indices[i::nr_cv] for i in range(nr_cv)]
    #
    # # 自定义 KFold 的生成器
    # class CustomKFold:
    #     def __init__(self, folds):
    #         self.folds = folds
    #     def split(self, X):
    #         for i in range(len(self.folds)):
    #             test_idx = self.folds[i]
    #             train_idx = np.setdiff1d(indices, test_idx)
    #             yield train_idx, test_idx
    #
    # kfold = CustomKFold(folds)

    class torch_data_loader(torch.utils.data.Dataset):
        def __init__(self, X, Y):
            self.X = torch.tensor(X, dtype=torch.float)
            self.Y = torch.tensor(Y, dtype=torch.float)

        def __len__(self):
            return len(self.Y)

        def __getitem__(self, idx):
            return idx, self.X[idx], self.Y[idx]

    # model_path = f"{args.model_path}_{args.model_name}_lr{args.lr}"
    os.makedirs(f"{args.model_path}", exist_ok=True)
    model_path = f"{args.model_path}/{args.model_name}/hiddensize{args.hidden_size}_numlayers{args.num_layers}_lr{args.lr}"
    os.makedirs(f"{args.model_path}/{args.model_name}", exist_ok=True)
    os.makedirs(f"{model_path}", exist_ok=True)
    tb_writer = SummaryWriter(model_path)

    base_dataset = torch_data_loader(X_std, Y_std)
    fold_results = []
    num_epochs = args.epochs
    batch_size = args.batch_size
    val_step = args.val_step
    lr = args.lr
    device = "cuda" if torch.cuda.is_available() else "cpu"

    xs_np = np.empty_like(X)
    preds_np = np.empty_like(Y)
    gts_np = np.empty_like(Y)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_std), 1):
        train_ds = Subset(base_dataset, train_idx)
        val_ds = Subset(base_dataset, val_idx)

        # model = SimpleLSTM(input_size=X_std.shape[1], hidden_size=256, num_layers=2, output_size=Y_std.shape[1])
        # model = SimpleMLP(input_size=X_std.shape[1], hidden_size=256, num_layers=2, output_size=Y_std.shape[1])

        model = Spectrum1DCNN(input_size=X_std.shape[1],hidden_size=args.hidden_size, num_layers=args.num_layers,output_size=input_layers)
        model.to(device)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        # train_model moves the model to device internally
        # train_model(model, train_ds, epochs=num_epochs, batch_size=batch_size, lr=lr, device=device)
        model.train()
        best_loss = float("inf")
        # for epoch in range(1, num_epochs + 1):
        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            for train_ix, train_X, train_Y in train_loader:
                train_X = train_X.to(device)
                train_Y = train_Y.to(device)
                # if features only, make seq_len=1
                if train_X.dim() == 2:
                    train_X = train_X.unsqueeze(1)
                preds = model(train_X)
                loss = criterion(preds, train_Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * train_X.size(0)
            avg_train = total_loss / len(train_loader)
            tb_writer.add_scalar(f'Folder-{fold}/train_loss', avg_train, epoch)

            if epoch % val_step == 0:
                print(f"Epoch {epoch}/{num_epochs} - loss: {avg_train:.6f}")
                # print("Evaluating...")
                with torch.no_grad():
                    total = 0.0
                    preds_val = []
                    gt_np_val = []
                    xs_val = []
                    for val_ix, val_X, val_Y in val_loader:
                        val_X, val_Y = val_X.to(device), val_Y.to(device)
                        x_val = sc_in.inverse_transform(val_X.detach().cpu().numpy())
                        xs_val.append(x_val)
                        if val_X.dim() == 2:
                            val_X = val_X.unsqueeze(1)
                        preds = model(val_X)
                        loss = criterion(preds, val_Y)
                        total += loss.item() * val_X.size(0)

                        pred_val = sc_out.inverse_transform(preds.detach().cpu().numpy())
                        preds_val.append(pred_val)
                        gt_np = sc_out.inverse_transform(val_Y.detach().cpu().numpy())
                        gt_np_val.append(gt_np)


                    avg_loss_val = total / len(val_ds)
                    preds_val = np.concatenate(preds_val)
                    gt_np_val = np.concatenate(gt_np_val)
                    xs_val = np.concatenate(xs_val)
                    print(f"Fold-{fold} Validation MSE after epoch {epoch}: {avg_loss_val:.6f}")
                    tb_writer.add_scalar(f'Folder-{fold}/test_loss', avg_loss_val, epoch)

                    for ix_pred in range(preds_val.shape[1]):
                        MAPE = np.mean(np.abs((preds_val[:, ix_pred] - gt_np_val[:, ix_pred]) / gt_np_val[:, ix_pred]))
                        # print(f"Fold-{fold} Validation MAPE for param {ix_pred} after epoch {epoch}: {MAPE:.6f}")
                        tb_writer.add_scalar(f'Folder-{fold}/MAPE{ix_pred}', MAPE, epoch)

                if avg_train < best_loss:
                    best_loss = avg_train
                    best_epoch = epoch
                    # torch.save(model.state_dict(), f"model_fold{fold}.pth")
                    torch.save(model, f"{model_path}/model_fold{fold}.pth")
                    preds_np[val_idx] = preds_val
                    gts_np[val_idx] = gt_np_val
                    xs_np[val_idx] = xs_val
                    print(f"New best model (epoch {epoch}) saved")

    MAPE = np.mean(np.abs((preds_np - gts_np) / gts_np))

    df_preds = np.concatenate((Y, preds_np), axis=1)
    # df_preds = pd.DataFrame(df_preds, columns=["thk_SiO2_1_gt", "thk_Si3N4_1_gt", "thk_SiO2_2_gt", \
    #                                            "thk_SiO2_1_pred", "thk_Si3N4_1_pred", "thk_SiO2_2_pred"])
    columns = [col+'_gt' for col in df_out.columns] + [col+'_pred' for col in df_out.columns]
    df_preds = pd.DataFrame(df_preds, columns=columns)
    df_preds.to_csv(f"{model_path}/pred.csv", index=False)

    MAPEs = []
    for i in range(input_layers):
        Y_true = df_preds.iloc[:, i].values.astype(float)
        Y_pred = df_preds.iloc[:, i + input_layers].values.astype(float)
        MAPE = np.mean(np.abs((Y_pred - Y_true) / Y_true))
        MAPEs.append(MAPE)
        print("MAPE:", MAPE)

    MAPEs = np.array(MAPEs)  # Ensure it's a 1D array
    df_eval = pd.DataFrame(MAPEs, index=df_out.columns, columns=["MAPE"])
    df_eval.to_csv(f"{model_path}/eval.csv")
    print("Training complete.")