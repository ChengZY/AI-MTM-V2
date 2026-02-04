import numpy as np
import pandas as pd

ml_names = ["MLP","XGB","GPR"]
ml_name = "GPR"

for ml in ml_names:
# pred_file = f'{ml_name}_pred.csv'
    df_pred = pd.read_csv(f'{ml_name}_pred.csv')
    # df_pred = pd.read_csv(f'{ml_name}_pred_10manual.csv')

    for i in range(3):
        Y_true = df_pred.iloc[:, i].values.astype(float)
        Y_pred = df_pred.iloc[:, i + 3].values.astype(float)
        MAPE = np.mean(np.abs((Y_pred - Y_true) / Y_true))
        print("MAPE:", MAPE)
    print("completed evaluation.")