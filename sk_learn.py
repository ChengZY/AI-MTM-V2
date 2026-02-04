import pandas as pd
import sklearn
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
# from sklearn.model_selection import GridSearchCV, KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, DotProduct

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from xgboost import XGBRegressor

file_path = "/home/zhongyao/dl/data/IPSR-AI-MTM/tmm3_smooth_1000_chunk_0000.csv"
df = pd.read_csv(file_path)


df_out = df.iloc[:, :3]
df_in = df.iloc[:, 3:]

X = df_in.values.astype(float)
Y = df_out.values.astype(float)

sc_in = StandardScaler()
X_std = sc_in.fit_transform(X)
sc_out = StandardScaler()
Y_std = sc_out.fit_transform(Y)

nr_cv = 10
kfold = KFold(n_splits=nr_cv, shuffle=True, random_state=42)

ml_name = "PLSR"
# ml_name = "GPR"
grid_search = True
if grid_search:
    if ml_name=="MLP":
        ml_model = MLPRegressor(max_iter=5000, random_state=42)
        param_grid = {
            "hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "alpha": [1e-4, 1e-3, 1e-2],
            "learning_rate_init": [1e-3, 1e-4],
            "activation": ["relu", "tanh"]
        }
        # Best params: {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (100,), 'learning_rate_init': 0.001}
    elif ml_name=="GPR":
        ml_model = GaussianProcessRegressor(
            random_state=42,
            optimizer="fmin_l_bfgs_b",
            copy_X_train=True
        )
        # 统一管理 kernel 构造
        def make_kernels(params=None):
            if params is None:
                params = [0.001, 0.01, 0.1, 0.5, 1, 1.5, 2, 10]

            kernels = []
            kernels += [Matern(length_scale=p, nu=1.5) for p in params]
            kernels += [RationalQuadratic(length_scale=p) for p in params]
            kernels += [DotProduct(sigma_0=p) for p in params]
            return kernels

        kernels = make_kernels()
        # GridSearch
        param_grid = {
            "kernel": kernels,
            "alpha": [1e-10, 1e-8, 1e-6, 1e-5, 1e-2],
            "n_restarts_optimizer": [5, 10],
        }
        # Best params: {'alpha': 1e-10, 'kernel': RationalQuadratic(alpha=1, length_scale=10), 'n_restarts_optimizer': 5}

    elif ml_name=="XGB":
        ml_model = XGBRegressor(random_state=42)
        param_grid = {'nthread': [3,5,10],  # when use hyperthread, xgboost may become slower
                      'learning_rate': [.03, 0.05, .07, .01],
                      'max_depth': [4, 5, 6],
                      'min_child_weight': [4],
                      'subsample': [0.7],
                      'colsample_bytree': [0.7]
                      }

    elif ml_name == "PLSR":
        ml_model = PLSRegression(max_iter=5000)
        param_grid = {'n_components': [10, 50, 500, 800],  # when use hyperthread, xgboost may become slower
                      }
    gs = GridSearchCV(ml_model, param_grid, cv=kfold, scoring="neg_mean_squared_error", n_jobs=-1, verbose=2, refit=True)
    gs.fit(X_std, Y_std)

    best = gs.best_estimator_
    best_params = gs.best_params_
    print("Best params:", gs.best_params_)
    print("Best CV neg MSE (grid search):", gs.best_score_)

    # Evaluate best estimator across kfold with two metrics
    mse_scores = -cross_val_score(best, X_std, Y_std, cv=kfold, scoring="neg_mean_squared_error", n_jobs=-1)
    r2_scores = cross_val_score(best, X_std, Y_std, cv=kfold, scoring="r2", n_jobs=-1)

else:
    if ml_name=="MLP":
        best_params = {"hidden_layer_sizes": (100,), "alpha": 0.01, "learning_rate_init": 0.001, "activation": "relu"}
    elif ml_name=="GPR":
        best_params = {'alpha': 1e-10, 'kernel': RationalQuadratic(alpha=1, length_scale=10), 'n_restarts_optimizer': 5}
    elif ml_name=="XGB":
        # Best params: {'colsample_bytree': 0.7, 'learning_rate': 0.07, 'max_depth': 6, 'min_child_weight': 4, 'nthread': 3, 'subsample': 0.7}
        best_params = {'colsample_bytree': 0.7, 'learning_rate': 0.07, 'max_depth': 6, 'min_child_weight': 4, 'nthread': 3, 'subsample': 0.7}
    elif ml_name=="PLSR":
        # Best params: {'colsample_bytree': 0.7, 'learning_rate': 0.07, 'max_depth': 6, 'min_child_weight': 4, 'nthread': 3, 'subsample': 0.7}
        best_params = {'n_components':50}

n_samples = X_std.shape[0]
preds = np.empty_like(Y)

for train_idx, test_idx in kfold.split(X_std):
    if ml_name == "MLP":
        ml_model = MLPRegressor(max_iter=5000, random_state=42,**best_params)
    elif ml_name == "GPR":
        ml_model = GaussianProcessRegressor(random_state=42, optimizer="fmin_l_bfgs_b", **best_params)
    elif ml_name == "XGB":
        ml_model = XGBRegressor(random_state=42,**best_params)
    elif ml_name == "PLSR":
        ml_model = PLSRegression(max_iter=5000,n_components=50)
    # mdl = MLPRegressor(max_iter=5000, random_state=42, **best_params)
    ml_model.fit(X_std[train_idx], Y_std[train_idx])

    Y_pred_std = ml_model.predict(X_std[test_idx])
    Y_pred = sc_out.inverse_transform(Y_pred_std)
    preds[test_idx] = Y_pred

MSE = np.mean((preds - Y) ** 2)
MAPE = np.mean(np.abs((preds - Y) / Y))
print(f"Fold-aligned MSE: {MSE:.6f}")
print(f"Fold-aligned MAPE: {MAPE:.6f}")

df_preds = np.concatenate((Y, preds), axis=1)
df_preds = pd.DataFrame(df_preds, columns=["thk_SiO2_1_gt", "thk_Si3N4_1_gt", "thk_SiO2_2_gt",\
                                           "thk_SiO2_1_pred", "thk_Si3N4_1_pred", "thk_SiO2_2_pred"])
df_preds.to_csv(f"{ml_name}_pred.csv", index=False)

for i in range(3):
    Y_true = df_preds.iloc[:, i].values.astype(float)
    Y_pred = df_preds.iloc[:, i + 3].values.astype(float)
    MAPE = np.mean(np.abs((Y_pred - Y_true) / Y_true))
    print(f"MAPE{i}:", MAPE)
print("completed evaluation.")
print("complete")