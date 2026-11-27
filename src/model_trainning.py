#%%

import pandas as  pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier

import optuna
import xgboost as xgb

from optuna_dashboard import run_server

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import shap
import datetime as dt
import pickle

import os
from sklearn.metrics import auc, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

#%%

parent_dir = os.path.dirname(os.getcwd())

results_folder_path = os.path.join(parent_dir, 'results')

base_data_folder_path = os.path.join(parent_dir, 'base_data')

model_result_path = os.path.join(results_folder_path, 'model')

#%%

base_df = pd.read_csv(os.path.join(base_data_folder_path, 'home_loan_applications_features_20k_extended.csv'))

master_df = pd.read_csv(os.path.join(base_data_folder_path, 'home_loan_applications_with_bad_flag.csv'))

master_df['login_date'] = pd.to_datetime(master_df['login_date'], format='%d-%m-%Y')

bad_def_res_df = pd.read_csv(os.path.join(results_folder_path, 'bad_def_base_data.csv'))

final_df = pd.merge(base_df, master_df, on='app_id', how='left')

final_df = pd.merge(final_df, bad_def_res_df[['app_id', 'max_mob', 'ever_60_dpd_24_mob']], on='app_id', how='left')

not_eligible = final_df[(final_df['ever_60_dpd_24_mob'] != 1) & (final_df['max_mob'] < 24)]['app_id']

final_df = final_df[~final_df['app_id'].isin(not_eligible)]

final_df.head()

#%%

metedata_columns = ['app_id', 'login_date', 'approve_reject_flag', 'ever_60_dpd_24_mob', 'max_mob']

train_df = final_df[final_df['login_date'] < '01-01-2024']

test_df = final_df[final_df['login_date'] >= '01-01-2024']

train_df.shape, test_df.shape

#%%

train_df.drop(metedata_columns, axis=1, inplace=True)
test_df.drop(metedata_columns, axis=1, inplace=True)

X_train = train_df.drop('bad', axis=1)
y_train = train_df['bad']

X_test = test_df.drop('bad', axis=1)
y_test = test_df['bad']

#%%


def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=10),
        "max_depth": trial.suggest_int("max_depth", 3, 8, step=1),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10, step=1),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 20, 50, step=5)
    }

    model = xgb.XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=15
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)

    f1_scr = f1_score(y_test, y_pred)

    return f1_scr


# Create persistent Optuna study
study = optuna.create_study(
    direction="maximize",
    storage="sqlite:///optuna_study.db",
    load_if_exists=True
)

# Run optimization
study.optimize(objective, n_trials=2)

print("Best F1-score:", study.best_value)
print("Best Parameters:", study.best_params)

#%%

base_xgb = xgb.XGBClassifier(**study.best_params, random_state=15)

base_xgb.fit(X_train, y_train)

y_pred_1 = base_xgb.predict(X_test)

y_prob_1 = base_xgb.predict_proba(X_test)

f1_scr = f1_score(y_test, y_pred_1)

print(f'final f1_score : {f1_scr}')

#%%

explainer = shap.TreeExplainer(base_xgb)
shap_values = explainer(X_test)

# 5. Generate and Display the SHAP Beeswarm Plot
plt.figure(figsize=(10, 8))
shap.plots.beeswarm(shap_values, max_display=20)

#%%

shap_ranking = pd.DataFrame({
    'feature': X_train.columns,
    'mean_abs_shap': abs(shap_values.values).mean(axis=0)
}).sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)

# Add rank column starting at 1
shap_ranking.insert(0, 'rank', shap_ranking.index + 1)

#%%

k = 10
top_k_feaatures = shap_ranking[shap_ranking['rank'] <= k]['feature'].to_list()

#%%

X_train_final = X_train[top_k_feaatures]
X_test_final = X_test[top_k_feaatures]


def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=10),
        "max_depth": trial.suggest_int("max_depth", 3, 8, step=1),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10, step=1),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, step=1),
        "reg_alpha": trial.suggest_float("reg_lambda", 1.0, 20.0, step=1),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 20, 50, step=5)
    }

    model = xgb.XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=15
    )

    model.fit(X_train_final, y_train)

    y_pred = model.predict(X_test_final)

    y_pred_proba = model.predict_proba(X_test_final)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)

    f1_scr = f1_score(y_test, y_pred)

    return auc


storage = optuna.storages.InMemoryStorage()

# Create persistent Optuna study
study = optuna.create_study(
    direction="maximize",
    storage=storage,
    load_if_exists=True
)

# Run optimization
study.optimize(objective, n_trials=2)

print("Best F1-score:", study.best_value)
print("Best Parameters:", study.best_params)

#%%

base_xgb_final = xgb.XGBClassifier(**study.best_params, random_state=15)

base_xgb_final.fit(X_train_final, y_train)

y_pred_final = base_xgb_final.predict(X_test_final)

y_prob_final = base_xgb_final.predict_proba(X_test_final)

f1_scr = f1_score(y_test, y_pred_final)

print(f'final f1_score : {f1_scr}')

auc = roc_auc_score(y_true=y_test, y_score=y_prob_final[:, 1])

print(f'final auc: {auc}')


#%%

# run_server(storage)

#%%

model_folder_path = os.path.join(model_result_path, f'model_{dt.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}')

os.makedirs(model_folder_path, exist_ok=True)

with open(os.path.join(model_folder_path, 'model.pkl'), 'wb') as file:

    pickle.dump(base_xgb_final, file)

#%%