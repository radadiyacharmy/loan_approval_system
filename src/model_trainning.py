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
from sklearn.metrics import auc, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve

#%%


def generate_decile_chart(model, X, y, n_deciles=10, bad_label=1):

    # Predicted probability of Bad (class 1)
    class_index = list(model.classes_).index(bad_label)

    probabilities = model.predict_proba(X)[:, class_index]

    decile_df = pd.DataFrame({
        "prob": probabilities,
        "actual": np.asarray(y)
    })

    # Rank from highest risk to lowest risk
    decile_df["rank"] = decile_df["prob"].rank(
        method="first",
        ascending=False
    )

    # Assign deciles: 1 = highest risk
    decile_df["decile"] = np.ceil(
        decile_df["rank"] * n_deciles / len(decile_df)
    ).astype(int)

    # Guard against edge cases
    decile_df["decile"] = decile_df["decile"].clip(
        upper=n_deciles
    )

    # Aggregate decile statistics
    decile_chart = (
        decile_df
        .groupby("decile")
        .agg(
            min_prob=("prob", "min"),
            max_prob=("prob", "max"),
            total_count=("prob", "size"),
            bad_count=("actual", lambda x: (x == bad_label).sum()),
        )
        .reset_index()
        .sort_values("decile")
    )

    decile_chart["bad_rate"] = (decile_chart["bad_count"] / decile_chart["total_count"]) * 100

    total_bad_rate = (decile_chart["bad_count"].sum() / decile_chart["total_count"].sum()) * 100

    decile_chart["lift"] = decile_chart["bad_rate"] / total_bad_rate

    return decile_chart, decile_df


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

# final_df.drop(cols_to_drop, axis=1, inplace=True)

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
study.optimize(objective, n_trials=20)

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

k = 15
top_k_feaatures = shap_ranking[shap_ranking['rank'] <= k]['feature'].to_list()


#%%

final_features = [
 'foir',
 'avg_dpd_active',
 'ltv',
 'disposable_income',
 'monthly_income',
 'overdue_to_income_ratio',
 'no_enq_12_months',
 'loan_amount',
 'emi',
 'total_unsecured_overdue',
 'no_enq_12_months_HL'
]


X_train_final = X_train[final_features]
X_test_final = X_test[final_features] 

# X_train_final = X_train[top_k_feaatures]
# X_test_final = X_test[top_k_feaatures]


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
study.optimize(objective, n_trials=30)

print("Best F1-score:", study.best_value)
print("Best Parameters:", study.best_params)

#%%

base_xgb_final = xgb.XGBClassifier(**study.best_params, random_state=15)

base_xgb_final.fit(X_train_final, y_train)

y_pred_final = base_xgb_final.predict(X_test_final)

y_prob_final = base_xgb_final.predict_proba(X_test_final)

f1_scr = f1_score(y_test, y_pred_final)

print(f'final f1_score : {f1_scr}')

auc_val = roc_auc_score(y_true=y_test, y_score=y_prob_final[:, 1])

print(f'final auc: {auc_val}')

#%%

fpr, tpr, _ = roc_curve(y_test, y_prob_final[:, 1])
roc_auc = auc(fpr, tpr)

# 3. Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'LightGBM ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess (AUC = 0.50)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%%


explainer = shap.TreeExplainer(base_xgb_final)
shap_values = explainer(X_test_final)

# 5. Generate and Display the SHAP Beeswarm Plot
plt.figure(figsize=(10, 8))
shap.plots.beeswarm(shap_values, max_display=20)



#%%

# run_server(storage)

#%%

decile_chart, decile_df = generate_decile_chart(base_xgb_final, X_test_final, y_test, n_deciles=10, bad_label=1)

decile_chart

#%%

model_folder_path = os.path.join(model_result_path, f'model_{dt.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}')

os.makedirs(model_folder_path, exist_ok=True)

with open(os.path.join(model_folder_path, 'model.pkl'), 'wb') as file:

    pickle.dump(base_xgb_final, file)

#%%

