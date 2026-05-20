#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import os
import json
import re
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

import xgboost as xgb
import lightgbm as lgb
import optuna


# In[ ]:


# Load datasets
go_data = pd.read_csv("output/go_ratio_PD.csv", index_col=0)
tpm_data = pd.read_csv("output/tpm_data_PD.csv", index_col=0)
bg_data = pd.read_csv("data_folder/background_all_small.csv", index_col=0)

# Merge GO and TPM data
merged_data = pd.concat([go_data, tpm_data], axis=0)

# Transpose go data for features (X) and extract labels (y)
X = go_data.T
y = bg_data['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[ ]:


# Baseline Random Forest Model
rf_baseline = RandomForestClassifier(random_state=0)
rf_baseline.fit(X_train, y_train)
y_pred_rf = rf_baseline.predict(X_test)
print(f"Baseline Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

# Define Optuna objective function for Random Forest
def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300) 
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 5, 30)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 5, 20)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=0,
        n_jobs=-1 
    )

    score = cross_val_score(
        clf, X_train, y_train, n_jobs=-1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        scoring='accuracy'
    )
    return score.mean()

# Execute Optuna Optimization
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=500)

best_params_rf = study_rf.best_params
print("-" * 50)
print("✅ Optimization Completed")
print("✅ Best Parameters (Optuna):", best_params_rf)
print(f"✅ Best Cross-Validation Accuracy: {study_rf.best_value:.4f}")
print("-" * 50)

# Train the final model using the best hyperparameters
best_rf = RandomForestClassifier(**best_params_rf, random_state=0, n_jobs=-1)
best_rf.fit(X_train, y_train)

# Evaluate final model accuracy
final_accuracy_rf = accuracy_score(y_test, best_rf.predict(X_test))
print(f"✅ Final Test Accuracy: {final_accuracy_rf:.4f}")

# Save the best parameters to a JSON file
param_save_path = 'parameter/RF_go_PD_params.json'
os.makedirs(os.path.dirname(param_save_path), exist_ok=True)
with open(param_save_path, 'w') as f:
    json.dump(best_params_rf, f, indent=4)
print("Hyperparameters successfully saved.")


# In[ ]:


# Function to sanitize and deduplicate column names for LightGBM
def sanitize_and_deduplicate_columns(df):
    """
    Sanitizes column names by replacing special characters and then
    appends a suffix to any resulting duplicates.
    """
    import re
    from collections import Counter
    sanitized_cols = [re.sub(r'[^a-zA-Z0-9_]', '_', str(col)) for col in df.columns]
    counts = Counter(sanitized_cols)
    final_cols = []

    for i in range(len(sanitized_cols) - 1, -1, -1):
        col = sanitized_cols[i]
        if counts[col] > 1:
            final_cols.append(f"{col}_{counts[col]}")
            counts[col] -= 1
        else:
            final_cols.append(col)

    df.columns = final_cols[::-1]
    return df

# Apply sanitization to the dataset
X_sanitized = sanitize_and_deduplicate_columns(X.copy())
X_train_lgb, X_test_lgb, y_train_lgb, y_test_lgb = train_test_split(X_sanitized, y, test_size=0.3, random_state=0)

# Baseline LightGBM Model
lgb_baseline = lgb.LGBMClassifier(random_state=0, verbosity=-1)
lgb_baseline.fit(X_train_lgb, y_train_lgb)
print(f"Baseline LightGBM Accuracy: {accuracy_score(y_test_lgb, lgb_baseline.predict(X_test_lgb)):.4f}")

# Define Optuna objective function for LightGBM
def objective_lgbm(trial):
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss', # 評価指標に合わせて変更可能
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'random_state': 0,
        'n_jobs': -1
    }

    clf = lgb.LGBMClassifier(**param)

    # Cross-Validation
    score = cross_val_score(
        clf, X_train_lgb, y_train_lgb, n_jobs=-1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        scoring='accuracy'
    )
    return score.mean()

# Execute Optuna Optimization for LightGBM
study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(objective_lgbm, n_trials=500)

best_params_lgbm = study_lgbm.best_params
print("-" * 50)
print("✅ LightGBM Optimization Completed")
print("✅ Best Parameters (Optuna):", best_params_lgbm)
print(f"✅ Best Cross-Validation Accuracy: {study_lgbm.best_value:.4f}")
print("-" * 50)

# Train the final model using the best hyperparameters
best_lgbm = lgb.LGBMClassifier(**best_params_lgbm, random_state=0, n_jobs=-1, verbosity=-1)
best_lgbm.fit(X_train_lgb, y_train_lgb)

# Evaluate final model accuracy
final_accuracy_lgbm = accuracy_score(y_test_lgb, best_lgbm.predict(X_test_lgb))
print(f"✅ Final LightGBM Test Accuracy: {final_accuracy_lgbm:.4f}")

# Save the best parameters to a JSON file
param_save_path_lgbm = 'parameter/LGBM_go_PD_params.json'
os.makedirs(os.path.dirname(param_save_path_lgbm), exist_ok=True)
with open(param_save_path_lgbm, 'w') as f:
    json.dump(best_params_lgbm, f, indent=4)
print("LightGBM hyperparameters successfully saved.")


# In[ ]:


from sklearn.preprocessing import LabelEncoder

# Encode target labels to numeric values (required by XGBoost)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the sanitized data for XGBoost
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_sanitized, y_encoded, test_size=0.3, random_state=0
)

# Baseline XGBoost Model
xgb_baseline = xgb.XGBClassifier(random_state=0, eval_metric='logloss', n_jobs=-1)
xgb_baseline.fit(X_train_xgb, y_train_xgb)
y_pred_xgb = xgb_baseline.predict(X_test_xgb)
print(f"Baseline XGBoost Accuracy: {accuracy_score(y_test_xgb, y_pred_xgb):.4f}")

# Define Optuna objective function for XGBoost
def objective_xgb(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'random_state': 0,
        'eval_metric': 'logloss',
        'n_jobs': -1
    }

    clf = xgb.XGBClassifier(**param)

    # Cross-Validation
    score = cross_val_score(
        clf, X_train_xgb, y_train_xgb, n_jobs=-1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        scoring='accuracy'
    )
    return score.mean()

# Execute Optuna Optimization for XGBoost
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=500)

best_params_xgb = study_xgb.best_params
print("-" * 50)
print("✅ XGBoost Optimization Completed")
print("✅ Best Parameters (Optuna):", best_params_xgb)
print(f"✅ Best Cross-Validation Accuracy: {study_xgb.best_value:.4f}")
print("-" * 50)

# Train the final model using the best hyperparameters
best_xgb = xgb.XGBClassifier(**best_params_xgb, random_state=0, n_jobs=-1, eval_metric='logloss')
best_xgb.fit(X_train_xgb, y_train_xgb)

# Evaluate final model accuracy
final_accuracy_xgb = accuracy_score(y_test_xgb, best_xgb.predict(X_test_xgb))
print(f"✅ Final XGBoost Test Accuracy: {final_accuracy_xgb:.4f}")

# Save the best parameters to a JSON file
param_save_path_xgb = 'parameter/XGB_go_PD_params.json'
os.makedirs(os.path.dirname(param_save_path_xgb), exist_ok=True)
with open(param_save_path_xgb, 'w') as f:
    json.dump(best_params_xgb, f, indent=4)
print("XGBoost hyperparameters successfully saved.")

