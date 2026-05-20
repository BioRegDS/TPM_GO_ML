#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =====================================================================
# 1. Setup and Data Preparation
# =====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, 
    average_precision_score, roc_curve, auc, accuracy_score
)
import xgboost as xgb
import lightgbm as lgb

# Load datasets
go_data = pd.read_csv("../../1209/output/go_ratio_PD.csv", index_col=0)
tpm_data = pd.read_csv("../../1209/output/tpm_data_PD.csv", index_col=0)
bg_data = pd.read_csv("../../1209/data_folder/background_all_small.csv", index_col=0)

# Preprocessing
if "Gene_Count_PD" in go_data.columns:
    go_data = go_data.drop("Gene_Count_PD", axis=1)

# Merge GO and TPM data
merged_data = pd.concat([go_data, tpm_data], axis=0)

# Define Features (X) and Target (y)
X = merged_data.T
y = bg_data['Class']

# ---------------------------------------------------------
# Feature Name Cleaning & Deduplication (Crucial for LightGBM/XGBoost)
# ---------------------------------------------------------
# 1. Ensure strings and remove special characters
clean_cols = X.columns.astype(str).str.replace(r'[^A-Za-z0-9_]', '_', regex=True)

# 2. Make duplicated column names unique by appending _1, _2, etc.
cols_series = pd.Series(clean_cols)
dup_counts = cols_series.groupby(cols_series).cumcount()
X.columns = np.where(dup_counts > 0, cols_series + '_' + dup_counts.astype(str), cols_series)

# Encode target variable: 'PD' -> 1, 'CL' (Control) -> 0
y_encoded = np.where(y == 'PD', 1, 0)

# Train-Test Split (Stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=0, stratify=y_encoded
)

print(f"Data ready. Features: {X_train.shape[1]}, Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
# Check if there are any remaining duplicates (Should output 0)
print(f"Duplicate features remaining: {X_train.columns.duplicated().sum()}")


# In[ ]:


# =====================================================================
# 2. Parameter Loading and Model Initialization
# =====================================================================

# Load Random Forest parameters
with open("../../1209/parameter/RF_merged_PD_params-500.json", "r") as f:
    rf_params = json.load(f)

# Load LightGBM parameters (Adjust filename as needed)
with open("../../1209/parameter/lgbm_best_merged_PD_params-500.json", "r") as f:
    lgbm_params = json.load(f)

# Load XGBoost parameters (Adjust filename as needed)
with open("../../1209/parameter/xgboost_best_merged_PD_params-500.json", "r") as f:
    xgb_params = json.load(f)

print("All model parameters loaded successfully.")

# Initialize models with loaded parameters
models = {
    "Random Forest": RandomForestClassifier(**rf_params, random_state=0),
    "LightGBM": lgb.LGBMClassifier(**lgbm_params, random_state=0),
    "XGBoost": xgb.XGBClassifier(**xgb_params, random_state=0)
}


# In[ ]:


# =====================================================================
# 3. Model Training and Evaluation (Standard vs Youden Index)
# =====================================================================
results = {}

for name, clf in models.items():
    print(f"Training {name}...")
    # Train the model
    clf.fit(X_train, y_train)

    # Predict probabilities for the positive class (PD = 1)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Calculate ROC metrics
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y_test, y_prob)

    # Find optimal threshold using Youden's Index (J = Sensitivity + Specificity - 1)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[best_idx]

    # Predictions using standard threshold (0.5) and optimal threshold
    y_pred_std = (y_prob >= 0.5).astype(int)
    y_pred_youden = (y_prob >= optimal_threshold).astype(int)

    # Store all results for later visualization
    results[name] = {
        "y_prob": y_prob,
        "y_pred_std": y_pred_std,
        "y_pred_youden": y_pred_youden,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "optimal_threshold": optimal_threshold
    }

print("\nAll models trained and evaluated.")


# In[ ]:


# =====================================================================
# 4. Classification Reports and CSV Export
# =====================================================================
import os

# Create output directory if it doesn't exist
output_dir = "../../output_fig"
os.makedirs(output_dir, exist_ok=True)

metrics_list = []

for name, data in results.items():
    print(f"\n" + "="*50)
    print(f" MODEL: {name}")
    print("="*50)
    print(f"ROC AUC: {data['roc_auc']:.4f} | PR AUC: {data['pr_auc']:.4f}\n")

    # [1] Standard Threshold
    acc_std = accuracy_score(y_test, data['y_pred_std'])
    print("[1] Standard Threshold (0.5)")
    print(f"Accuracy: {acc_std:.4f}")
    print(classification_report(y_test, data['y_pred_std'], target_names=['Control', 'PD']))

    # [2] Optimal Threshold
    acc_youden = accuracy_score(y_test, data['y_pred_youden'])
    print("-" * 50)
    print(f"[2] Optimal Threshold (Youden Index = {data['optimal_threshold']:.4f})")
    print(f"Accuracy: {acc_youden:.4f}")
    print(classification_report(y_test, data['y_pred_youden'], target_names=['Control', 'PD']))

    # Append metrics for CSV export
    metrics_list.append({
        "Model": name,
        "Threshold_Type": "Standard (0.5)",
        "Threshold_Value": 0.5,
        "Accuracy": acc_std,
        "ROC_AUC": data['roc_auc'],
        "PR_AUC": data['pr_auc']
    })
    metrics_list.append({
        "Model": name,
        "Threshold_Type": "Optimal (Youden)",
        "Threshold_Value": data['optimal_threshold'],
        "Accuracy": acc_youden,
        "ROC_AUC": data['roc_auc'],
        "PR_AUC": data['pr_auc']
    })

# Export metrics to CSV
metrics_df = pd.DataFrame(metrics_list)
csv_path = os.path.join(output_dir, "model_evaluation_metrics_PD_merged.csv")
metrics_df.to_csv(csv_path, index=False)

print(f"\n✅ Evaluation metrics successfully saved to: {csv_path}")


# In[ ]:


# =====================================================================
# 5. Confusion Matrix Visualization and Image Export
# =====================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for i, (name, data) in enumerate(results.items()):

    # --- Top Row: Standard Threshold (0.5) ---
    cm_std = confusion_matrix(y_test, data['y_pred_std'])
    df_cm_std = pd.DataFrame(np.rot90(cm_std, 2), index=["actual_PD", "actual_Control"], columns=["predict_PD", "predict_Control"])

    sns.heatmap(df_cm_std, annot=True, fmt="g", cmap='Blues', ax=axes[0, i], cbar=False)
    axes[0, i].set_title(f"{name}\nStandard Threshold (0.5)")
    axes[0, i].set_yticklabels(axes[0, i].get_yticklabels(), va='center')

    # --- Bottom Row: Optimal Threshold (Youden) ---
    cm_youden = confusion_matrix(y_test, data['y_pred_youden'])
    df_cm_youden = pd.DataFrame(np.rot90(cm_youden, 2), index=["actual_PD", "actual_Control"], columns=["predict_PD", "predict_Control"])

    sns.heatmap(df_cm_youden, annot=True, fmt="g", cmap='Oranges', ax=axes[1, i], cbar=False)
    axes[1, i].set_title(f"{name}\nOptimal Threshold ({data['optimal_threshold']:.4f})")
    axes[1, i].set_yticklabels(axes[1, i].get_yticklabels(), va='center')

plt.tight_layout()

# Save figure before showing
cm_fig_path = os.path.join(output_dir, "confusion_matrices_PD_merged.png")
plt.savefig(cm_fig_path, dpi=300, bbox_inches='tight')
print(f"✅ Confusion matrices figure saved to: {cm_fig_path}")

plt.show()


# In[ ]:


# =====================================================================
# 6. Combined ROC Curve Analysis and Image Export
# =====================================================================
plt.figure(figsize=(8, 7))

# Define colors for each model
colors = {'Random Forest': 'darkorange', 'LightGBM': 'red', 'XGBoost': 'blue'}

for name, data in results.items():
    plt.plot(
        data["fpr"], data["tpr"], 
        color=colors[name], lw=2, 
        label=f'{name} (AUC = {data["roc_auc"]:.4f})'
    )

# Plot random guess line
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')

# Formatting the plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Combined ROC Curve for PD Classification (TPM + GO data)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# Save figure before showing
roc_fig_path = os.path.join(output_dir, "combined_roc_curve_PD_merged.png")
plt.savefig(roc_fig_path, dpi=300, bbox_inches='tight')
print(f"✅ Combined ROC curve figure saved to: {roc_fig_path}")

plt.show()

