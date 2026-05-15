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
import shap
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# Load datasets
go_data = pd.read_csv("../../output/go_ratio_ALS.csv", index_col=0)
tpm_data = pd.read_csv("../../output/tpm_data_ALS.csv", index_col=0)
bg_data = pd.read_excel("../../data_folder/ALS_sample_state.xlsx", index_col=0)

# Preprocessing
if "Gene_Count_ALS" in go_data.columns:
    go_data = go_data.drop("Gene_Count_ALS", axis=1)

# Merge GO and TPM data
merged_data = pd.concat([go_data, tpm_data], axis=0)
X = go_data.T
y = bg_data['State']

# ---------------------------------------------------------
# Feature Name Cleaning & Deduplication
# ---------------------------------------------------------
# Replace special characters with underscores
clean_cols = X.columns.astype(str).str.replace(r'[^A-Za-z0-9_]', '_', regex=True)

# Make duplicated column names unique (_1, _2)
cols_series = pd.Series(clean_cols)
dup_counts = cols_series.groupby(cols_series).cumcount()
X.columns = np.where(dup_counts > 0, cols_series + '_' + dup_counts.astype(str), cols_series)

# Encode target: 'ALS' -> 1, 'CL' -> 0
y_encoded = np.where(y == 'ALS', 1, 0)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=0, stratify=y_encoded
)

# Output directory setup
output_dir = "../../output_fig"
os.makedirs(output_dir, exist_ok=True)

print(f"Data ready. Features: {X_train.shape[1]}, Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")


# In[ ]:


# =====================================================================
# 2. Parameter Loading and Model Initialization
# =====================================================================
# Load parameters from JSON
with open("../../parameter/RF_GO_ALS_params-500.json", "r") as f:
    rf_params = json.load(f)
with open("../../parameter/lgbm_best_GO_ALS_params-500.json", "r") as f:
    lgbm_params = json.load(f)
with open("../../parameter/xgboost_best_GO_ALS_params-500.json", "r") as f:
    xgb_params = json.load(f)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(**rf_params, random_state=0),
    "LightGBM": lgb.LGBMClassifier(**lgbm_params, random_state=0),
    "XGBoost": xgb.XGBClassifier(**xgb_params, random_state=0)
}

# Train all models
print("Training models...")
for name, clf in models.items():
    clf.fit(X_train, y_train)
    print(f"✅ {name} trained successfully.")


# In[ ]:


# =====================================================================
# 3. SHAP Analysis: Global Feature Importance (Bar Plots)
# =====================================================================
# Dictionary to store SHAP values for later use
shap_results = {}

for name, clf in models.items():
    print(f"\nCalculating SHAP values for {name}...")

    # Initialize TreeExplainer
    explainer = shap.TreeExplainer(clf)

    # Calculate SHAP values for the test set
    # Using X_test evaluates how features drive predictions on unseen data
    shap_values_raw = explainer.shap_values(X_test)

    # Extract SHAP values for the target class (ALS = 1)
    # Random Forest usually returns a list of arrays [class_0, class_1]
    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1]
    # LightGBM/XGBoost might return a 3D array or 2D array depending on version/objective
    elif len(shap_values_raw.shape) == 3:
        shap_values = shap_values_raw[:, :, 1]
    else:
        shap_values = shap_values_raw

    shap_results[name] = shap_values

    # --- Generate and Save SHAP Bar Plot ---
    plt.figure(figsize=(10, 6))
    plt.title(f"SHAP Feature Importance (Bar) - {name}", fontsize=14, pad=20)

    # Plot top 20 features
    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=20, show=False)

    # Save figure
    fig_path = os.path.join(output_dir, f"shap_bar_{name.replace(' ', '_')}_go_ALS.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"💾 Saved: {fig_path}")


# In[ ]:


# =====================================================================
# 4. SHAP Analysis: Directional Impact (Beeswarm Plots)
# =====================================================================
for name, shap_values in shap_results.items():

    plt.figure(figsize=(10, 6))
    plt.title(f"SHAP Summary (Beeswarm) - {name}", fontsize=14, pad=20)

    # Plot top 20 features
    shap.summary_plot(shap_values, X_test, max_display=20, show=False)

    # Save figure
    fig_path = os.path.join(output_dir, f"shap_beeswarm_{name.replace(' ', '_')}_go_ALS.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"💾 Saved: {fig_path}")


# In[ ]:


# =====================================================================
# 5. Export Top Features to CSV
# =====================================================================
for name, shap_values in shap_results.items():

    # Calculate Mean Absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean_Abs_SHAP': mean_abs_shap
    })

    # Sort by importance (descending)
    importance_df = importance_df.sort_values(by='Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

    # Save Top 50 features to CSV
    csv_path = os.path.join(output_dir, f"top50_features_shap_{name.replace(' ', '_')}_go_ALS.csv")
    importance_df.head(50).to_csv(csv_path, index=False)

    print(f"\nTop 5 features for {name}:")
    print(importance_df.head(5))
    print(f"💾 Exported top 50 features to: {csv_path}")


# In[ ]:


# =====================================================================
# 6. Identify Common Important Features Across Models (Top 10)
# =====================================================================
# Define how many top features to consider for overlap analysis
top_n_for_overlap = 10

# Extract top features for each model
top_features_sets = {}
for name, shap_values in shap_results.items():
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': mean_abs_shap
    }).sort_values(by='Importance', ascending=False)

    # Display Top 10 in the console
    print(f"\nTop 10 features for {name}:")
    print(importance_df.head(top_n_for_overlap)['Feature'].tolist())

    # Store top N for overlap calculation
    top_features_sets[name] = set(importance_df.head(top_n_for_overlap)['Feature'])

# Get all unique features that appeared in any of the top N lists
all_top_features = sorted(list(set().union(*top_features_sets.values())))

# Create a comparison table (Binary matrix: 1 if present in top N, 0 otherwise)
common_features_df = pd.DataFrame({'Feature': all_top_features})

for name, feature_set in top_features_sets.items():
    common_features_df[name] = common_features_df['Feature'].apply(lambda x: 1 if x in feature_set else 0)

# Add a 'Count' column to see how many models shared the feature
model_cols = list(models.keys())
common_features_df['Count'] = common_features_df[model_cols].sum(axis=1)

# Filter: Keep only features found in at least 2 models
common_features_final = common_features_df[common_features_df['Count'] >= 2].copy()

# Sort by Count (descending) and Feature name
common_features_final = common_features_final.sort_values(by=['Count', 'Feature'], ascending=[False, True])

print(f"\n✅ Found {len(common_features_final)} features common to 2 or more models (out of top {top_n_for_overlap}).")
display(common_features_final)


# In[ ]:


# =====================================================================
# 7. Export Common Features to CSV
# =====================================================================
# Drop the 'Count' column before final export to match the requested format
# (The columns will be: Feature, Random Forest, LightGBM, XGBoost)
export_df = common_features_final.drop(columns=['Count'])

# Save to the output directory
csv_output_path = os.path.join(output_dir, "common_features_across_models_go_ALS.csv")
export_df.to_csv(csv_output_path, index=False)

print(f"✅ Common features data successfully exported to: {csv_output_path}")

# Display first few rows of the final export format
display(export_df.head())

