#!/bin/bash

# エラーが発生した時点でスクリプトを停止する設定
set -e

# ディレクトリの定義
# スクリプトを home ディレクトリ上で実行することを想定しています
BASE_DIR="home"
SCRIPT_DIR="${BASE_DIR}/script"
OUTPUT_DIR="${BASE_DIR}/output"

# 1. outputディレクトリの作成（念のためここでも実行します）
echo "========================================"
echo "Creating/Verifying output directory: ${OUTPUT_DIR}"
echo "========================================"
mkdir -p "${OUTPUT_DIR}"

# 2. 解析手順の実行
echo "Starting ALS dataset analysis pipeline..."

# Step 1: data_processing_ALS.py
echo "----------------------------------------"
echo "Step 1: Running data_processing_ALS.py"
python "${SCRIPT_DIR}/data_processing/data_processing_ALS.py"

# Step 2: GO_count2_ALS.R
# ※ディレクトリ構造の提示に合わせて '2' を含めています
echo "----------------------------------------"
echo "Step 2: Running GO_count2_ALS.R"
Rscript "${SCRIPT_DIR}/data_processing/GO_count2_ALS.R"

# Step 3: optuna_tpm_ALS.py
echo "----------------------------------------"
echo "Step 3: Running optuna_tpm_ALS.py"
python "${SCRIPT_DIR}/model_tuning/optuna_tpm_ALS.py"

# Step 4: optuna_go_ALS.py
echo "----------------------------------------"
echo "Step 4: Running optuna_go_ALS.py"
python "${SCRIPT_DIR}/model_tuning/optuna_go_ALS.py"

# Step 5: eval_tpm_ALS.py
echo "----------------------------------------"
echo "Step 5: Running eval_tpm_ALS.py"
python "${SCRIPT_DIR}/model_evaluation/eval_tpm_ALS.py"

# Step 6: shap_tpm_ALS.py
echo "----------------------------------------"
echo "Step 6: Running shap_tpm_ALS.py"
python "${SCRIPT_DIR}/model_evaluation/shap_tpm_ALS.py"

echo "========================================"
echo "ALS Pipeline completed successfully!"
echo "Outputs should be verified in: ${OUTPUT_DIR}"
echo "========================================"