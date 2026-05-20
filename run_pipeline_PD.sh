#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define directories
BASE_DIR="."
SCRIPT_DIR="${BASE_DIR}/script"
OUTPUT_DIR="${BASE_DIR}/output"
PARAM_DIR="${BASE_DIR}/parameter"

# 1. Create required directories
echo "========================================"
echo "Creating required directories..."
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PARAM_DIR}"
echo " - ${OUTPUT_DIR}"
echo " - ${PARAM_DIR}"
echo "========================================"

# 2. Execute the analysis pipeline
echo "Starting PD dataset analysis pipeline..."

# Step 1: data_processing
echo "----------------------------------------"
echo "Step 1: Running data_processing_PD.py"
python "${SCRIPT_DIR}/data_processing/data_processing_PD.py"

# Step 2: GO_count
echo "----------------------------------------"
echo "Step 2: Running GO_count_PD.R"
Rscript "${SCRIPT_DIR}/data_processing/GO_count_PD.R"

# Step 3-5: Optuna Tuning (tpm, go, merged)
echo "----------------------------------------"
echo "Step 3: Running optuna_tpm_PD.py"
python "${SCRIPT_DIR}/model_tuning/optuna_tpm_PD.py"

echo "----------------------------------------"
echo "Step 4: Running optuna_go_PD.py"
python "${SCRIPT_DIR}/model_tuning/optuna_go_PD.py"

echo "----------------------------------------"
echo "Step 5: Running optuna_merged_PD.py"
python "${SCRIPT_DIR}/model_tuning/optuna_merged_PD.py"

# Step 6-8: Model Evaluation (tpm, go, merged)
echo "----------------------------------------"
echo "Step 6: Running eval_tpm_PD.py"
python "${SCRIPT_DIR}/model_evaluation/eval_tpm_PD.py"

echo "----------------------------------------"
echo "Step 7: Running eval_go_PD.py"
python "${SCRIPT_DIR}/model_evaluation/eval_go_PD.py"

echo "----------------------------------------"
echo "Step 8: Running eval_merged_PD.py"
python "${SCRIPT_DIR}/model_evaluation/eval_merged_PD.py"

# Step 9-11: SHAP Analysis (tpm, go, merged)
echo "----------------------------------------"
echo "Step 9: Running shap_tpm_PD.py"
python "${SCRIPT_DIR}/model_evaluation/shap_tpm_PD.py"

echo "----------------------------------------"
echo "Step 10: Running shap_go_PD.py"
python "${SCRIPT_DIR}/model_evaluation/shap_go_PD.py"

echo "----------------------------------------"
echo "Step 11: Running shap_merged_PD.py"
python "${SCRIPT_DIR}/model_evaluation/shap_merged_PD.py"

echo "========================================"
echo "PD Pipeline completed successfully!"
echo "Outputs can be found in: ${OUTPUT_DIR}"
echo "========================================"