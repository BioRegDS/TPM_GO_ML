# TPM GO ML

<p align="center">
  <img width="960" height="672" alt="GA_parkinson_github" src="https://github.com/user-attachments/assets/11ef86ff-b2e1-4480-bfc2-d597d04a8486" />
</p>

## Overview
Short description of the study.

## Paper
- Title: Accurate and Explainable AI for Neurodegenerative Diseases: A Novel Feature Engineering Approach Using Gene Ontology from Gene Expression Data
- Journal: xxx
- DOI: xxx

## Requirements
pip install -r requirements.txt

## Usage
1. Preprocessing
notebook/data_processing/data_processing_PD.ipynb
notebook/data_processing/GO_count2_PD.R

3. Training
notebook/model_tuning/optuna_tpm_PD.ipynb
notebook/model_tuning/optuna_go_PD.ipynb
notebook/model_tuning/optuna_merged_PD.ipynb

5. Evaluation
notebook/model_evaluation/eval_tpm_PD.ipynb
notebook/model_evaluation/eval_go_PD.ipynb
notebook/model_evaluation/eval_merged_PD.ipynb
notebook/model_evaluation/shap_tpm_PD.ipynb
notebook/model_evaluation/shap_go_PD.ipynb
notebook/model_evaluation/shap_merged_PD.ipynb

## Data Availability
The datasets utilized in this study were obtained from the following sources:

Parkinson’s Disease (PD) Dataset: Gene expression profiles and patient clinical data were retrieved from the GitHub repository (https://github.com/sssSSLp/PD_2021.git).

Amyotrophic Lateral Sclerosis (ALS) Dataset: Gene expression and patient data were sourced from the NCBI Gene Expression Omnibus (GEO) repository under accession number GSE234297.

## Author
Hayato Nakahara, Hiroaki Iwata
Department of Biological Regulation, Faculty of Medicine, Tottori University, 86 Nishi-cho, Yonago 683-8503, Japan
