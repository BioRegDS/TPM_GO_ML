# TPM GO ML

<p align="center">
  <img width="960" height="672" alt="GA_parkinson4github" src="https://github.com/user-attachments/assets/ef500202-8f06-43f3-9cdd-0506940eda92" />
</p>

## Overview
Short description of the study.

## Paper
- Title: Accurate and Explainable AI for Neurodegenerative Diseases: A Novel Feature Engineering Approach Using Gene Ontology from Gene Expression Data
- Journal: xxx
- DOI: xxx

## Requirements
Recommended (Conda)
```bash
conda env create -f requirements.txt
conda activate go_ml
```
Alternative (pip only)
```bash
pip install -r requirements.txt
```

## Usage
Make scripts executable:
```bash
chmod +x *.sh
```
Run script for each disease datasets
・Parkinson's Disease (PD) dataset
```bash
./run_pipeline_PD.sh
```
・Amyotrophic Lateral Sclerosis (ALS) dataset
```bash
./run_pipeline_ALS.sh
```

## Data Availability
The datasets utilized in this study were obtained from the following sources:

Parkinson’s Disease (PD) Dataset: Gene expression profiles and patient clinical data were retrieved from the GitHub repository (https://github.com/sssSSLp/PD_2021.git).

Amyotrophic Lateral Sclerosis (ALS) Dataset: Gene expression and patient data were sourced from the NCBI Gene Expression Omnibus (GEO) repository under accession number GSE234297.

A small subset is included in `data_folder` for demonstration.

## Author
Hayato Nakahara, Hiroaki Iwata

Department of Biological Regulation, Faculty of Medicine, Tottori University, 86 Nishi-cho, Yonago 683-8503, Japan
