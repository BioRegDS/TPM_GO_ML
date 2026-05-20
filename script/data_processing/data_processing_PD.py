#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing modules
get_ipython().system('pip install pandas')
get_ipython().system('pip install goatools')
import pandas as pd
import goatools


# In[ ]:


#loading data
bg_data = pd.read_csv("../../data_folder/background_all_small.csv",index_col=0)
rc_data = pd.read_csv("../../data_folder/readcount_all_small.csv",index_col=0)
pd_disc = pd.read_csv("../../data_folder/pd_discreption.csv",index_col=1)
pd_tran = pd.read_csv("../../data_folder/pd_transcript.csv")


# In[ ]:


#filtering genes which have no expression
rc_data = rc_data[rc_data.sum(axis=1) > 0]
rc_data = rc_data.groupby('Gene').mean()


# In[ ]:


#remaking discreption data
pd_disc = pd_disc.drop("Unnamed: 0",axis=1)
pd_tran = pd_tran.rename(columns={"hgnc_symbol" : "Gene"})
pd_tran.set_index("Gene",inplace=True)
pd_discreption = pd_tran.join(pd_disc,on="Gene")
pd_discreption.dropna(subset=["entrezgene_id"], inplace=True)
gene_id = pd_tran.entrezgene_id
gene_id= gene_id[~gene_id.index.duplicated(keep='first')]
gene_id


# In[ ]:


#getting gene id and gene length
gene_length = pd_discreption["Gene Length"]


# In[ ]:


# 1. Obtain the intersection of indices between rc_data and gene_length.
#    This list ensures that only Gene names present in both datasets are processed.
common_genes = rc_data.index.intersection(gene_length.index)

# 2. Filter rc_data to include only common Genes.
#    This step prevents KeyErrors that arise when referencing indices that exist 
#    in rc_data but are missing in the gene_length metadata.
rc_data = rc_data.loc[common_genes]

# 3. Synchronize gene_length by filtering for the same set of common Genes.
gene_length = gene_length.loc[common_genes]

# Verify the count of overlapping genes and the final dimensions of rc_data.
print(f"Number of common genes: {len(common_genes)}")
print(f"Final rc_data shape (filtered): {rc_data.shape}")


# In[ ]:


#making control-data
ctrl_list = bg_data[bg_data["Class"]=="HL"] #regarded CL as control
ctrl_list=ctrl_list.index

rc_ctrl=rc_data[ctrl_list]
rc_ctrl


# In[ ]:


# =============================================================================
# TPM (Transcripts Per Million) Calculation (Corrected Version)
# =============================================================================

# Function definitions for TPM normalization
# Note: These functions handle length and depth normalization sequentially.
gene_length = gene_length.groupby(level=0).mean()

def normalize_per_million_reads(df):
    """Normalize by sequencing depth (scaling to 1 million reads)."""
    sum_count = df.sum()
    return 10**6 * df / sum_count

def normalize_per_kilobase(df, gene_length):
    """Normalize by gene length (scaling to kilobase units)."""
    df_tmp = df.copy()
    df_tmp = (df.T * 10**3 / gene_length).T
    return df_tmp

# 1. Convert raw read counts of all samples to TPM.
# The logic ensures samples are comparable regardless of library size or gene length.
rc_all_tpm = normalize_per_kilobase(rc_data, gene_length)
rc_all_tpm = normalize_per_million_reads(rc_all_tpm)
rc_all_tpm = pd.DataFrame(rc_all_tpm)

# 2. Extract columns corresponding to the control group from the TPM DataFrame.
# (ctrl_list was previously defined in Cell 11).
rc_ctrl_tpm = rc_all_tpm[ctrl_list]

# 3. Calculate the mean TPM values for the control group.
# These values serve as the baseline for fold-change calculations.
ctrl_mean_tpm = rc_ctrl_tpm.mean(axis="columns")

# Verify results
print("Corrected Control Group Mean TPM:")
print(ctrl_mean_tpm)

# The original calculation is retained below for comparison purposes.
print("\nOriginal (Inappropriate) Control Group Mean Read Counts:")
ctrl_mean = rc_ctrl.mean(axis="columns")
print(ctrl_mean)


# In[ ]:


#saving tpm data as tpm_data.csv
rc_all_tpm.to_csv("../../output/tpm_data_PD.csv")


# In[ ]:


import numpy as np
import pandas as pd

all_samples = bg_data.index
log2fold = {}
tpm_filtered = {}

for i in all_samples:
    tpm_filtered[i] = rc_all_tpm.loc[:, i]
    tpm_filtered[i] = tpm_filtered[i][tpm_filtered[i].abs() > 0.0]

    # ctrl_mean_tpmから必要な部分を抽出
    # 元のコードではctrl_mean_tpm[i]に再代入していましたが、ここでは変数に格納します
    ctrl_subset = ctrl_mean_tpm.loc[tpm_filtered[i].index]

    # fold changeを計算
    fold_change = (tpm_filtered[i] + 10**-6) / (ctrl_subset + 10**-6)

    # データ型をfloatに変換してからlog2を計算【ここが修正点】
    log2fold[i] = np.log2(fold_change.astype(float))

log2fold = pd.DataFrame(log2fold)

log2fold


# In[ ]:


#joining gene id to log2fold data
log2fold4go = log2fold.join(gene_id,on="Gene")
log2fold4go.reset_index(inplace=True)
log2fold4go.set_index("entrezgene_id",inplace=True)
log2fold4go


# In[ ]:


#saving log2fold data as log2fold4go.csv
log2fold4go.to_csv("../../output/log2fold4go_PD.csv")


# In[ ]:


# =============================================================================
# Preparation for Gene Ontology (GO) Enrichment Analysis
# =============================================================================
import goatools

# --- Step 1: Acquire External Resources ---
# Download the latest Gene Ontology structure (OBO format)
from goatools.base import download_go_basic_obo
obo_fname = download_go_basic_obo()

# Download the NCBI Gene-to-GO association file (Annotation mapping)
from goatools.base import download_ncbi_associations
fin_gene2go = download_ncbi_associations()

# --- Step 2: Parse Ontology and Annotations ---
# Parse the OBO file to construct the Directed Acyclic Graph (DAG)
from goatools.obo_parser import GODag
obodag = GODag("go-basic.obo")

# Load and parse the NCBI gene2go file
# Taxids=[9606] limits the annotations to Homo sapiens
from __future__ import print_function
from goatools.anno.genetogo_reader import Gene2GoReader
objanno = Gene2GoReader(fin_gene2go, taxids=[9606])

# Generate a mapping of GO namespaces (BP, MF, CC) to gene associations
ns2assoc = objanno.get_ns2assc()

# Display the count of annotated human genes per namespace
for nspc, id2gos in ns2assoc.items():
    print("{NS}: {N:,} annotated human genes identified".format(NS=nspc, N=len(id2gos)))

# --- Step 3: Define Background Gene Set ---
# Load the reference population (Background) for statistical testing
# This set typically contains all detected protein-coding genes in the experiment
from genes_ncbi_9606_proteincoding import GENEID2NT as GeneID2nt_human
print(f"Background gene set size: {len(GeneID2nt_human)}")

# --- Step 4: Initialize the GO Enrichment Study Object ---
# Configure parameters for statistical enrichment testing:
# - propagate_counts: If True, child annotations are counted towards parent terms.
# - alpha: Significance threshold (0.05).
# - methods: Multiple test correction method (Benjamini-Hochberg / FDR).
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS

goeaobj = GOEnrichmentStudyNS(
    GeneID2nt_human.keys(), # Population genes
    ns2assoc,               # Gene-to-GO associations
    obodag,                 # Ontology DAG
    propagate_counts=False, # Set to True to enable True Path Rule propagation
    alpha=0.05,             # Significance level
    methods=['fdr_bh']      # Benjamini-Hochberg FDR correction
)


# In[ ]:


#calculating how many genes associated wtih each go-term
import concurrent.futures
import numpy as np
import pandas as pd
all_samples=bg_data.index
go_list = all_samples
fingerprint = {sample: {} for sample in go_list}
simple_results = []

for sample in go_list:
    before_go = log2fold4go.loc[:, [sample,"Gene"]].replace([np.inf, -np.inf], np.nan).dropna()
    before_go = before_go[(before_go[sample].abs() >= 1.5)]
    print(before_go)
    geneid2symbol = before_go["Gene"].to_dict()
    geneids_study = list(geneid2symbol.keys())
    goea_results_all = goeaobj.run_study(geneids_study)

    # ▼▼▼【変更点】'BP' (Biological Process) のみに絞り込む条件を追加 ▼▼▼
    goea_results_sig = [r for r in goea_results_all if r.p_fdr_bh < 0.05 and r.NS == 'BP']

    print(goea_results_sig)

    for result in goea_results_sig:
        simple_results.append(f"{sample} {result.GO} {result.NS} {result.enrichment} {result.name} {result.study_count}")
        fingerprint[sample][result.GO] = result.study_count


# saving study_count
finger_df = pd.DataFrame.from_dict(fingerprint, orient="index").fillna(0).astype(int)




# In[ ]:


finger_df


# In[ ]:


# =============================================================================
# Standardizing the Feature Matrix (Metadata Alignment)
# =============================================================================

# Transpose the DataFrame so that GO Terms are Rows and Samples are Columns
# (or vice-versa, depending on your preferred input format for ML models)
finger_df = finger_df.T

# Extract the full list of sample IDs from the metadata (Gold Standard)
sample_ID = bg_data.index

# Ensure all samples in the metadata are present in the feature matrix.
# If a sample had zero significantly enriched GO terms, it won't appear in finger_df.
# We must perform 'Zero-padding' for these samples to maintain data consistency.
for case_name in sample_ID:
    if case_name not in finger_df.columns:
        # Add a new column for the missing sample and fill it with zeros
        finger_df[case_name] = 0

# Sort columns to match the order in metadata (Optional but recommended)
finger_df = finger_df.reindex(columns=sample_ID)

# Export the standardized feature matrix for downstream machine learning pipelines
finger_df.to_csv("../../output/study_in_count_PD.csv")


# In[ ]:


finger_df = finger_df.T
finger_df


# In[ ]:


# =============================================================================
# 4. Metadata Alignment and Zero-padding
# =============================================================================
# To ensure the feature matrix (X) is compatible with the labels (y) in 
# downstream machine learning pipelines, we must synchronize the samples 
# in the feature matrix with the master metadata list (bg_data.index).

# Samples that had zero significantly enriched GO terms will be missing 
# from the columns of finger_df. We use 'reindex' to automatically add these 
# missing samples and populate them with zeros (Zero-padding).
finger_df = finger_df.reindex(columns=bg_data.index, fill_value=0)

# Ensure the data type remains integer after reindexing
finger_df = finger_df.astype(int)

# --- Final Validation: Check Sample Integrity ---
# Verify that the dimensions and order of the matrix perfectly match the metadata
print(f"Total samples in metadata: {len(bg_data.index)}")
print(f"Total samples in final feature matrix: {len(finger_df.columns)}")

if all(finger_df.columns == bg_data.index):
    print("✅ Success: Feature matrix columns are perfectly aligned with the metadata index.")
else:
    print("⚠️ Warning: Sample alignment mismatch detected. Please verify the sorting of indices.")

# Export the standardized functional feature matrix
# The resulting CSV will have GO IDs as rows and Sample IDs as columns.
finger_df.to_csv("../../output/standardized_go_matrix_PD.csv")


# In[ ]:


finger_df

