#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function

# In[ ]:


#importing modules
import pandas as pd
import goatools


# In[ ]:


#loading data
bg_data = pd.read_excel("data_folder/ALS_sample_state_small.xlsx",index_col=0)
rc_data = pd.read_csv("data_folder/gene_expression_small.csv",index_col=0)
ALS_disc = pd.read_csv("data_folder/ALS_gene_discreption.csv",index_col=2)


# In[ ]:


#getting gene id and gene length
gene_id = ALS_disc.index
gene_length = ALS_disc.transcript_length


# In[ ]:


#reshape rc_data
rc_data = rc_data.loc[gene_id]
rc_data = rc_data.groupby(level=0).mean()
rc_data


# In[ ]:


#making control-data
ctrl_list = bg_data[bg_data["State"]=="CL"] #regarded CL as control
ctrl_list=ctrl_list.index

rc_ctrl=rc_data[ctrl_list]
rc_ctrl


# In[ ]:


# =============================================================================
# TPM (Transcripts Per Million) Calculation (Corrected Version)
# =============================================================================

# Pre-processing: Collapse gene lengths by calculating the mean for each gene index
gene_length = gene_length.groupby(level=0).mean()

def normalize_per_million_reads(df):
    """Normalize by sequencing depth (Scaling to 1 million reads)."""
    sum_count = df.sum()
    return 10**6 * df / sum_count

def normalize_per_kilobase(df, gene_length):
    """Normalize by gene length (Scaling to kilobase units)."""
    df_tmp = df.copy()
    df_tmp = (df.T * 10**3 / gene_length).T
    return df_tmp

# 1. Convert raw read counts of all samples to TPM.
# This transformation accounts for both gene length and library size,
# allowing for robust cross-sample comparison.
rc_all_tpm = normalize_per_kilobase(rc_data, gene_length)
rc_all_tpm = normalize_per_million_reads(rc_all_tpm)
rc_all_tpm = pd.DataFrame(rc_all_tpm)

# 2. Extract control group columns from the TPM-normalized DataFrame.
# (Assumes 'ctrl_list' was defined in a preceding cell).
rc_ctrl_tpm = rc_all_tpm[ctrl_list]

# 3. Calculate the mean TPM values for the control group.
# This mean serves as the reference baseline for downstream fold-change calculations.
ctrl_mean_tpm = rc_ctrl_tpm.mean(axis="columns")

# --- Verification of results ---
print("Corrected mean TPM for the control group:")
print(ctrl_mean_tpm)

# Retaining the original calculation for diagnostic comparison
print("\nOriginal (Inappropriate) mean read counts for the control group:")
ctrl_mean = rc_ctrl.mean(axis="columns")
print(ctrl_mean)


# In[ ]:


# Create a local copy of the TPM data to avoid modifying the original dataframe
tpm_data = rc_all_tpm.copy()

# --- Step 1: Deduplicate Metadata ---
# Handle duplicate indices by taking the first occurrence to ensure unique mapping.
# This step preserves column information (such as hgnc_symbol).
ALS_disc = ALS_disc.groupby(level=0).first() 

# Select only the 'hgnc_symbol' column for index reassignment
ALS_disc = ALS_disc[['hgnc_symbol']]

# --- Step 2: Standardize Index Mapping ---
# Realign the metadata index to match the TPM-normalized dataset
ALS_disc = ALS_disc.reindex(tpm_data.index)

# Map raw identifiers to HGNC gene symbols for biological interpretability
tpm_data.index = ALS_disc.hgnc_symbol

# Remove entries that could not be mapped (NaN indices) to ensure a clean dataset
tpm_data = tpm_data[tpm_data.index.notna()]


# In[ ]:


#saving tpm data as tpm_data.csv
tpm_data.to_csv("output/tpm_data_ALS.csv")


# In[ ]:


# =============================================================================
# Differential Expression Analysis: Calculating Log2 Fold Change
# =============================================================================
import numpy as np
import pandas as pd

# 1. Initialize data structures
all_samples = bg_data.index
log2fold = {}
tpm_filtered = {}

# 2. Iterate through each sample to calculate relative expression levels
for i in all_samples:
    # Filter for expressed genes: Keep only genes with TPM > 0 for the current sample
    # This reduces computational noise and prevents log-transformation errors
    tpm_filtered[i] = rc_all_tpm.loc[:, i]
    tpm_filtered[i] = tpm_filtered[i][tpm_filtered[i] > 0]

    # Align the control baseline (mean TPM) with the gene set of the current sample
    ctrl_subset = ctrl_mean_tpm.loc[tpm_filtered[i].index]

    # Calculate Fold Change relative to the control group
    # A small pseudocount (10^-6) is added to both numerator and denominator 
    # to ensure numerical stability and handle near-zero values.
    fold_change = (tpm_filtered[i] + 10**-6) / (ctrl_subset + 10**-6)

    # Perform Log2 transformation
    # Fixed: Ensure the data type is float before transformation to prevent casting errors
    log2fold[i] = np.log2(fold_change.astype(float))

# 3. Consolidate individual sample results into a single Master DataFrame
log2fold = pd.DataFrame(log2fold)

# Display the resulting Log2 Fold Change matrix
log2fold


# In[ ]:


# =============================================================================
# Feature Annotation: Mapping Entrez IDs to HGNC Symbols
# =============================================================================

# Extract the mapping series (Entrez ID -> HGNC Symbol)
gene_list = ALS_disc.hgnc_symbol

# Create a dedicated copy for GO analysis to maintain data integrity
log2fold4go = log2fold.copy()

# Step 1: Prepare the matrix for joining
# Resetting the index ensures 'entrezgene_id' can be used as a join key
log2fold4go.reset_index(inplace=True)

# Step 2: Perform the Join operation
# Map HGNC symbols to the log2 fold change data using Entrez IDs
log2fold4go = log2fold.join(gene_list, on="entrezgene_id")

# Step 3: Re-structure the DataFrame
# Restore the primary index and ensure the HGNC symbols are correctly integrated
log2fold4go.reset_index(inplace=True)
log2fold4go.set_index("entrezgene_id", inplace=True)

# Display the final annotated matrix for verification
log2fold4go


# In[ ]:


#saving log2fold data as log2fold4go.csv
log2fold4go.to_csv("output/log2fold4go_ALS.csv")


# In[ ]:


# =============================================================================
# Preparation for Gene Ontology (GO) Enrichment Analysis
# =============================================================================
import goatools

# --- Step 1: Resource Acquisition ---
# Download the latest GO structure definitions (OBO format)
from goatools.base import download_go_basic_obo
obo_fname = download_go_basic_obo()

# Download the NCBI Gene-to-GO association file (Gene annotation mapping)
from goatools.base import download_ncbi_associations
fin_gene2go = download_ncbi_associations()

# --- Step 2: Ontology Parsing and DAG Construction ---
# Parse the OBO file to build the Directed Acyclic Graph (DAG) for GO terms
from goatools.obo_parser import GODag
obodag = GODag("go-basic.obo")

# --- Step 3: Association Mapping and Filtering ---
# Load and parse NCBI associations, filtering for Homo sapiens (Taxon ID: 9606)
from goatools.anno.genetogo_reader import Gene2GoReader
objanno = Gene2GoReader(fin_gene2go, taxids=[9606])

# Generate a dictionary mapping of namespaces (BP, MF, CC) to gene associations
ns2assoc = objanno.get_ns2assc()

# Display the count of annotated human genes identified for each namespace
for nspc, id2gos in ns2assoc.items():
    print("{NS}: {N:,} annotated human genes identified".format(NS=nspc, N=len(id2gos)))

# --- Step 4: Define Reference Population (Background) ---
# Load the protein-coding gene background for statistical comparison
from genes_ncbi_9606_proteincoding import GENEID2NT as GeneID2nt_human
print(f"Background gene set size: {len(GeneID2nt_human)}")

# --- Step 5: Statistical Framework Initialization ---
# Initialize the GO Enrichment Study object with defined statistical thresholds:
# - alpha: Significance level (0.05)
# - methods: Multiple test correction method (Benjamini-Hochberg FDR)
# - propagate_counts: Set to False for term-specific enrichment
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS

goeaobj = GOEnrichmentStudyNS(
    GeneID2nt_human.keys(), # Population genes (Background)
    ns2assoc,               # Gene-to-GO associations
    obodag,                 # Ontology DAG structure
    propagate_counts=False, # Count inheritance from child to parent terms
    alpha=0.05,             # Type I error rate
    methods=['fdr_bh'])     # False Discovery Rate (FDR) correction


# In[ ]:


# =============================================================================
# Functional Profiling: Calculating Gene Counts per GO Term
# =============================================================================
import concurrent.futures
import numpy as np
import pandas as pd

# Initialize lists and storage structures
all_samples = bg_data.index
go_list = all_samples
# 'fingerprint' will store the functional signature for each sample
fingerprint = {sample: {} for sample in go_list}
simple_results = []

for sample in go_list:
    # --- 1. Data Cleaning and Feature Selection ---
    # Extract genes with significant expression changes (|log2FC| >= 1.5).
    # HGNC symbols are kept for reference, while Entrez IDs (index) are used for calculations.
    # Handle infinities and NaNs to ensure numerical stability.
    before_go = log2fold4go.loc[:, [sample, "hgnc_symbol"]].replace([np.inf, -np.inf], np.nan).dropna()
    before_go = before_go[(before_go[sample].abs() >= 1.5)]

    # Extract the study gene set (list of Entrez IDs) for enrichment analysis
    geneids_study = before_go.index.tolist()

    # --- 2. Execute GO Enrichment Analysis (GOEA) ---
    # Perform statistical enrichment testing against the background gene set
    goea_results_all = goeaobj.run_study(geneids_study)

    # --- 3. Statistical Filtering ---
    # Strictly filter for terms that meet the following criteria:
    # - Namespace: Biological Process (BP)
    # - Significance: Adjusted p-value (FDR via Benjamini-Hochberg) < 0.05
    goea_results_sig = [r for r in goea_results_all if r.p_fdr_bh < 0.05 and r.NS == 'BP']

    # Log the progress for each sample
    print(f"--- Processing Sample: {sample} ---")
    print(f"Found {len(goea_results_sig)} significant BP GO terms.")

    # --- 4. Store Results ---
    for result in goea_results_sig:
        # Save a summary string for qualitative logging
        simple_results.append(f"{sample} {result.GO} {result.NS} {result.enrichment} {result.name} {result.study_count}")
        # Map GO IDs to their 'study_count' (abundance) to build the functional profile
        fingerprint[sample][result.GO] = result.study_count

# =============================================================================
# 5. Construct Functional Feature Matrix (GO-ratio Foundation)
# =============================================================================
# Convert the dictionary to a DataFrame (Samples as rows, GO terms as columns)
# Apply 'Zero-padding' (fillna(0)) for terms that were not significantly enriched in specific samples
finger_df = pd.DataFrame.from_dict(fingerprint, orient="index").fillna(0).astype(int)

# Transpose the matrix: GO IDs as rows, Samples as columns
finger_df = finger_df.T

print("\n✅ Functional profiling analysis complete.")


# In[ ]:


# =============================================================================
# Standardizing the Feature Matrix (Metadata Alignment)
# =============================================================================

# Extract the complete list of sample IDs from the metadata (Gold Standard)
# This ensures that all expected samples are accounted for in the final matrix.
sample_ID = bg_data.index

# Iterate through the metadata sample IDs to check for completeness in finger_df.
# Note: Samples with zero significantly enriched GO terms will be missing from 
# the finger_df columns. We must perform 'Zero-padding' to maintain consistency.
for case_name in sample_ID:
    if case_name not in finger_df.columns:
        # Add a missing sample column and initialize with zeros
        # This prevents data shape mismatch during machine learning training.
        finger_df[case_name] = 0

# (Optional) Reorder columns to perfectly match the metadata index for safety.
finger_df = finger_df.reindex(columns=sample_ID)

# Display the standardized functional feature matrix
finger_df


# In[ ]:


finger_df.to_csv("output/study_in_count_ALS.csv")

