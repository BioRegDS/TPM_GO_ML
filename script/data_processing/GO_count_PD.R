# =============================================================================
# Functional Profiling: Calculating GO-ratio (Strictly Filtering for BP)
# =============================================================================

# --- Environment Setup: Package Installation and Loading ---

# CRAN packages for data manipulation
cran_packages <- c("dplyr")
# Bioconductor packages for genomic annotation and ontology metadata
bioc_packages <- c("org.Hs.eg.db", "AnnotationDbi", "GO.db")

# Install BiocManager if not already present
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cran.r-project.org")
}

# Install missing CRAN packages
new_cran_packages <- cran_packages[!cran_packages %in% installed.packages()[, "Package"]]
if (length(new_cran_packages) > 0) {
    install.packages(new_cran_packages, repos = "https://cran.r-project.org")
}

# Install missing Bioconductor packages
new_bioc_packages <- bioc_packages[!bioc_packages %in% installed.packages()[, "Package"]]
if (length(new_bioc_packages) > 0) {
    BiocManager::install(new_bioc_packages, update = FALSE)
}

# Load all required libraries
all_packages <- c(cran_packages, bioc_packages)
lapply(all_packages, library, character.only = TRUE)

print("✅ Environment setup complete.")


# --- Data Analysis Pipeline ---

# 1. Import Data
# Load the study count matrix (extracted GO terms and their associated gene counts)
go_result_PD <- read.csv("output/study_in_count_PD.csv")
sample_names_PD <- setdiff(colnames(go_result_PD), "X")
go_list_PD <- go_result_PD$X

# --- Feature Filtering: Restricting to Biological Process (BP) Ontology ---

# Retrieve ontology information (BP, MF, CC) for each GO ID using GO.db
go_ontology_info <- AnnotationDbi::select(GO.db, 
                                          keys = go_list_PD, 
                                          columns = "ONTOLOGY", 
                                          keytype = "GOID")

# Filter for 'Biological Process' (BP) terms only
bp_go_ids <- go_ontology_info %>% 
  filter(ONTOLOGY == "BP") %>% 
  pull(GOID)

# Retain only BP-related rows in the primary dataframe
go_result_PD <- go_result_PD %>% 
  filter(X %in% bp_go_ids)

# Update the active GO list
go_list_PD <- go_result_PD$X

print(paste("Filtering complete: Reduced to", length(go_list_PD), "Biological Process (BP) terms."))

# --- Quantitative Calculation: Population Background and GO-ratio ---

# Retrieve the total number of genes (Population Count) mapped to each GO ID
# Using org.Hs.eg.db to identify unique Entrez IDs associated with each term
results_direct_PD <- AnnotationDbi::select(org.Hs.eg.db,
                                           keys = go_list_PD,
                                           columns = c("GO","ENTREZID"),
                                           keytype = "GO")

# Summarize the population background for each term
count_summary_PD <- results_direct_PD %>%
  na.omit() %>% 
  group_by(GO) %>%
  summarize(Gene_Count_PD = n_distinct(ENTREZID))

# Compute the GO-ratio: (Study Count) / (Population Background Count)
# This normalization accounts for the size bias of individual GO terms
go_ratio_PD <- left_join(go_result_PD, count_summary_PD, by=c("X"="GO")) %>%
  mutate(across(all_of(sample_names_PD), ~ . / Gene_Count_PD))

# Export the normalized feature matrix
write.csv(go_ratio_PD, "output/go_ratio_PD.csv", row.names = FALSE)

print("✅ Analysis complete: BP-specific GO-ratio matrix exported to go_ratio_PD.csv.")