# =============================================================================
# Functional Profiling: Calculating GO-ratio for ALS Down-regulated Data
# =============================================================================

# --- Part 1: Environment Setup ---

# 1. Define required packages
# CRAN packages for data manipulation
cran_packages <- c("dplyr")
# Bioconductor packages for genomic annotation and database interface
bioc_packages <- c("org.Hs.eg.db", "AnnotationDbi")

# 2. Install BiocManager if not already present
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cran.r-project.org")
}

# 3. Check and install missing CRAN packages
new_cran_packages <- cran_packages[!cran_packages %in% installed.packages()[, "Package"]]
if (length(new_cran_packages) > 0) {
    install.packages(new_cran_packages, repos = "https://cran.r-project.org")
}

# 4. Check and install missing Bioconductor packages
new_bioc_packages <- bioc_packages[!bioc_packages %in% installed.packages()[, "Package"]]
if (length(new_bioc_packages) > 0) {
    BiocManager::install(new_bioc_packages, update = FALSE)
}

# 5. Load all libraries
print("Loading the following packages:")
all_packages <- c(cran_packages, bioc_packages)
print(all_packages)
lapply(all_packages, library, character.only = TRUE)

print("✅ Package preparation complete.")


# --- Part 2: Data Analysis Pipeline ---

# 1. Import Study Results
# Load the count matrix of significant GO terms identified in the ALS down-regulated set
go_result_ALS <- read.csv("output/study_in_count_ALS.csv")

# Identify sample columns and the list of GO terms
sample_names_ALS <- setdiff(colnames(go_result_ALS), "X")
go_list_ALS <- go_result_ALS$X

# 2. Retrieve Population Background (Gene-to-term Mapping)
# Use org.Hs.eg.db to fetch all Entrez IDs associated with each GO term in the list.
# This serves as the 'Population Count' for normalization.
results_direct_ALS <- AnnotationDbi::select(org.Hs.eg.db,
                                           keys = go_list_ALS,
                                           columns = c("GO","ENTREZID"),
                                           keytype = "GO")

# 3. Summarize Population Gene Counts
# Calculate the number of unique genes mapped to each GO ID in the human background
count_summary_ALS <- results_direct_ALS %>%
  na.omit() %>% 
  group_by(GO) %>%
  summarize(Gene_Count_ALS = n_distinct(ENTREZID))

# 4. Compute GO-ratio
# Normalize the study count by dividing it by the population count.
# This ratio accounts for the size bias of individual GO terms.
go_ration_ALS <- left_join(go_result_ALS, count_summary_ALS, by=c("X"="GO")) %>%
  mutate(across(all_of(sample_names_ALS), ~ . / Gene_Count_ALS))

# 5. Export Standardized Feature Matrix
write.csv(go_ration_ALS, "output/go_ratio_ALS.csv", row.names = FALSE)

print("✅ Analysis complete: go_ratio_ALS.csv has been generated.")