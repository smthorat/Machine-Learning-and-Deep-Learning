# Install TCGAbiolinks package if not already installed
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("TCGAbiolinks")

# Load the TCGAbiolinks package
library(TCGAbiolinks)
library(limma)
library(edgeR)

# Now you can use GDCquery and related functions
query_TCGA <- GDCquery(
  project = "TCGA-PRAD",
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification",
  experimental.strategy = "RNA-Seq",
  workflow.type = "STAR - Counts"  # Specify the workflow type
)

# Download data
GDCdownload(query = query_TCGA, method = "api", files.per.chunk = 100)

# Prepare data as a SummarizedExperiment object
tcga_data <- GDCprepare(query = query_TCGA, summarizedExperiment = TRUE)

# Check the class of the object
class(tcga_data)

# Save data to .rds file
saveRDS(tcga_data, file = "TCGA_PRAD.rds")

###########Data Preprocessing###############

# Use the full data, make a working copy
sedf <- tcga_data

# Filter out only "Metastatic" samples (keep everything else)
sedf <- sedf[, sedf@colData@listData$sample_type != "Metastatic"]

# Extract gene IDs and sample names
geneslist  <- sedf@rowRanges$gene_id
samplelist <- sedf@colData@listData$sample

# Get the TPM expression matrix
expr <- sedf@assays@data@listData$tpm_unstrand

# Assign row and column names for clarity
rownames(expr) <- geneslist
colnames(expr) <- samplelist

# Filter genes: Remove high-expression outliers
drop <- apply(expr, 1, min) > 2000
expr_filtered <- expr[!drop, ]

# Filter genes: Remove low-expression genes
drop2 <- apply(expr_filtered, 1, max) < 100
expr_filtered <- expr_filtered[!drop2, ]

# Check resulting dimensions
dim(expr_filtered)

# Optional: Clean memory
rm(expr)
gc()

#######Visualized sample relationship#########
library(edgeR)

# expr_filtered should be a matrix of genes x samples
dge <- DGEList(counts = expr_filtered)

sample_types <- sedf@colData@listData$sample_type  # same order as expr_filtered columns
color_palette <- rainbow(length(unique(sample_types)))
names(color_palette) <- unique(sample_types)
sample_colors <- color_palette[sample_types]

plotMDS(dge, col = sample_colors, cex = 0.8)
legend("topright", legend = names(color_palette), fill = color_palette, cex = 0.8)

############## Min-Max + Log2 Normalization of the expression data ################
normalize <- function(x) {
  # Avoid division by zero
  if(max(x) == min(x)) {
    return(rep(0, length(x)))
  } else {
    return((x - min(x)) / (max(x) - min(x)))
  }
}

# Apply normalization across genes (rows)
expr_norm <- t(apply(expr_filtered, 1, normalize))

# Add small constant to avoid log(0) and then apply log2 transformation
log2_expr <- log2(expr_norm + 0.001)

# Verify dimensions match original data
dim(log2_expr)
head(log2_expr[, 1:5])

# Create a heatmap to visualize normalized data (optional)
if(requireNamespace("pheatmap", quietly = TRUE)) {
  library(pheatmap)
  # Select top variable genes for visualization
  var_genes <- apply(log2_expr, 1, var)
  top_var_genes <- names(sort(var_genes, decreasing = TRUE))[1:100]
  
  # Create annotation for columns (samples)
  sample_annotation <- data.frame(
    SampleType = sedf@colData@listData$sample_type
  )
  rownames(sample_annotation) <- colnames(log2_expr)
  
  # Plot heatmap
  pheatmap(log2_expr[top_var_genes, ], 
           annotation_col = sample_annotation,
           show_rownames = FALSE,
           main = "Top 100 variable genes (Min-Max + Log2 normalized)")
}



train_data <- unlist(t(expr_filtered))
train_data <- log2(train_data + 0.5)
dim(train_data) <- dim(t(expr_filtered))  # Reset dimensions

hist(train_data[1, ])
hist(train_data[2, ])
hist(train_data[3, ])

##############PCA############

# Define train_data (assuming you want to use your normalized expression data)
train_data <- t(log2_expr)  # Transpose since PCA expects samples as rows, features as columns

# Run PCA on the log-transformed expression matrix
pca.gse <- PCA(train_data, graph = FALSE)

# Visualize PCA with individuals colored by Gleason grade
fviz_pca_ind(
  pca.gse,
  geom = "point",
  col.ind = sedf@colData@listData[["secondary_gleason_grade"]]
)

# View frequency of each Gleason grade
table(sedf@colData@listData[["secondary_gleason_grade"]])


############## Preparing training label ###########
library(keras)
train_label <- sedf@colData@listData[["secondary_gleason_grade"]]
train_label <- train_label %>% as.factor() %>% as.numeric()
train_label <- train_label - 1  # shift to start from 0 for keras

dim(train_label) <- c(dim(expr_filtered)[2], 1)
train_label <- to_categorical(train_label, num_classes = 3)

## Examine the label data 
table(sedf@colData@listData[["secondary_gleason_grade"]], useNA = "ifany")
non_na_samples <- !is.na(sedf@colData@listData[["secondary_gleason_grade"]])

expr_filtered_clean <- expr_filtered[, non_na_samples]
clean_labels <- sedf@colData@listData[["secondary_gleason_grade"]][non_na_samples]
expr_filtered_clean <- expr_filtered[, non_na_samples]
clean_labels <- sedf@colData@listData[["secondary_gleason_grade"]][non_na_samples]

clean_labels_factor <- factor(clean_labels, levels = c("Pattern 3", "Pattern 4", "Pattern 5"))
clean_labels_numeric <- as.numeric(clean_labels_factor) - 1
dim(clean_labels_numeric) <- c(length(clean_labels_numeric), 1)
clean_labels_cat <- to_categorical(clean_labels_numeric, num_classes = 3)

pheatmap(train_label,
         cluster_rows = FALSE,
         cluster_cols = FALSE,
         main = "One-Hot Encoded Labels (Gleason Grade)")


####### Constructing Neural Network #############
####### Constructing Neural Network #############
# Number of units in each layer
NArray <- c(4096, 2048, 1024, 512, 32)

# Define the model
model <- keras_model_sequential() %>%
  layer_dense(units = NArray[1], activation = "relu", input_shape = dim(train_data)[2]) %>%
  layer_dense(units = NArray[2], activation = "relu") %>%
  layer_dense(units = NArray[3], activation = "relu") %>%
  layer_dense(units = NArray[4], activation = "relu") %>%
  layer_dense(units = dim(train_label)[2], activation = "sigmoid")  # softmax for multi-class

# Compile the model
model %>% compile(
  optimizer = "sgd",  # Simple string specification
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

# Train the model
history <- model %>% fit(
  x = train_data,  # Your input data
  y = train_label, # Your output labels
  epochs = 200,
  batch_size = 32,
  validation_split = 0.2  # Use 20% of data for validation
)



# Feature selection: top 1000 most variable genes
top_genes <- order(apply(expr_filtered, 1, var), decreasing = TRUE)[1:1000]
train_data <- t(log2_expr[top_genes, ])

# Define one-hot encoded labels (already prepared earlier as `train_label`)
train_label <- to_categorical(clean_labels_numeric, num_classes = 3)

# Define model architecture
model <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = ncol(train_data)) %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 3, activation = "softmax")

# Compile model
model %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# Train model with early stopping
history <- model %>% fit(
  x = train_data,
  y = train_label,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 10))
)




NArray <- c(4096, 2048, 1024, 512)

model <- keras_model_sequential() %>%
  layer_dense(
    units = NArray[1],
    activation = "relu",
    input_shape = dim(train_data)[2]
  ) %>%
  layer_dense(units = NArray[2], activation = "relu") %>%
  layer_dense(units = NArray[3], activation = "relu") %>%
  layer_dense(units = NArray[4], activation = "relu") %>%
  layer_dense(units = dim(train_label)[2], activation = "sigmoid")  # Consider changing to "softmax" for multi-class

model %>% compile(
  optimizer = 'sgd',
  loss = "binary_crossentropy",  # Consider using "categorical_crossentropy" if using softmax
  metrics = c("accuracy")
)

history <- model %>% fit(
  x = train_data,
  y = train_label,
  epochs = 100,
  use_multiprocessing = TRUE,
  batch_size = dim(train_data)[1] / 25
  # validation_split = 0.1  # Uncomment and fix if you want validation split
)