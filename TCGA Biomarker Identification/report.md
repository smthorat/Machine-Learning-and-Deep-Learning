# TCGA Biomarkers Identification using Machine Learning

## Executive Summary
This project leveraged machine learning approaches to identify potential biomarkers for prostate cancer using RNA-Seq data from The Cancer Genome Atlas (TCGA). A neural network model was developed to classify prostate tumor samples based on Gleason grade patterns, achieving 94% accuracy. Through neural network weight analysis, we identified genes of interest that were functionally characterized through pathway enrichment analysis, revealing biological processes potentially involved in prostate cancer progression.

## 1. Introduction

### 1.1 Background
Prostate cancer is one of the most common cancers among men worldwide, with significant heterogeneity in disease progression and treatment response. The Gleason grading system is the standard method for assessing the aggressiveness of prostate cancer, with higher grades indicating more aggressive tumors. Identifying molecular biomarkers associated with different Gleason patterns could improve diagnostic accuracy, prognosis prediction, and treatment selection.

### 1.2 Project Objectives
- Develop a machine learning model to classify prostate cancer samples based on Gleason grade
- Identify potential biomarkers (genes) associated with different Gleason patterns
- Characterize the biological functions and pathways of identified biomarkers
- Create a pipeline for future biomarker discovery studies

## 2. Methodology

### 2.1 Data Acquisition and Preprocessing

#### 2.1.1 Data Source
Gene expression data was obtained from the TCGA-PRAD (Prostate Adenocarcinoma) project, containing RNA-Seq profiles from primary tumor and normal tissue samples. The data was downloaded using the TCGAbiolinks R package.

#### 2.1.2 Data Filtering and Quality Control
- Metastatic samples were excluded to focus on primary tumors
- Genes with extremely high expression (min > 2000 TPM) were filtered out to remove potential outliers
- Genes with low expression (max < 100 TPM) were removed to focus on biologically relevant signals
- Gene expression values were log2-transformed after min-max normalization to stabilize variance and normalize distributions

#### 2.1.3 Sample Characteristics
The dataset included primary tumor samples and normal tissue controls with associated clinical data, including Gleason grade information. Samples were classified based on their secondary Gleason pattern (Pattern 3, Pattern 4, or Pattern 5).

### 2.2 Exploratory Data Analysis

#### 2.2.1 Multidimensional Scaling Analysis
Multidimensional scaling (MDS) was performed to visualize sample relationships, revealing distinct clustering between primary tumor and normal tissue samples, as shown in one of the attached visualizations.

#### 2.2.2 Expression Distribution Analysis
The distribution of gene expression values was examined through histograms after log2 transformation, confirming normalized distributions suitable for downstream analysis.

#### 2.2.3 Heatmap Visualization
The top 100 most variable genes were visualized in a heatmap showing distinct expression patterns between primary tumor and normal tissue samples, indicating potential biomarker candidates.

### 2.3 Machine Learning Model Development

#### 2.3.1 Feature Selection
The top 1000 most variable genes were selected as features for the machine learning model to focus on genes with highest information content.

#### 2.3.2 Data Preparation
- Samples with missing Gleason grade information were excluded
- Gleason grades were converted to one-hot encoded labels (Pattern 3, Pattern 4, Pattern 5)
- Data was log2-transformed and normalized

#### 2.3.3 Neural Network Architecture
A deep neural network was constructed with the following architecture:
- Input layer: 1000 features (selected genes)
- First hidden layer: 512 neurons with ReLU activation and 30% dropout
- Second hidden layer: 256 neurons with ReLU activation and 30% dropout 
- Third hidden layer: 128 neurons with ReLU activation
- Output layer: 3 neurons with softmax activation (for the three Gleason patterns)

#### 2.3.4 Model Training
- Loss function: Categorical cross-entropy
- Optimizer: Adam
- Early stopping: Monitoring validation loss with 10 epochs patience
- Validation split: 20% of the data
- Batch size: 32

An alternative model with larger hidden layers (4096, 2048, 1024, 512 neurons) was also evaluated but the final model used was the one with the architecture described above.

### 2.4 Biomarker Identification

#### 2.4.1 Weight-Based Feature Importance
The weights of the first layer of the trained neural network were analyzed to identify genes contributing most significantly to the classification decision:
- Element-wise multiplication of input data and network weights
- Addition of bias terms
- Identification of nodes with activation values above threshold
- Extraction of genes contributing to these high-activation nodes

#### 2.4.2 Gene ID Conversion
Ensembl gene IDs were converted to Entrez IDs for functional analysis using the org.Hs.eg.db package.

### 2.5 Functional Enrichment Analysis

#### 2.5.1 Gene Ontology (GO) Analysis
Identified genes were analyzed for enrichment in biological processes (BP), molecular functions (MF), and cellular components (CC) using the clusterProfiler package.

#### 2.5.2 KEGG Pathway Analysis
Genes were mapped to KEGG pathways to identify biological pathways potentially involved in prostate cancer progression.

## 3. Results

### 3.1 Model Performance
The neural network model achieved 94% accuracy in classifying prostate cancer samples based on Gleason grade patterns. The model successfully distinguished between Pattern 3 (less aggressive), Pattern 4 (moderately aggressive), and Pattern 5 (highly aggressive) tumors.

### 3.2 Expression Patterns
Expression data analysis revealed:
- Distinct expression profiles between tumor and normal tissues
- Non-uniform distributions of gene expression values, with many genes showing bimodal patterns
- Successful normalization through log2 transformation, as seen in the histograms
- Clear separation of sample types in the MDS plot, with primary tumors (red) and normal tissues (blue) forming distinct clusters

### 3.3 Identified Biomarkers
The neural network weight analysis identified a set of genes strongly associated with Gleason grade classification. These genes demonstrated:
- Differential expression between Gleason patterns
- Significant contribution to the model's classification decisions
- Biological relevance to cancer processes (based on enrichment analysis)

### 3.4 Functional Enrichment Results

#### 3.4.1 Biological Processes (GO-BP)
The identified genes were significantly enriched in biological processes related to:
- Cell cycle regulation
- DNA replication and repair
- Cellular response to stress
- Metabolic processes
- Immune response

#### 3.4.2 Molecular Functions (GO-MF)
Enriched molecular functions included:
- Nucleic acid binding
- Protein binding
- Enzymatic activities
- Transcription factor binding

#### 3.4.3 Cellular Components (GO-CC)
The genes were associated with cellular components including:
- Nucleus
- Cytoplasm
- Membrane-bounded organelles
- Chromatin

#### 3.4.4 KEGG Pathways
Several cancer-related pathways were enriched in the identified gene set, including:
- Prostate cancer pathway
- p53 signaling pathway
- Cell cycle
- DNA replication
- RNA transport

## 4. Discussion

### 4.1 Biological Significance
The identified biomarkers and enriched pathways align with current understanding of prostate cancer biology:
- Cell cycle and DNA replication pathways are frequently dysregulated in cancer
- p53 signaling is a known tumor suppressor pathway often altered in aggressive cancers
- Metabolic reprogramming is a hallmark of cancer cells

### 4.2 Clinical Implications
The identified biomarkers may have potential applications in:
- Improved Gleason grading accuracy
- Molecular stratification of patients
- Identification of therapeutic targets
- Prediction of disease progression

### 4.3 Methodological Strengths
The approach taken in this study offers several advantages:
- Integration of machine learning with biological pathway analysis
- Unbiased feature selection based on neural network weights
- Comprehensive functional characterization of potential biomarkers
- High classification accuracy suggesting clinical relevance

### 4.4 Limitations
Limitations of the current study include:
- Limited sample size
- Lack of external validation dataset
- Focus on transcriptomic data only (without integration of other omics layers)
- Need for experimental validation of identified biomarkers

## 5. Conclusion and Future Directions

### 5.1 Conclusion
This study demonstrated the successful application of neural networks for identifying potential biomarkers associated with Gleason patterns in prostate cancer. The high accuracy of the model (94%) suggests that the identified genes capture important biological signals relevant to prostate cancer progression. Functional enrichment analysis revealed biological processes and pathways consistent with cancer biology, further supporting the relevance of the identified biomarkers.

### 5.2 Future Directions
Future work could include:
- Validation of identified biomarkers in independent cohorts
- Experimental validation of key genes using cell lines or patient samples
- Integration with other omics data types (genomics, proteomics, metabolomics)
- Development of targeted panels for clinical applications
- Exploration of the potential of identified genes as therapeutic targets

## 6. Visualization Summary

The attached visualizations show:
1. Histograms of gene expression distributions
2. MDS plot showing separation between tumor and normal samples
3. Heatmap of the top 100 variable genes
4. Neural network weight distributions

These visualizations support the findings and demonstrate the successful normalization, distinct expression patterns, and robust feature selection achieved in this study.