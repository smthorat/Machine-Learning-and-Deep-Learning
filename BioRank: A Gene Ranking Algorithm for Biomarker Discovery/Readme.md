# BioRank: Gene Ranking Algorithm for Biomarker Discovery

## 📌 Overview
**BioRank** is a simple yet effective gene ranking algorithm designed for biomarker discovery from gene expression data. It ranks genes based on a custom score that integrates **statistical significance (p-value)** and **magnitude of expression change (log₂ fold change)**.

## 🚀 Features
- **Custom Ranking Algorithm**: Prioritizes genes using a combination of statistical significance and effect size.
- **Differential Expression Analysis**: Uses t-tests to assess gene expression changes.
- **Scalability**: Works on datasets ranging from 1,000 to 10,000+ genes.
- **Classifier Validation**: Uses logistic regression to assess biomarker effectiveness.

## 📊 Methodology
1. **Data Preprocessing**:
   - Load gene expression data (counts or normalized values).
   - Apply log₂ transformation for normalization.

2. **Differential Expression Analysis**:
   - Perform t-tests between conditions (e.g., Normoxia vs. Hypoxia).
   - Compute p-values and log₂ fold changes.

3. **Custom Gene Ranking**:
   - Compute a **custom score** using:
     ```
     Score = -log10(p-value) * |log₂ fold change|
     ```
   - Higher scores indicate stronger biomarker potential.

4. **Visualization**:
   - **Bar Plot**: Displays top-ranked genes.
   - **Volcano Plot**: Shows significance vs. effect size.

5. **Validation with Classification**:
   - Select top-ranked genes as features.
   - Train a **logistic regression model** to test classification accuracy.

## 📂 How to Use
1. Prepare a **gene expression dataset** (CSV format).
2. Define **sample groups** (e.g., control vs. disease).
3. Run the algorithm to compute **gene rankings**.
4. (Optional) Use the ranked genes for **classification analysis**.

## 🔬 Applications
- **Biomarker Discovery**: Identify genes that differentiate conditions.
- **Feature Selection**: Select relevant genes for downstream analysis.
- **Machine Learning Pipelines**: Improve model performance with ranked genes.

## 📌 Future Enhancements
- Support for **real-world datasets** from GEO.
- Integration with **machine learning models** for improved predictions.
- More advanced **statistical weighting** for gene selection.

## 📝 Author
Developed as an **algorithm development project** in **bioinformatics**.