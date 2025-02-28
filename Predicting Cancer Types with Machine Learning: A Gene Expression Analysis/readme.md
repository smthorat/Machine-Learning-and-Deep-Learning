# Predicting Cancer Types with Machine Learning: A Gene Expression Analysis

## Overview  
This project focuses on classifying different types of cancer using gene expression data. A **Random Forest Classifier** is implemented with a **One-vs-Rest (OvR) strategy** to handle the multiclass classification problem. The model is evaluated using various performance metrics, including **accuracy, precision, recall, F1-score, confusion matrix, and ROC curves**.

## Dataset  
The dataset contains gene expression data for different cancer types. The last column represents the cancer type (label), while the remaining columns are the gene expression values (features).

### Data Preprocessing  
To ensure the dataset is ready for machine learning, the following preprocessing steps are applied:  
- **Handling Missing Values**: Checking and addressing missing values in the dataset.  
- **Feature and Label Separation**: Separating feature values (`X`) from target labels (`y`).  
- **Encoding Labels**: Converting categorical cancer types into numeric values using `LabelEncoder`.  
- **Data Splitting**: Dividing the dataset into **training (80%)** and **testing (20%)** subsets.  
- **Feature Scaling**: Normalizing feature values using **MinMaxScaler** to scale values between **0 and 1**.

## Model Training  
The **Random Forest Classifier** is trained using the **One-vs-Rest (OvR) strategy**, where each class is predicted against the others. The model learns from the training data and applies the learned patterns to predict the class of unseen test samples.

## Model Evaluation  
To assess the model's performance, several evaluation metrics are used:  

- **Accuracy**: Measures overall correctness of predictions.  
- **Precision**: Indicates the proportion of correctly predicted positive cases.  
- **Recall**: Evaluates how well the model identifies actual positive cases.  
- **F1-Score**: A balance between precision and recall.  
- **Confusion Matrix**: Displays correct and incorrect classifications for each class.  
- **ROC Curve**: Plots the trade-off between sensitivity and specificity for each class.

## Results and Visualization  
### 1. **Class Distribution**  
A bar chart is used to visualize the distribution of different cancer types in the dataset.

## **Results**
### 1. **Dataset Summary**  
- **Number of Samples:** 801  
- **Number of Features:** 8000  
- **Number of Cancer Types:** 5  

### 2. **Class Distribution**  
- **BRCA**: 300 samples  
- **KIRC**: 146 samples  
- **LUAD**: 141 samples  
- **PRAD**: 136 samples  
- **COAD**: 78 samples  

### 3. **Performance Metrics**  
#### **Overall Model Performance**  
- **Accuracy:** 97.08%  
- **Precision:** 98.15%  
- **Recall:** 98.14%  
- **F1 Score:** 98.12%  

### 2. **Confusion Matrix**  
A **heatmap** of the confusion matrix is generated to show how well the model classifies each cancer type.

### 3. **ROC Curve**  
ROC curves are generated for each cancer type, illustrating the model's ability to distinguish between different classes.

          precision    recall  f1-score   support

    BRCA       0.97      1.00      0.98        60
    COAD       1.00      0.94      0.97        16
    KIRC       1.00      1.00      1.00        28
    LUAD       0.96      0.92      0.94        24
    PRAD       1.00      1.00      1.00        33

accuracy                           0.98       161

## Conclusion  
This project successfully classifies cancer types based on gene expression data using machine learning. The **Random Forest Classifier** with the **One-vs-Rest strategy** demonstrates strong performance in multiclass classification. Future improvements could include **feature selection techniques, hyperparameter tuning, and exploring deep learning models** for enhanced accuracy.

