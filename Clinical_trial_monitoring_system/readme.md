# Clinical Trial Monitoring System

![Clinical Trial Dashboard](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![SQLite](https://img.shields.io/badge/Database-SQLite-blue)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A comprehensive dashboard application built with Streamlit for monitoring and analyzing clinical trial data with a focus on biomarker tracking, adverse event monitoring, patient exploration, and dropout risk prediction.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technical Architecture](#technical-architecture)
- [Data Schema](#data-schema)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Machine Learning Models](#machine-learning-models)
- [Known Issues and Solutions](#known-issues-and-solutions)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

The Clinical Trial Monitoring System is a powerful web-based dashboard designed for clinical researchers, data scientists, and medical professionals to monitor ongoing clinical trials. It provides real-time analytics, visualization, and risk prediction capabilities to help improve trial outcomes and patient care.

The system integrates multi-omics data (proteomics and genomics), clinical observations, and adverse event tracking to provide a holistic view of trial progress. It features machine learning-based risk assessment to identify patients at high risk of dropping out, allowing for proactive intervention.

## ✨ Features

### 📊 Overview Dashboard
- Comprehensive trial summary with key metrics
- Patient demographics and clinical characteristics visualization
- Summary of biomarker distribution and adverse events

### 🧬 Biomarker Tracker
- Dynamic protein expression and copy number variation (CNV) analysis
- Statistical summary of selected biomarkers
- Distribution visualization by demographic and clinical factors
- Multi-biomarker correlation analysis

### ⚠️ Adverse Event Monitor
- Temporal tracking of adverse events
- Severity distribution analysis
- Demographic and clinical correlation with adverse events
- Biomarker-adverse event relationship analysis

### 👤 Patient Explorer
- Detailed individual patient profiles
- Clinical status visualization with severity scoring
- Timeline of adverse events
- Comparison of patient biomarker data to population averages

### 🔮 Dropout Risk Analysis
- Machine learning-based prediction of patient dropout risk
- Risk factor identification and visualization
- High-risk patient cohort analysis
- Intervention recommendation system

## 🏗️ Technical Architecture

The system is built using:

- **Frontend**: Streamlit for interactive web interface
- **Backend**: Python for data processing and analysis
- **Database**: SQLite for data storage
- **Visualization**: Plotly and Plotly Express for interactive charts
- **Machine Learning**: Scikit-learn for dropout risk prediction
- **Statistical Analysis**: Pandas and NumPy for data manipulation

## 🗄️ Data Schema

The application uses a SQLite database with the following tables:

1. **patients** - Core patient information
   - patient_id (primary key)
   - age
   - sex
   - histologic_grade
   - path_stage_pt (tumor stage)
   - path_stage_pn (node stage)
   - tumor_size_cm
   - tp53_mutation
   - pik3ca_mutation
   - os_days (overall survival days)
   - os_event (mortality event flag)

2. **proteomics** - Protein expression data
   - id (primary key)
   - patient_id (foreign key)
   - protein_id
   - expression_level
   - collection_date

3. **genomics** - Copy number variation data
   - id (primary key)
   - patient_id (foreign key)
   - gene_id
   - copy_number_variation
   - collection_date

4. **adverse_events** - Adverse event records
   - id (primary key)
   - patient_id (foreign key)
   - event_type
   - event_grade (severity 1-5)
   - event_date
   - related_to_treatment

## 🚀 Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/clinical-trial-monitor.git
   cd clinical-trial-monitor
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a requirements.txt file with the following dependencies:
   ```
   streamlit>=1.10.0
   pandas>=1.3.0
   numpy>=1.20.0
   plotly>=5.3.0
   scikit-learn>=1.0.0
   matplotlib>=3.4.0
   seaborn>=0.11.0
   ```

5. Run the application:
   ```bash
   streamlit run clinical_trial_dashboard.py
   ```

6. Open your browser and navigate to `http://localhost:8501`

## 💻 Usage

### Filtering Data
Use the sidebar filters to select specific patient cohorts based on:
- Age range
- Sex
- Tumor stage

### Navigating Between Pages
Use the sidebar navigation to switch between the five main dashboard pages:
- Overview
- Biomarker Tracker
- Adverse Event Monitor
- Patient Explorer
- Dropout Risk Analysis

### Analyzing Biomarkers
1. Select the "Biomarker Tracker" page
2. Choose between "Protein Expression" or "Copy Number Variation" tabs
3. Select a specific biomarker from the dropdown menu
4. View statistical summaries and visualizations
5. For correlation analysis, select multiple proteins and genes

### Exploring Patient Data
1. Navigate to the "Patient Explorer" page
2. Select a patient ID from the dropdown
3. View patient details, clinical status, and disease severity
4. Explore adverse events, protein expression, and CNV data for the patient
5. Compare patient values to population averages

### Analyzing Dropout Risk
1. Go to the "Dropout Risk Analysis" page
2. View risk distribution and summary statistics
3. Examine risk factors and their impact
4. Identify high-risk patients and review intervention recommendations

## 🤖 Machine Learning Models

The dashboard incorporates a Random Forest classifier to predict patient dropout risk based on:
- Demographic features (age, sex)
- Clinical characteristics (tumor stage, size)
- Genetic mutations (TP53, PIK3CA)

The model is either loaded from a saved file (`dropout_model.pkl`) or dynamically created with sample data if the file is not available.

## 🐛 Known Issues and Solutions

The application had several technical challenges that have been resolved:

1. **SQLite Variance Calculation**: 
   - **Issue**: SQLite doesn't allow referencing aliases in the same SELECT statement.
   - **Error**: `pandas.errors.DatabaseError: Execution failed on sql: no such column: avg_expression`
   - **Solution**: Modified the variance calculation to use `AVG(value * value) - AVG(value) * AVG(value)` formula.

2. **Plotly Timeline Visualization**: 
   - **Issue**: Invalid marker property in timeline plots.
   - **Error**: `ValueError: Invalid property specified for object of type plotly.graph_objs.bar.Marker: 'size'`
   - **Solution**: Used `marker=dict(size=15)` instead of `marker_size=15`.

3. **Adverse Event Visualization**: 
   - **Issue**: Data length mismatch in events by type over time.
   - **Error**: `ValueError: All arguments should have the same length.`
   - **Solution**: Created proper aggregated dataframe before visualization.

4. **Machine Learning Feature Name Mismatch**: 
   - **Issue**: Feature names during prediction didn't match those during training.
   - **Error**: `ValueError: The feature names should match those that were passed during fit.`
   - **Solution**: Added feature name compatibility check and dynamic model recreation when necessary.

## 🔮 Future Enhancements

Planned features for future releases:

1. **Advanced Analytics**
   - Survival analysis with Kaplan-Meier curves
   - Multivariate regression modeling for outcome prediction
   - Time-series analysis for biomarker trends

2. **User Interface Improvements**
   - Customizable dashboards with drag-and-drop widgets
   - Report generation and export functionality
   - Dark mode and accessibility features

3. **Technical Enhancements**
   - Migration to a more robust database (PostgreSQL)
   - RESTful API for data integration with other systems
   - Containerization with Docker for easier deployment

4. **Data Capabilities**
   - Integration with electronic health records
   - Image data analysis (radiology, histopathology)
   - Natural language processing for clinical notes

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License
