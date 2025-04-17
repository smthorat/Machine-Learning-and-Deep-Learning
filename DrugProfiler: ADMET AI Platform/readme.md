# ADMET AI: Drug Property Predictor

An interactive platform for predicting and visualizing ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) properties of drug candidates using machine learning.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Available-blue.svg)](https://www.docker.com/)
[![RDKit](https://img.shields.io/badge/RDKit-2022.03+-green.svg)](https://www.rdkit.org/)
[![ChemProp](https://img.shields.io/badge/ChemProp-Latest-orange.svg)](https://github.com/chemprop/chemprop)

## 🧪 Overview

ADMET AI is a comprehensive tool for drug discovery researchers to quickly assess the pharmacokinetic and toxicity profiles of potential drug candidates. Using machine learning models trained on TDC (Therapeutics Data Commons) datasets, it predicts key ADMET properties that are critical for drug development.

### Key Features

- **Interactive Web Interface**: User-friendly Streamlit dashboard for molecule input and analysis
- **Multiple Input Methods**: Single SMILES or batch processing
- **Comprehensive Predictions**: Classification and regression models for diverse ADMET properties
- **Interactive Visualizations**: Radar charts, molecular structures, and property distributions
- **Drug-likeness Assessment**: Lipinski's Rule of Five evaluation
- **Toxicity Screening**: Identification of structural alerts
- **Batch Analysis**: Process and compare multiple compounds
- **Export Options**: Download results as CSV, images, and charts

## 📋 Requirements

- Python 3.7+
- RDKit
- Streamlit
- Chemprop
- Pandas
- NumPy
- Matplotlib
- scikit-learn

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/admet_ai.git
cd admet_ai

# Create and activate a conda environment (recommended)
conda create -n admet_ai python=3.8
conda activate admet_ai

# Install dependencies
pip install -r requirements.txt

# Install RDKit (if not included in requirements)
conda install -c conda-forge rdkit

# Install Chemprop
pip install chemprop
```

## 🏃‍♂️ Running the Application

```bash
# Run the Streamlit app
streamlit run app.py
```

## 🧠 Model Training Pipeline

The machine learning models are trained using the following pipeline:

```bash
# Run the full training pipeline
bash python_run.sh
```

The pipeline includes:

1. Data preparation from TDC sources
2. Feature computation using RDKit
3. Model training with Chemprop
4. Evaluation and benchmarking
5. Visualization generation

## 📊 Data

The models are trained on data from the Therapeutics Data Commons (TDC), a comprehensive platform for machine learning in drug discovery. The datasets include:

- **ADMET Group**: Benchmark datasets for absorption, distribution, metabolism, excretion, and toxicity
- **ADMET All**: Comprehensive collection of ADMET properties
- **DrugBank Approved**: FDA-approved drugs for reference

## 🛠️ Technical Details

### Model Architecture

The platform uses a combination of:

- **Chemprop**: Message passing neural networks for molecular property prediction
- **RDKit Features**: Traditional molecular descriptors
- **Hybrid Approach**: Combining graph neural networks with molecular descriptors

### Prediction Categories

- **Absorption**: Blood-brain barrier penetration, bioavailability, Caco-2 permeability, HIA
- **Distribution**: Plasma protein binding, volume of distribution
- **Metabolism**: CYP450 enzyme interactions (1A2, 2C9, 2C19, 2D6, 3A4)
- **Excretion**: Clearance, half-life
- **Toxicity**: hERG inhibition, AMES mutagenicity, hepatotoxicity, skin sensitization

## 📚 Documentation

For more detailed information on model parameters, datasets, and performance benchmarks, see the [documentation](docs/README.md).

## 🔄 How to Use

1. **Input**: Enter a SMILES string or batch of SMILES
2. **Select Model**: Choose between classification and regression
3. **Run Prediction**: Click "Run Prediction" to generate results
4. **Analyze**: View molecular properties, ADMET profile, and interpretations
5. **Export**: Download results for further analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Therapeutics Data Commons](https://tdcommons.ai/) for providing the datasets
- [Swansonk14](https://github.com/swansonk14/admet_ai) Referring 
- [Chemprop](https://github.com/chemprop/chemprop) for the molecular property prediction models
- [RDKit](https://www.rdkit.org/) for cheminformatics functionality
- [Streamlit](https://streamlit.io/) for the web interface

## 📧 Contact

For questions or feedback, please open an issue or contact: swarajmthorat@gmail.com





















