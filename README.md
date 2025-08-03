# Fraud Detection System

## Overview

This project implements a fraud detection system for financial transactions using machine learning and anomaly detection techniques. It addresses the challenge of detecting fraudulent transactions in highly imbalanced datasets, while providing explainable AI outputs for investigator insights.

The system employs feature engineering, scaling, and multiple modeling approaches — including Isolation Forest, Autoencoder, Random Forest, and an ensemble voting method. It also incorporates SHAP explainability, concept drift detection, and a Streamlit-based web app for real-time risk scoring and feedback.

## Repository Contents

- `Ironclad_Dynamics_Week_13-2.ipynb` — Jupyter notebook containing the complete data preprocessing, EDA, modeling pipeline, evaluation, and explainability analysis.
- `app.py` — Streamlit application for interactive fraud prediction, concept drift monitoring, and investigator feedback.
- `Models/` — Folder containing saved models, scaler, and feature column metadata:
  - `random_forest_model.pkl`
  - `isolation_forest_model.pkl`
  - `autoencoder_model.h5`
  - `scaler.pkl`
  - `feature_columns.pkl`
- `dataset/fraudTest.csv` — Sample or processed transaction data.
- `requirements.txt` — Python package dependencies, including ML libraries, TensorFlow/Keras, Streamlit, and SHAP.

## Features

- **Data Preprocessing & Feature Engineering:**  
  Datetime feature extraction, label encoding, numerical scaling, and identifier removal.

- **Anomaly Detection Models:**  
  Isolation Forest and Autoencoder for unsupervised detection of suspicious transactions.

- **Supervised Model:**  
  Random Forest classifier trained on oversampled, balanced data.

- **Ensemble Voting:**  
  Majority vote ensemble combining predictions of all three models for improved robustness.

- **Explainability:**  
  Feature importance visualization and SHAP values for both global and local interpretation.

- **Concept Drift Detection:**  
  Monitors feature distribution shifts over time to identify evolving fraud patterns.

- **Streamlit Web App:**  
  Upload transaction CSVs, select detection models, get fraud risk scores, download results, and provide investigator feedback.

## Setup and Installation

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. **Create and Activate a Python Virtual Environment**

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Git LFS Setup (if necessary)**  
Large files like models and datasets are tracked via Git LFS. Make sure you have [Git LFS installed](https://git-lfs.github.com/) and initialized:

```bash
git lfs install
```

## How to Run

### 1. Run Jupyter Notebook (Optional)

Explore the preprocessing, modeling, evaluation, and explainability workflow interactively:

```bash
jupyter notebook Ironclad_Dynamics_Week_13-2.ipynb
```

### 2. Run the Streamlit App

Make sure all models and required files are in the proper folder (`Models/`). Then run:

```bash
streamlit run app.py
```

- Upload your transaction CSV file.
- Select a model for fraud prediction.
- View and download results.
- Monitor concept drift.
- Submit investigator feedback.

## Project Highlights

- Handles highly imbalanced fraud data using oversampling and ensemble methods.
- Combines unsupervised and supervised approaches for enhanced fraud detection.
- Provides interpretable insights through SHAP explainability.
- Offers real-time anomaly risk scoring via a user-friendly web app.
- Includes feedback integration to improve model accuracy over time.


## Contact

For questions or support, please contact:

- Sarojkumar Lal
- GitHub: [https://github.com/sarojlal](https://github.com/sarojlal)
