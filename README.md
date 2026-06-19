# Credit Scoring XAI — Comparative Study

**Thesis:** Evaluating the Trade-off Between Predictive Accuracy and Interpretability in Credit Scoring: A Comparative Study of XAI Techniques

**Author:** TONS-BASUO TONTE SUPREME  
**Institution:** Department of Computer Science, Faculty of Science, University of Abuja  
**Year:** 2025

---

## Overview

This repository contains the complete source code for the empirical analysis conducted in the above undergraduate thesis. The study evaluates three machine learning models (Logistic Regression, Random Forest, and XGBoost) across three credit scoring datasets, applying three explainable artificial intelligence (XAI) techniques — SHAP, LIME, and Partial Dependence Plots (PDP) — to analyse the trade-off between predictive accuracy and model interpretability.

A key finding of the study is the identification of algorithmic sex discrimination in the German Credit dataset, with a counterfactual sex gap trajectory of +6.18pp (LR) → +2.68pp (RF) → +3.89pp (XGB), demonstrating that boosting re-amplifies protected-attribute bias that bagging suppresses.

---

## Datasets

The following publicly available datasets were used. They are **not included** in this repository due to size and licensing constraints. Download links are provided below.

| Dataset | Source | Task |
|---|---|---|
| Credit Score Classification (Rohan Paris) | [Kaggle](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) | Multiclass (Poor / Standard / Good) |
| Give Me Some Credit | [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit) | Binary default prediction |
| German Credit | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) | Binary credit risk |

After downloading, place each dataset in its corresponding folder before running the scripts.

---

## Repository Structure

```
credit-scoring-xai-thesis/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── dataset_1_rohan_paris/             # Scripts for Dataset 1
│   ├── lr_model_notebook.py           # Logistic Regression training and evaluation
│   ├── rf_model_notebook.py           # Random Forest training and evaluation
│   ├── credit_score_xgboost.py        # XGBoost training and evaluation
│   ├── lr_shap_notebook.py            # SHAP for LR
│   ├── rf_shap_notebook.py            # SHAP for RF
│   ├── shap_notebook.py               # SHAP for XGBoost
│   ├── lr_lime_notebook.py            # LIME for LR
│   ├── rf_lime_notebook.py            # LIME for RF
│   ├── lime_notebook.py               # LIME for XGBoost
│   ├── lr_pdp_notebook.py             # PDP for LR
│   ├── rf_pdp_notebook.py             # PDP for RF
│   └── pdp_notebook.py                # PDP for XGBoost
│
├── dataset_2_give_me_some_credit/     # Scripts for Dataset 2
│   ├── 01_load_and_clean.py
│   ├── 02_logistic_regression.py
│   ├── 03_random_forest.py
│   ├── 04_xgboost.py
│   ├── 05_shap_lr.py
│   ├── 06_shap_rf.py
│   ├── 07_shap_xgb.py
│   ├── 08_lime_lr.py
│   ├── 09_lime_rf.py
│   ├── 10_lime_xgb.py
│   ├── 11_pdp_lr.py
│   ├── 12_pdp_rf.py
│   └── 13_pdp_xgb.py
│
└── dataset_3_german_credit/           # Scripts for Dataset 3
    ├── 01_load_and_clean_german.py
    ├── 02_eda_german.py
    ├── 03_logistic_regression_german.py
    ├── 04_shap_lr_german.py
    ├── 05_pdp_lr_german.py
    ├── 06_random_forest_german.py
    ├── 07_shap_rf_german.py
    ├── 08_lime_rf_german.py
    ├── 09_pdp_rf_german.py
    ├── 10_xgboost_german.py
    ├── 11_shap_xgb_german.py
    ├── 12_lime_xgb_german.py
    └── 13_pdp_xgb_german.py
```

---

## Installation

### Requirements

- Python 3.10 or above
- pip

### Steps

1. Clone the repository:

```bash
git clone https://github.com/[YOUR-USERNAME]/credit-scoring-xai-thesis.git
cd credit-scoring-xai-thesis
```

2. Install all dependencies:

```bash
pip install -r requirements.txt
```

3. Download the datasets and place them in their respective folders as described above.

---

## Running the Scripts

All scripts are designed to be run in order, numbered 01 through 13 for Datasets 2 and 3. Each script is self-contained and can be run independently provided the dataset CSV is in the same directory.

### Example — German Credit full pipeline:

```bash
cd dataset_3_german_credit

python 01_load_and_clean_german.py
python 02_eda_german.py
python 03_logistic_regression_german.py
python 04_shap_lr_german.py
python 05_pdp_lr_german.py
python 06_random_forest_german.py
python 07_shap_rf_german.py
python 08_lime_rf_german.py
python 09_pdp_rf_german.py
python 10_xgboost_german.py
python 11_shap_xgb_german.py
python 12_lime_xgb_german.py
python 13_pdp_xgb_german.py
```

### Note on random state

All scripts use `random_state=42` throughout to ensure full reproducibility of results.

---

## Key Results Summary

| Dataset | LR ROC-AUC | RF ROC-AUC | XGB ROC-AUC |
|---|---|---|---|
| Rohan Paris (macro OvR) | 0.7665 | ~0.852 | ~0.871 |
| Give Me Some Credit | 0.8501 | 0.8540 | 0.8470 |
| German Credit | 0.6635 | 0.7667 | 0.7732 |

### LIME Surrogate R² (mean across borrower types)

| Model | Give Me Some Credit | German Credit |
|---|---|---|
| LR | 0.447 | 0.575 |
| RF | 0.273 | 0.630 |
| XGB | 0.060 | 0.326 |

### Sex Bias Trajectory — German Credit

| Model | Counterfactual Sex Gap |
|---|---|
| Logistic Regression | +6.18pp |
| Random Forest | +2.68pp |
| XGBoost | +3.89pp |

---

## Dependencies

See `requirements.txt` for the full list. Key libraries:

- scikit-learn 1.3
- xgboost 1.7
- shap 0.42
- lime 0.2.0.1
- imbalanced-learn 0.11
- pandas 2.0
- numpy 1.24
- matplotlib 3.7

---

## Reproducibility

All experiments were run on a system with:
- Intel Core i5 processor
- 16 GB RAM
- No GPU acceleration
- Python 3.10

All random states are fixed at 42. Running the scripts in order on the same dataset versions will reproduce all results reported in the thesis.

---

## Citation

If you use any part of this code or methodology, please cite:

```
TONS-BASUO, T. S. (2025). Evaluating the Trade-off Between Predictive Accuracy 
and Interpretability in Credit Scoring: A Comparative Study of XAI Techniques. 
Undergraduate thesis, Department of Computer Science, University of Abuja.
```

---

## Contact

For questions about this repository, please contact: [YOUR EMAIL ADDRESS]
