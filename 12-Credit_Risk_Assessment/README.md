# 12 - Credit Risk Assessment with Neural Network

## Overview
A feed-forward neural network that classifies loan applicants into **Low Risk** or **High Risk** of default. Uses carefully designed synthetic data that mimics real credit features (income, credit score, debt-to-income, late payments, etc.).

## Features
- Synthetic but realistic credit dataset (6,000 samples)
- Fully-connected NN with BatchNorm + Dropout
- Binary classification (default / no-default)
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Visualizations: Loss curves, Confusion Matrix, ROC Curve

## Project Structure
```
12-Credit_Risk_Assessment/
├── train.py
├── requirements.txt
├── README.md
└── outputs/
    ├── credit_risk_results.png
    ├── sample_credit_data.csv
    └── credit_risk_model.pth
```

## How to Run
```bash
pip install -r requirements.txt
python train.py
```

## Key Hyperparameters
- Hidden layers: 64 → 32
- Dropout: 0.3 / 0.2
- Loss: Binary Cross Entropy
- Optimizer: Adam (lr=0.001)
- Epochs: 40

## Notes
- No internet required after installation (data is generated on the fly).
- You can increase `n_samples` in the code for larger experiments.
