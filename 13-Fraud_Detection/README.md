# 13 - Credit Card Fraud Detection with Neural Network

## Overview
Detects fraudulent credit card transactions using a deep feed-forward neural network. The dataset is highly imbalanced (~1.5% fraud), so the training pipeline uses **WeightedRandomSampler** to balance classes during training.

## Features
- Synthetic transaction data with realistic feature distributions
- Features: amount, hour of day, distance from home, balances, merchant risk score, online flag
- Deep NN: 128 → 64 → 32 → 1 with BatchNorm & Dropout
- Handles class imbalance
- Metrics: Precision, Recall, F1, Average Precision (PR-AUC)
- Plots: Loss curve, Confusion Matrix, Precision-Recall curve

## Project Structure
```
13-Fraud_Detection/
├── train.py
├── requirements.txt
├── README.md
└── outputs/
    ├── fraud_detection_results.png
    ├── sample_transactions.csv
    └── fraud_model.pth
```

## How to Run
```bash
pip install -r requirements.txt
python train.py
```

## Notes
- Fully offline after package installation.
- Fraud rate is intentionally low to simulate real-world conditions.
- You can adjust `fraud_ratio` and `n_samples` in the code.
