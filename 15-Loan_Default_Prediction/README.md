# 15 - Loan Default Prediction with Deep Neural Network

## Overview
Predicts whether a personal loan applicant will default using a multi-layer feed-forward neural network. Features include demographic, financial, and loan-specific variables (credit score, DTI, interest rate, home ownership, purpose, etc.).

## Features
- Synthetic loan dataset (10,000 samples)
- Categorical encoding + Standard scaling
- Deep NN with BatchNorm and Dropout
- Multi-class style output (CrossEntropy) for binary problem
- Full training history (loss + accuracy)
- Confusion Matrix + ROC Curve

## Project Structure
```
15-Loan_Default_Prediction/
├── train.py
├── requirements.txt
├── README.md
└── outputs/
    ├── loan_default_results.png
    ├── sample_loans.csv
    └── loan_default_model.pth
```

## How to Run
```bash
pip install -r requirements.txt
python train.py
```

## Notes
- Completely offline after installing packages.
- Default rate is realistic (~15-25% depending on random seed).
- Easy to extend with real LendingClub-style datasets.
