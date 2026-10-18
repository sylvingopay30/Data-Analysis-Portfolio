# Neural Network Projects on Finance (Folders 11-15)

Five complete, self-contained neural network projects focused on finance applications.

## Projects

| Folder | Project Name | Description | Data Source |
|--------|--------------|-------------|-------------|
| **11-Stock_Price_Prediction** | LSTM Stock Price Predictor | Predict next-day closing prices using LSTM | Real (Yahoo Finance) |
| **12-Credit_Risk_Assessment** | Credit Risk Classifier | Binary classification of loan default risk | Synthetic |
| **13-Fraud_Detection** | Credit Card Fraud Detector | Highly imbalanced fraud detection NN | Synthetic |
| **14-Portfolio_Optimization** | Multi-Asset Return Predictor | LSTM for portfolio returns + simple allocation | Real (Yahoo Finance) |
| **15-Loan_Default_Prediction** | Loan Default Predictor | Deep NN for personal loan default | Synthetic |

## How to Use Any Project
```bash
cd <project-folder>
pip install -r requirements.txt
python train.py
```

Each project will:
- Train a neural network
- Print metrics
- Save high-quality plots (PNG) in an `outputs/` folder
- Save the trained model (.pth)

## Requirements (common)
- Python 3.8+
- PyTorch, pandas, numpy, matplotlib, seaborn, scikit-learn
- yfinance (only for projects 11 and 14)

## Notes
- Projects 12, 13, 15 are fully offline.
- Projects 11 and 14 need internet once to download stock data.
- All code is designed to run cleanly on CPU in a few minutes.
- Pictures (plots) are generated automatically on every run.

Enjoy experimenting!
