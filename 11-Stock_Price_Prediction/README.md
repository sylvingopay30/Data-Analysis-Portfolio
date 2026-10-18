# 11 - Stock Price Prediction with LSTM

## Overview
This project implements a Long Short-Term Memory (LSTM) neural network to predict the next-day closing price of a stock (default: Apple - AAPL) using historical price data.

## Features
- Downloads real historical stock data via Yahoo Finance
- Min-Max scaling of prices
- Sequence-based LSTM model (60-day lookback)
- Training with MSE loss and Adam optimizer
- Evaluation metrics: MSE, MAE, RMSE
- Visualization of training loss and predictions vs actual prices

## Project Structure
```
11-Stock_Price_Prediction/
├── train.py              # Main training & evaluation script
├── requirements.txt      # Dependencies
├── README.md             # This file
└── outputs/              # Generated plots and model (created on run)
    ├── prediction_results.png
    ├── closeup_predictions.png
    └── lstm_stock_model.pth
```

## How to Run
```bash
pip install -r requirements.txt
python train.py
```

## Requirements
- Python 3.8+
- Internet connection (to download stock data)
- PyTorch, yfinance, pandas, numpy, matplotlib, seaborn, scikit-learn

## Expected Output
- Console logs of training progress and test metrics
- Two high-quality plots saved in `outputs/`
- Trained model checkpoint

## Notes
- Change `TICKER`, `START_DATE`, `END_DATE` in `train.py` to experiment with other stocks.
- Training takes ~1-3 minutes on CPU depending on data size.
