# 14 - Portfolio Return Prediction & Optimization with LSTM

## Overview
Uses a multi-asset LSTM neural network to predict next-day returns for a basket of stocks (AAPL, MSFT, GOOGL, AMZN, JPM). Based on predicted returns, it suggests a simple allocation and compares cumulative performance against equal-weight portfolio.

## Features
- Real historical data via Yahoo Finance
- Multivariate LSTM (sequences of returns across assets)
- Predicted-return weighted portfolio vs Equal Weight
- Correlation heatmap of assets
- Metrics: MSE, R²

## Project Structure
```
14-Portfolio_Optimization/
├── train.py
├── requirements.txt
├── README.md
└── outputs/
    ├── portfolio_results.png
    ├── correlation_heatmap.png
    └── portfolio_lstm.pth
```

## How to Run
```bash
pip install -r requirements.txt
python train.py
```

## Notes
- Requires internet to download stock data.
- You can change the `TICKERS` list to any set of liquid assets.
- This is a research-style demo; real portfolio optimization needs risk models, transaction costs, constraints, etc.
