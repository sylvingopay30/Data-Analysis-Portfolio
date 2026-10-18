"""
Portfolio Return Prediction & Simple Optimization with Neural Network
Predicts next-period returns for a basket of assets and suggests equal-risk or predicted-return weighted allocation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
START = "2019-01-01"
END = "2024-12-31"
SEQ_LEN = 30
BATCH_SIZE = 32
EPOCHS = 40
LR = 0.001


class MultiAssetDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class PortfolioLSTM(nn.Module):
    def __init__(self, n_assets, hidden=64, num_layers=2):
        super(PortfolioLSTM, self).__init__()
        self.lstm = nn.LSTM(n_assets, hidden, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, n_assets),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def download_data(tickers, start, end):
    print(f"Downloading data for {tickers}...")
    data = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    data = data.dropna()
    print(f"Shape: {data.shape}")
    return data


def create_sequences(returns, seq_len):
    X, y = [], []
    for i in range(len(returns) - seq_len):
        X.append(returns[i : i + seq_len])
        y.append(returns[i + seq_len])
    return np.array(X), np.array(y)


def train_model(model, loader, criterion, optimizer, device, epochs):
    losses = []
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / len(loader)
        losses.append(avg)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg:.6f}")
    return losses


def plot_results(losses, y_true, y_pred, tickers, data):
    sns.set_style("whitegrid")
    n_assets = len(tickers)
    fig = plt.figure(figsize=(14, 10))

    # Loss
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(losses, color="navy")
    ax1.set_title("Training Loss (MSE)")
    ax1.set_xlabel("Epoch")
    ax1.grid(True, alpha=0.3)

    # Predicted vs Actual returns (scatter for first asset)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.4, s=15, c="teal")
    ax2.plot([y_true[:, 0].min(), y_true[:, 0].max()],
             [y_true[:, 0].min(), y_true[:, 0].max()], "r--", lw=1)
    ax2.set_title(f"Predicted vs Actual Returns ({tickers[0]})")
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.grid(True, alpha=0.3)

    # Cumulative returns comparison (last period predictions)
    ax3 = fig.add_subplot(2, 2, 3)
    # Simple equal weight vs predicted return weight
    equal_w = np.ones(n_assets) / n_assets
    # Use mean predicted return as signal for weights
    mean_pred = y_pred.mean(axis=0)
    pred_w = np.maximum(mean_pred, 0)
    if pred_w.sum() == 0:
        pred_w = equal_w
    else:
        pred_w = pred_w / pred_w.sum()

    # Simulate last 100 days portfolio
    test_rets = y_true[-100:]
    eq_port = (test_rets * equal_w).sum(axis=1)
    pred_port = (test_rets * pred_w).sum(axis=1)
    ax3.plot(np.cumsum(eq_port), label="Equal Weight", color="gray")
    ax3.plot(np.cumsum(pred_port), label="NN Predicted Weight", color="darkgreen")
    ax3.set_title("Cumulative Portfolio Return (Test Period)")
    ax3.set_xlabel("Days")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Weight comparison
    ax4 = fig.add_subplot(2, 2, 4)
    x = np.arange(n_assets)
    width = 0.35
    ax4.bar(x - width/2, equal_w, width, label="Equal", color="lightgray")
    ax4.bar(x + width/2, pred_w, width, label="NN Suggested", color="steelblue")
    ax4.set_xticks(x)
    ax4.set_xticklabels(tickers, rotation=45)
    ax4.set_title("Portfolio Weights")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "portfolio_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")

    # Correlation heatmap of assets
    fig2, ax = plt.subplots(figsize=(7, 5))
    corr = data.pct_change().dropna().corr()
    sns.heatmap(corr, annot=True, cmap="RdYlGn", center=0, ax=ax, fmt=".2f")
    ax.set_title("Asset Return Correlations")
    path2 = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path2}")


def main():
    print("=" * 60)
    print("Portfolio Return Prediction with LSTM Neural Network")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = download_data(TICKERS, START, END)
    returns = data.pct_change().dropna().values  # daily returns

    X, y = create_sequences(returns, SEQ_LEN)
    print(f"Sequences: {X.shape}, Targets: {y.shape}")

    # Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_ds = MultiAssetDataset(X_train, y_train)
    test_ds = MultiAssetDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = PortfolioLSTM(n_assets=len(TICKERS)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("\nTraining multi-asset LSTM...")
    losses = train_model(model, train_loader, criterion, optimizer, device, EPOCHS)

    # Evaluate
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            out = model(Xb).cpu().numpy()
            preds.append(out)
            actuals.append(yb.numpy())
    y_pred = np.vstack(preds)
    y_true = np.vstack(actuals)

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\nTest MSE: {mse:.6f}")
    print(f"Test R² : {r2:.4f}")

    plot_results(losses, y_true, y_pred, TICKERS, data)

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "portfolio_lstm.pth"))
    print("Model saved.")
    print("\nDone! Check outputs/ for plots.")


if __name__ == "__main__":
    main()
