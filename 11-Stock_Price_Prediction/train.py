"""
Stock Price Prediction using LSTM Neural Network
Predicts next-day closing prices for a stock using historical data.
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
TICKER = "AAPL"
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
SEQ_LENGTH = 60
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return torch.FloatTensor(x), torch.FloatTensor(y).view(-1)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def fetch_data(ticker, start, end):
    print(f"Downloading {ticker} data from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError("No data downloaded. Check internet connection or ticker.")
    df = df[["Close"]].dropna()
    print(f"Downloaded {len(df)} trading days.")
    return df


def prepare_data(df, seq_length, train_ratio=0.8):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df.values)

    train_size = int(len(scaled) * train_ratio)
    train_data = scaled[:train_size]
    test_data = scaled[train_size - seq_length :]

    train_dataset = StockDataset(train_data, seq_length)
    test_dataset = StockDataset(test_data, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, scaler, train_size


def train_model(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    return losses


def evaluate_model(model, test_loader, scaler, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)
    print(f"\nTest Metrics:")
    print(f"  MSE  : {mse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    return predictions, actuals


def plot_results(losses, predictions, actuals, ticker):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(losses, color="steelblue", linewidth=2)
    axes[0].set_title("Training Loss (MSE)", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    # Predictions vs Actual
    axes[1].plot(actuals, label="Actual", color="darkgreen", linewidth=1.5, alpha=0.8)
    axes[1].plot(predictions, label="Predicted", color="crimson", linewidth=1.5, alpha=0.8)
    axes[1].set_title(f"{ticker} Stock Price Prediction", fontsize=14)
    axes[1].set_xlabel("Time Steps (Test Set)")
    axes[1].set_ylabel("Price (USD)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "prediction_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")

    # Additional close-up plot
    fig, ax = plt.subplots(figsize=(10, 5))
    n = min(100, len(actuals))
    ax.plot(actuals[-n:], label="Actual", color="darkgreen", marker="o", markersize=3)
    ax.plot(predictions[-n:], label="Predicted", color="crimson", marker="x", markersize=3)
    ax.set_title(f"{ticker} - Last {n} Predictions vs Actual", fontsize=14)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path2 = os.path.join(OUTPUT_DIR, "closeup_predictions.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path2}")


def main():
    print("=" * 60)
    print("Stock Price Prediction with LSTM Neural Network")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = fetch_data(TICKER, START_DATE, END_DATE)
    train_loader, test_loader, scaler, train_size = prepare_data(df, SEQ_LENGTH)

    model = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nTraining LSTM model...")
    losses = train_model(model, train_loader, criterion, optimizer, device, EPOCHS)

    print("\nEvaluating on test set...")
    predictions, actuals = evaluate_model(model, test_loader, scaler, device)

    plot_results(losses, predictions, actuals, TICKER)

    # Save model
    model_path = os.path.join(OUTPUT_DIR, "lstm_stock_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

    print("\nDone! Check the 'outputs' folder for results and plots.")


if __name__ == "__main__":
    main()
