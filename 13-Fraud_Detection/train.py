"""
Credit Card Fraud Detection using a Neural Network
Handles highly imbalanced data with class weighting and synthetic minority oversampling concepts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score

torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 256
EPOCHS = 25
LEARNING_RATE = 0.001


def generate_fraud_data(n_samples=20000, fraud_ratio=0.02):
    """Generate synthetic credit card transaction data with fraud labels."""
    np.random.seed(42)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legitimate transactions
    amount_legit = np.random.lognormal(mean=3.5, sigma=1.2, size=n_legit)
    hour_legit = np.random.normal(14, 5, n_legit) % 24
    distance_legit = np.random.exponential(5, n_legit)
    old_balance_legit = np.random.lognormal(7, 1.5, n_legit)
    new_balance_legit = old_balance_legit - amount_legit + np.random.normal(0, 50, n_legit)
    merchant_risk_legit = np.random.beta(1, 8, n_legit)
    is_online_legit = np.random.binomial(1, 0.3, n_legit)

    # Fraudulent transactions (different distributions)
    amount_fraud = np.random.lognormal(mean=5.0, sigma=1.5, size=n_fraud)
    hour_fraud = np.random.normal(2, 3, n_fraud) % 24  # more at night
    distance_fraud = np.random.exponential(50, n_fraud)  # higher distance
    old_balance_fraud = np.random.lognormal(6, 1.8, n_fraud)
    new_balance_fraud = old_balance_fraud - amount_fraud * np.random.uniform(0.8, 1.2, n_fraud)
    merchant_risk_fraud = np.random.beta(5, 2, n_fraud)  # higher risk merchants
    is_online_fraud = np.random.binomial(1, 0.7, n_fraud)

    X_legit = np.column_stack([
        amount_legit, hour_legit, distance_legit, old_balance_legit,
        new_balance_legit, merchant_risk_legit, is_online_legit
    ])
    X_fraud = np.column_stack([
        amount_fraud, hour_fraud, distance_fraud, old_balance_fraud,
        new_balance_fraud, merchant_risk_fraud, is_online_fraud
    ])

    X = np.vstack([X_legit, X_fraud])
    y = np.array([0] * n_legit + [1] * n_fraud)

    # Shuffle
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]

    columns = ["amount", "hour", "distance_from_home", "old_balance",
               "new_balance", "merchant_risk_score", "is_online"]
    df = pd.DataFrame(X, columns=columns)
    df["is_fraud"] = y
    return df


class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FraudNet(nn.Module):
    def __init__(self, input_dim):
        super(FraudNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_model(model, train_loader, criterion, optimizer, device, epochs):
    losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg = running_loss / len(train_loader)
        losses.append(avg)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg:.5f}")
    return losses


def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            probs = model(Xb).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            all_labels.extend(yb.numpy().flatten())
            all_preds.extend(preds)
            all_probs.extend(probs)
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_results(losses, y_true, y_pred, y_prob):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(losses, color="teal", lw=2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].grid(True, alpha=0.3)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="OrRd", ax=axes[1],
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    axes[1].set_title("Confusion Matrix")
    axes[1].set_ylabel("Actual")
    axes[1].set_xlabel("Predicted")

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    axes[2].plot(recall, precision, color="darkred", lw=2, label=f"AP = {ap:.3f}")
    axes[2].set_title("Precision-Recall Curve")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fraud_detection_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")
    return ap


def main():
    print("=" * 60)
    print("Credit Card Fraud Detection with Neural Network")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nGenerating synthetic transaction data...")
    df = generate_fraud_data(25000, fraud_ratio=0.015)
    print(f"Total transactions: {len(df)}")
    print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")

    df.sample(300).to_csv(os.path.join(OUTPUT_DIR, "sample_transactions.csv"), index=False)

    feature_cols = [c for c in df.columns if c != "is_fraud"]
    X = df[feature_cols].values
    y = df["is_fraud"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # Weighted sampler for imbalance
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_ds = FraudDataset(X_train, y_train)
    test_ds = FraudDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = FraudNet(input_dim=X.shape[1]).to(device)

    # Class weight for loss
    pos_weight = torch.tensor([class_counts[0] / class_counts[1]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # but model has sigmoid, so use BCE
    # Actually model ends with Sigmoid, so use BCELoss with weight
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nTraining model (with weighted sampling for imbalance)...")
    losses = train_model(model, train_loader, criterion, optimizer, device, EPOCHS)

    print("\nEvaluating on test set...")
    y_true, y_pred, y_prob = evaluate(model, test_loader, device)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"], digits=4))

    ap = plot_results(losses, y_true, y_pred, y_prob)
    print(f"Average Precision (AP): {ap:.4f}")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "fraud_model.pth"))
    print(f"Model saved to outputs/fraud_model.pth")
    print("\nDone!")


if __name__ == "__main__":
    main()
