"""
Credit Risk Assessment using a Feed-Forward Neural Network
Classifies loan applicants as high-risk or low-risk based on financial features.
Uses synthetic but realistic data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 0.001
HIDDEN1 = 64
HIDDEN2 = 32


def generate_synthetic_credit_data(n_samples=5000):
    """Generate realistic synthetic credit application data."""
    np.random.seed(42)

    age = np.random.randint(21, 70, n_samples)
    income = np.random.lognormal(mean=10.5, sigma=0.6, size=n_samples)  # ~$36k median
    debt_to_income = np.random.beta(2, 5, n_samples) * 0.8  # 0-0.8
    credit_score = np.clip(np.random.normal(680, 80, n_samples), 300, 850).astype(int)
    employment_years = np.random.exponential(5, n_samples).clip(0, 40)
    loan_amount = np.random.lognormal(mean=9.5, sigma=0.7, size=n_samples)
    num_late_payments = np.random.poisson(1.5, n_samples).clip(0, 15)
    has_mortgage = np.random.binomial(1, 0.4, n_samples)
    num_credit_cards = np.random.poisson(3, n_samples).clip(0, 12)

    # Risk score (latent)
    risk_logit = (
        -0.015 * (credit_score - 600)
        + 2.5 * debt_to_income
        + 0.4 * num_late_payments
        - 0.00002 * income
        + 0.00005 * loan_amount
        - 0.03 * employment_years
        + 0.3 * (1 - has_mortgage)
        + np.random.normal(0, 0.5, n_samples)
    )
    risk_prob = 1 / (1 + np.exp(-risk_logit))
    default = (np.random.rand(n_samples) < risk_prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": income.round(0),
        "debt_to_income_ratio": debt_to_income.round(3),
        "credit_score": credit_score,
        "employment_years": employment_years.round(1),
        "loan_amount": loan_amount.round(0),
        "num_late_payments": num_late_payments,
        "has_mortgage": has_mortgage,
        "num_credit_cards": num_credit_cards,
        "default": default,
    })
    return df


class CreditDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CreditNet(nn.Module):
    def __init__(self, input_dim):
        super(CreditNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN1),
            nn.ReLU(),
            nn.BatchNorm1d(HIDDEN1),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.BatchNorm1d(HIDDEN2),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    train_losses, val_losses = [], []
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
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = model(Xb)
                val_loss += criterion(preds, yb).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")
    return train_losses, val_losses


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            probs = model(Xb).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(yb.numpy().flatten())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_results(train_losses, val_losses, y_true, y_pred, y_prob):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Loss curves
    axes[0].plot(train_losses, label="Train", color="steelblue")
    axes[0].plot(val_losses, label="Validation", color="darkorange")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=["Low Risk", "High Risk"],
                yticklabels=["Low Risk", "High Risk"])
    axes[1].set_title("Confusion Matrix")
    axes[1].set_ylabel("Actual")
    axes[1].set_xlabel("Predicted")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    axes[2].plot(fpr, tpr, color="crimson", lw=2, label=f"AUC = {auc:.3f}")
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[2].set_title("ROC Curve")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].legend(loc="lower right")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "credit_risk_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")

    # Feature importance style plot (correlation with default)
    return auc


def main():
    print("=" * 60)
    print("Credit Risk Assessment with Neural Network")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nGenerating synthetic credit data...")
    df = generate_synthetic_credit_data(6000)
    print(f"Dataset shape: {df.shape}")
    print(f"Default rate: {df['default'].mean()*100:.1f}%")

    # Save sample data
    df.head(200).to_csv(os.path.join(OUTPUT_DIR, "sample_credit_data.csv"), index=False)

    feature_cols = [c for c in df.columns if c != "default"]
    X = df[feature_cols].values
    y = df["default"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dataset = CreditDataset(X_scaled, y)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = CreditNet(input_dim=X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nTraining model...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, device, EPOCHS)

    print("\nEvaluating on test set...")
    y_true, y_pred, y_prob = evaluate(model, test_loader, device)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Low Risk", "High Risk"]))

    auc = plot_results(train_losses, val_losses, y_true, y_pred, y_prob)
    print(f"Test ROC-AUC: {auc:.4f}")

    model_path = os.path.join(OUTPUT_DIR, "credit_risk_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")
    print("\nDone! Check the 'outputs' folder.")


if __name__ == "__main__":
    main()
