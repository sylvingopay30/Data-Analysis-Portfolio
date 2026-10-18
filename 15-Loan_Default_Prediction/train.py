"""
Loan Default Prediction using a Deep Neural Network
Predicts whether a borrower will default on a personal loan.
Synthetic data with rich borrower features.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_fscore_support)

torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 35
LR = 0.001


def generate_loan_data(n=8000):
    np.random.seed(42)

    age = np.random.randint(22, 65, n)
    income = np.random.lognormal(10.8, 0.55, n)
    loan_amount = np.random.lognormal(9.2, 0.65, n)
    interest_rate = np.random.uniform(5.0, 24.0, n)
    term_months = np.random.choice([12, 24, 36, 48, 60], n, p=[0.1, 0.2, 0.35, 0.25, 0.1])
    credit_score = np.clip(np.random.normal(700, 70, n), 300, 850).astype(int)
    dti = np.random.beta(2.2, 4.5, n) * 0.7
    employment_length = np.random.exponential(4.5, n).clip(0, 30)
    home_ownership = np.random.choice(["RENT", "OWN", "MORTGAGE"], n, p=[0.35, 0.25, 0.4])
    purpose = np.random.choice(
        ["debt_consolidation", "home_improvement", "major_purchase", "medical", "vacation"],
        n, p=[0.4, 0.2, 0.15, 0.15, 0.1]
    )
    num_open_accounts = np.random.poisson(4, n).clip(1, 15)
    delinquencies_2yrs = np.random.poisson(0.4, n).clip(0, 8)

    # Latent risk
    risk = (
        -0.012 * (credit_score - 650)
        + 3.0 * dti
        + 0.08 * interest_rate
        + 0.35 * delinquencies_2yrs
        - 0.000015 * income
        + 0.00004 * loan_amount
        - 0.04 * employment_length
        + np.where(home_ownership == "RENT", 0.25, 0)
        + np.random.normal(0, 0.6, n)
    )
    prob = 1 / (1 + np.exp(-risk))
    default = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": np.round(income, 0),
        "loan_amount": np.round(loan_amount, 0),
        "interest_rate": np.round(interest_rate, 2),
        "term_months": term_months,
        "credit_score": credit_score,
        "debt_to_income": np.round(dti, 3),
        "employment_length": np.round(employment_length, 1),
        "home_ownership": home_ownership,
        "purpose": purpose,
        "num_open_accounts": num_open_accounts,
        "delinquencies_2yrs": delinquencies_2yrs,
        "default": default,
    })
    return df


class LoanDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LoanDefaultNet(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(LoanDefaultNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.BatchNorm1d(96),
            nn.Dropout(0.35),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.BatchNorm1d(48),
            nn.Dropout(0.25),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_labels.extend(yb.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_results(history, y_true, y_pred, y_prob):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(history["train_loss"], label="Train Loss", color="steelblue")
    axes[0, 0].plot(history["val_loss"], label="Val Loss", color="darkorange")
    axes[0, 0].set_title("Loss Curves")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history["train_acc"], label="Train Acc", color="steelblue")
    axes[0, 1].plot(history["val_acc"], label="Val Acc", color="darkorange")
    axes[0, 1].set_title("Accuracy Curves")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=axes[1, 0],
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"])
    axes[1, 0].set_title("Confusion Matrix")
    axes[1, 0].set_ylabel("Actual")
    axes[1, 0].set_xlabel("Predicted")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    axes[1, 1].plot(fpr, tpr, color="purple", lw=2, label=f"AUC = {auc:.3f}")
    axes[1, 1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1, 1].set_title("ROC Curve")
    axes[1, 1].set_xlabel("FPR")
    axes[1, 1].set_ylabel("TPR")
    axes[1, 1].legend(loc="lower right")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "loan_default_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return auc


def main():
    print("=" * 60)
    print("Loan Default Prediction with Deep Neural Network")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nGenerating synthetic loan data...")
    df = generate_loan_data(10000)
    print(f"Samples: {len(df)}, Default rate: {df['default'].mean()*100:.1f}%")

    df.sample(250).to_csv(os.path.join(OUTPUT_DIR, "sample_loans.csv"), index=False)

    # Encode categoricals
    le_home = LabelEncoder()
    le_purpose = LabelEncoder()
    df["home_ownership"] = le_home.fit_transform(df["home_ownership"])
    df["purpose"] = le_purpose.fit_transform(df["purpose"])

    feature_cols = [c for c in df.columns if c != "default"]
    X = df[feature_cols].values.astype(float)
    y = df["default"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    dataset = LoanDataset(X, y)
    n = len(dataset)
    train_n = int(0.7 * n)
    val_n = int(0.15 * n)
    test_n = n - train_n - val_n
    train_ds, val_ds, test_ds = random_split(dataset, [train_n, val_n, test_n],
                                             generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = LoanDefaultNet(input_dim=X.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\nTraining...")
    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # val
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                val_loss += criterion(logits, yb).item()
                preds = logits.argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 7 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
                  f"Train Loss {tr_loss:.4f} Acc {tr_acc:.3f} | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc:.3f}")

    print("\nTest Evaluation:")
    y_true, y_pred, y_prob = evaluate(model, test_loader, device)
    print(classification_report(y_true, y_pred, target_names=["No Default", "Default"]))

    auc = plot_results(history, y_true, y_pred, y_prob)
    print(f"Test ROC-AUC: {auc:.4f}")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "loan_default_model.pth"))
    print("Model saved. Done!")


if __name__ == "__main__":
    main()
