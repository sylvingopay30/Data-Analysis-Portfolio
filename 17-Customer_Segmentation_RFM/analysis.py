"""
Customer Segmentation using RFM Analysis
Classic and highly relevant project for Data Analyst / Business Analyst roles.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

np.random.seed(42)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 140

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_customer_transactions(n_customers=5000, n_transactions=30000):
    """Generate realistic customer transaction history."""
    start = datetime(2023, 1, 1)
    end = datetime(2024, 12, 31)
    days_range = (end - start).days

    customer_ids = [f"CUST-{10000 + i}" for i in range(n_customers)]

    # Different customer behaviors
    # High-value frequent, occasional high, low-value frequent, dormant, etc.
    records = []
    for cid in customer_ids:
        # Number of orders per customer (power-law-ish)
        n_orders = int(np.random.choice(
            [1, 2, 3, 4, 5, 8, 12, 20],
            p=[0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04]
        ))
        for _ in range(n_orders):
            order_date = start + timedelta(days=np.random.randint(0, days_range))
            # Monetary value
            monetary = round(np.random.lognormal(mean=3.8, sigma=0.9), 2)
            records.append({
                "customer_id": cid,
                "order_date": order_date,
                "order_value": monetary,
            })

    df = pd.DataFrame(records)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def compute_rfm(df, analysis_date=None):
    if analysis_date is None:
        analysis_date = df["order_date"].max() + timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        recency=("order_date", lambda x: (analysis_date - x.max()).days),
        frequency=("order_date", "count"),
        monetary=("order_value", "sum"),
    ).reset_index()

    return rfm


def score_rfm(rfm):
    """Classic RFM scoring (1-5)."""
    rfm = rfm.copy()
    rfm["R_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["RFM_Score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]
    return rfm


def segment_customers(rfm):
    """Business-friendly segments based on RFM scores."""
    def label(row):
        if row["R_score"] >= 4 and row["F_score"] >= 4 and row["M_score"] >= 4:
            return "Champions"
        elif row["R_score"] >= 3 and row["F_score"] >= 3 and row["M_score"] >= 3:
            return "Loyal Customers"
        elif row["R_score"] >= 4 and row["F_score"] <= 2:
            return "Promising / New"
        elif row["R_score"] <= 2 and row["F_score"] >= 3:
            return "At Risk"
        elif row["R_score"] <= 2 and row["F_score"] <= 2 and row["M_score"] <= 2:
            return "Lost / Hibernating"
        elif row["M_score"] >= 4:
            return "Big Spenders"
        else:
            return "Need Attention"

    rfm["Segment"] = rfm.apply(label, axis=1)
    return rfm


def run_analysis():
    print("=" * 70)
    print("CUSTOMER SEGMENTATION – RFM ANALYSIS")
    print("=" * 70)

    print("\nGenerating transaction data...")
    tx = generate_customer_transactions()
    print(f"Transactions: {len(tx):,} | Customers: {tx['customer_id'].nunique():,}")

    rfm = compute_rfm(tx)
    rfm = score_rfm(rfm)
    rfm = segment_customers(rfm)

    print("\n📊 Segment Distribution:")
    print(rfm["Segment"].value_counts())

    # ---------- Visualizations ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1. Segment size
    seg_counts = rfm["Segment"].value_counts()
    colors = sns.color_palette("viridis", len(seg_counts))
    axes[0, 0].barh(seg_counts.index, seg_counts.values, color=colors)
    axes[0, 0].set_title("Customer Count by Segment")
    axes[0, 0].set_xlabel("Number of Customers")

    # 2. Average Monetary by Segment
    mon = rfm.groupby("Segment")["monetary"].mean().sort_values(ascending=True)
    axes[0, 1].barh(mon.index, mon.values, color="#FF9900")
    axes[0, 1].set_title("Average Monetary Value by Segment ($)")
    axes[0, 1].set_xlabel("Avg Lifetime Value ($)")

    # 3. Recency vs Frequency scatter (colored by segment)
    sample = rfm.sample(min(2000, len(rfm)), random_state=42)
    sns.scatterplot(
        data=sample, x="recency", y="frequency", hue="Segment",
        ax=axes[1, 0], palette="tab10", alpha=0.7, s=40
    )
    axes[1, 0].set_title("Recency vs Frequency by Segment")
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    # 4. RFM Score distribution
    axes[1, 1].hist(rfm["RFM_Score"], bins=13, color="#232F3E", edgecolor="white")
    axes[1, 1].set_title("Distribution of RFM Scores")
    axes[1, 1].set_xlabel("RFM Score (3–15)")
    axes[1, 1].set_ylabel("Customers")

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "01_rfm_segmentation.png")
    plt.savefig(path1, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Saved: {path1}")

    # Heatmap of average R, F, M by segment
    pivot = rfm.groupby("Segment")[["R_score", "F_score", "M_score"]].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
    ax.set_title("Average R / F / M Scores by Segment")
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "02_rfm_heatmap.png")
    plt.savefig(path2, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path2}")

    # ---------- Business Recommendations ----------
    print("\n" + "=" * 70)
    print("📌 SEGMENT STRATEGY RECOMMENDATIONS")
    print("=" * 70)
    print("""
• Champions          → Reward with exclusive offers, early access, loyalty points.
• Loyal Customers    → Upsell premium products / Amazon Prime.
• Big Spenders       → Personalized high-value recommendations.
• Promising / New    → Onboarding campaigns, first-purchase incentives.
• At Risk            → Win-back campaigns, limited-time discounts.
• Need Attention     → Re-engagement emails, category recommendations.
• Lost / Hibernating → Aggressive win-back or reduce marketing spend.
""")

    # Save outputs
    rfm.to_csv(os.path.join(OUTPUT_DIR, "customer_rfm_segments.csv"), index=False)
    tx.sample(1000).to_csv(os.path.join(OUTPUT_DIR, "sample_transactions.csv"), index=False)

    summary = rfm.groupby("Segment").agg(
        customers=("customer_id", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        total_revenue=("monetary", "sum"),
    ).round(1)
    summary.to_csv(os.path.join(OUTPUT_DIR, "segment_summary.csv"))
    print(f"✅ All outputs saved to {OUTPUT_DIR}/")
    print("\nAnalysis complete.")


if __name__ == "__main__":
    run_analysis()
