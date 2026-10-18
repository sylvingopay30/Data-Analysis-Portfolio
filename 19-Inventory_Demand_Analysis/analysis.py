"""
Inventory & Demand Analysis
Focuses on demand patterns, stock health, and inventory recommendations.
Strong project for Operations / Business Analyst roles at Amazon.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

np.random.seed(42)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 140

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CATEGORIES = ["Electronics", "Home", "Fashion", "Beauty", "Grocery", "Toys"]


def generate_inventory_data(n_products=400, days=180):
    """Generate product-level daily demand and inventory snapshot."""
    start = datetime(2024, 6, 1)
    products = []

    for i in range(n_products):
        cat = np.random.choice(CATEGORIES)
        base_demand = np.random.randint(5, 120)
        seasonality = np.random.uniform(0.7, 1.4)
        price = round(np.random.uniform(8, 350), 2)
        cost = round(price * np.random.uniform(0.45, 0.75), 2)
        current_stock = np.random.randint(0, 500)
        lead_time = np.random.choice([3, 5, 7, 10, 14])

        products.append({
            "product_id": f"SKU-{10000 + i}",
            "category": cat,
            "unit_price": price,
            "unit_cost": cost,
            "current_stock": current_stock,
            "avg_daily_demand": round(base_demand * seasonality / 30, 1),
            "lead_time_days": lead_time,
            "reorder_point": None,  # calculated later
        })

    products_df = pd.DataFrame(products)

    # Calculate key inventory metrics
    products_df["daily_demand"] = products_df["avg_daily_demand"]
    products_df["days_of_supply"] = (products_df["current_stock"] / products_df["daily_demand"]).round(1)
    products_df["reorder_point"] = (products_df["daily_demand"] * products_df["lead_time_days"] * 1.3).round(0)
    products_df["stock_status"] = products_df.apply(
        lambda r: "Stockout Risk" if r["current_stock"] < r["reorder_point"] * 0.5
        else ("Low Stock" if r["current_stock"] < r["reorder_point"]
              else ("Overstock" if r["days_of_supply"] > 60 else "Healthy")),
        axis=1
    )
    products_df["inventory_value"] = products_df["current_stock"] * products_df["unit_cost"]
    products_df["margin"] = ((products_df["unit_price"] - products_df["unit_cost"]) / products_df["unit_price"] * 100).round(1)

    return products_df


def run_analysis():
    print("=" * 70)
    print("INVENTORY & DEMAND ANALYSIS")
    print("=" * 70)

    df = generate_inventory_data()
    print(f"\nProducts analyzed: {len(df)}")

    # Key Metrics
    total_value = df["inventory_value"].sum()
    stockout_risk = (df["stock_status"] == "Stockout Risk").sum()
    overstock = (df["stock_status"] == "Overstock").sum()
    healthy = (df["stock_status"] == "Healthy").sum()

    print(f"\n📊 Inventory Health Snapshot")
    print(f"   Total Inventory Value : ${total_value:,.0f}")
    print(f"   Stockout Risk SKUs    : {stockout_risk}")
    print(f"   Low Stock SKUs        : {(df['stock_status']=='Low Stock').sum()}")
    print(f"   Healthy SKUs          : {healthy}")
    print(f"   Overstock SKUs        : {overstock}")

    # ---------- Visualizations ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Stock Status Distribution
    status_counts = df["stock_status"].value_counts()
    colors = {"Healthy": "#2ecc71", "Low Stock": "#f39c12", "Stockout Risk": "#e74c3c", "Overstock": "#3498db"}
    axes[0, 0].bar(status_counts.index, status_counts.values,
                   color=[colors.get(x, "gray") for x in status_counts.index])
    axes[0, 0].set_title("Inventory Health by Status")
    axes[0, 0].set_ylabel("Number of SKUs")
    axes[0, 0].tick_params(axis="x", rotation=15)

    # 2. Inventory Value by Category
    cat_val = df.groupby("category")["inventory_value"].sum().sort_values(ascending=True)
    axes[0, 1].barh(cat_val.index, cat_val.values / 1000, color="#FF9900")
    axes[0, 1].set_title("Inventory Value by Category ($K)")
    axes[0, 1].set_xlabel("Value ($K)")

    # 3. Days of Supply distribution
    axes[1, 0].hist(df["days_of_supply"].clip(0, 120), bins=30, color="#232F3E", edgecolor="white")
    axes[1, 0].axvline(30, color="red", linestyle="--", label="30-day target")
    axes[1, 0].set_title("Days of Supply Distribution")
    axes[1, 0].set_xlabel("Days of Supply")
    axes[1, 0].legend()

    # 4. Margin vs Days of Supply
    sample = df.sample(min(300, len(df)))
    scatter = axes[1, 1].scatter(
        sample["days_of_supply"].clip(0, 100),
        sample["margin"],
        c=sample["inventory_value"],
        cmap="YlOrRd", alpha=0.7, s=40
    )
    axes[1, 1].set_title("Margin % vs Days of Supply")
    axes[1, 1].set_xlabel("Days of Supply")
    axes[1, 1].set_ylabel("Gross Margin %")
    plt.colorbar(scatter, ax=axes[1, 1], label="Inventory Value")

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "01_inventory_health.png")
    plt.savefig(path1, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Saved: {path1}")

    # Category deep dive
    fig, ax = plt.subplots(figsize=(10, 5))
    status_cat = pd.crosstab(df["category"], df["stock_status"])
    status_cat.plot(kind="bar", stacked=True, ax=ax,
                    color=["#e74c3c", "#f39c12", "#2ecc71", "#3498db"])
    ax.set_title("Stock Status Breakdown by Category")
    ax.set_ylabel("Number of SKUs")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Status")
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "02_status_by_category.png")
    plt.savefig(path2, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path2}")

    # ---------- Recommendations ----------
    print("\n" + "=" * 70)
    print("📌 INVENTORY RECOMMENDATIONS")
    print("=" * 70)
    print(f"""
1. Immediate Action – Stockout Risk ({stockout_risk} SKUs)
   → Place urgent purchase orders for high-velocity items below reorder point.

2. Overstock Management ({overstock} SKUs)
   → Run targeted promotions or transfer inventory to high-demand regions.
   → Review slow-moving high-value items for markdowns.

3. Category Focus
   → Prioritize Electronics and Home for tighter inventory control (high value).
   → Grocery needs higher safety stock due to shorter shelf life.

4. Process Improvement
   → Review lead times for frequent stockout categories.
   → Consider dynamic reorder points based on recent demand trends.
""")

    # Save outputs
    df.to_csv(os.path.join(OUTPUT_DIR, "product_inventory.csv"), index=False)
    summary = df.groupby("stock_status").agg(
        skus=("product_id", "count"),
        total_value=("inventory_value", "sum"),
        avg_days_supply=("days_of_supply", "mean"),
    ).round(1)
    summary.to_csv(os.path.join(OUTPUT_DIR, "inventory_summary.csv"))
    print(f"✅ Files saved to {OUTPUT_DIR}/")
    print("\nAnalysis complete.")


if __name__ == "__main__":
    run_analysis()
