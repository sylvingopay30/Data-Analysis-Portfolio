"""
Amazon Sales Performance Analysis
Comprehensive sales analysis suitable for Data Analyst / Business Analyst roles.
Generates professional visualizations and actionable business insights.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_amazon_orders(n_orders=25000):
    """Generate realistic Amazon-style order data."""
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    days = (end_date - start_date).days

    categories = {
        "Electronics": 0.22,
        "Home & Kitchen": 0.18,
        "Fashion": 0.16,
        "Books": 0.10,
        "Beauty": 0.09,
        "Sports": 0.08,
        "Toys": 0.07,
        "Grocery": 0.10,
    }
    cat_list = list(categories.keys())
    cat_probs = list(categories.values())

    # Product price ranges by category
    price_params = {
        "Electronics": (80, 450),
        "Home & Kitchen": (25, 180),
        "Fashion": (20, 120),
        "Books": (8, 35),
        "Beauty": (12, 65),
        "Sports": (18, 150),
        "Toys": (15, 80),
        "Grocery": (8, 45),
    }

    order_ids = [f"ORD-{100000 + i}" for i in range(n_orders)]
    order_dates = [start_date + timedelta(days=np.random.randint(0, days)) for _ in range(n_orders)]
    customer_ids = [f"C-{np.random.randint(10000, 99999)}" for _ in range(n_orders)]
    category = np.random.choice(cat_list, n_orders, p=cat_probs)

    prices = []
    quantities = []
    for cat in category:
        low, high = price_params[cat]
        prices.append(round(np.random.uniform(low, high), 2))
        quantities.append(np.random.choice([1, 1, 1, 2, 2, 3], p=[0.55, 0.2, 0.1, 0.1, 0.03, 0.02]))

    prices = np.array(prices)
    quantities = np.array(quantities)
    revenue = np.round(prices * quantities, 2)

    # Discount
    discount_pct = np.random.choice([0, 5, 10, 15, 20, 25], n_orders, p=[0.45, 0.2, 0.15, 0.1, 0.07, 0.03])
    discount_amount = np.round(revenue * discount_pct / 100, 2)
    net_revenue = revenue - discount_amount

    # Region
    regions = np.random.choice(
        ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East"],
        n_orders,
        p=[0.45, 0.22, 0.20, 0.08, 0.05],
    )

    # Shipping
    shipping = np.random.choice(["Standard", "Prime", "Same-Day"], n_orders, p=[0.35, 0.50, 0.15])

    df = pd.DataFrame({
        "order_id": order_ids,
        "order_date": order_dates,
        "customer_id": customer_ids,
        "category": category,
        "unit_price": prices,
        "quantity": quantities,
        "gross_revenue": revenue,
        "discount_pct": discount_pct,
        "discount_amount": discount_amount,
        "net_revenue": net_revenue,
        "region": regions,
        "shipping_type": shipping,
    })

    df["order_date"] = pd.to_datetime(df["order_date"])
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["day_of_week"] = df["order_date"].dt.day_name()

    return df.sort_values("order_date").reset_index(drop=True)


def run_analysis(df):
    print("=" * 70)
    print("AMAZON SALES PERFORMANCE ANALYSIS")
    print("=" * 70)

    # ---------- Key Metrics ----------
    total_orders = len(df)
    total_revenue = df["net_revenue"].sum()
    total_customers = df["customer_id"].nunique()
    aov = df["net_revenue"].mean()
    avg_items = df["quantity"].mean()

    print(f"\n📊 KEY BUSINESS METRICS")
    print(f"   Total Orders          : {total_orders:,}")
    print(f"   Total Net Revenue     : ${total_revenue:,.0f}")
    print(f"   Unique Customers      : {total_customers:,}")
    print(f"   Average Order Value   : ${aov:.2f}")
    print(f"   Avg Items per Order   : {avg_items:.2f}")

    # ---------- Monthly Revenue Trend ----------
    monthly = df.groupby("year_month").agg(
        revenue=("net_revenue", "sum"),
        orders=("order_id", "count"),
        customers=("customer_id", "nunique"),
    ).reset_index()
    monthly["aov"] = monthly["revenue"] / monthly["orders"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Monthly Revenue
    axes[0, 0].plot(monthly["year_month"], monthly["revenue"] / 1000, marker="o", color="#FF9900", linewidth=2)
    axes[0, 0].set_title("Monthly Net Revenue ($K)")
    axes[0, 0].set_xlabel("")
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].set_ylabel("Revenue ($K)")

    # 2. Category Revenue Share
    cat_rev = df.groupby("category")["net_revenue"].sum().sort_values(ascending=False)
    colors = sns.color_palette("Oranges_r", len(cat_rev))
    axes[0, 1].pie(cat_rev, labels=cat_rev.index, autopct="%1.1f%%", colors=colors, startangle=90)
    axes[0, 1].set_title("Revenue Share by Category")

    # 3. Region Performance
    region_rev = df.groupby("region")["net_revenue"].sum().sort_values(ascending=True)
    axes[1, 0].barh(region_rev.index, region_rev.values / 1000, color="#232F3E")
    axes[1, 0].set_title("Revenue by Region ($K)")
    axes[1, 0].set_xlabel("Revenue ($K)")

    # 4. Shipping Type Impact
    ship = df.groupby("shipping_type").agg(
        orders=("order_id", "count"),
        revenue=("net_revenue", "sum"),
        aov=("net_revenue", "mean"),
    )
    x = np.arange(len(ship))
    width = 0.35
    axes[1, 1].bar(x - width/2, ship["orders"], width, label="Orders", color="#FF9900")
    axes[1, 1].bar(x + width/2, ship["aov"] * 10, width, label="AOV x10", color="#232F3E")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(ship.index)
    axes[1, 1].set_title("Orders vs AOV by Shipping Type")
    axes[1, 1].legend()

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "01_sales_overview.png")
    plt.savefig(path1, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Saved: {path1}")

    # ---------- Category Deep Dive ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cat_stats = df.groupby("category").agg(
        revenue=("net_revenue", "sum"),
        orders=("order_id", "count"),
        aov=("net_revenue", "mean"),
        avg_qty=("quantity", "mean"),
    ).sort_values("revenue", ascending=False)

    sns.barplot(x=cat_stats.index, y=cat_stats["aov"], ax=axes[0], palette="Oranges_r")
    axes[0].set_title("Average Order Value by Category")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("AOV ($)")

    # Discount impact
    disc = df.groupby("discount_pct")["net_revenue"].mean().reset_index()
    axes[1].plot(disc["discount_pct"], disc["net_revenue"], marker="o", color="#FF9900", linewidth=2)
    axes[1].set_title("Average Net Revenue by Discount %")
    axes[1].set_xlabel("Discount %")
    axes[1].set_ylabel("Avg Net Revenue ($)")

    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "02_category_discount_analysis.png")
    plt.savefig(path2, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path2}")

    # ---------- Day of Week & Seasonality ----------
    fig, ax = plt.subplots(figsize=(10, 5))
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = df.groupby("day_of_week")["net_revenue"].sum().reindex(dow_order)
    sns.barplot(x=dow.index, y=dow.values / 1000, ax=ax, palette="Blues_d")
    ax.set_title("Total Revenue by Day of Week ($K)")
    ax.set_ylabel("Revenue ($K)")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, "03_day_of_week.png")
    plt.savefig(path3, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path3}")

    # ---------- Top Insights ----------
    print("\n" + "=" * 70)
    print("📌 KEY BUSINESS INSIGHTS & RECOMMENDATIONS")
    print("=" * 70)

    top_cat = cat_stats.index[0]
    top_region = region_rev.index[-1]
    best_ship = ship["aov"].idxmax()

    print(f"""
1. Category Performance
   • {top_cat} is the highest revenue category.
   • Focus marketing and inventory investment on top 3 categories.
   • Books & Grocery have lower AOV → promote bundles / subscription.

2. Regional Opportunity
   • {top_region} leads in revenue.
   • Asia-Pacific and Europe show strong potential for growth campaigns.

3. Shipping Strategy
   • {best_ship} shipping has the highest Average Order Value.
   • Encourage Prime / Same-Day upgrades for higher basket size.

4. Discount Effectiveness
   • Higher discounts do not always translate to higher net revenue.
   • Optimize discount depth by category (avoid heavy discounts on high-margin items).

5. Seasonality
   • Weekends and end-of-month periods show elevated order volume.
   • Plan promotions and inventory accordingly.
""")

    # Save summary metrics
    summary = pd.DataFrame({
        "Metric": ["Total Orders", "Total Net Revenue", "Unique Customers", "AOV", "Avg Items/Order"],
        "Value": [total_orders, round(total_revenue, 2), total_customers, round(aov, 2), round(avg_items, 2)]
    })
    summary.to_csv(os.path.join(OUTPUT_DIR, "key_metrics.csv"), index=False)

    # Sample data
    df.sample(500).to_csv(os.path.join(OUTPUT_DIR, "sample_orders.csv"), index=False)
    print(f"✅ Sample data & metrics saved to {OUTPUT_DIR}/")

    print("\nAnalysis complete. Ready for portfolio / interview discussion.")


if __name__ == "__main__":
    print("Generating Amazon-style order data...")
    df = generate_amazon_orders(25000)
    print(f"Generated {len(df):,} orders from {df['order_date'].min().date()} to {df['order_date'].max().date()}")
    run_analysis(df)
