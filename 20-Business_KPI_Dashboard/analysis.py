"""
Business KPI Dashboard – E-commerce Executive View
Creates a comprehensive set of business metrics and visualizations
suitable for Data Analyst / Business Analyst portfolio (Amazon style).
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


def generate_kpi_data(n_days=365):
    """Generate daily business metrics for an e-commerce platform."""
    start = datetime(2024, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]

    # Base trends + seasonality + noise
    trend = np.linspace(1.0, 1.35, n_days)
    weekly = 1 + 0.08 * np.sin(np.arange(n_days) * 2 * np.pi / 7)
    monthly = 1 + 0.12 * np.sin(np.arange(n_days) * 2 * np.pi / 30)
    noise = np.random.normal(1, 0.04, n_days)

    sessions = (185000 * trend * weekly * monthly * noise).astype(int)
    conversion_rate = np.clip(0.095 + 0.015 * np.sin(np.arange(n_days)/40) + np.random.normal(0, 0.004, n_days), 0.07, 0.14)
    orders = (sessions * conversion_rate).astype(int)
    aov = np.clip(68 + 8 * np.sin(np.arange(n_days)/50) + np.random.normal(0, 2.5, n_days), 55, 95)
    revenue = orders * aov
    new_customers = (orders * np.random.uniform(0.28, 0.42, n_days)).astype(int)
    returning_customers = orders - new_customers
    refund_rate = np.clip(0.035 + np.random.normal(0, 0.006, n_days), 0.015, 0.07)
    refunds = (revenue * refund_rate)

    df = pd.DataFrame({
        "date": dates,
        "sessions": sessions,
        "orders": orders,
        "conversion_rate": conversion_rate,
        "aov": aov,
        "revenue": revenue,
        "new_customers": new_customers,
        "returning_customers": returning_customers,
        "refund_rate": refund_rate,
        "refund_amount": refunds,
    })
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["week"] = df["date"].dt.isocalendar().week
    df["net_revenue"] = df["revenue"] - df["refund_amount"]
    return df


def run_dashboard():
    print("=" * 70)
    print("BUSINESS KPI DASHBOARD – E-commerce Executive View")
    print("=" * 70)

    df = generate_kpi_data()
    print(f"\nPeriod: {df['date'].min().date()} → {df['date'].max().date()}")

    # Latest 30 days vs previous 30 days
    latest = df.tail(30)
    previous = df.iloc[-60:-30]

    def pct_change(new, old):
        return (new - old) / old * 100

    metrics = {
        "Revenue": (latest["revenue"].sum(), previous["revenue"].sum()),
        "Orders": (latest["orders"].sum(), previous["orders"].sum()),
        "Sessions": (latest["sessions"].sum(), previous["sessions"].sum()),
        "Conversion Rate": (latest["conversion_rate"].mean(), previous["conversion_rate"].mean()),
        "AOV": (latest["aov"].mean(), previous["aov"].mean()),
        "New Customers": (latest["new_customers"].sum(), previous["new_customers"].sum()),
    }

    print("\n📊 Last 30 Days vs Previous 30 Days")
    print("-" * 55)
    for name, (curr, prev) in metrics.items():
        change = pct_change(curr, prev)
        arrow = "▲" if change > 0 else "▼"
        if "Rate" in name or name == "AOV":
            print(f"   {name:<18}: {curr:.2%} → {change:+.1f}% {arrow}" if "Rate" in name
                  else f"   {name:<18}: ${curr:.1f} → {change:+.1f}% {arrow}")
        else:
            print(f"   {name:<18}: {curr:,.0f} → {change:+.1f}% {arrow}")

    # ---------- Dashboard Visuals ----------
    fig = plt.figure(figsize=(16, 12))

    # 1. Revenue Trend
    ax1 = fig.add_subplot(3, 2, 1)
    monthly_rev = df.groupby("month")["revenue"].sum() / 1000
    ax1.plot(monthly_rev.index, monthly_rev.values, marker="o", color="#FF9900", linewidth=2)
    ax1.set_title("Monthly Gross Revenue ($K)")
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_ylabel("$K")

    # 2. Conversion Rate Trend
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(df["date"], df["conversion_rate"] * 100, color="#232F3E", alpha=0.7, linewidth=1)
    ax2.axhline(df["conversion_rate"].mean() * 100, color="red", linestyle="--", label="Average")
    ax2.set_title("Daily Conversion Rate (%)")
    ax2.legend()
    ax2.set_ylabel("%")

    # 3. Orders vs Sessions (dual axis concept via twin)
    ax3 = fig.add_subplot(3, 2, 3)
    monthly = df.groupby("month").agg(orders=("orders", "sum"), sessions=("sessions", "sum"))
    x = range(len(monthly))
    ax3.bar(x, monthly["orders"] / 1000, color="#FF9900", label="Orders (K)")
    ax3_t = ax3.twinx()
    ax3_t.plot(x, monthly["sessions"] / 1000, color="#232F3E", marker="o", label="Sessions (K)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(monthly.index, rotation=45)
    ax3.set_title("Monthly Orders & Sessions")
    ax3.legend(loc="upper left")
    ax3_t.legend(loc="upper right")

    # 4. New vs Returning Customers
    ax4 = fig.add_subplot(3, 2, 4)
    cust = df.groupby("month")[["new_customers", "returning_customers"]].sum()
    cust.plot(kind="bar", stacked=True, ax=ax4, color=["#3498db", "#2ecc71"])
    ax4.set_title("New vs Returning Customers")
    ax4.tick_params(axis="x", rotation=45)
    ax4.legend(["New", "Returning"])

    # 5. AOV Trend
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(df["date"], df["aov"], color="#9b59b6", alpha=0.8)
    ax5.set_title("Average Order Value ($)")
    ax5.set_ylabel("AOV ($)")

    # 6. Refund Rate
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.plot(df["date"], df["refund_rate"] * 100, color="#e74c3c", alpha=0.8)
    ax6.axhline(df["refund_rate"].mean() * 100, color="black", linestyle="--", label="Avg")
    ax6.set_title("Daily Refund Rate (%)")
    ax6.legend()

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "01_kpi_dashboard.png")
    plt.savefig(path1, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Saved: {path1}")

    # Funnel style summary
    fig, ax = plt.subplots(figsize=(9, 5))
    funnel_labels = ["Sessions", "Orders", "New Customers"]
    funnel_values = [
        latest["sessions"].sum(),
        latest["orders"].sum(),
        latest["new_customers"].sum(),
    ]
    colors_f = ["#232F3E", "#FF9900", "#146EB4"]
    bars = ax.barh(funnel_labels[::-1], funnel_values[::-1], color=colors_f[::-1])
    ax.set_title("Last 30 Days Funnel Snapshot")
    for bar, val in zip(bars, funnel_values[::-1]):
        ax.text(val * 1.01, bar.get_y() + bar.get_height()/2, f"{val:,.0f}", va="center")
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "02_funnel_snapshot.png")
    plt.savefig(path2, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path2}")

    # ---------- Executive Summary ----------
    print("\n" + "=" * 70)
    print("📌 EXECUTIVE SUMMARY & INSIGHTS")
    print("=" * 70)
    print(f"""
• Revenue and order volume show healthy growth trajectory over the year.
• Conversion rate is stable with mild seasonal fluctuation – opportunity to test
  homepage and recommendation improvements.
• New customer acquisition remains strong; focus on improving retention to increase
  the share of returning customers.
• AOV has room for improvement via bundling, cross-sell, and premium product push.
• Refund rate should be monitored closely; investigate high-refund categories.

Recommended Next Actions:
1. Launch A/B tests on high-traffic pages to lift conversion.
2. Deep-dive retention cohorts for customers acquired in the last 90 days.
3. Review top refund reasons and product quality signals.
""")

    # Save data
    df.to_csv(os.path.join(OUTPUT_DIR, "daily_kpis.csv"), index=False)
    summary_df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Last_30_Days": [v[0] for v in metrics.values()],
        "Previous_30_Days": [v[1] for v in metrics.values()],
        "Pct_Change": [pct_change(v[0], v[1]) for v in metrics.values()],
    })
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "kpi_comparison.csv"), index=False)
    print(f"✅ Data saved to {OUTPUT_DIR}/")
    print("\nDashboard generation complete.")


if __name__ == "__main__":
    run_dashboard()
