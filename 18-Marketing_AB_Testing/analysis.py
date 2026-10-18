"""
Marketing A/B Testing Analysis
Classic experiment analysis project for Data Analyst / Business Analyst roles.
Includes statistical significance, confidence intervals, and business decision framework.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(42)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 140

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_ab_test_data(n_control=12000, n_treatment=12000):
    """
    Simulate an A/B test for a new product recommendation banner on Amazon.
    Primary metric: Conversion Rate (Add-to-Cart or Purchase).
    Secondary: Average Order Value.
    """
    # Control: current experience
    control_conv = np.random.binomial(1, 0.112, n_control)  # 11.2% conversion
    control_aov = np.where(
        control_conv == 1,
        np.random.lognormal(3.9, 0.55),
        0
    )

    # Treatment: new personalized banner
    treatment_conv = np.random.binomial(1, 0.127, n_treatment)  # 12.7% conversion (lift)
    treatment_aov = np.where(
        treatment_conv == 1,
        np.random.lognormal(3.95, 0.55),
        0
    )

    df_control = pd.DataFrame({
        "user_id": [f"U-C-{i}" for i in range(n_control)],
        "variant": "Control",
        "converted": control_conv,
        "order_value": np.round(control_aov, 2),
    })

    df_treatment = pd.DataFrame({
        "user_id": [f"U-T-{i}" for i in range(n_treatment)],
        "variant": "Treatment",
        "converted": treatment_conv,
        "order_value": np.round(treatment_aov, 2),
    })

    return pd.concat([df_control, df_treatment], ignore_index=True)


def calculate_metrics(df):
    summary = df.groupby("variant").agg(
        users=("user_id", "count"),
        conversions=("converted", "sum"),
        conversion_rate=("converted", "mean"),
        total_revenue=("order_value", "sum"),
        avg_order_value=("order_value", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
    ).round(4)
    return summary


def two_proportion_ztest(success_a, n_a, success_b, n_b):
    """Two-proportion z-test."""
    p_a = success_a / n_a
    p_b = success_b / n_b
    p_pool = (success_a + success_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    z = (p_b - p_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value, p_a, p_b


def confidence_interval(success, n, confidence=0.95):
    p = success / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    se = np.sqrt(p * (1 - p) / n)
    return p - z * se, p + z * se


def run_analysis():
    print("=" * 70)
    print("MARKETING A/B TEST ANALYSIS – New Recommendation Banner")
    print("=" * 70)

    df = generate_ab_test_data()
    print(f"\nTotal users in experiment: {len(df):,}")

    summary = calculate_metrics(df)
    print("\n📊 Experiment Summary:")
    print(summary)

    # Statistical Test
    ctrl = df[df["variant"] == "Control"]
    treat = df[df["variant"] == "Treatment"]

    z_stat, p_value, p_ctrl, p_treat = two_proportion_ztest(
        ctrl["converted"].sum(), len(ctrl),
        treat["converted"].sum(), len(treat)
    )

    lift = (p_treat - p_ctrl) / p_ctrl * 100
    ci_ctrl = confidence_interval(ctrl["converted"].sum(), len(ctrl))
    ci_treat = confidence_interval(treat["converted"].sum(), len(treat))

    print("\n📈 Statistical Results (Conversion Rate)")
    print(f"   Control CR     : {p_ctrl*100:.2f}%  (95% CI: {ci_ctrl[0]*100:.2f}% – {ci_ctrl[1]*100:.2f}%)")
    print(f"   Treatment CR   : {p_treat*100:.2f}%  (95% CI: {ci_treat[0]*100:.2f}% – {ci_treat[1]*100:.2f}%)")
    print(f"   Absolute Lift  : {(p_treat - p_ctrl)*100:.2f} pp")
    print(f"   Relative Lift  : {lift:.1f}%")
    print(f"   Z-statistic    : {z_stat:.3f}")
    print(f"   P-value        : {p_value:.6f}")
    print(f"   Significant?   : {'YES ✅' if p_value < 0.05 else 'NO ❌'} (α = 0.05)")

    # ---------- Visualizations ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Conversion Rate comparison
    rates = [p_ctrl, p_treat]
    labels = ["Control", "Treatment"]
    colors = ["#232F3E", "#FF9900"]
    bars = axes[0].bar(labels, [r*100 for r in rates], color=colors, edgecolor="white")
    axes[0].set_ylabel("Conversion Rate (%)")
    axes[0].set_title("Conversion Rate by Variant")
    for bar, r in zip(bars, rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                     f"{r*100:.2f}%", ha="center", fontweight="bold")

    # Revenue impact
    rev = summary["total_revenue"]
    axes[1].bar(labels, rev.values / 1000, color=colors)
    axes[1].set_ylabel("Total Revenue ($K)")
    axes[1].set_title("Total Revenue by Variant")

    # AOV of converters
    aov = summary["avg_order_value"]
    axes[2].bar(labels, aov.values, color=colors)
    axes[2].set_ylabel("Average Order Value ($)")
    axes[2].set_title("AOV (among converters)")

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "01_ab_test_results.png")
    plt.savefig(path1, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Saved: {path1}")

    # Distribution of order values
    fig, ax = plt.subplots(figsize=(9, 5))
    converted = df[df["converted"] == 1]
    sns.histplot(data=converted, x="order_value", hue="variant", bins=40,
                 kde=True, ax=ax, palette=["#232F3E", "#FF9900"])
    ax.set_title("Order Value Distribution (Converters Only)")
    ax.set_xlabel("Order Value ($)")
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "02_order_value_distribution.png")
    plt.savefig(path2, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path2}")

    # ---------- Decision Framework ----------
    print("\n" + "=" * 70)
    print("📌 BUSINESS DECISION FRAMEWORK")
    print("=" * 70)
    print(f"""
Experiment: New Personalized Recommendation Banner

Primary Metric (Conversion Rate):
  • Relative Lift : {lift:.1f}%
  • P-value       : {p_value:.5f}
  • Result        : Statistically significant at 95% confidence

Recommendation:
  {'→ SHIP THE TREATMENT. The new banner drives a meaningful and statistically significant lift in conversion.' if p_value < 0.05 else '→ DO NOT SHIP. Results are not statistically significant.'}

Next Steps:
  1. Monitor secondary metrics (AOV, bounce rate, latency) for any negative impact.
  2. Run the experiment longer if we want tighter confidence intervals.
  3. Segment the lift (new vs returning customers, device type) for deeper insights.
""")

    # Save data
    df.to_csv(os.path.join(OUTPUT_DIR, "ab_test_raw_data.csv"), index=False)
    summary.to_csv(os.path.join(OUTPUT_DIR, "ab_test_summary.csv"))
    print(f"✅ Data saved to {OUTPUT_DIR}/")
    print("\nAnalysis complete.")


if __name__ == "__main__":
    run_analysis()
