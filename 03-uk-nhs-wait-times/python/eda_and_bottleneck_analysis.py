"""
UK NHS Wait Times & Resource Allocation
EDA, trust/specialty bottleneck identification, and capacity gap analysis.
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/nhs_wait_times.csv", parse_dates=["month"])

# ---- Trust-level performance ----
trust_perf = df.groupby(["trust","region"]).agg(
    avg_median_wait_weeks=("median_wait_weeks","mean"),
    avg_pct_over_18weeks=("pct_over_18weeks","mean"),
    total_waiting_list=("waiting_list_size","sum")
).round(1).sort_values("avg_pct_over_18weeks", ascending=False)
print("=== Trust-level performance (worst first) ===")
print(trust_perf.to_string())

# ---- Specialty bottlenecks ----
spec_perf = df.groupby("specialty").agg(
    avg_median_wait_weeks=("median_wait_weeks","mean"),
    avg_pct_over_18weeks=("pct_over_18weeks","mean"),
    avg_monthly_backlog_growth=("referrals","mean")
).round(1)
spec_perf["avg_monthly_backlog_growth"] = (df.groupby("specialty")["referrals"].mean() - df.groupby("specialty")["treated"].mean()).round(0)
spec_perf = spec_perf.sort_values("avg_pct_over_18weeks", ascending=False)
print("\n=== Specialty-level bottlenecks (worst first) ===")
print(spec_perf.to_string())

# ---- System-wide trend ----
trend = df.groupby("month").agg(
    total_waiting_list=("waiting_list_size","sum"),
    avg_median_wait_weeks=("median_wait_weeks","mean")
)
print(f"\n=== System-wide waiting list growth: {trend['total_waiting_list'].iloc[0]:,.0f} -> {trend['total_waiting_list'].iloc[-1]:,.0f} "
      f"({(trend['total_waiting_list'].iloc[-1]/trend['total_waiting_list'].iloc[0]-1):.1%} change) ===")

# ---- Capacity gap: trust x specialty combinations with worst shortfall ----
gap = df.groupby(["trust","specialty"]).agg(
    avg_referrals=("referrals","mean"),
    avg_treated=("treated","mean")
)
gap["avg_shortfall"] = (gap["avg_referrals"] - gap["avg_treated"]).round(1)
gap = gap.sort_values("avg_shortfall", ascending=False).head(10)
print("\n=== Top 10 trust x specialty capacity gaps (highest referral-treatment shortfall) ===")
print(gap.to_string())

# ---- Charts ----
fig, ax = plt.subplots(figsize=(8,5))
trust_perf["avg_pct_over_18weeks"].sort_values().plot(kind="barh", ax=ax, color="#B23A48")
ax.set_title("Avg % of Patients Waiting Over 18 Weeks, by Trust")
ax.set_xlabel("% over 18 weeks")
plt.tight_layout()
plt.savefig("../outputs/trust_18week_performance.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8,5))
spec_perf["avg_pct_over_18weeks"].sort_values().plot(kind="barh", ax=ax, color="#457B9D")
ax.set_title("Avg % of Patients Waiting Over 18 Weeks, by Specialty")
ax.set_xlabel("% over 18 weeks")
plt.tight_layout()
plt.savefig("../outputs/specialty_18week_performance.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8,4.5))
trend["total_waiting_list"].plot(ax=ax, color="#1D3557")
ax.set_title("System-Wide Waiting List Size Over Time")
ax.set_ylabel("Patients waiting")
plt.tight_layout()
plt.savefig("../outputs/waiting_list_trend.png", dpi=150)
plt.close()

trust_perf.to_csv("../outputs/trust_performance.csv")
spec_perf.to_csv("../outputs/specialty_performance.csv")
gap.to_csv("../outputs/capacity_gaps.csv")

print("\nSaved charts and tables to ../outputs/")
