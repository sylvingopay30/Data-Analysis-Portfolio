"""
Australia Public Transport On-Time Performance & Cost-Benefit Analysis
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/au_transport_performance.csv", parse_dates=["date"])

# ---- Route-level OTP ----
route_otp = df.groupby(["route","mode","city"]).apply(
    lambda g: pd.Series({
        "otp_pct": round(100 * g["on_time_services"].sum() / g["scheduled_services"].sum(), 1),
        "avg_delay_minutes": round(g["avg_delay_minutes"].mean(), 1),
        "total_patronage": g["daily_patronage"].sum()
    })
).reset_index().sort_values("otp_pct")
print("=== Route-level OTP (worst first) ===")
print(route_otp.to_string(index=False))

# ---- Mode-level comparison ----
mode_otp = df.groupby("mode").apply(
    lambda g: pd.Series({
        "otp_pct": round(100 * g["on_time_services"].sum() / g["scheduled_services"].sum(), 1),
        "avg_delay_minutes": round(g["avg_delay_minutes"].mean(), 1),
        "total_patronage": g["daily_patronage"].sum()
    })
).reset_index().sort_values("otp_pct")
print("\n=== Mode-level OTP ===")
print(mode_otp.to_string(index=False))

# ---- Monthly trend ----
df["month"] = df["date"].dt.to_period("M")
monthly_otp = df.groupby("month").apply(
    lambda g: round(100 * g["on_time_services"].sum() / g["scheduled_services"].sum(), 1)
)
monthly_otp.index = monthly_otp.index.to_timestamp()

# ---- Rider impact score (prioritisation for cost-benefit) ----
impact = df.groupby(["route","mode","city"]).apply(
    lambda g: pd.Series({
        "total_disrupted_services": g["disrupted_services"].sum(),
        "total_patronage": g["daily_patronage"].sum(),
        "avg_daily_patronage": g["daily_patronage"].mean()
    })
).reset_index()
impact["rider_impact_score"] = (impact["total_disrupted_services"] * impact["avg_daily_patronage"]).round(0)
impact = impact.sort_values("rider_impact_score", ascending=False)
print("\n=== Rider impact score (disruption x ridership, highest first) ===")
print(impact[["route","mode","city","total_disrupted_services","avg_daily_patronage","rider_impact_score"]].to_string(index=False))

# ---- Simple cost-benefit for the top-impact route ----
top = impact.iloc[0]
print(f"\n=== Illustrative cost-benefit for top-impact route: {top['route']} ({top['mode']}, {top['city']}) ===")
assumed_intervention_cost = 250000  # e.g. additional rolling stock / signalling fix, illustrative
assumed_otp_uplift_pct = 8.0
assumed_value_per_avoided_delay_min = 0.45  # illustrative AUD value of rider time per minute saved, aggregated
avoided_disrupted_services_per_year = top["total_disrupted_services"] * (assumed_otp_uplift_pct/100) * (365/730)
print(f"Assumed intervention cost: AUD {assumed_intervention_cost:,.0f}")
print(f"Assumed OTP uplift: {assumed_otp_uplift_pct}%")
print(f"Estimated avoided disrupted services/year: {avoided_disrupted_services_per_year:,.0f}")

# ---- Charts ----
fig, ax = plt.subplots(figsize=(8,5))
ax.barh(route_otp["route"] + " (" + route_otp["city"] + ")", route_otp["otp_pct"], color="#264653")
ax.set_title("On-Time Performance % by Route")
ax.set_xlabel("OTP %")
plt.tight_layout()
plt.savefig("../outputs/route_otp.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7,4.5))
ax.bar(mode_otp["mode"], mode_otp["otp_pct"], color="#2A9D8F")
ax.set_title("On-Time Performance % by Mode")
ax.set_ylabel("OTP %")
plt.tight_layout()
plt.savefig("../outputs/mode_otp.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8,4.5))
monthly_otp.plot(ax=ax, color="#E76F51")
ax.set_title("System-Wide OTP Trend")
ax.set_ylabel("OTP %")
plt.tight_layout()
plt.savefig("../outputs/otp_trend.png", dpi=150)
plt.close()

route_otp.to_csv("../outputs/route_otp.csv", index=False)
mode_otp.to_csv("../outputs/mode_otp.csv", index=False)
impact.to_csv("../outputs/rider_impact_scores.csv", index=False)

print("\nSaved charts and tables to ../outputs/")
