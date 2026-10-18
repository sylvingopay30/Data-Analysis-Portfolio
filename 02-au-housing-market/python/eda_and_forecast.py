"""
Australia Housing Market Price Trends
EDA, YoY growth, and a simple linear trend forecast per metro area.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("../data/au_property_sales.csv", parse_dates=["sale_date"])

# ---- Median price by metro/quarter ----
df["quarter"] = df["sale_date"].dt.to_period("Q")
metro_quarter = df.groupby(["metro_area", "quarter"])["sale_price_aud"].median().reset_index()
metro_quarter["quarter_ts"] = metro_quarter["quarter"].dt.to_timestamp()

print("=== Median price by metro area (latest quarter) ===")
latest_q = metro_quarter["quarter"].max()
print(metro_quarter[metro_quarter["quarter"] == latest_q].sort_values("sale_price_aud", ascending=False).to_string(index=False))

# ---- YoY growth by suburb ----
df["year"] = df["sale_date"].dt.year
yearly = df.groupby(["suburb", "state", "year"])["sale_price_aud"].median().reset_index()
yearly_pivot = yearly.pivot_table(index=["suburb","state"], columns="year", values="sale_price_aud")
first_year, last_year = yearly["year"].min(), yearly["year"].max()
yearly_pivot["total_growth_pct"] = ((yearly_pivot[last_year] - yearly_pivot[first_year]) / yearly_pivot[first_year] * 100).round(1)
print(f"\n=== Total growth {first_year}->{last_year} by suburb ===")
print(yearly_pivot[["total_growth_pct"]].sort_values("total_growth_pct", ascending=False).to_string())

# ---- Affordability: price per bedroom ----
afford = df[df["bedrooms"] > 0].groupby(["suburb","state"]).apply(
    lambda g: (g["sale_price_aud"] / g["bedrooms"]).mean()
).reset_index(name="avg_price_per_bedroom").sort_values("avg_price_per_bedroom")
print("\n=== Most affordable suburbs (price per bedroom) ===")
print(afford.head(5).to_string(index=False))
print("\n=== Least affordable suburbs (price per bedroom) ===")
print(afford.tail(5).to_string(index=False))

# ---- Simple linear trend forecast per metro (next 2 quarters) ----
forecasts = []
for metro, g in metro_quarter.groupby("metro_area"):
    g = g.sort_values("quarter_ts")
    g["t"] = np.arange(len(g))
    model = LinearRegression().fit(g[["t"]], g["sale_price_aud"])
    future_t = np.array([[len(g)], [len(g)+1]])
    preds = model.predict(future_t)
    forecasts.append({"metro_area": metro, "next_q_forecast": round(preds[0]), "q_after_forecast": round(preds[1]),
                       "trend_aud_per_quarter": round(model.coef_[0])})
forecast_df = pd.DataFrame(forecasts).sort_values("trend_aud_per_quarter", ascending=False)
print("\n=== Simple linear trend forecast (next 2 quarters, median price) ===")
print(forecast_df.to_string(index=False))

# ---- Charts ----
fig, ax = plt.subplots(figsize=(9,5))
for metro, g in metro_quarter.groupby("metro_area"):
    g = g.sort_values("quarter_ts")
    ax.plot(g["quarter_ts"], g["sale_price_aud"], marker="o", label=metro)
ax.set_title("Median Sale Price by Metro Area Over Time")
ax.set_ylabel("AUD")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("../outputs/median_price_trend.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8,5))
afford_sorted = afford.sort_values("avg_price_per_bedroom")
ax.barh(afford_sorted["suburb"], afford_sorted["avg_price_per_bedroom"], color="#3E6990")
ax.set_title("Average Price per Bedroom by Suburb")
ax.set_xlabel("AUD")
plt.tight_layout()
plt.savefig("../outputs/price_per_bedroom.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7,4.5))
dom = df.groupby("property_type")["days_on_market"].mean().sort_values()
dom.plot(kind="bar", ax=ax, color="#7A6C5D")
ax.set_title("Average Days on Market by Property Type")
ax.set_ylabel("Days")
plt.tight_layout()
plt.savefig("../outputs/days_on_market_by_type.png", dpi=150)
plt.close()

metro_quarter.to_csv("../outputs/metro_quarter_median_price.csv", index=False)
forecast_df.to_csv("../outputs/metro_forecast.csv", index=False)
afford.to_csv("../outputs/affordability_by_suburb.csv", index=False)

print("\nSaved charts and tables to ../outputs/")
