"""
UK Retail Customer Churn & Revenue Impact
EDA + RFM segmentation (in pandas, mirroring sql/rfm_analysis.sql) + logistic regression churn model.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

pd.set_option("display.width", 120)

customers = pd.read_csv("../data/customers.csv", parse_dates=["signup_date"])
transactions = pd.read_csv("../data/transactions.csv", parse_dates=["transaction_date"])

snapshot_date = transactions["transaction_date"].max() + pd.Timedelta(days=1)

# ---- RFM ----
rfm = transactions.groupby("customer_id").agg(
    frequency=("transaction_id", "count"),
    monetary=("order_value_gbp", "sum"),
    last_order=("transaction_date", "max")
).reset_index()
rfm["recency_days"] = (snapshot_date - rfm["last_order"]).dt.days

rfm["r_score"] = pd.qcut(rfm["recency_days"], 4, labels=[4,3,2,1]).astype(int)
rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1,2,3,4]).astype(int)
rfm["m_score"] = pd.qcut(rfm["monetary"], 4, labels=[1,2,3,4]).astype(int)

def segment(row):
    if row.r_score >= 3 and row.f_score >= 3 and row.m_score >= 3:
        return "Champions"
    if row.r_score >= 3 and row.f_score <= 2:
        return "Recent but Low Engagement"
    if row.r_score <= 2 and row.f_score >= 3 and row.m_score >= 3:
        return "At Risk (High Value)"
    if row.r_score <= 2 and row.f_score <= 2:
        return "Lost / Churned"
    return "Regular"

rfm["segment"] = rfm.apply(segment, axis=1)
seg_summary = rfm.groupby("segment").agg(
    customers=("customer_id", "count"),
    avg_spend=("monetary", "mean"),
    avg_recency=("recency_days", "mean")
).sort_values("avg_spend", ascending=False)
print("=== Segment summary ===")
print(seg_summary.round(1))

revenue_at_risk = rfm.loc[rfm["segment"].isin(["At Risk (High Value)", "Lost / Churned"]), "monetary"].sum()
total_revenue = rfm["monetary"].sum()
print(f"\nRevenue at risk: GBP {revenue_at_risk:,.2f} ({revenue_at_risk/total_revenue:.1%} of total historical revenue)")

# ---- Churn label: no purchase in the most recent 90 days (relative to snapshot) ----
rfm["churned"] = (rfm["recency_days"] > 90).astype(int)
churn_rate = rfm["churned"].mean()
print(f"Observed churn rate (>90 days inactive): {churn_rate:.1%}")

# ---- Feature engineering for churn model ----
model_df = rfm.merge(customers, on="customer_id")
model_df["tenure_days"] = (snapshot_date - model_df["signup_date"]).dt.days
model_df = pd.get_dummies(model_df, columns=["acquisition_channel", "city"], drop_first=True)

# Note: recency_days is excluded as a feature because it directly defines the
# churn label (>90 days inactive) - including it would leak the target and
# produce an artificially perfect model.
feature_cols = [c for c in model_df.columns if c not in
                ["customer_id","last_order","signup_date","segment","churned","r_score","f_score","m_score","recency_days"]]
X = model_df[feature_cols]
y = model_df["churned"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
print(f"\nChurn model ROC-AUC: {auc:.3f}")
print(classification_report(y_test, clf.predict(X_test)))

# ---- Feature importance (coefficients) ----
coef_df = pd.DataFrame({"feature": feature_cols, "coef": clf.coef_[0]}).sort_values("coef", key=abs, ascending=False).head(10)
print("\nTop churn drivers (by |coefficient|):")
print(coef_df.to_string(index=False))

# ---- Charts ----
fig, ax = plt.subplots(figsize=(7,4.5))
seg_summary["customers"].plot(kind="bar", ax=ax, color="#2E5266")
ax.set_title("Customer Count by RFM Segment")
ax.set_ylabel("Customers")
plt.tight_layout()
plt.savefig("../outputs/segment_counts.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7,4.5))
seg_summary["avg_spend"].plot(kind="bar", ax=ax, color="#A63446")
ax.set_title("Average Lifetime Spend (GBP) by Segment")
ax.set_ylabel("GBP")
plt.tight_layout()
plt.savefig("../outputs/segment_avg_spend.png", dpi=150)
plt.close()

monthly = transactions.set_index("transaction_date").resample("ME")["order_value_gbp"].sum()
fig, ax = plt.subplots(figsize=(8,4.5))
monthly.plot(ax=ax, color="#1B998B")
ax.set_title("Monthly Revenue Trend")
ax.set_ylabel("GBP")
plt.tight_layout()
plt.savefig("../outputs/monthly_revenue_trend.png", dpi=150)
plt.close()

rfm.to_csv("../outputs/rfm_segments.csv", index=False)
seg_summary.to_csv("../outputs/segment_summary.csv")

print("\nSaved charts and segment tables to ../outputs/")
