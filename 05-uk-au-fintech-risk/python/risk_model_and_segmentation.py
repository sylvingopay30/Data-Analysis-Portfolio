"""
UK/AU Fintech Loan Default & Risk Segmentation
EDA, risk-tier segmentation, and a classification model with feature importance.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, roc_curve

df = pd.read_csv("../data/loan_applications.csv")
print(f"Overall default rate: {df['default'].mean():.1%}\n")

# ---- Credit score band ----
bins = [0, 580, 670, 740, 800, 999]
labels = ["<580 (Poor)", "580-669 (Fair)", "670-739 (Good)", "740-799 (Very Good)", "800+ (Excellent)"]
df["credit_band"] = pd.cut(df["credit_score"], bins=bins, labels=labels)
band_summary = df.groupby("credit_band", observed=True).agg(
    applications=("loan_id","count"),
    default_rate=("default","mean")
)
band_summary["default_rate"] = (band_summary["default_rate"]*100).round(1)
print("=== Default rate by credit band ===")
print(band_summary.to_string())

# ---- Country x employment ----
country_emp = df.groupby(["country","employment_status"]).agg(
    applications=("loan_id","count"),
    default_rate=("default","mean"),
    avg_loan=("loan_amount","mean")
).round(3).sort_values("default_rate", ascending=False)
country_emp["default_rate"] = (country_emp["default_rate"]*100).round(1)
print("\n=== Default rate by country x employment status ===")
print(country_emp.to_string())

# ---- Debt ratio band ----
df["dti_band"] = pd.cut(df["existing_debt_ratio"], bins=[-0.01,0.2,0.4,0.6,1.0],
                          labels=["Low (<0.2)","Moderate (0.2-0.4)","Elevated (0.4-0.6)","High (0.6+)"])
dti_summary = df.groupby("dti_band", observed=True).agg(
    applications=("loan_id","count"),
    default_rate=("default","mean")
)
dti_summary["default_rate"] = (dti_summary["default_rate"]*100).round(1)
print("\n=== Default rate by debt-to-income band ===")
print(dti_summary.to_string())

# ---- Purpose risk ranking ----
purpose_summary = df.groupby("purpose").agg(
    applications=("loan_id","count"),
    default_rate=("default","mean"),
    total_defaulted_exposure=("loan_amount", lambda x: (x * df.loc[x.index,"default"]).sum())
).sort_values("default_rate", ascending=False)
purpose_summary["default_rate"] = (purpose_summary["default_rate"]*100).round(1)
print("\n=== Default rate and exposure by loan purpose ===")
print(purpose_summary.to_string())

# ---- Classification model ----
model_df = df.copy()
le_emp = LabelEncoder()
le_purpose = LabelEncoder()
le_country = LabelEncoder()
model_df["employment_status_enc"] = le_emp.fit_transform(model_df["employment_status"])
model_df["purpose_enc"] = le_purpose.fit_transform(model_df["purpose"])
model_df["country_enc"] = le_country.fit_transform(model_df["country"])

feature_cols = ["age","annual_income","credit_score","loan_amount","loan_term_months",
                 "existing_debt_ratio","employment_status_enc","purpose_enc","country_enc"]
X = model_df[feature_cols]
y = model_df["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, probs)
print(f"\n=== Random Forest default model: ROC-AUC = {auc:.3f} ===")
print(classification_report(y_test, clf.predict(X_test)))

importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n=== Feature importance ===")
print(importances.round(3).to_string())

# ---- Charts ----
fig, ax = plt.subplots(figsize=(7,4.5))
band_summary["default_rate"].plot(kind="bar", ax=ax, color="#6A0572")
ax.set_title("Default Rate % by Credit Score Band")
ax.set_ylabel("Default rate %")
plt.tight_layout()
plt.savefig("../outputs/default_rate_by_credit_band.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7,4.5))
importances.plot(kind="barh", ax=ax, color="#AB4967")
ax.set_title("Feature Importance (Random Forest)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("../outputs/feature_importance.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, probs)
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(fpr, tpr, color="#2E5266", label=f"AUC = {auc:.3f}")
ax.plot([0,1],[0,1], linestyle="--", color="grey")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - Default Prediction Model")
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/roc_curve.png", dpi=150)
plt.close()

band_summary.to_csv("../outputs/default_by_credit_band.csv")
country_emp.to_csv("../outputs/default_by_country_employment.csv")
purpose_summary.to_csv("../outputs/default_by_purpose.csv")
importances.to_csv("../outputs/feature_importance.csv")

print("\nSaved charts and tables to ../outputs/")
