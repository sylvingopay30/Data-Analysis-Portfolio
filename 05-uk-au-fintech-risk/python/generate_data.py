import pandas as pd
import numpy as np

np.random.seed(55)

n = 8000
countries = np.random.choice(["UK","Australia"], n, p=[0.55,0.45])
ages = np.random.randint(21, 70, n)
annual_income = np.round(np.random.gamma(shape=3.0, scale=15000, size=n) + 18000, -2)
employment_status = np.random.choice(["Employed","Self-Employed","Contract","Unemployed"], n, p=[0.65,0.15,0.15,0.05])
credit_score = np.clip(np.round(np.random.normal(680, 80, n)), 300, 850).astype(int)
loan_amount = np.round(np.random.gamma(shape=2.0, scale=4000, size=n) + 1000, -2)
loan_term_months = np.random.choice([12,24,36,48,60], n, p=[0.15,0.25,0.3,0.2,0.1])
existing_debt_ratio = np.round(np.clip(np.random.beta(2,5,n), 0, 0.95), 2)
purpose = np.random.choice(["Debt Consolidation","Auto","Home Improvement","Personal","Business","Education"], n)

# default probability driven by realistic factors (for a meaningful model)
logit = (
    -3.0
    + (700 - credit_score) * 0.012
    + existing_debt_ratio * 3.2
    + (employment_status == "Unemployed") * 1.8
    + (employment_status == "Contract") * 0.4
    + (loan_amount / annual_income) * 1.5
    - (annual_income / 100000) * 0.6
    + np.random.normal(0, 0.6, n)
)
prob_default = 1 / (1 + np.exp(-logit))
default = np.random.binomial(1, prob_default)

df = pd.DataFrame({
    "loan_id": [f"LN{str(i).zfill(6)}" for i in range(1, n+1)],
    "country": countries,
    "age": ages,
    "annual_income": annual_income,
    "employment_status": employment_status,
    "credit_score": credit_score,
    "loan_amount": loan_amount,
    "loan_term_months": loan_term_months,
    "existing_debt_ratio": existing_debt_ratio,
    "purpose": purpose,
    "default": default
})

df.to_csv("../data/loan_applications.csv", index=False)
print(df.shape)
print("Default rate:", df["default"].mean().round(3))
print(df.head())
