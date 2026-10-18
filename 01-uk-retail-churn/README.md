# UK Retail Customer Churn & Revenue Impact

**Type:** Data Analyst + Business Analyst combined project
**Market focus:** UK e-commerce/retail
**Tools:** SQL (RFM segmentation), Python (pandas, scikit-learn), Power BI/Tableau-ready CSV outputs

## Business Problem

A mid-sized UK online retailer wants to understand how much revenue is at risk from
disengaged customers and which customer segments deserve retention investment.
Marketing has no formal segmentation model and retention spend is currently
allocated evenly across the customer base, regardless of value or churn risk.

**Stakeholders:** Head of CRM, Marketing Director, Finance (for ROI sign-off)

**Objective:** Segment the customer base by recency/frequency/monetary (RFM)
value, quantify revenue at risk, and identify the strongest churn drivers to
inform a targeted retention campaign.

## Data

Synthetic dataset generated to mirror a realistic UK retail transaction log
(`python/generate_data.py`), because a licensed real-world dataset can't be
redistributed publicly. Structure mirrors real retailer exports:

- `data/customers.csv` — customer_id, city, signup_date, acquisition_channel (1,200 customers)
- `data/transactions.csv` — transaction_id, customer_id, transaction_date, category, order_value_gbp, discount_used (15,000 transactions)

To swap in a real dataset (e.g. the UCI/Kaggle "Online Retail II" UK dataset),
keep the same column names and re-run the pipeline unchanged.

## Method

1. **SQL** (`sql/rfm_analysis.sql`) — RFM scoring with window functions
   (`NTILE`), segment classification, and a monthly cohort retention query.
2. **Python** (`python/eda_and_churn_model.py`) — replicates the RFM logic in
   pandas, quantifies revenue at risk, and fits a logistic regression churn
   model (with an explicit note on avoiding target leakage — recency is
   excluded as a feature because it defines the churn label itself).
3. **Outputs** — segment tables and charts saved to `outputs/` for direct use
   in Power BI/Tableau.

## Key Findings (from the synthetic dataset — will change with real data)

| Segment | Customers | Avg. Lifetime Spend | Avg. Recency (days) |
|---|---|---|---|
| Champions | 297 | £737.80 | 15.9 |
| At Risk (High Value) | 209 | £706.80 | 68.9 |
| Regular | 94 | £489.30 | 43.5 |
| Recent but Low Engagement | 252 | £436.90 | 16.3 |
| Lost / Churned | 348 | £424.20 | 89.0 |

- **~44% of historical revenue** sits with customers who are either "At Risk"
  or already "Lost / Churned" — this is the segment retention spend should
  prioritise.
- Observed churn rate (>90 days inactive): **14.1%**.
- **Purchase frequency is the strongest behavioural churn signal** (declining
  order frequency precedes churn more reliably than acquisition channel or
  city). This is what a real-world model should lean on most heavily.
- The churn model's accuracy here is deliberately modest (ROC-AUC ~0.59)
  because the synthetic data has no engineered churn drivers baked in — the
  point of this repo is the *methodology*, which transfers directly once
  real behavioural data is loaded.

## Business Recommendation

Reallocate the next quarter's retention budget away from blanket discounting
and toward the "At Risk (High Value)" segment specifically — see
`business_case.md` for the full write-up, cost-benefit reasoning, and
proposed KPIs to track post-launch.

## How to Run

```bash
cd python
python3 generate_data.py          # creates data/customers.csv, data/transactions.csv
python3 eda_and_churn_model.py    # runs RFM + churn model, writes outputs/
```

Load `sql/rfm_analysis.sql` against the two CSVs in PostgreSQL/SQLite to
reproduce the segmentation in pure SQL.

## Repo Structure

```
01-uk-retail-churn/
├── data/                # synthetic transaction data
├── sql/                 # RFM + cohort SQL
├── python/              # data generation + EDA/model scripts
├── outputs/             # charts + segment tables (Power BI/Tableau-ready)
├── business_case.md     # BA-style stakeholder document
└── README.md
```
