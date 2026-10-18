# UK/AU Fintech Loan Default & Risk Segmentation

**Type:** Data Analyst + Business Analyst combined project
**Market focus:** UK and Australian consumer lending/fintech
**Tools:** SQL (segmentation), Python (pandas, scikit-learn Random Forest),
Power BI/Tableau-ready CSV outputs

## Business Problem

A fintech lender operating in both the UK and Australia wants a clearer,
data-backed view of which applicant segments carry disproportionate default
risk, so underwriting policy and pricing can be adjusted without relying
solely on a black-box credit score.

**Stakeholders:** Head of Credit Risk, Compliance (FCA in the UK / ASIC in
Australia), Product/Pricing team

**Objective:** Segment default rate by credit band, employment status,
country, and debt-to-income ratio; build a classification model to rank
feature importance; and translate this into an underwriting policy
recommendation.

## Data

Synthetic dataset (`python/generate_data.py`) with a default probability
deliberately driven by realistic factors (credit score, debt-to-income,
employment status, loan-to-income ratio) so the resulting model behaves the
way a real credit risk model would — this is different from Project 1's
churn model, which intentionally used data with no built-in signal. Real
lending data is never publicly redistributable due to privacy regulation, so
this is built to mirror the *structure and relationships* of a real consumer
lending book. Structure:

- `data/loan_applications.csv` — loan_id, country, age, annual_income,
  employment_status, credit_score, loan_amount, loan_term_months,
  existing_debt_ratio, purpose, default (8,000 applications, 20.8% default
  rate)

## Method

1. **SQL** (`sql/risk_segmentation.sql`) — default rate by credit band,
   country × employment status, debt-to-income band, and loan purpose, plus
   a combined high-risk segment exposure query.
2. **Python** (`python/risk_model_and_segmentation.py`) — replicates the SQL
   segmentation and fits a Random Forest classifier with feature importance
   and an ROC curve.

## Key Findings (from the synthetic dataset)

- **Default rate drops sharply with credit score**: 48.3% for sub-580
  applicants vs. 3.0% for 800+ — as expected, but the size of the gap
  matters for pricing tiers.
- **Unemployed applicants default at ~2.5x the rate of employed applicants**
  in both countries (Australia 48.8%, UK 45.6%, vs. ~18.6% for employed in
  both).
- **Debt-to-income ratio is the second-strongest driver** after credit
  score: applicants with existing debt ratio 0.6+ default at 46.5%, more
  than 3x the rate of the <0.2 band.
- The Random Forest model achieves **ROC-AUC 0.743** on held-out data —
  credit score alone accounts for ~49% of feature importance, with
  debt-to-income ratio a distant but meaningful second (~20%).
- Loan purpose and country show only minor differences in default rate,
  suggesting underwriting policy should weight credit score and DTI far more
  heavily than either of those two factors.

## Business Recommendation

See `risk_policy_memo.md` for the underwriting policy recommendation,
including where the current findings would and wouldn't justify a policy
change, and the compliance considerations for adjusting pricing by segment.

## How to Run

```bash
cd python
python3 generate_data.py                    # creates data/loan_applications.csv
python3 risk_model_and_segmentation.py      # runs analysis + model, writes outputs/
```

## Repo Structure

```
05-uk-au-fintech-risk/
├── data/                     # synthetic loan application data
├── sql/                      # segmentation SQL
├── python/                   # data generation + model/segmentation scripts
├── outputs/                  # charts + tables (Power BI/Tableau-ready)
├── risk_policy_memo.md       # BA-style underwriting policy recommendation
└── README.md
```
