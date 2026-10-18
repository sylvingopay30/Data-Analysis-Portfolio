# UK NHS Wait Times & Resource Allocation

**Type:** Data Analyst + Business Analyst combined project
**Market focus:** UK public sector / healthcare (NHS)
**Tools:** SQL (window functions, ranking), Python (pandas), Power BI/Tableau-
ready CSV outputs

## Business Problem

NHS trusts are measured against the 18-week referral-to-treatment target.
Regional/system leadership needs to identify which trusts and specialties
are furthest from that target and where referral demand is consistently
outpacing treatment capacity, so resource reallocation (staffing, additional
clinics, outsourced capacity) can be prioritised.

**Stakeholders:** Trust Performance Managers, Integrated Care Board (ICB)
analysts, Clinical Directors by specialty

**Objective:** Rank trusts and specialties by 18-week target performance,
identify the specific trust×specialty combinations with the largest capacity
gap, and translate this into a resourcing recommendation.

## Data

Synthetic dataset (`python/generate_data.py`) modelled on the structure of
NHS England's published Referral to Treatment (RTT) waiting times statistics
(monthly, by trust and specialty), since the live published tables are large
and reformatted often. Structure:

- `data/nhs_wait_times.csv` — record_id, trust, region, specialty, month,
  referrals, treated, waiting_list_size, median_wait_weeks, pct_over_18weeks
  (10 trusts × 10 specialties × 36 months = 3,600 rows)

To use real data, download NHS England's monthly RTT waiting times statistics
(NHS England Statistics — "RTT Waiting Times") and map columns to this
schema.

## Method

1. **SQL** (`sql/wait_time_analysis.sql`) — trust and specialty performance
   ranking against the 18-week target, system-wide trend, and a capacity-gap
   query using `RANK()`.
2. **Python** (`python/eda_and_bottleneck_analysis.py`) — replicates the
   analysis in pandas and produces charts for a dashboard.

## Key Findings (from the synthetic dataset)

- **University Hospitals Birmingham** is the worst-performing trust in the
  sample: 42.0% of patients waiting over 18 weeks, vs. a best-in-sample 16.1%
  at NHS Lothian.
- **Cardiology, Gastroenterology, and Neurology** are the weakest specialties
  system-wide on the 18-week measure (~26% over target).
- The single largest capacity gap is **Neurology at University Hospitals
  Birmingham** — referrals exceed treatment capacity by ~43 patients/month on
  average, the highest shortfall of any trust×specialty pair in the dataset.
- System-wide waiting list size was roughly flat over the study period in
  this sample (-1.5%), masking significant divergence at trust level — the
  aggregate number would have hidden the Birmingham problem entirely.

## Business Recommendation

See `requirements_and_recommendation.md` for the BA-style write-up:
problem framing, current-state vs. proposed-state process notes, and the
resourcing recommendation aimed at the two worst-performing trust×specialty
combinations first.

## How to Run

```bash
cd python
python3 generate_data.py                    # creates data/nhs_wait_times.csv
python3 eda_and_bottleneck_analysis.py       # runs analysis, writes outputs/
```

## Repo Structure

```
03-uk-nhs-wait-times/
├── data/                                # synthetic RTT-style data
├── sql/                                 # performance + capacity-gap SQL
├── python/                              # data generation + EDA scripts
├── outputs/                             # charts + tables (Power BI/Tableau-ready)
├── requirements_and_recommendation.md   # BA-style stakeholder document
└── README.md
```
