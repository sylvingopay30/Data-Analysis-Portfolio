# Australia Housing Market Price Trends

**Type:** Data Analyst + Business Analyst combined project
**Market focus:** Australian residential property (Sydney, Melbourne,
Brisbane, Perth, Adelaide)
**Tools:** SQL (window functions, percentiles), Python (pandas, scikit-learn
linear trend), Power BI/Tableau-ready CSV outputs

## Business Problem

A property investment advisory firm wants a data-driven view of which
Australian metro areas and suburbs offer the best growth trajectory versus
affordability trade-off, to brief clients on where to allocate capital over
the next 12 months.

**Stakeholders:** Investment Advisors, Client-facing Relationship Managers,
Compliance (for how recommendations are worded/caveated)

**Objective:** Quantify price growth trends by metro area and suburb,
identify the most/least affordable suburbs on a price-per-bedroom basis, and
produce a short forward-looking trend estimate to support client
conversations.

## Data

Synthetic dataset (`python/generate_data.py`) modelled on the structure of
public sources like the ABS Residential Property Price Indexes and
Domain/CoreLogic-style listing exports, since live commercial data can't be
redistributed. Structure:

- `data/au_property_sales.csv` — listing_id, suburb, state, metro_area,
  property_type, bedrooms, sale_price_aud, sale_date, days_on_market
  (6,000 sales across 12 suburbs, 2021–2024)

To use real data, point the pipeline at the ABS Residential Property Price
Indexes (abs.gov.au) or a state land registry export with matching columns.

## Method

1. **SQL** (`sql/price_trend_analysis.sql`) — median price by metro/quarter,
   year-on-year suburb growth via self-join, affordability ranking, and
   days-on-market outlier detection.
2. **Python** (`python/eda_and_forecast.py`) — replicates the SQL logic,
   adds a simple linear trend forecast per metro area for the next two
   quarters, and saves charts.

## Key Findings (from the synthetic dataset)

- **Sydney and Melbourne** lead on both absolute median price (~$976K and
  ~$856K in the latest quarter) and quarterly growth trend (~$6,500–$7,600
  per quarter).
- **Brisbane's trend has flattened** relative to Sydney/Melbourne/Adelaide in
  this sample — worth flagging to clients expecting continued rapid growth.
- **Most affordable on a price-per-bedroom basis:** Elizabeth (SA) and
  Midland (WA). **Least affordable:** Bondi (NSW) and Richmond (VIC).
- Total growth 2021→2024 was highest in **Glenelg (SA, +18.3%)** and
  **Blacktown (NSW, +17.2%)** — both outer/middle-ring suburbs, consistent
  with a real-world pattern of affordability-driven demand shifting outward.

## Business Recommendation

See `investment_brief.md` for the client-facing recommendation, including
which segment of buyer each suburb suits and the caveats that must accompany
any forward-looking estimate.

## How to Run

```bash
cd python
python3 generate_data.py       # creates data/au_property_sales.csv
python3 eda_and_forecast.py    # runs analysis, writes outputs/
```

Load `sql/price_trend_analysis.sql` against the CSV (import into
PostgreSQL/SQLite as table `au_property_sales`) to reproduce the SQL-only
version.

## Repo Structure

```
02-au-housing-market/
├── data/                    # synthetic property sales data
├── sql/                     # trend + affordability SQL
├── python/                  # data generation + EDA/forecast scripts
├── outputs/                 # charts + tables (Power BI/Tableau-ready)
├── investment_brief.md      # BA-style client-facing recommendation
└── README.md
```
