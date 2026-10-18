# Australia Public Transport On-Time Performance & Cost-Benefit Analysis

**Type:** Data Analyst + Business Analyst combined project
**Market focus:** Australian public transport (Melbourne, Sydney, Brisbane —
modelled on PTV / Transport for NSW / TransLink style operations)
**Tools:** SQL (aggregation, window functions), Python (pandas), Power
BI/Tableau-ready CSV outputs

## Business Problem

A state transport agency's board wants to know which routes deliver the
worst rider experience once ridership is accounted for (a route with 60%
on-time performance but 200 daily riders matters less than one at 90% with
50,000 daily riders), and whether a proposed reliability investment is
justified on a cost-benefit basis.

**Stakeholders:** Network Performance team, Transport Agency board (budget
sign-off), Customer Experience team

**Objective:** Calculate on-time performance (OTP) by route and mode,
combine it with ridership to produce a "rider impact score" for
prioritisation, and build an illustrative cost-benefit case for the
highest-impact route.

## Data

Synthetic dataset (`python/generate_data.py`) modelled on the structure of
public operator performance reporting (PTV Victoria, Transport for NSW open
data, TransLink Queensland), since live operational data isn't publicly
redistributable at this granularity. Structure:

- `data/au_transport_performance.csv` — trip_log_id, route, mode, city, date,
  scheduled_services, on_time_services, disrupted_services,
  avg_delay_minutes, daily_patronage (10 routes × 730 days = 7,300 rows)

To use real data, the PTV and Transport for NSW open data portals publish
timetable adherence and patronage datasets with a compatible structure.

## Method

1. **SQL** (`sql/otp_analysis.sql`) — OTP % by route and mode, monthly
   system-wide trend, weekday/weekend comparison, and a patronage-weighted
   "rider impact score" query.
2. **Python** (`python/eda_and_cost_benefit.py`) — replicates the SQL
   analysis, then runs an illustrative cost-benefit calculation for the
   single highest-impact route.

## Key Findings (from the synthetic dataset)

- **Train services underperform other modes on OTP** (91.1% vs. 94.4% for
  bus, 94.5% for tram) despite carrying by far the most riders (7.88M vs.
  1.99M for bus over the study period) — this is where reliability
  investment has the most leverage.
- **Route 5 (Sydney Train)** has the highest rider impact score in the
  sample — not the worst raw OTP, but the combination of high disruption
  volume and very high ridership makes it the top prioritisation candidate.
- Ferry and train modes show the highest average delay minutes when
  disrupted (~4.6–4.9 min), consistent with real-world patterns where
  rail/water modes have less schedule flexibility than road-based transport.

## Business Recommendation

See `cost_benefit_case.md` for the full board-ready cost-benefit write-up on
Route 5, including the explicit assumptions that would need validating with
real operational cost data before this goes to a funding decision.

## How to Run

```bash
cd python
python3 generate_data.py            # creates data/au_transport_performance.csv
python3 eda_and_cost_benefit.py     # runs analysis, writes outputs/
```

## Repo Structure

```
04-au-public-transport/
├── data/                      # synthetic OTP/patronage data
├── sql/                       # OTP + rider-impact SQL
├── python/                    # data generation + EDA/cost-benefit scripts
├── outputs/                   # charts + tables (Power BI/Tableau-ready)
├── cost_benefit_case.md       # BA-style board-ready business case
└── README.md
```
