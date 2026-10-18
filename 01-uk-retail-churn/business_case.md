# Business Case: Targeted Retention Programme for At-Risk High-Value Customers

## 1. Background
The retail business currently spends its retention/marketing budget uniformly
across the customer base. Analysis of transaction history shows this is
inefficient: a specific, identifiable segment ("At Risk — High Value")
represents a large share of revenue but is trending toward churn.

## 2. Problem Statement
Without segmentation, retention spend is diluted across customers who would
have stayed anyway (Champions) and customers unlikely to respond to any offer
(Lost/Churned), while the highest-value at-risk group receives no
differentiated treatment.

## 3. Stakeholders

| Stakeholder | Interest |
|---|---|
| Head of CRM | Owns retention campaign design and execution |
| Marketing Director | Owns budget allocation and campaign ROI |
| Finance | Requires ROI justification before releasing budget |
| Data/Analytics team | Delivers segmentation and tracks post-launch performance |

## 4. Proposed Solution
Launch a targeted win-back campaign for the "At Risk (High Value)" segment
(209 customers in the sample data, ~£147K in historical lifetime spend),
using:
- A personalised discount or loyalty incentive (not a blanket sitewide sale)
- Triggered email/SMS sequences timed to each customer's historical
  purchase cadence
- A control group held out to measure incremental lift

## 5. Cost-Benefit Analysis (illustrative — recalibrate with real campaign costs)

| Item | Estimate |
|---|---|
| Target segment size | 209 customers |
| Average historical spend per customer | £706.80 |
| Assumed reactivation rate | 20% |
| Assumed average order value post-reactivation | £75 |
| Campaign cost (design + incentive + platform) | £6,000 |
| Projected incremental revenue (209 × 20% × £75) | £3,135 per campaign cycle |
| Break-even reactivation rate needed | ~38% cost-to-incremental-revenue ratio at these assumptions |

**Note:** a single campaign cycle at these illustrative assumptions does not
clear the cost — the business case depends on repeat-purchase value over a
12-month horizon, not a single order. The recommendation below reflects that.

## 6. Recommendation
Run a 3-month pilot on a randomly selected half of the "At Risk (High Value)"
segment, holding out the other half as a control. Track 12-month retained
revenue per customer (not just the first reactivation order) before scaling
spend. This avoids committing the full budget on unvalidated assumptions.

## 7. Success Metrics (KPIs)

- Reactivation rate within 30/60/90 days of campaign send
- 12-month retained revenue per targeted customer vs. control group
- Cost per incremental retained customer
- Segment migration: % of "At Risk" customers moving to "Regular" or
  "Champions" within two quarters

## 8. Risks & Assumptions
- Synthetic data used for this repo has no real churn-cause signal built in;
  a production rollout requires validating the RFM thresholds and churn
  model against actual behavioural data before sizing the campaign budget.
- Discount-led reactivation can train high-value customers to wait for
  discounts — monitor margin impact, not just revenue.
