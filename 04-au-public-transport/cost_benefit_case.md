# Cost-Benefit Case: Reliability Investment on Route 5 (Sydney Train)

## 1. Problem
Route 5 has the highest "rider impact score" in the network — a combination
of disruption frequency and ridership volume that means service failures
here affect more people than on any other route in the sample, even though
its raw OTP (90.8%) is not the single worst in the network.

## 2. Why Rider Impact Score, Not Raw OTP
Raw OTP ranks routes by reliability alone and would send investment toward
low-ridership routes with poor percentages. Weighting disruption by ridership
answers the question the board actually cares about: **where does a
reliability fix affect the most people?**

## 3. Proposed Intervention
Illustrative reliability investment (e.g. additional rolling stock buffer,
signalling upgrade, or timetable padding at known pinch points) — **exact
scope and cost must come from Engineering/Operations**, not from this
analysis. Financial modelling below uses placeholder figures to demonstrate
the method.

## 4. Cost-Benefit Model (illustrative — replace all inputs with real figures)

| Input | Value | Source |
|---|---|---|
| Intervention cost | AUD 250,000 | **Placeholder** — needs Engineering estimate |
| Assumed OTP uplift | 8 percentage points | **Placeholder** — needs Ops estimate based on intervention type |
| Estimated avoided disrupted services/year | ~140 | Derived from historical disruption rate × assumed uplift |
| Assumed value of rider time saved per avoided delay (aggregated) | AUD 0.45/min | **Placeholder** — Transport agencies typically have a standard "value of travel time" figure; use the agency's official rate |

**This case is not fundable as-is** — it demonstrates the calculation
structure, not a validated recommendation. Before board submission, replace
every placeholder with a real Engineering/Finance estimate.

## 5. Recommendation
1. Commission an Engineering cost estimate specifically for Route 5's known
   disruption causes (the data alone can't diagnose *why* Route 5 disrupts —
   only *how much it matters when it does*).
2. Use the agency's official value-of-travel-time figure to replace the
   placeholder in this model.
3. Re-run the cost-benefit with real figures before requesting budget
   sign-off.

## 6. Success Metrics (if funded)
- OTP % on Route 5, tracked monthly against a control period
- Rider impact score trend (should fall even if OTP improvement is modest,
  because it accounts for the ridership actually affected)
- Cost per percentage-point of OTP improvement, to compare against
  alternative interventions on other high-impact routes

## 7. Limitations
- This dataset is synthetic; the specific route flagged is illustrative of
  method, not a real network finding.
- The model treats "value of rider time" as a single aggregated figure —
  a production version should account for peak vs. off-peak trip purposes,
  since time value assumptions differ by trip type (commute vs. leisure).
