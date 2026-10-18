# Investment Brief: Metro & Suburb Allocation Recommendation

## 1. Purpose
This brief translates the price-trend and affordability analysis into a
client-ready recommendation, structured the way an advisory firm would
present it internally before client delivery.

## 2. Requirement
Relationship Managers need a defensible, data-backed short-list of
suburbs/metro areas to discuss with two client profiles:
- **Growth-focused investors** — willing to pay a premium for stronger
  historical trend
- **Yield/affordability-focused investors** — prioritising entry price and
  future upside from a lower base

## 3. Findings Summary

| Metro Area | Latest Median Price | Quarterly Trend | Read |
|---|---|---|---|
| Sydney | $976,000 | +$6,519/qtr | Strong but expensive entry point |
| Melbourne | $855,500 | +$7,550/qtr | Strongest trend momentum of the majors |
| Perth | $593,500 | +$5,686/qtr | Solid growth, lower entry price |
| Adelaide | $566,000 | +$7,620/qtr | Highest trend-per-dollar of the five |
| Brisbane | $627,000 | +$329/qtr | Growth has stalled in this sample |

## 4. Recommendation by Client Profile

**Growth-focused clients:** Melbourne and Adelaide show the strongest
quarterly trend in this dataset relative to their price base — Adelaide in
particular offers the highest trend-per-dollar-invested ratio.

**Affordability-focused clients:** Elizabeth (SA) and Midland (WA) offer the
lowest price-per-bedroom in the sample, with Glenelg (SA) and Blacktown (NSW)
showing the strongest total growth over the study period from a comparable
starting base.

**Caution flag:** Brisbane's trend has flattened in this sample and should
not be presented as a high-growth market without more recent data — this is
exactly the kind of claim Compliance will ask to see the underlying numbers
for.

## 5. How This Should Be Used
This is a **directional screening tool**, not investment advice. Every
number here comes from a synthetic dataset built to mirror the *structure* of
real property data (ABS price indices, Domain/CoreLogic exports) — before any
client-facing use, the pipeline must be re-run against real, current data,
and figures must carry standard investment disclaimers.

## 6. Next Steps
1. Replace synthetic data with a live ABS/CoreLogic feed (same schema).
2. Extend the trend model beyond linear (e.g. seasonal decomposition) once
   real multi-year data is available — 12 quarters of synthetic data is too
   short to trust a linear extrapolation for genuine forecasting.
3. Add a rental-yield dimension so "growth" and "yield" can be compared side
   by side, since Growth vs. Affordability clients often actually want yield.
