# Requirements & Recommendation: RTT Performance Improvement

## 1. Current State

Performance against the 18-week Referral to Treatment (RTT) target is
currently monitored at trust level only, in aggregate, on a monthly cycle.
This masks two things:
- Which **specific specialty** within an underperforming trust is driving
  the breach
- Whether the root cause is a **referral surge** or a **treatment capacity
  shortfall** — these require entirely different interventions

## 2. Stakeholder Requirements

| Stakeholder | Requirement |
|---|---|
| ICB Performance Lead | A ranked view of trust×specialty combinations by capacity gap, refreshed monthly |
| Clinical Director (per specialty) | Visibility into whether their specialty's shortfall is system-wide or trust-specific |
| Trust Performance Manager | Early warning when a specialty's shortfall trend is worsening, not just a snapshot |

## 3. Proposed Future State

A trust×specialty level dashboard (see `outputs/` for the underlying charts)
that:
1. Ranks combinations by average monthly shortfall (referrals − treated)
2. Flags whether the shortfall is trending up, flat, or improving over the
   last 6 months
3. Separates "demand problem" (referrals rising) from "capacity problem"
   (treated declining) — these need different fixes and the current
   aggregate reporting cannot distinguish them

## 4. Recommendation

Prioritise resourcing review for the two combinations with the largest and
most persistent shortfall in this analysis:
1. **Neurology, University Hospitals Birmingham** (~43 patients/month
   shortfall)
2. **Neurology, Barts Health NHS Trust** (~41 patients/month shortfall)

Neurology appearing twice in the top gaps suggests this may be a
**specialty-wide capacity issue** (e.g. national neurologist shortage) rather
than something either trust can fully solve independently — this is worth
escalating to the ICB/specialty commissioning level rather than treating as
two isolated local problems.

## 5. Success Metrics

- Reduction in average monthly shortfall for the flagged trust×specialty
  pairs within two quarters
- % of patients over 18 weeks trending down at trust level, tracked monthly
  rather than only at year-end
- Time-to-escalation: how quickly a newly emerging shortfall (e.g. a
  specialty going from healthy to backlogged) is flagged vs. the current
  aggregate-only reporting cycle

## 6. Assumptions & Limitations

- This analysis uses synthetic data structured to mirror NHS England's
  published RTT statistics format — real referral/treatment volumes and
  regional variation will differ, and the specific trusts/specialties
  flagged here are illustrative of the *method*, not a live finding.
- Real analysis should also incorporate case-mix complexity (a hip
  replacement and a routine dermatology check are not equivalent "treated"
  units) before comparing specialties directly.
