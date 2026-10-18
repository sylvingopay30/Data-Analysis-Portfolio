# Underwriting Policy Memo: Risk-Based Segmentation Findings

## 1. Purpose
This memo translates the risk segmentation analysis into a proposed
underwriting policy adjustment, framed the way it would be presented to a
Credit Risk Committee.

## 2. Current Policy Gap
Underwriting currently relies primarily on the aggregate credit score
threshold. This analysis shows that **debt-to-income ratio and employment
status add meaningful, independent predictive power** beyond credit score
alone — two applicants with identical credit scores can have very different
default risk depending on these factors.

## 3. Findings Relevant to Policy

| Factor | Finding | Policy Relevance |
|---|---|---|
| Credit score | Default rate ranges 48.3% (sub-580) to 3.0% (800+) | Already the primary underwriting lever — correctly weighted |
| Existing debt ratio | 46.5% default at 0.6+ DTI vs. 14.1% at <0.2 | **Not currently a hard cutoff** — recommend introducing a DTI ceiling for marginal credit bands |
| Employment status | Unemployed applicants default at ~2.5x the employed rate | Should be a required declarative field with a pricing/eligibility flag, not just informational |
| Country (UK vs. AU) | Minimal difference in default rate | Do not need separate underwriting models per country on this evidence — a shared model is defensible |
| Loan purpose | Minimal difference (18.9%–21.8% range) | Not a strong enough signal to justify purpose-based pricing tiers |

## 4. Recommendation
1. Introduce a **combined credit score + DTI eligibility matrix** rather than
   scoring DTI as a minor secondary factor — applicants in the "Fair"
   credit band (580–669) with high DTI (0.6+) should be treated closer to
   "Poor" band risk, not "Fair" band risk, based on this data.
2. Add employment status as an **explicit underwriting flag**, not just a
   model input — unemployed applicants show default rates close to the
   worst credit band regardless of their credit score.
3. Do **not** introduce country-specific or purpose-specific pricing tiers
   based on this analysis — the differences observed are too small to be
   confidently actionable and risk adding complexity without benefit.

## 5. Compliance Considerations
- Any segmentation used for pricing or eligibility decisions must be
  reviewed against **FCA Consumer Duty** requirements (UK) and **ASIC
  Responsible Lending obligations** (Australia) before implementation —
  this analysis identifies statistical patterns, not a compliance-cleared
  policy.
- Employment status and DTI-based adjustments must be checked for
  disparate impact on protected characteristics before deployment, even
  though those characteristics aren't in this dataset.

## 6. Model Limitations
- ROC-AUC of 0.743 is a reasonable starting point but not production-grade;
  a deployed model would need broader features (payment history, existing
  relationship with the lender, bureau data) and ongoing monitoring for
  drift.
- This dataset is synthetic, constructed so that credit score, DTI, and
  employment status drive default risk in a realistic direction — the
  *relationships* are illustrative of real lending patterns, but the exact
  percentages will differ from any real loan book and must not be used as
  actual risk parameters.

## 7. Next Steps
1. Validate the proposed credit score × DTI matrix against a real
   (anonymised) historical loan book.
2. Route the compliance review with Legal before any pricing change.
3. Pilot the revised matrix on new originations only, with a control group,
   before full rollout.
