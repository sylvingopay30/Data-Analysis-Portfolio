-- ============================================================
-- UK NHS Wait Times & Resource Allocation
-- Target: PostgreSQL / SQLite compatible
-- ============================================================

-- 1. Trust-level performance against the 18-week referral-to-treatment target
SELECT
    trust,
    region,
    ROUND(AVG(median_wait_weeks), 1) AS avg_median_wait_weeks,
    ROUND(AVG(pct_over_18weeks), 1)  AS avg_pct_over_18weeks,
    SUM(waiting_list_size)           AS total_waiting_list
FROM nhs_wait_times
GROUP BY trust, region
ORDER BY avg_pct_over_18weeks DESC;

-- 2. Specialty-level bottleneck identification (worst performers system-wide)
SELECT
    specialty,
    ROUND(AVG(median_wait_weeks), 1) AS avg_median_wait_weeks,
    ROUND(AVG(pct_over_18weeks), 1)  AS avg_pct_over_18weeks,
    ROUND(AVG(referrals - treated), 0) AS avg_monthly_backlog_growth
FROM nhs_wait_times
GROUP BY specialty
ORDER BY avg_pct_over_18weeks DESC;

-- 3. Trend over time: system-wide waiting list size by month
SELECT
    month,
    SUM(waiting_list_size) AS total_waiting_list,
    ROUND(AVG(median_wait_weeks), 1) AS avg_median_wait_weeks
FROM nhs_wait_times
GROUP BY month
ORDER BY month;

-- 4. Capacity gap: trusts/specialties where referrals consistently exceed
--    treatment capacity (candidates for resource reallocation)
SELECT
    trust,
    specialty,
    ROUND(AVG(referrals), 0)  AS avg_monthly_referrals,
    ROUND(AVG(treated), 0)    AS avg_monthly_treated,
    ROUND(AVG(referrals - treated), 0) AS avg_monthly_shortfall,
    RANK() OVER (ORDER BY AVG(referrals - treated) DESC) AS shortfall_rank
FROM nhs_wait_times
GROUP BY trust, specialty
ORDER BY shortfall_rank
LIMIT 15;
