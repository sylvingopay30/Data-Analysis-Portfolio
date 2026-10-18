-- ============================================================
-- UK/AU Fintech Loan Default & Risk Segmentation
-- Target: PostgreSQL / SQLite compatible
-- NOTE: the CSV column "default" is a reserved word in SQL. When loading
-- loan_applications.csv into a database, rename it to "default_flag"
-- (as used throughout this file) or quote it as "default" consistently.
-- ============================================================

-- 1. Default rate by credit score band
SELECT
    CASE
        WHEN credit_score < 580 THEN '<580 (Poor)'
        WHEN credit_score < 670 THEN '580-669 (Fair)'
        WHEN credit_score < 740 THEN '670-739 (Good)'
        WHEN credit_score < 800 THEN '740-799 (Very Good)'
        ELSE '800+ (Excellent)'
    END AS credit_band,
    COUNT(*) AS applications,
    SUM(default_flag) AS defaults,
    ROUND(100.0 * SUM(default_flag) / COUNT(*), 1) AS default_rate_pct
FROM loan_applications
GROUP BY credit_band
ORDER BY MIN(credit_score);

-- 2. Default rate by country and employment status
SELECT
    country,
    employment_status,
    COUNT(*) AS applications,
    ROUND(100.0 * SUM(default_flag) / COUNT(*), 1) AS default_rate_pct,
    ROUND(AVG(loan_amount), 0) AS avg_loan_amount
FROM loan_applications
GROUP BY country, employment_status
ORDER BY default_rate_pct DESC;

-- 3. Debt-to-income proxy risk tiering
SELECT
    CASE
        WHEN existing_debt_ratio < 0.2 THEN 'Low (<0.2)'
        WHEN existing_debt_ratio < 0.4 THEN 'Moderate (0.2-0.4)'
        WHEN existing_debt_ratio < 0.6 THEN 'Elevated (0.4-0.6)'
        ELSE 'High (0.6+)'
    END AS debt_ratio_band,
    COUNT(*) AS applications,
    ROUND(100.0 * SUM(default_flag) / COUNT(*), 1) AS default_rate_pct
FROM loan_applications
GROUP BY debt_ratio_band
ORDER BY MIN(existing_debt_ratio);

-- 4. Loan purpose risk ranking
SELECT
    purpose,
    COUNT(*) AS applications,
    ROUND(100.0 * SUM(default_flag) / COUNT(*), 1) AS default_rate_pct,
    ROUND(AVG(loan_amount), 0) AS avg_loan_amount,
    ROUND(SUM(CASE WHEN default_flag=1 THEN loan_amount ELSE 0 END), 0) AS total_defaulted_exposure
FROM loan_applications
GROUP BY purpose
ORDER BY default_rate_pct DESC;

-- 5. High-risk segment exposure (candidates for tightened underwriting)
SELECT
    country,
    CASE WHEN credit_score < 620 THEN 'Sub-620' ELSE '620+' END AS credit_tier,
    CASE WHEN existing_debt_ratio >= 0.4 THEN 'High DTI' ELSE 'Normal DTI' END AS dti_tier,
    COUNT(*) AS applications,
    ROUND(100.0 * SUM(default_flag) / COUNT(*), 1) AS default_rate_pct,
    SUM(loan_amount) AS total_exposure
FROM loan_applications
GROUP BY country, credit_tier, dti_tier
ORDER BY default_rate_pct DESC;
