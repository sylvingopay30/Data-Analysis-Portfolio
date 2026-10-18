-- ============================================================
-- UK Retail Customer Churn & Revenue Impact
-- RFM (Recency, Frequency, Monetary) Segmentation
-- Target: PostgreSQL / SQLite compatible (minor syntax tweaks may be needed)
-- ============================================================

-- 1. Base RFM metrics per customer
WITH customer_orders AS (
    SELECT
        customer_id,
        COUNT(transaction_id)                                   AS frequency,
        SUM(order_value_gbp)                                    AS monetary,
        MAX(transaction_date)                                   AS last_order_date
    FROM transactions
    GROUP BY customer_id
),
rfm_base AS (
    SELECT
        co.customer_id,
        co.frequency,
        co.monetary,
        (SELECT MAX(transaction_date) FROM transactions) - co.last_order_date AS recency_days
    FROM customer_orders co
),

-- 2. Score each dimension into quartiles (1 = worst, 4 = best)
rfm_scored AS (
    SELECT
        customer_id,
        frequency,
        monetary,
        recency_days,
        NTILE(4) OVER (ORDER BY recency_days DESC) AS r_score,   -- lower recency_days = more recent = higher score
        NTILE(4) OVER (ORDER BY frequency ASC)     AS f_score,
        NTILE(4) OVER (ORDER BY monetary ASC)      AS m_score
    FROM rfm_base
),

-- 3. Combine into a segment label
rfm_segments AS (
    SELECT
        *,
        (r_score + f_score + m_score) AS rfm_total,
        CASE
            WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3 THEN 'Champions'
            WHEN r_score >= 3 AND f_score <= 2                   THEN 'Recent but Low Engagement'
            WHEN r_score <= 2 AND f_score >= 3 AND m_score >= 3 THEN 'At Risk (High Value)'
            WHEN r_score <= 2 AND f_score <= 2                   THEN 'Lost / Churned'
            ELSE 'Regular'
        END AS segment
    FROM rfm_scored
)

SELECT
    segment,
    COUNT(*)                       AS customer_count,
    ROUND(AVG(monetary), 2)        AS avg_lifetime_spend_gbp,
    ROUND(AVG(recency_days), 0)    AS avg_recency_days
FROM rfm_segments
GROUP BY segment
ORDER BY avg_lifetime_spend_gbp DESC;

-- ============================================================
-- 4. Revenue at risk: total historical spend from customers
--    flagged "At Risk" or "Lost / Churned"
-- ============================================================
-- (run against rfm_segments CTE materialized as a table/view in practice)

-- ============================================================
-- 5. Monthly cohort retention (signup month vs activity month)
-- ============================================================
WITH cohort AS (
    SELECT
        c.customer_id,
        DATE_TRUNC('month', c.signup_date) AS signup_month,
        DATE_TRUNC('month', t.transaction_date) AS activity_month
    FROM customers c
    JOIN transactions t ON t.customer_id = c.customer_id
)
SELECT
    signup_month,
    activity_month,
    COUNT(DISTINCT customer_id) AS active_customers
FROM cohort
GROUP BY signup_month, activity_month
ORDER BY signup_month, activity_month;
