-- ============================================================
-- Australia Public Transport On-Time Performance (OTP)
-- Target: PostgreSQL / SQLite compatible
-- ============================================================

-- 1. On-time performance % by route
SELECT
    route,
    mode,
    city,
    ROUND(100.0 * SUM(on_time_services) / NULLIF(SUM(scheduled_services),0), 1) AS otp_pct,
    ROUND(AVG(avg_delay_minutes), 1) AS avg_delay_minutes,
    SUM(daily_patronage) AS total_patronage
FROM au_transport_performance
GROUP BY route, mode, city
ORDER BY otp_pct ASC;

-- 2. Mode-level comparison (Train vs Tram vs Bus vs Ferry)
SELECT
    mode,
    ROUND(100.0 * SUM(on_time_services) / NULLIF(SUM(scheduled_services),0), 1) AS otp_pct,
    ROUND(AVG(avg_delay_minutes), 1) AS avg_delay_minutes,
    SUM(daily_patronage) AS total_patronage
FROM au_transport_performance
GROUP BY mode
ORDER BY otp_pct ASC;

-- 3. Monthly OTP trend, system-wide
SELECT
    DATE_TRUNC('month', date) AS month,
    ROUND(100.0 * SUM(on_time_services) / NULLIF(SUM(scheduled_services),0), 1) AS otp_pct
FROM au_transport_performance
GROUP BY DATE_TRUNC('month', date)
ORDER BY month;

-- 4. Weekday vs weekend reliability comparison
SELECT
    CASE WHEN EXTRACT(DOW FROM date) IN (0,6) THEN 'Weekend' ELSE 'Weekday' END AS day_type,
    route,
    ROUND(100.0 * SUM(on_time_services) / NULLIF(SUM(scheduled_services),0), 1) AS otp_pct
FROM au_transport_performance
GROUP BY day_type, route
ORDER BY route, day_type;

-- 5. Patronage-weighted impact of disruption (routes where disruption hits
--    the most riders — prioritisation input for the cost-benefit case)
SELECT
    route,
    mode,
    city,
    SUM(disrupted_services) AS total_disrupted_services,
    SUM(daily_patronage) AS total_patronage,
    ROUND(SUM(disrupted_services) * 1.0 * (SUM(daily_patronage)*1.0/COUNT(*)), 0) AS rider_impact_score
FROM au_transport_performance
GROUP BY route, mode, city
ORDER BY rider_impact_score DESC;
