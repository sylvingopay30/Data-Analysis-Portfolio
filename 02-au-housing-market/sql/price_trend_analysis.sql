-- ============================================================
-- Australia Housing Market Price Trend Analysis
-- Target: PostgreSQL / SQLite compatible
-- ============================================================

-- 1. Median price by metro area and quarter (median approximated via
--    PERCENTILE_CONT in PostgreSQL)
SELECT
    metro_area,
    DATE_TRUNC('quarter', sale_date) AS quarter,
    COUNT(*) AS sales_count,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sale_price_aud) AS median_price_aud,
    ROUND(AVG(days_on_market), 1) AS avg_days_on_market
FROM au_property_sales
GROUP BY metro_area, DATE_TRUNC('quarter', sale_date)
ORDER BY metro_area, quarter;

-- 2. Year-on-year growth rate by suburb (self-join on year)
WITH yearly AS (
    SELECT
        suburb,
        state,
        EXTRACT(YEAR FROM sale_date) AS sale_year,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sale_price_aud) AS median_price
    FROM au_property_sales
    GROUP BY suburb, state, EXTRACT(YEAR FROM sale_date)
)
SELECT
    curr.suburb,
    curr.state,
    curr.sale_year,
    curr.median_price,
    prev.median_price AS prev_year_median,
    ROUND(100.0 * (curr.median_price - prev.median_price) / NULLIF(prev.median_price,0), 1) AS yoy_growth_pct
FROM yearly curr
LEFT JOIN yearly prev
    ON curr.suburb = prev.suburb AND curr.sale_year = prev.sale_year + 1
ORDER BY curr.suburb, curr.sale_year;

-- 3. Affordability proxy: price per bedroom by suburb, ranked
SELECT
    suburb,
    state,
    bedrooms,
    ROUND(AVG(sale_price_aud) / NULLIF(bedrooms,0), 0) AS avg_price_per_bedroom,
    RANK() OVER (ORDER BY AVG(sale_price_aud) / NULLIF(bedrooms,0) ASC) AS affordability_rank
FROM au_property_sales
WHERE bedrooms > 0
GROUP BY suburb, state, bedrooms
ORDER BY affordability_rank;

-- 4. Days-on-market outliers (slow-moving stock) by property type
SELECT
    property_type,
    metro_area,
    ROUND(AVG(days_on_market), 1) AS avg_dom,
    ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY days_on_market), 1) AS p90_dom
FROM au_property_sales
GROUP BY property_type, metro_area
ORDER BY avg_dom DESC;
