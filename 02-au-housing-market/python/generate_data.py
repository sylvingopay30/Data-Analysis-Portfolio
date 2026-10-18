import pandas as pd
import numpy as np

np.random.seed(7)

suburbs = [
    ("Bondi", "NSW", "Sydney", 1200000),
    ("Parramatta", "NSW", "Sydney", 850000),
    ("Blacktown", "NSW", "Sydney", 700000),
    ("Richmond", "VIC", "Melbourne", 950000),
    ("Dandenong", "VIC", "Melbourne", 650000),
    ("Footscray", "VIC", "Melbourne", 780000),
    ("Fortitude Valley", "QLD", "Brisbane", 720000),
    ("Logan Central", "QLD", "Brisbane", 480000),
    ("Fremantle", "WA", "Perth", 690000),
    ("Midland", "WA", "Perth", 450000),
    ("Glenelg", "SA", "Adelaide", 630000),
    ("Elizabeth", "SA", "Adelaide", 380000),
]

n_listings = 6000
rows = []
start = pd.Timestamp("2021-01-01")
for i in range(n_listings):
    suburb, state, metro, base_price = suburbs[np.random.randint(len(suburbs))]
    days_offset = np.random.randint(0, 1400)
    sale_date = start + pd.Timedelta(days=days_offset)
    # mild upward trend over time + noise
    trend_factor = 1 + (days_offset / 1400) * 0.18
    price = base_price * trend_factor * np.random.normal(1.0, 0.12)
    bedrooms = np.random.choice([1,2,3,4,5], p=[0.12,0.28,0.32,0.2,0.08])
    proptype = np.random.choice(["House","Apartment","Townhouse"], p=[0.45,0.4,0.15])
    days_on_market = max(1, int(np.random.gamma(2.0, 15)))
    rows.append({
        "listing_id": f"AU{str(i+1).zfill(6)}",
        "suburb": suburb,
        "state": state,
        "metro_area": metro,
        "property_type": proptype,
        "bedrooms": bedrooms,
        "sale_price_aud": round(price, -3),
        "sale_date": sale_date,
        "days_on_market": days_on_market
    })

df = pd.DataFrame(rows)
df.to_csv("../data/au_property_sales.csv", index=False)
print(df.shape)
print(df.head())
