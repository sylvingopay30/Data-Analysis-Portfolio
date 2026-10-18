import pandas as pd
import numpy as np

np.random.seed(33)

routes = [
    ("Route 1", "Train", "Melbourne", 45),
    ("Route 2", "Train", "Melbourne", 38),
    ("Route 3", "Tram", "Melbourne", 22),
    ("Route 4", "Bus", "Melbourne", 15),
    ("Route 5", "Train", "Sydney", 50),
    ("Route 6", "Train", "Sydney", 42),
    ("Route 7", "Bus", "Sydney", 18),
    ("Route 8", "Ferry", "Sydney", 12),
    ("Route 9", "Bus", "Brisbane", 20),
    ("Route 10","Train", "Brisbane", 35),
]

n_days = 730  # 2 years
start = pd.Timestamp("2023-01-01")
rows = []
tid = 1
for route, mode, city, avg_trip_min in routes:
    reliability_factor = np.random.uniform(0.85, 0.98)  # route-level baseline reliability
    for d in range(n_days):
        date = start + pd.Timedelta(days=d)
        is_weekend = date.dayofweek >= 5
        scheduled_services = np.random.randint(40, 80) if not is_weekend else np.random.randint(20, 45)
        # peak-hour disruption + weather-like noise
        disruption_chance = 0.08 if mode in ["Train","Ferry"] else 0.05
        disrupted = np.random.binomial(scheduled_services, disruption_chance * (1/reliability_factor))
        on_time = scheduled_services - disrupted
        avg_delay_min = round(max(0, np.random.gamma(1.5, 3) * (1/reliability_factor)), 1) if disrupted > 0 else 0.0
        patronage = int(np.random.poisson(1800 if not is_weekend else 900) * (avg_trip_min/30))
        rows.append({
            "trip_log_id": f"TRN{str(tid).zfill(7)}",
            "route": route,
            "mode": mode,
            "city": city,
            "date": date,
            "scheduled_services": scheduled_services,
            "on_time_services": on_time,
            "disrupted_services": disrupted,
            "avg_delay_minutes": avg_delay_min,
            "daily_patronage": patronage
        })
        tid += 1

df = pd.DataFrame(rows)
df.to_csv("../data/au_transport_performance.csv", index=False)
print(df.shape)
print(df.head())
