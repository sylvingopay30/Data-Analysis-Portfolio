import pandas as pd
import numpy as np

np.random.seed(21)

trusts = [
    ("Barts Health NHS Trust", "London"),
    ("Manchester University NHS FT", "North West"),
    ("University Hospitals Birmingham", "Midlands"),
    ("Leeds Teaching Hospitals NHS Trust", "Yorkshire"),
    ("Sheffield Teaching Hospitals NHS FT", "Yorkshire"),
    ("University Hospitals Bristol NHS FT", "South West"),
    ("Cardiff and Vale University Health Board", "Wales"),
    ("NHS Lothian", "Scotland"),
    ("Liverpool University Hospitals NHS FT", "North West"),
    ("Newcastle upon Tyne Hospitals NHS FT", "North East"),
]
specialties = ["Orthopaedics","Cardiology","General Surgery","Ophthalmology",
               "ENT","Dermatology","Gynaecology","Urology","Gastroenterology","Neurology"]

rows = []
months = pd.date_range("2022-01-01", "2024-12-01", freq="MS")
rid = 1
for trust, region in trusts:
    # each trust has a baseline capacity/efficiency factor
    trust_factor = np.random.normal(1.0, 0.15)
    for specialty in specialties:
        spec_factor = np.random.normal(1.0, 0.2)
        for m in months:
            # rising waiting list pressure over time, with seasonal winter spikes
            month_index = (m.year - 2022) * 12 + m.month
            trend = 1 + month_index * 0.01
            seasonal = 1.15 if m.month in [12,1,2] else 1.0
            referrals = max(20, int(np.random.poisson(180) * trust_factor * spec_factor))
            treated = max(10, int(referrals * np.random.uniform(0.75, 0.95)))
            waiting_list = max(0, int(referrals - treated + np.random.normal(50,20)))
            median_wait_weeks = round(max(2, np.random.normal(14,4) * trend * seasonal * trust_factor), 1)
            pct_over_18weeks = round(min(95, max(2, (median_wait_weeks - 8) * 3.2 + np.random.normal(0,5))), 1)
            rows.append({
                "record_id": f"NHS{str(rid).zfill(6)}",
                "trust": trust,
                "region": region,
                "specialty": specialty,
                "month": m,
                "referrals": referrals,
                "treated": treated,
                "waiting_list_size": waiting_list,
                "median_wait_weeks": median_wait_weeks,
                "pct_over_18weeks": pct_over_18weeks
            })
            rid += 1

df = pd.DataFrame(rows)
df.to_csv("../data/nhs_wait_times.csv", index=False)
print(df.shape)
print(df.head())
