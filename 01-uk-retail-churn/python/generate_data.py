import pandas as pd
import numpy as np

np.random.seed(42)

n_customers = 1200
n_transactions = 15000

cities = ["London","Manchester","Birmingham","Leeds","Glasgow","Bristol","Liverpool","Edinburgh","Sheffield","Cardiff"]
categories = ["Homeware","Electronics","Fashion","Beauty","Garden","Toys","Groceries","Books","Stationery","Sports"]

customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(1, n_customers+1)]
signup_dates = pd.to_datetime("2022-01-01") + pd.to_timedelta(np.random.randint(0, 700, n_customers), unit="D")
city_choice = np.random.choice(cities, n_customers, p=[0.22,0.13,0.11,0.09,0.08,0.08,0.08,0.07,0.07,0.07])
customers = pd.DataFrame({
    "customer_id": customer_ids,
    "city": city_choice,
    "signup_date": signup_dates,
    "acquisition_channel": np.random.choice(["Paid Search","Organic","Referral","Email","Social"], n_customers, p=[0.3,0.25,0.15,0.2,0.1])
})

cust_for_txn = np.random.choice(customer_ids, n_transactions)
txn_dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(np.random.randint(0, 640, n_transactions), unit="D")
order_values = np.round(np.random.gamma(shape=2.2, scale=18, size=n_transactions) + 5, 2)
categories_choice = np.random.choice(categories, n_transactions)
discount_used = np.random.choice([0,1], n_transactions, p=[0.72,0.28])

transactions = pd.DataFrame({
    "transaction_id": [f"TXN{str(i).zfill(6)}" for i in range(1, n_transactions+1)],
    "customer_id": cust_for_txn,
    "transaction_date": txn_dates,
    "category": categories_choice,
    "order_value_gbp": order_values,
    "discount_used": discount_used
})

customers.to_csv("../data/customers.csv", index=False)
transactions.to_csv("../data/transactions.csv", index=False)
print("customers:", customers.shape, "transactions:", transactions.shape)
