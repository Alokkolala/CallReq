#!/usr/bin/env python
"""Generate synthetic PaySim-style transaction data for testing."""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

n_transactions = 500
n_customers = 50
n_merchants = 30

steps = np.random.randint(1, 1000, n_transactions)
types = np.random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN"], n_transactions)
amounts = np.abs(np.random.lognormal(4, 2, n_transactions))
names_orig = np.random.choice([f"C{i:06d}" for i in range(n_customers)], n_transactions)
names_dest = np.random.choice([f"M{i:06d}" for i in range(n_merchants)], n_transactions)

old_balance_org = np.abs(np.random.normal(5000, 2000, n_transactions))
new_balance_orig = np.maximum(old_balance_org - amounts, 0)

old_balance_dest = np.abs(np.random.normal(3000, 1500, n_transactions))
new_balance_dest = old_balance_dest + amounts * 0.9

is_fraud = np.random.binomial(1, 0.01, n_transactions)

df = pd.DataFrame({
    "step": steps,
    "type": types,
    "amount": amounts,
    "nameOrig": names_orig,
    "oldbalanceOrg": old_balance_org,
    "newbalanceOrig": new_balance_orig,
    "nameDest": names_dest,
    "oldbalanceDest": old_balance_dest,
    "newbalanceDest": new_balance_dest,
    "isFraud": is_fraud,
    "isFlaggedFraud": np.zeros(n_transactions, dtype=int),
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/transactions.csv", index=False)
print(f"✓ Generated data/transactions.csv with {len(df)} rows")
print(df.head())
