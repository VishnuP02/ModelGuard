import os
import numpy as np
import pandas as pd

np.random.seed(42)

n = 2000

income = np.random.normal(65000, 18000, n).clip(18000, 180000)
debt_to_income = np.random.normal(0.32, 0.12, n).clip(0.02, 0.95)
credit_utilization = np.random.normal(0.45, 0.20, n).clip(0.01, 0.99)
late_payments_12m = np.random.poisson(1.2, n).clip(0, 12)
loan_amount = np.random.normal(18000, 9000, n).clip(1000, 50000)
credit_score = np.random.normal(690, 60, n).clip(500, 850)
account_age_years = np.random.normal(7, 4, n).clip(0.5, 25)

risk_signal = (
    1.8 * debt_to_income
    + 1.7 * credit_utilization
    + 0.18 * late_payments_12m
    + 0.000015 * loan_amount
    - 0.000012 * income
    - 0.008 * (credit_score - 650)
    - 0.03 * account_age_years
)

prob_default = 1 / (1 + np.exp(-risk_signal))
default = (np.random.rand(n) < prob_default).astype(int)

df = pd.DataFrame({
    "income": income,
    "debt_to_income": debt_to_income,
    "credit_utilization": credit_utilization,
    "late_payments_12m": late_payments_12m,
    "loan_amount": loan_amount,
    "credit_score": credit_score,
    "account_age_years": account_age_years,
    "default": default
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/credit_risk_synthetic.csv", index=False)

print("Saved: data/credit_risk_synthetic.csv")
print(df.head())