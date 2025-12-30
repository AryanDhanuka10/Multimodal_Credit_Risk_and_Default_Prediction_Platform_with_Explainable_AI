import numpy as np
import pandas as pd
import os

np.random.seed(42)

OUTPUT_DIR = "artifacts/data_ingestion/timeseries"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "transactions.csv")

N_CUSTOMERS = 1200
N_MONTHS = 12


def generate_customer_profile():
    income = np.random.normal(50000, 15000)
    expense_ratio = np.random.uniform(0.4, 0.9)
    volatility = np.random.uniform(0.05, 0.35)

    risk_score = (
        0.5 * (expense_ratio > 0.75) +
        0.3 * (volatility > 0.25) +
        0.2 * (income < 30000)
    )

    default = int(risk_score > 0.6)
    return income, expense_ratio, volatility, default


def generate_transactions():
    rows = []

    for customer_id in range(N_CUSTOMERS):
        income, expense_ratio, volatility, default = generate_customer_profile()
        balance = np.random.uniform(10000, 50000)

        for month in range(1, N_MONTHS + 1):
            monthly_income = max(0, np.random.normal(income, income * volatility))
            expense = monthly_income * expense_ratio * np.random.uniform(0.9, 1.1)
            txn_count = int(np.random.normal(30, 10))

            balance += monthly_income - expense

            if default and month > 8:
                expense *= 1.2
                balance -= np.random.uniform(5000, 15000)

            rows.append([
                customer_id,
                month,
                round(monthly_income, 2),
                round(expense, 2),
                round(balance, 2),
                txn_count,
                round(volatility, 2),
                default
            ])

    return pd.DataFrame(rows, columns=[
        "customer_id",
        "month",
        "income",
        "expense",
        "balance",
        "transaction_count",
        "volatility",
        "default_flag"
    ])


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = generate_transactions()
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved synthetic transactions to {OUTPUT_FILE}")
