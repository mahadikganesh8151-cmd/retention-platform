import pandas as pd
from faker import Faker
import random

fake = Faker()
random.seed(42)  # makes the "random" data reproducible every run

def generate_customer(customer_id: int) -> dict:
    tenure = random.randint(0, 72)  # months as a customer
    contract = random.choice(["month-to-month", "one-year", "two-year"])
    monthly_charges = round(random.uniform(20, 120), 2)

    return {
        "customer_id": customer_id,
        "name": fake.name(),
        "signup_date": fake.date_between(start_date="-6y", end_date="today"),
        "tenure_months": tenure,
        "contract_type": contract,
        "monthly_charges": monthly_charges,
        "total_charges": round(monthly_charges * tenure, 2),
        "support_tickets": random.randint(0, 10),
    }
def generate_dataset(n: int = 5000) -> pd.DataFrame:
    records = [generate_customer(i) for i in range(1, n + 1)]
    return pd.DataFrame(records)

if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/raw_customers.csv", index=False)
    print(f"Generated {len(df)} customer records → data/raw_customers.csv")