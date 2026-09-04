import pandas as pd

def load_raw_data(path: str = "data/raw_customers.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["signup_date"])
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Tenure bucket — groups raw months into meaningful ranges
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"],
        bins=[-1, 12, 24, 48, 100],
        labels=["0-1yr", "1-2yr", "2-4yr", "4yr+"]
    )

    # Average monthly spend so far (guards against divide-by-zero for brand-new customers)
    df["avg_monthly_spend"] = df["total_charges"] / df["tenure_months"].replace(0, 1)

    # High support-ticket flag :- a common churn signal
    df["high_support_usage"] = (df["support_tickets"] >= 5).astype(int)

    return df
def save_transformed(df: pd.DataFrame, path: str = "data/transformed_customers.csv"):
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} transformed records → {path}")

if __name__ == "__main__":
    raw = load_raw_data()
    transformed = engineer_features(raw)
    save_transformed(transformed)