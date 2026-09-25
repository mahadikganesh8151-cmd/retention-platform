import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def load_billing_data(path: str = "data/telco_churn_raw.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df


def train_fraud_detector(df: pd.DataFrame):
    features = df[["tenure", "MonthlyCharges", "TotalCharges"]].copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # contamination=0.02 means we expect ~2% of records to be anomalous - a
    # reasonable, explicit assumption for rare-event fraud detection
    model = IsolationForest(contamination=0.02, random_state=42)
    model.fit(scaled)

    return model, scaler


def flag_anomalies(df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    features = df[["tenure", "MonthlyCharges", "TotalCharges"]]
    scaled = scaler.transform(features)

    df = df.copy()
    df["anomaly_score"] = model.decision_function(scaled)
    df["is_anomalous"] = model.predict(scaled) == -1  # -1 means flagged as outlier

    return df


if __name__ == "__main__":
    print("Loading billing data...")
    df = load_billing_data()

    print("Training anomaly detector...")
    model, scaler = train_fraud_detector(df)

    print("Flagging anomalies...")
    results = flag_anomalies(df, model, scaler)

    flagged = results[results["is_anomalous"]]
    print(f"\nFlagged {len(flagged)} anomalous billing records out of {len(df)}")
    print(flagged[["customerID", "tenure", "MonthlyCharges", "TotalCharges", "anomaly_score"]].head(10))

    joblib.dump(model, "app/fraud_model.pkl")
    joblib.dump(scaler, "app/fraud_scaler.pkl")
    print("\nSaved fraud detection model")