import shap
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_data(path: str = "data/telco_churn_raw.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # Fix TotalCharges: blank strings for tenure=0 customers, forcing it to load as text
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)  # brand-new customers, 0 total charged so far

    return df
def prepare_features(df: pd.DataFrame):
    df = df.copy()

    # Drop customerID — it's just an identifier, carries no predictive signal
    df = df.drop(columns=["customerID"])

    # Convert target to binary: Yes -> 1, No -> 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # One-hot encode all remaining categorical columns
    categorical_cols = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    }

    fitted_lr = None

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print(f"\n=== {name} ===")
        print(f"5-fold CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
        print(classification_report(y_test, y_pred))

        if name == "Logistic Regression":
            fitted_lr = model

    return fitted_lr, X_train, X_test
def explain_model(model, X_train, X_test):
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    X_test_scaled = scaler.transform(X_test)

    explainer = shap.LinearExplainer(clf, scaler.transform(X_train))
    shap_values = explainer.shap_values(X_test_scaled)

    import matplotlib.pyplot as plt

    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("reports_shap_importance.png")
    plt.close()

    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("reports_shap_beeswarm.png")
    plt.close()

    print("SHAP plots saved: reports_shap_importance.png, reports_shap_beeswarm.png")
if __name__ == "__main__":
    df = load_data()
    X, y = prepare_features(df)
    fitted_lr, X_train, X_test = train_and_evaluate(X, y)
    explain_model(fitted_lr, X_train, X_test)