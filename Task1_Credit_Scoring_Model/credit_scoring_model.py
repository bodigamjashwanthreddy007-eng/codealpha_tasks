
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, classification_report
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "credit_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_dataset(n_samples=1500):
    """Generate a realistic synthetic credit-scoring dataset."""
    age = np.random.randint(21, 66, n_samples)
    income = np.random.normal(60000, 22000, n_samples).clip(15000, 160000)
    total_debt = np.random.normal(18000, 12000, n_samples).clip(0, 90000)
    credit_history_years = np.random.uniform(0.5, 30, n_samples)
    late_payments = np.random.poisson(1.8, n_samples).clip(0, 12)
    credit_utilization = np.random.beta(2.2, 3.5, n_samples)
    existing_loans = np.random.poisson(1.6, n_samples).clip(0, 8)
    savings = np.random.normal(14000, 12000, n_samples).clip(0, 100000)
    employment_years = np.minimum(
        np.random.uniform(0, 25, n_samples),
        np.maximum(age - 18, 0)
    )

    debt_to_income = total_debt / np.maximum(income, 1)

    # Hidden score used only to create labels for this synthetic dataset.
    score = (
        2.2
        + 0.000018 * income
        + 0.045 * credit_history_years
        + 0.000012 * savings
        + 0.028 * employment_years
        - 2.4 * debt_to_income
        - 1.8 * credit_utilization
        - 0.38 * late_payments
        - 0.12 * existing_loans
        + np.random.normal(0, 0.65, n_samples)
    )

    probability_good = 1 / (1 + np.exp(-score))
    creditworthy = (np.random.rand(n_samples) < probability_good).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": income.round(2),
        "total_debt": total_debt.round(2),
        "credit_history_years": credit_history_years.round(2),
        "late_payments": late_payments,
        "credit_utilization": credit_utilization.round(3),
        "existing_loans": existing_loans,
        "savings": savings.round(2),
        "employment_years": employment_years.round(2),
        "debt_to_income": debt_to_income.round(3),
        "creditworthy": creditworthy
    })
    df.to_csv(DATA_PATH, index=False)
    return df

def load_or_generate_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return generate_dataset()

def main():
    df = load_or_generate_data()

    print("Dataset shape:", df.shape)
    print("\nTarget distribution:")
    print(df["creditworthy"].value_counts())

    X = df.drop(columns=["creditworthy"])
    y = df["creditworthy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }

    print("\nModel Evaluation:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save metrics
    pd.DataFrame([metrics]).to_csv(
        os.path.join(RESULTS_DIR, "model_metrics.csv"), index=False
    )

    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.set_title("Credit Scoring - Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=160)
    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title("Credit Scoring - ROC Curve")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=160)
    plt.close(fig)

    # Model coefficients
    coef_df = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_[0]
    }).sort_values("coefficient", key=abs, ascending=False)
    coef_df.to_csv(os.path.join(RESULTS_DIR, "feature_coefficients.csv"), index=False)

    # Example prediction
    sample = X_test.iloc[[0]]
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0, 1]

    print("\nExample Prediction:")
    print(sample.to_string(index=False))
    print("Prediction:", "Creditworthy" if prediction == 1 else "Not Creditworthy")
    print(f"Probability of being creditworthy: {probability:.4f}")

    with open(os.path.join(RESULTS_DIR, "sample_prediction.txt"), "w") as f:
        f.write(sample.to_string(index=False))
        f.write("\n")
        f.write(f"Prediction: {'Creditworthy' if prediction == 1 else 'Not Creditworthy'}\n")
        f.write(f"Probability of being creditworthy: {probability:.4f}\n")

if __name__ == "__main__":
    main()
