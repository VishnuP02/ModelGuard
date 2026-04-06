import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss
)


def evaluate_split(y_true, y_pred, y_prob, split_name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "split": split_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "brier_score": round(brier_score_loss(y_true, y_prob), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def main():
    df = pd.read_csv("data/credit_risk_synthetic.csv")

    X = df.drop(columns=["default"])
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    train_prob = pipeline.predict_proba(X_train)[:, 1]
    test_prob = pipeline.predict_proba(X_test)[:, 1]

    train_metrics = evaluate_split(y_train, train_pred, train_prob, "train")
    test_metrics = evaluate_split(y_test, test_pred, test_prob, "test")

    metrics_df = pd.DataFrame([train_metrics, test_metrics])

    os.makedirs("outputs", exist_ok=True)
    metrics_df.to_csv("outputs/model_metrics.csv", index=False)

    auc_gap = abs(train_metrics["roc_auc"] - test_metrics["roc_auc"])
    acc_gap = abs(train_metrics["accuracy"] - test_metrics["accuracy"])

    findings = []
    findings.append("MODEL VALIDATION SUMMARY")
    findings.append("=" * 50)
    findings.append("")
    findings.append("Train Metrics:")
    for k, v in train_metrics.items():
        if k != "split":
            findings.append(f"  {k}: {v}")

    findings.append("")
    findings.append("Test Metrics:")
    for k, v in test_metrics.items():
        if k != "split":
            findings.append(f"  {k}: {v}")

    findings.append("")
    findings.append("RISK ASSESSMENT")
    findings.append("-" * 50)

    if auc_gap > 0.08 or acc_gap > 0.08:
        findings.append("Overfitting Risk: FLAGGED")
    else:
        findings.append("Overfitting Risk: LOW")

    if test_metrics["brier_score"] > 0.20:
        findings.append("Probability Calibration Risk: FLAGGED")
    else:
        findings.append("Probability Calibration Risk: ACCEPTABLE")

    if test_metrics["recall"] < 0.60:
        findings.append("Missed Risk Events: HIGH (low recall)")
    else:
        findings.append("Missed Risk Events: ACCEPTABLE")

    findings.append("")
    findings.append("Recommended Actions:")
    findings.append("- Monitor stability across new data periods")
    findings.append("- Compare against challenger models")
    findings.append("- Review threshold strategy for business risk tolerance")
    findings.append("- Perform drift checks on incoming production data")

    with open("outputs/validation_report.txt", "w") as f:
        f.write("\n".join(findings))

    print("Saved: outputs/model_metrics.csv")
    print("Saved: outputs/validation_report.txt")
    print(metrics_df)


if __name__ == "__main__":
    main()