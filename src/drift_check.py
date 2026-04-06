import pandas as pd
from scipy.stats import ks_2samp


def main():
    df = pd.read_csv("data/credit_risk_synthetic.csv")

    baseline = df.sample(frac=0.5, random_state=42).copy()
    new_data = df.drop(index=baseline.index).copy()

    new_data["credit_utilization"] = (
        new_data["credit_utilization"] + 0.10
    ).clip(0, 0.99)

    new_data["debt_to_income"] = (
        new_data["debt_to_income"] + 0.05
    ).clip(0, 0.99)

    numeric_cols = [col for col in df.columns if col != "default"]

    print("DRIFT CHECK RESULTS")
    print("=" * 50)

    drift_summary = []

    for col in numeric_cols:
        stat, p_value = ks_2samp(baseline[col], new_data[col])
        drift_flag = "FLAGGED" if p_value < 0.05 else "OK"

        print(
            f"{col:<22} | KS={stat:.4f} | p-value={p_value:.6f} | drift={drift_flag}"
        )

        drift_summary.append(
            {
                "feature": col,
                "ks_stat": round(stat, 4),
                "p_value": round(p_value, 6),
                "drift": drift_flag,
            }
        )

    drift_df = pd.DataFrame(drift_summary)
    drift_df.to_csv("outputs/drift_report.csv", index=False)

    print("\nSaved: outputs/drift_report.csv")


if __name__ == "__main__":
    main()