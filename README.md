# ModelGuard: ML Model Monitoring System

ModelGuard is an end-to-end machine learning monitoring pipeline designed to simulate real-world financial risk modeling and post-deployment monitoring.

## Features

- Synthetic financial dataset generation
- Logistic Regression model training
- Model performance evaluation (accuracy, precision, recall, ROC-AUC)
- Overfitting and calibration risk checks
- Data drift detection using Kolmogorov-Smirnov test
- Exportable reports for monitoring and auditing

## Workflow

1. Generate dataset
2. Train and validate model
3. Detect drift in incoming data

## Run

```bash
python3 src/generate_data.py
python3 src/train_and_validate.py
python3 src/drift_check.py