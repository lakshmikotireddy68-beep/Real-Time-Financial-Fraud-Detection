
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

print("=" * 60)
print("REAL-TIME FINANCIAL FRAUD DETECTION")
print("RANDOM FOREST MODEL TRAINING")
print("=" * 60)

# 1. Load processed dataset
file_path = "outputs/processed_transactions.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded:", df.shape)

# 2. Separate genuine and fraudulent transactions
fraud = df[df["isFraud"] == 1]
genuine = df[df["isFraud"] == 0]

print("Fraud records:", len(fraud))
print("Genuine records:", len(genuine))

# 3. Sample genuine transactions
genuine_sample = genuine.sample(
    n=min(150000, len(genuine)),
    random_state=42
)

# Retain every fraudulent transaction
balanced_df = pd.concat(
    [genuine_sample, fraud],
    ignore_index=True
)

balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nTraining dataset:", balanced_df.shape)

# 4. Separate features and target
X = balanced_df.drop(columns=["isFraud"])
y = balanced_df["isFraud"]

# 5. Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
# Save held-out test data for independent prediction testing

test_data = X_test.copy()
test_data["isFraud"] = y_test

test_data.to_csv(
    "outputs/heldout_test_data.csv",
    index=False
)

print("Held-out test data saved!")
print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 6. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=2
)

# 7. Train model
print("\nTraining started...")

model.fit(X_train, y_train)

print("Model training completed!")

# 8. Predict test data
y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

# 9. Evaluation metrics
precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

pr_auc = average_precision_score(
    y_test, y_probability
)

print("\n========== MODEL EVALUATION ==========")

print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("PR-AUC:", round(pr_auc, 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Genuine", "Fraud"],
    zero_division=0
))

# 10. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Genuine", "Fraud"],
    yticklabels=["Genuine", "Fraud"]
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

plt.savefig("outputs/confusion_matrix.png")
plt.close()

# 11. Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))

importance.head(12).sort_values().plot(
    kind="barh"
)

plt.title("Top 12 Important Fraud Detection Features")
plt.xlabel("Feature Importance")
plt.tight_layout()

plt.savefig("outputs/feature_importance.png")
plt.close()

# 12. Save trained model and feature names
joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist()
    },
    "models/fraud_model.pkl"
)

# 13. Save evaluation report
with open("outputs/model_evaluation.txt", "w") as f:

    f.write("RANDOM FOREST MODEL EVALUATION\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"PR-AUC: {pr_auc:.4f}\n")
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nClassification Report:\n")
    f.write(classification_report(
        y_test, y_pred, zero_division=0
    ))

print("\nModel saved: models/fraud_model.pkl")
print("Confusion matrix saved!")
print("Feature importance graph saved!")
print("Evaluation report saved!")

print("\nMODEL DEVELOPMENT COMPLETED!")
