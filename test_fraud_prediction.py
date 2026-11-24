
import pandas as pd
import joblib

# Load trained model
saved = joblib.load("models/fraud_model.pkl")

model = saved["model"]
features = saved["features"]

print("=" * 50)
print("FINANCIAL FRAUD PREDICTION SYSTEM")
print("=" * 50)

# Load an unseen transaction from the dataset
df = pd.read_csv(
    "outputs/processed_transactions.csv"
)

# Select one transaction for demonstration
transaction = df.drop(
    columns=["isFraud"]
).iloc[[0]].copy()

actual = df["isFraud"].iloc[0]

# Ensure correct feature order
transaction = transaction[features]

# Predict
prediction = model.predict(transaction)[0]

probability = model.predict_proba(
    transaction
)[0][1]

print("\nTransaction details:")
print(transaction)

print("\nActual label:", actual)

print("\nPrediction probability:",
      round(probability * 100, 2), "%")

if prediction == 1:
    print("\nALERT: SUSPICIOUS TRANSACTION!")
else:
    print("\nGENUINE TRANSACTION")

print("\nPrediction completed!")
