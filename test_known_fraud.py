
import pandas as pd
import joblib

# Load model
saved = joblib.load("models/fraud_model.pkl")
model = saved["model"]
features = saved["features"]

# Load held-out test records
df = pd.read_csv("outputs/heldout_test_data.csv")

# Select an actual fraudulent test transaction
fraud_rows = df[df["isFraud"] == 1]

if fraud_rows.empty:
    print("No fraud records found!")
    raise SystemExit

transaction = fraud_rows.iloc[[0]]

actual = transaction["isFraud"].iloc[0]

X = transaction[features]

prediction = model.predict(X)[0]
probability = model.predict_proba(X)[0][1]

print("=" * 45)
print("FRAUD TRANSACTION TEST")
print("=" * 45)

print("Actual label:", actual)
print("Predicted label:", prediction)
print("Fraud probability:", round(probability * 100, 2), "%")

if prediction == 1:
    print("FRAUD DETECTED!")
else:
    print("FRAUD MISSED!")

print("\nTesting completed!")
