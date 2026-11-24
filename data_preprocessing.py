
import pandas as pd
import numpy as np
import os

print("=" * 55)
print("FINANCIAL FRAUD - DATA PREPROCESSING")
print("=" * 55)

# 1. Load dataset
file_path = "dataset/PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(file_path)

print("\nOriginal dataset shape:", df.shape)

# 2. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# 3. Check duplicate records
print("\nDuplicate records:", df.duplicated().sum())

# 4. Create additional features
df["balance_error_orig"] = (
    df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
)

df["balance_error_dest"] = (
    df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
)

df["is_large_transaction"] = (
    df["amount"] > df["amount"].quantile(0.95)
).astype(int)

# 5. Encode transaction type
df = pd.get_dummies(
    df,
    columns=["type"],
    dtype=int
)

# 6. Remove account identifiers and leakage columns
drop_columns = [
    "nameOrig",
    "nameDest",
    "isFlaggedFraud"
]

df = df.drop(columns=drop_columns)

# 7. Separate input features and target
X = df.drop(columns=["isFraud"])
y = df["isFraud"]

print("\nInput feature shape:", X.shape)
print("Target shape:", y.shape)

print("\nFraud distribution:")
print(y.value_counts())

# 8. Save processed dataset
os.makedirs("outputs", exist_ok=True)

processed = X.copy()
processed["isFraud"] = y

processed.to_csv(
    "outputs/processed_transactions.csv",
    index=False
)

print("\nPreprocessed dataset saved!")
print("\nDATA PREPROCESSING COMPLETED!")
