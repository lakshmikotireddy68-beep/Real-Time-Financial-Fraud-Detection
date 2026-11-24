
import pandas as pd
import os

file_path = "dataset/PS_20174392719_1491204439457_log.csv"

print("PAYSim DATASET VERIFICATION")
print("=" * 45)

if not os.path.exists(file_path):
    print("Dataset not found!")
    raise SystemExit

total_rows = 0
fraud_count = 0
genuine_count = 0

for chunk in pd.read_csv(file_path, chunksize=100000):

    total_rows += len(chunk)
    fraud_count += (chunk["isFraud"] == 1).sum()
    genuine_count += (chunk["isFraud"] == 0).sum()

print("Total transactions:", total_rows)
print("Fraud transactions:", fraud_count)
print("Genuine transactions:", genuine_count)

print("\nVerification completed!")
