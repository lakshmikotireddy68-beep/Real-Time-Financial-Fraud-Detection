
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------------------------
# 1. PROJECT CONFIGURATION
# ----------------------------------

FILE_PATH = "dataset/PS_20174392719_1491204439457_log.csv"

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

print("=" * 60)
print("FINANCIAL FRAUD DETECTION - DATA ANALYSIS")
print("=" * 60)

# ----------------------------------
# 2. CHECK DATASET
# ----------------------------------

if not os.path.exists(FILE_PATH):
    print("Dataset not found!")
    raise SystemExit

print("\nDataset found successfully!")

# ----------------------------------
# 3. PROCESS DATA IN CHUNKS
# ----------------------------------

total_rows = 0
fraud_counts = {0: 0, 1: 0}
transaction_types = {}
missing_values = {}
amount_by_fraud = {}
samples = []

for chunk in pd.read_csv(FILE_PATH, chunksize=100000):

    total_rows += len(chunk)

    # Fraud distribution
    counts = chunk["isFraud"].value_counts()

    for label, count in counts.items():
        fraud_counts[int(label)] += int(count)

    # Transaction types
    for name, count in chunk["type"].value_counts().items():
        transaction_types[name] = (
            transaction_types.get(name, 0) + int(count)
        )

    # Missing values
    for col, count in chunk.isnull().sum().items():
        missing_values[col] = (
            missing_values.get(col, 0) + int(count)
        )

    # Transaction amount by fraud label
    grouped = chunk.groupby("isFraud")["amount"].agg(
        ["sum", "count"]
    )

    for label, row in grouped.iterrows():
        label = int(label)

        if label not in amount_by_fraud:
            amount_by_fraud[label] = [0.0, 0]

        amount_by_fraud[label][0] += float(row["sum"])
        amount_by_fraud[label][1] += int(row["count"])

    # Keep a small representative sample for graphs
    sample_size = min(10000, len(chunk))

    samples.append(
        chunk.sample(n=sample_size, random_state=42)
    )

print("\nDATASET SUMMARY")
print("Total transactions:", total_rows)
print("Fraud transactions:", fraud_counts[1])
print("Genuine transactions:", fraud_counts[0])

print("\nFraud percentage:",
      round(fraud_counts[1] / total_rows * 100, 4), "%")

# ----------------------------------
# 4. MISSING VALUE ANALYSIS
# ----------------------------------

print("\nMISSING VALUES")

for column, count in missing_values.items():
    print(column, ":", count)

# ----------------------------------
# 5. TRANSACTION TYPE ANALYSIS
# ----------------------------------

print("\nTRANSACTION TYPES")

for name, count in transaction_types.items():
    print(name, ":", count)

# ----------------------------------
# 6. PREPARE GRAPH SAMPLE
# ----------------------------------

eda_df = pd.concat(samples, ignore_index=True)

print("\nSample size for visualization:", len(eda_df))

# ----------------------------------
# 7. FRAUD DISTRIBUTION GRAPH
# ----------------------------------

fraud_df = pd.DataFrame({
    "Transaction": ["Genuine", "Fraudulent"],
    "Count": [fraud_counts[0], fraud_counts[1]]
})

plt.figure(figsize=(8, 5))

sns.barplot(
    data=fraud_df,
    x="Transaction",
    y="Count"
)

plt.title("Genuine vs Fraudulent Transactions")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/fraud_distribution.png"
)

plt.close()

# ----------------------------------
# 8. TRANSACTION TYPE GRAPH
# ----------------------------------

type_df = pd.DataFrame(
    list(transaction_types.items()),
    columns=["Type", "Count"]
)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=type_df,
    x="Type",
    y="Count"
)

plt.title("Transaction Type Distribution")
plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/transaction_types.png"
)

plt.close()

# ----------------------------------
# 9. TRANSACTION AMOUNT GRAPH
# ----------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=eda_df,
    x="amount",
    bins=50
)

plt.title("Transaction Amount Distribution")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/transaction_amount.png"
)

plt.close()

# ----------------------------------
# 10. FRAUD VS AMOUNT GRAPH
# ----------------------------------

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=eda_df,
    x="isFraud",
    y="amount"
)

plt.title("Transaction Amount by Fraud Label")
plt.xlabel("Fraud (0 = Genuine, 1 = Fraud)")
plt.ylabel("Transaction Amount")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/fraud_amount_comparison.png"
)

plt.close()

# ----------------------------------
# 11. SAVE SUMMARY REPORT
# ----------------------------------

with open(
    f"{OUTPUT_DIR}/eda_summary.txt",
    "w"
) as report:

    report.write("FINANCIAL FRAUD DETECTION EDA\n")
    report.write(f"Total transactions: {total_rows}\n")
    report.write(f"Fraud transactions: {fraud_counts[1]}\n")
    report.write(f"Genuine transactions: {fraud_counts[0]}\n")

    report.write("\nTransaction types:\n")

    for name, count in transaction_types.items():
        report.write(f"{name}: {count}\n")

    report.write("\nMissing values:\n")

    for column, count in missing_values.items():
        report.write(f"{column}: {count}\n")

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY!")
print("Check the outputs folder for graphs and reports.")
print("=" * 60)
