import pandas as pd
import joblib
import time
import json
from kafka import KafkaProducer

print("=" * 60)
print("REAL-TIME FINANCIAL FRAUD DETECTION PIPELINE")
print("=" * 60)

# ---------------------------------------------------------
# 1. Load trained fraud detection model
# ---------------------------------------------------------

saved = joblib.load("models/fraud_model.pkl")

model = saved["model"]
features = saved["features"]

print("\nFraud detection model loaded successfully.")

# ---------------------------------------------------------
# 2. Connect to Kafka
# ---------------------------------------------------------

producer = KafkaProducer(
    bootstrap_servers="127.0.0.1:9092",
    value_serializer=lambda value: json.dumps(
        value,
        default=str
    ).encode("utf-8")
)

print("Connected to Kafka successfully.")

TOPIC = "financial-transactions"

# ---------------------------------------------------------
# 3. Load held-out transactions
# ---------------------------------------------------------

data = pd.read_csv(
    "outputs/heldout_test_data.csv"
)

print("Transactions loaded:", len(data))
print("\nStarting real-time transaction simulation...\n")

# ---------------------------------------------------------
# 4. Send transactions to Kafka
# ---------------------------------------------------------

for index, row in data.head(20).iterrows():

    # -----------------------------------------------------
    # Prepare data for ML model
    # -----------------------------------------------------

    transaction_features = pd.DataFrame(
        [row[features].to_dict()]
    )

    # -----------------------------------------------------
    # Model prediction
    # -----------------------------------------------------

    prediction = model.predict(
        transaction_features
    )[0]

    probability = model.predict_proba(
        transaction_features
    )[0][1]

    # -----------------------------------------------------
    # Reconstruct original transaction type
    # -----------------------------------------------------

    transaction_type = next(
        (
            transaction_type.replace("type_", "")
            for transaction_type in [
                "type_CASH_IN",
                "type_CASH_OUT",
                "type_DEBIT",
                "type_PAYMENT",
                "type_TRANSFER"
            ]
            if row.get(transaction_type, 0) == 1
        ),
        "UNKNOWN"
    )

    # -----------------------------------------------------
    # Create FULL transaction message
    # -----------------------------------------------------

    transaction_message = {
        "transaction_id": int(index + 1),

        "step": int(row["step"]),
        "amount": float(row["amount"]),

        "oldbalanceOrg": float(row["oldbalanceOrg"]),
        "newbalanceOrig": float(row["newbalanceOrig"]),

        "oldbalanceDest": float(row["oldbalanceDest"]),
        "newbalanceDest": float(row["newbalanceDest"]),

        "type": transaction_type,

        "fraud_probability": float(probability),
        "prediction": int(prediction),
        "actual_label": int(row["isFraud"])
    }

    # -----------------------------------------------------
    # Send transaction to Kafka
    # -----------------------------------------------------

    producer.send(
        TOPIC,
        value=transaction_message
    )

    producer.flush()

    # -----------------------------------------------------
    # Display transaction information
    # -----------------------------------------------------

    print("-" * 60)

    print(
        "Transaction ID:",
        index + 1
    )

    print(
        "Type:",
        transaction_message["type"]
    )

    print(
        "Amount:",
        round(
            transaction_message["amount"],
            2
        )
    )

    print(
        "Fraud probability:",
        round(
            probability * 100,
            2
        ),
        "%"
    )

    if prediction == 1:
        print(
            "STATUS: FRAUD ALERT!"
        )
    else:
        print(
            "STATUS: GENUINE TRANSACTION"
        )

    print(
        "Kafka Status: Transaction sent successfully"
    )

    print(
        "Actual label:",
        int(row["isFraud"])
    )

    time.sleep(1)

# ---------------------------------------------------------
# 5. Close Kafka producer
# ---------------------------------------------------------

producer.close()

print("\n" + "=" * 60)
print("TRANSACTION SIMULATION COMPLETED!")
print("=" * 60)