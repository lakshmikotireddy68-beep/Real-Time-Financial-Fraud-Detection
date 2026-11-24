import json
import joblib
import pandas as pd

from database.db_manager import save_transaction

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.functions import col


# ---------------------------------------------------------
# 1. Create Spark session
# ---------------------------------------------------------
spark = (
    SparkSession.builder
    .master("local[2]")
    .appName("RealTimeFinancialFraudDetection")
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.13:4.3.0"
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ---------------------------------------------------------
# 2. Load trained fraud detection model
# ---------------------------------------------------------
MODEL_PATH = "models/fraud_model.pkl"

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
features = model_data["features"]

print("=" * 70)
print("REAL-TIME FINANCIAL FRAUD DETECTION")
print("=" * 70)
print("Fraud detection model loaded successfully.")
print("Model:", type(model).__name__)
print("Number of features:", len(features))
print("Features:", features)


# ---------------------------------------------------------
# 3. Read transactions from Kafka
# ---------------------------------------------------------
kafka_stream = (
    spark.readStream
    .format("kafka")
    .option(
        "kafka.bootstrap.servers",
        "127.0.0.1:9092"
    )
    .option(
        "subscribe",
        "financial-transactions"
    )
    .option(
        "startingOffsets",
        "latest"
    )
    .option(
        "failOnDataLoss",
        "false"
    )
    .load()
)


# ---------------------------------------------------------
# 4. Convert Kafka value from binary to JSON string
# ---------------------------------------------------------
transactions = (
    kafka_stream
    .select(
        col("value")
        .cast(StringType())
        .alias("json_value")
    )
)


# ---------------------------------------------------------
# 5. Process each micro-batch
# ---------------------------------------------------------
def process_batch(batch_df, batch_id):

    if batch_df.rdd.isEmpty():
        return

    print("\n" + "=" * 70)
    print(f"PROCESSING BATCH: {batch_id}")
    print("=" * 70)

    rows = batch_df.collect()

    predictions = []

    for row in rows:

        try:

            # -------------------------------------------------
            # Read JSON transaction
            # -------------------------------------------------
            transaction = json.loads(
                row["json_value"]
            )

            # Convert transaction to Pandas DataFrame
            transaction_df = pd.DataFrame(
                [transaction]
            )

            # -------------------------------------------------
            # Create engineered features
            # -------------------------------------------------
            transaction_df["balance_error_orig"] = (
                transaction_df["oldbalanceOrg"]
                - transaction_df["amount"]
                - transaction_df["newbalanceOrig"]
            )

            transaction_df["balance_error_dest"] = (
                transaction_df["oldbalanceDest"]
                + transaction_df["amount"]
                - transaction_df["newbalanceDest"]
            )

            transaction_df["is_large_transaction"] = (
                transaction_df["amount"] > 200000
            ).astype(int)

            # -------------------------------------------------
            # One-hot encode transaction type
            # -------------------------------------------------
            transaction_df = pd.get_dummies(
                transaction_df,
                columns=["type"],
                prefix="type"
            )

            # -------------------------------------------------
            # Ensure ALL trained features exist
            # -------------------------------------------------
            for feature in features:

                if feature not in transaction_df.columns:
                    transaction_df[feature] = 0

            # Keep EXACT feature order used during training
            model_input = transaction_df[
                features
            ].copy()

            # -------------------------------------------------
            # Convert values to numeric
            # -------------------------------------------------
            model_input = model_input.apply(
                pd.to_numeric,
                errors="coerce"
            ).fillna(0)

            # -------------------------------------------------
            # Fraud probability
            # -------------------------------------------------
            fraud_probability = model.predict_proba(
                model_input
            )[0][1]

            # -------------------------------------------------
            # Classification
            # -------------------------------------------------
            prediction = int(
                fraud_probability >= 0.5
            )

            # -------------------------------------------------
            # Transaction information
            # -------------------------------------------------
            transaction_id = transaction.get(
                "transaction_id",
                transaction.get(
                    "step",
                    "N/A"
                )
            )

            amount = transaction.get(
                "amount",
                0
            )

            transaction_type = transaction.get(
                "type",
                "UNKNOWN"
            )

            predictions.append(
                {
                    "transaction_id": transaction_id,
                    "amount": amount,
                    "transaction_type": transaction_type,
                    "fraud_probability": fraud_probability,
                    "prediction": prediction
                }
            )

        except Exception as error:

            print(
                "Error processing transaction:",
                error
            )

    # ---------------------------------------------------------
    # 6. Display predictions and save to database
    # ---------------------------------------------------------
    for result in predictions:

        if result["prediction"] == 1:
            status = "🚨 FRAUD"
        else:
            status = "✅ GENUINE"

        # Save prediction to SQLite database
        try:

            save_transaction(
                result["transaction_id"],
                result["transaction_type"],
                result["amount"],
                result["fraud_probability"],
                result["prediction"]
            )

            database_status = "Database: SAVED"

        except Exception as database_error:

            database_status = (
                f"Database: ERROR - {database_error}"
            )

        print(
            f"Transaction: "
            f"{result['transaction_id']} | "
            f"Type: "
            f"{result['transaction_type']} | "
            f"Amount: "
            f"{result['amount']} | "
            f"Fraud Probability: "
            f"{result['fraud_probability']:.2%} | "
            f"Prediction: "
            f"{status} | "
            f"{database_status}"
        )


# ---------------------------------------------------------
# 7. Start Spark Structured Streaming
# ---------------------------------------------------------
query = (
    transactions
    .writeStream
    .foreachBatch(process_batch)
    .outputMode("append")
    .option(
        "checkpointLocation",
        "outputs/spark_checkpoint"
    )
    .start()
)

print("\n" + "=" * 70)
print("SPARK STREAMING STARTED")
print("=" * 70)
print(
    "Listening to Kafka topic: "
    "financial-transactions"
)
print("Waiting for new transactions...\n")


# ---------------------------------------------------------
# 8. Keep streaming application running
# ---------------------------------------------------------
query.awaitTermination()
