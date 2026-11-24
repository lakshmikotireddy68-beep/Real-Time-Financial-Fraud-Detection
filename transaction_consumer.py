import json
from kafka import KafkaConsumer

print("=" * 60)
print("KAFKA FINANCIAL TRANSACTION CONSUMER")
print("=" * 60)

consumer = KafkaConsumer(
    "financial-transactions",
    bootstrap_servers="127.0.0.1:9092",
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="fraud-detection-consumer",
    value_deserializer=lambda value: json.loads(
        value.decode("utf-8")
    )
)

print("\nConnected to Kafka successfully.")
print("Waiting for transactions...\n")

for message in consumer:

    transaction = message.value

    print("-" * 60)

    print("Transaction ID:", transaction["transaction_id"])
    print("Amount:", transaction["amount"])

    print(
        "Fraud Probability:",
        round(transaction["fraud_probability"] * 100, 2),
        "%"
    )

    if transaction["prediction"] == 1:
        print("STATUS: FRAUD ALERT!")
    else:
        print("STATUS: GENUINE TRANSACTION")

    print(
        "Actual Label:",
        transaction["actual_label"]
    )

    print("Kafka Partition:", message.partition)
    print("Kafka Offset:", message.offset)