import sqlite3

DB_PATH = "database/fraud_detection.db"

def save_transaction(transaction_id, transaction_type, amount, fraud_probability, prediction):
    connection = sqlite3.connect(DB_PATH)

    connection.execute(
        """
        INSERT INTO fraud_transactions
        (transaction_id, transaction_type, amount, fraud_probability, prediction)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            transaction_id,
            transaction_type,
            amount,
            fraud_probability,
            prediction
        )
    )

    connection.commit()
    connection.close()
