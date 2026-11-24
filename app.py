import streamlit as st
import sqlite3
import pandas as pd
import time

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Real-Time Financial Fraud Detection",
    page_icon="🚨",
    layout="wide"
)

DB_PATH = "database/fraud_detection.db"


# ---------------------------------------------------------
# DATABASE FUNCTION
# ---------------------------------------------------------

def load_transactions():

    connection = sqlite3.connect(DB_PATH)

    query = """
    SELECT
        id,
        transaction_id,
        transaction_type,
        amount,
        fraud_probability,
        prediction,
        detected_at
    FROM fraud_transactions
    ORDER BY id DESC
    """

    dataframe = pd.read_sql_query(
        query,
        connection
    )

    connection.close()

    return dataframe


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🚨 Real-Time Financial Fraud Detection System")

st.markdown(
    """
    **Kafka → PySpark → Machine Learning → SQLite → Streamlit**

    Real-time monitoring dashboard for detecting suspicious financial transactions.
    """
)

st.divider()


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

data = load_transactions()


if data.empty:

    st.warning("No transactions found in the database.")

    st.stop()


# ---------------------------------------------------------
# CALCULATE METRICS
# ---------------------------------------------------------

total_transactions = len(data)

fraud_transactions = int(
    data["prediction"].sum()
)

genuine_transactions = (
    total_transactions - fraud_transactions
)

fraud_rate = (
    fraud_transactions / total_transactions * 100
    if total_transactions > 0
    else 0
)

average_amount = data["amount"].mean()


# ---------------------------------------------------------
# KPI CARDS
# ---------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Transactions",
        total_transactions
    )

with col2:

    st.metric(
        "🚨 Fraud Detected",
        fraud_transactions
    )

with col3:

    st.metric(
        "✅ Genuine Transactions",
        genuine_transactions
    )

with col4:

    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )


st.divider()


# ---------------------------------------------------------
# ADD STATUS COLUMN
# ---------------------------------------------------------

data["status"] = data["prediction"].apply(
    lambda x: "🚨 FRAUD" if x == 1 else "✅ GENUINE"
)

data["fraud_probability_percent"] = (
    data["fraud_probability"] * 100
)


# ---------------------------------------------------------
# FRAUD ALERT
# ---------------------------------------------------------

fraud_data = data[
    data["prediction"] == 1
]


if not fraud_data.empty:

    st.error(
        f"🚨 FRAUD ALERT: {len(fraud_data)} suspicious "
        f"transaction(s) detected!"
    )

else:

    st.success(
        "✅ No fraudulent transactions detected."
    )


# ---------------------------------------------------------
# CHARTS
# ---------------------------------------------------------

chart_col1, chart_col2 = st.columns(2)


with chart_col1:

    st.subheader("Transaction Classification")

    classification_data = pd.DataFrame(
        {
            "Category": [
                "Genuine",
                "Fraud"
            ],
            "Count": [
                genuine_transactions,
                fraud_transactions
            ]
        }
    )

    st.bar_chart(
        classification_data.set_index("Category")
    )


with chart_col2:

    st.subheader("Transaction Amount")

    amount_chart = data[
        [
            "transaction_id",
            "amount"
        ]
    ].head(20)

    amount_chart = amount_chart.set_index(
        "transaction_id"
    )

    st.bar_chart(
        amount_chart
    )


st.divider()


# ---------------------------------------------------------
# FRAUD PROBABILITY CHART
# ---------------------------------------------------------

st.subheader("Fraud Probability of Recent Transactions")

probability_chart = data[
    [
        "transaction_id",
        "fraud_probability_percent"
    ]
].head(20)

probability_chart = probability_chart.set_index(
    "transaction_id"
)

st.line_chart(
    probability_chart
)


st.divider()


# ---------------------------------------------------------
# RECENT TRANSACTIONS
# ---------------------------------------------------------

st.subheader("📋 Recent Transactions")

display_data = data[
    [
        "transaction_id",
        "transaction_type",
        "amount",
        "fraud_probability_percent",
        "status",
        "detected_at"
    ]
].head(20)

display_data = display_data.rename(
    columns={
        "transaction_id": "Transaction ID",
        "transaction_type": "Type",
        "amount": "Amount",
        "fraud_probability_percent": "Fraud Probability (%)",
        "status": "Status",
        "detected_at": "Detected At"
    }
)

st.dataframe(
    display_data,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# FRAUD TRANSACTIONS TABLE
# ---------------------------------------------------------

if not fraud_data.empty:

    st.divider()

    st.subheader("🚨 Fraudulent Transactions")

    fraud_display = fraud_data[
        [
            "transaction_id",
            "transaction_type",
            "amount",
            "fraud_probability_percent",
            "detected_at"
        ]
    ].copy()

    fraud_display = fraud_display.rename(
        columns={
            "transaction_id": "Transaction ID",
            "transaction_type": "Type",
            "amount": "Amount",
            "fraud_probability_percent": "Fraud Probability (%)",
            "detected_at": "Detected At"
        }
    )

    st.dataframe(
        fraud_display,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# SYSTEM INFORMATION
# ---------------------------------------------------------

st.divider()

st.subheader("⚙️ System Information")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:

    st.info(
        "Kafka Topic: financial-transactions"
    )

with info_col2:

    st.info(
        "ML Model: Random Forest"
    )

with info_col3:

    st.info(
        "Database: SQLite"
    )


# ---------------------------------------------------------
# AUTO REFRESH
# ---------------------------------------------------------

time.sleep(2)

st.rerun()