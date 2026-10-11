
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datetime import date

st.set_page_config(
    page_title="Smart Expense Analyzer",
    page_icon="💰",
    layout="wide"
)

# -----------------------------
# 1. TRAIN A SIMPLE ML MODEL
# -----------------------------
training_data = {
    "description": [
        "bought vegetables groceries supermarket milk bread",
        "ordered pizza restaurant lunch dinner food",
        "coffee tea snacks breakfast meal",
        "bus ticket train metro taxi auto fare",
        "petrol diesel fuel car bike charging",
        "uber cab ride travel transport",
        "bought clothes shoes shopping mall",
        "online shopping bag watch accessories",
        "bought phone laptop headphones electronics",
        "electricity bill water bill gas bill",
        "mobile recharge internet wifi bill",
        "rent house maintenance apartment",
        "doctor hospital medicine pharmacy",
        "medical checkup dental clinic treatment",
        "movie cinema concert music entertainment",
        "game subscription streaming ticket",
        "school college tuition books course",
        "online class exam fees education",
        "haircut salon personal care",
        "gift donation miscellaneous other"
    ],
    "category": [
        "Food", "Food", "Food",
        "Transport", "Transport", "Transport",
        "Shopping", "Shopping", "Shopping",
        "Bills", "Bills", "Bills",
        "Health", "Health",
        "Entertainment", "Entertainment",
        "Education", "Education",
        "Other", "Other"
    ]
}

df_train = pd.DataFrame(training_data)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(
    df_train["description"]
)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, df_train["category"])

# -----------------------------
# 2. APP DESIGN
# -----------------------------
st.title("💰 Smart Expense Analyzer")
st.write(
    "Track your spending and predict expense categories "
    "using Machine Learning."
)

if "expenses" not in st.session_state:
    st.session_state.expenses = []

# -----------------------------
# 3. ENTER AN EXPENSE
# -----------------------------
st.subheader("➕ Add an Expense")

with st.form("expense_form"):
    description = st.text_input(
        "Expense description",
        placeholder="e.g. Bought groceries"
    )

    amount = st.number_input(
        "Amount (₹)",
        min_value=0.0,
        step=50.0
    )

    expense_date = st.date_input(
        "Date",
        value=date.today()
    )

    submitted = st.form_submit_button(
        "Predict Category & Save"
    )

    if submitted:
        if not description.strip():
            st.error("Please enter an expense description.")
        elif amount <= 0:
            st.error("Please enter an amount greater than zero.")
        else:
            features = vectorizer.transform(
                [description.lower()]
            )
            prediction = model.predict(features)[0]

            st.session_state.expenses.append({
                "Date": expense_date,
                "Description": description.strip(),
                "Amount": float(amount),
                "Category": prediction
            })

            st.success(
                f"Predicted category: {prediction}"
            )

# -----------------------------
# 4. DASHBOARD
# -----------------------------
st.divider()
st.subheader("📊 Spending Dashboard")

if st.session_state.expenses:
    expenses_df = pd.DataFrame(
        st.session_state.expenses
    )

    total = expenses_df["Amount"].sum()
    count = len(expenses_df)
    average = total / count

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Spending", f"₹{total:,.2f}")
    c2.metric("Number of Expenses", count)
    c3.metric("Average Expense", f"₹{average:,.2f}")

    category_totals = (
        expenses_df.groupby("Category", as_index=False)
        ["Amount"].sum()
    )

    left, right = st.columns(2)

    with left:
        st.write("### Spending by Category")
        fig = px.pie(
            category_totals,
            names="Category",
            values="Amount",
            hole=0.45
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.write("### Category Comparison")
        bar = px.bar(
            category_totals,
            x="Category",
            y="Amount",
            color="Category"
        )
        st.plotly_chart(bar, use_container_width=True)

    st.write("### Expense History")
    st.dataframe(
        expenses_df.sort_values(
            "Date", ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    csv_data = expenses_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Expenses as CSV",
        data=csv_data,
        file_name="my_expenses.csv",
        mime="text/csv"
    )

    if st.button("Clear All Expenses"):
        st.session_state.expenses = []
        st.rerun()

else:
    st.info(
        "No expenses yet. Add your first expense above "
        "to see the dashboard."
    )

st.caption(
    "Educational prototype. Category predictions may be incorrect. "
    "Expense records are stored only in the current app session."
)
