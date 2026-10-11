
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datetime import date

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Smart Expense Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. SESSION STATE
# ==========================================
if "expenses" not in st.session_state:
    st.session_state.expenses = []

if "budget" not in st.session_state:
    st.session_state.budget = 10000.0

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ==========================================
# 3. PREMIUM INTERFACE STYLING
# ==========================================
dark = st.session_state.dark_mode

if dark:
    bg = "#101827"
    panel = "#192538"
    text = "#F1F5F9"
    muted = "#A7B4C8"
    border = "#34445A"
else:
    bg = "#F4F7FB"
    panel = "#FFFFFF"
    text = "#17243A"
    muted = "#64748B"
    border = "#E2E8F0"

st.markdown(f"""
<style>
.stApp {{
    background-color: {bg};
    color: {text};
}}
[data-testid="stSidebar"] {{
    background-color: {panel};
}}
.block-container {{
    padding-top: 1.8rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}}
h1, h2, h3, p, label {{
    color: {text};
}}
.hero {{
    background: linear-gradient(120deg, #13233F, #235B55);
    padding: 28px;
    border-radius: 20px;
    margin-bottom: 24px;
    color: white;
}}
.hero h1 {{
    color: white !important;
    margin-bottom: 5px;
}}
.hero p {{
    color: #D9E7F2 !important;
}}
.metric-card {{
    background: {panel};
    border: 1px solid {border};
    border-radius: 16px;
    padding: 19px;
    min-height: 120px;
}}
.metric-label {{
    color: {muted};
    font-size: 14px;
    margin-bottom: 10px;
}}
.metric-value {{
    color: {text};
    font-size: 27px;
    font-weight: 700;
    overflow-wrap: anywhere;
}}
.metric-note {{
    color: {muted};
    font-size: 12px;
    margin-top: 6px;
}}
.section-card {{
    background: {panel};
    border: 1px solid {border};
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
}}
div.stButton > button,
div.stFormSubmitButton > button {{
    border-radius: 10px;
    min-height: 42px;
    font-weight: 600;
}}
[data-testid="stDataFrame"] {{
    border-radius: 12px;
}}
footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. MACHINE LEARNING TRAINING DATA
# ==========================================
training_data = {
    "description": [
        "vegetables groceries supermarket milk bread",
        "pizza restaurant lunch dinner food",
        "coffee tea snacks breakfast meal",
        "bus ticket train metro taxi fare",
        "petrol diesel fuel car bike",
        "cab ride travel transport",
        "clothes shoes shopping mall",
        "online shopping bag watch accessories",
        "phone laptop headphones electronics",
        "electricity bill water bill gas bill",
        "mobile recharge internet wifi bill",
        "rent house apartment maintenance",
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

train_df = pd.DataFrame(training_data)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_df["description"])

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, train_df["category"])

# ==========================================
# 5. HEADER AND DARK MODE
# ==========================================
with st.sidebar:
    st.markdown("## 💰 Expense Analyzer")
    st.caption("PERSONAL FINANCE DASHBOARD")
    st.divider()

    st.session_state.dark_mode = st.toggle(
        "🌙 Dark mode",
        value=st.session_state.dark_mode
    )

    st.divider()
    st.markdown("### 💡 Quick tip")
    st.write(
        "Add clear descriptions such as "
        "'grocery shopping' or 'electricity bill' "
        "to help the ML model classify expenses."
    )
    st.caption("Predictions may not always be correct.")

st.markdown("""
<div class="hero">
    <h1>Welcome to your Finance Dashboard 👋</h1>
    <p>Track spending, manage your monthly budget, and understand your money.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 6. MONTHLY BUDGET TRACKER
# ==========================================
st.markdown("## 🎯 Monthly Budget")

st.number_input(
    "Set your monthly budget (₹)",
    min_value=1.0,
    step=500.0,
    key="budget"
)

today = date.today()

expenses_df = pd.DataFrame(
    st.session_state.expenses,
    columns=["Date", "Description", "Amount", "Category"]
)

if not expenses_df.empty:
    expenses_df["Date"] = pd.to_datetime(
        expenses_df["Date"]
    ).dt.date

    monthly_df = expenses_df[
        (pd.to_datetime(expenses_df["Date"]).dt.month == today.month)
        & (pd.to_datetime(expenses_df["Date"]).dt.year == today.year)
    ].copy()
else:
    monthly_df = pd.DataFrame(
        columns=["Date", "Description", "Amount", "Category"]
    )

monthly_spending = float(monthly_df["Amount"].sum())
budget = float(st.session_state.budget)
remaining = budget - monthly_spending
budget_used = monthly_spending / budget * 100

st.progress(min(budget_used / 100, 1.0))

b1, b2, b3 = st.columns(3)

b1.metric("Monthly Budget", f"₹{budget:,.0f}")
b2.metric("Spent This Month", f"₹{monthly_spending:,.2f}")
b3.metric("Budget Remaining", f"₹{remaining:,.2f}")

if remaining < 0:
    st.error(
        f"⚠️ You exceeded your monthly budget by "
        f"₹{abs(remaining):,.2f}."
    )
elif budget_used >= 80:
    st.warning(
        f"You have used {budget_used:.1f}% of your budget. "
        "Consider limiting extra spending."
    )
else:
    st.success(f"You have used {budget_used:.1f}% of your monthly budget.")

# ==========================================
# 7. ADD EXPENSE FORM
# ==========================================
st.divider()
st.markdown("## ➕ Add a New Expense")

with st.form("expense_form", clear_on_submit=True):
    description = st.text_input(
        "What did you spend on?",
        placeholder="e.g. Bought groceries and milk"
    )

    c1, c2 = st.columns(2)

    with c1:
        amount = st.number_input(
            "Amount (₹)",
            min_value=0.0,
            step=50.0
        )

    with c2:
        expense_date = st.date_input(
            "Expense date",
            value=today
        )

    submitted = st.form_submit_button(
        "✨ Predict Category & Save",
        use_container_width=True
    )

    if submitted:
        if not description.strip():
            st.error("Please enter an expense description.")
        elif amount <= 0:
            st.error("Enter an amount greater than zero.")
        else:
            transformed = vectorizer.transform(
                [description.strip().lower()]
            )
            prediction = model.predict(transformed)[0]

            st.session_state.expenses.append({
                "Date": expense_date,
                "Description": description.strip(),
                "Amount": float(amount),
                "Category": prediction
            })

            st.session_state.last_prediction = prediction
            st.rerun()

if "last_prediction" in st.session_state:
    st.success(
        f"Expense saved! ML-predicted category: "
        f"{st.session_state.last_prediction}"
    )
    del st.session_state["last_prediction"]

# ==========================================
# 8. SUMMARY CARDS
# ==========================================
st.divider()
st.markdown("## 📊 Spending Overview")

expenses_df = pd.DataFrame(
    st.session_state.expenses,
    columns=["Date", "Description", "Amount", "Category"]
)

if not expenses_df.empty:
    expenses_df["Date"] = pd.to_datetime(
        expenses_df["Date"]
    ).dt.date

total_spending = float(expenses_df["Amount"].sum())
expense_count = len(expenses_df)
average_expense = (
    total_spending / expense_count if expense_count else 0
)

m1, m2, m3, m4 = st.columns(4)

cards = [
    (m1, "TOTAL SPENDING", f"₹{total_spending:,.2f}", "All saved expenses"),
    (m2, "THIS MONTH", f"₹{monthly_spending:,.2f}", "Current month"),
    (m3, "TRANSACTIONS", str(expense_count), "Expenses recorded"),
    (m4, "AVERAGE EXPENSE", f"₹{average_expense:,.2f}", "Per transaction")
]

for col, label, value, note in cards:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 9. INTERACTIVE CHARTS
# ==========================================
st.markdown("## 📈 Expense Analytics")

if not expenses_df.empty:
    chart_left, chart_right = st.columns(2)

    category_totals = (
        expenses_df.groupby("Category", as_index=False)["Amount"]
        .sum()
    )

    with chart_left:
        st.markdown("### Spending by Category")
        pie = px.pie(
            category_totals,
            names="Category",
            values="Amount",
            hole=0.55,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        pie.update_layout(
            template="plotly_dark" if dark else "plotly_white",
            paper_bgcolor=panel,
            plot_bgcolor=panel,
            font_color=text,
            margin=dict(t=20, b=20, l=10, r=10),
            legend_title_text="Category"
        )
        st.plotly_chart(pie, use_container_width=True)

    with chart_right:
        st.markdown("### Category Comparison")
        bar = px.bar(
            category_totals,
            x="Category",
            y="Amount",
            color="Category",
            text_auto=".2s",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        bar.update_layout(
            template="plotly_dark" if dark else "plotly_white",
            paper_bgcolor=panel,
            plot_bgcolor=panel,
            font_color=text,
            showlegend=False,
            xaxis_title="Category",
            yaxis_title="Amount (₹)",
            margin=dict(t=20, b=20, l=10, r=10)
        )
        st.plotly_chart(bar, use_container_width=True)

    st.markdown("### Spending Over Time")

    daily = expenses_df.copy()
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily_totals = (
        daily.groupby("Date", as_index=False)["Amount"].sum()
        .sort_values("Date")
    )

    line = px.line(
        daily_totals,
        x="Date",
        y="Amount",
        markers=True,
        labels={"Amount": "Spending (₹)", "Date": "Date"}
    )
    line.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=panel,
        plot_bgcolor=panel,
        font_color=text
    )
    st.plotly_chart(line, use_container_width=True)

    # ==========================================
    # 10. TRANSACTION HISTORY & CSV EXPORT
    # ==========================================
    st.divider()
    st.markdown("## 🧾 Transaction History")

    selected_category = st.selectbox(
        "Filter by category",
        ["All"] + sorted(expenses_df["Category"].unique().tolist())
    )

    filtered_df = expenses_df.copy()

    if selected_category != "All":
        filtered_df = filtered_df[
            filtered_df["Category"] == selected_category
        ]

    st.dataframe(
        filtered_df.sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    csv_data = expenses_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Expenses CSV",
        data=csv_data,
        file_name="smart_expenses.csv",
        mime="text/csv"
    )

    if st.button("🗑️ Clear All Expenses"):
        st.session_state.expenses = []
        st.rerun()

else:
    st.info(
        "Your dashboard is ready! Add your first expense above "
        "to unlock the summary cards and interactive charts."
    )

# ==========================================
# 11. FOOTER
# ==========================================
st.divider()
st.caption(
    "Smart Expense Analyzer · Python + TF-IDF + Logistic Regression + Streamlit"
)
st.caption(
    "Prototype note: expenses are kept in the current session only. "
    "They may disappear after a session reset. Download CSV to keep a copy."
)
