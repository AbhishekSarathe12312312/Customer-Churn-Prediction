import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.title("Customer Churn Prediction")
st.write("Predict whether a customer may leave.")

df = pd.read_csv("customer_data.csv")

X = df[["Age", "Tenure", "MonthlyCharges", "SupportCalls"]]
y = df["Churn"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

age = st.number_input("Customer Age", 18, 100, 30)
tenure = st.number_input("Tenure (months)", 0, 300, 2)
charges = st.number_input("Monthly Charges (₹)", 0, 100000, 850)
calls = st.number_input("Support Calls", 0, 50, 5)

if st.button("Predict Churn"):
    customer = pd.DataFrame(
        [[age, tenure, charges, calls]],
        columns=["Age", "Tenure", "MonthlyCharges", "SupportCalls"]
    )
    prediction = model.predict(customer)[0]

    if prediction == 1:
        st.warning("Customer May Leave")
    else:
        st.success("Customer May Stay")