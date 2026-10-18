import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("customer_data.csv")

X = df[["Age", "Tenure", "MonthlyCharges", "SupportCalls"]]
y = df["Churn"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

customer = pd.DataFrame(
    [[30, 2, 850, 5]],
    columns=["Age", "Tenure", "MonthlyCharges", "SupportCalls"]
)

prediction = model.predict(customer)[0]

print("Customer Churn Prediction")
print("Prediction:", "Customer May Leave" if prediction == 1 else "Customer May Stay")
