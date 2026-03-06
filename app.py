import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# load data
data = pd.read_csv("Employee_Dataset.csv")

# convert to numeric
data["age"] = pd.to_numeric(data["age"], errors="coerce")
data["salary"] = pd.to_numeric(data["salary"], errors="coerce")

# remove missing values
data.dropna(inplace=True)

# convert to integer
data["age"] = data["age"].astype(int)
data["salary"] = data["salary"].astype(int)

X = data[["age"]]
y = data["salary"]

model = LinearRegression()
model.fit(X, y)

st.title("Salary Prediction App")

age = st.number_input("Enter Age", min_value=18, max_value=65, step=1)

if st.button("Predict Salary"):
    result = model.predict([[age]])
    predicted_salary = int(result[0])

    st.success(f"Predicted Salary: {predicted_salary}")