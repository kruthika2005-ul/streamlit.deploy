import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# load data
data = pd.read_csv("Employee_Dataset.csv")

# cleaning
data["age"] = pd.to_numeric(data["age"], errors="coerce")
data["salary"] = pd.to_numeric(data["salary"], errors="coerce")

data.dropna(inplace=True)

X = data[["age"]]
y = data["salary"]

model = LinearRegression()
model.fit(X, y)

st.title("Salary Prediction App")

age = st.number_input("Enter Age")

if st.button("Predict Salary"):
    result = model.predict([[age]])
    st.success(f"Predicted Salary: {result[0]}")