import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# load data
data = pd.read_csv("Employee_Dataset.csv")

# cleaning
data["age"] = pd.to_numeric(data["age"], errors="coerce").astype(int)
data["salary"] = pd.to_numeric(data["salary"], errors="coerce").astype(int)

data.dropna(inplace=True)

X = data[["age"]]
y = data["salary"]

model = LinearRegression()
model.fit(X, y)

st.title("Salary Prediction App")

# make age input integer
age = int(st.number_input("Enter Age", step=1))

if st.button("Predict Salary"):
    result = model.predict([[age]])
    
    # convert prediction to int
    predicted_salary = int(result[0])
    
    st.success(f"Predicted Salary: {predicted_salary}")