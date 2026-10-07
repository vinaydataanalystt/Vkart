import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("📚Student Marks Prediction")

# Sample data
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [20, 30, 40, 50, 55, 65, 70, 80, 90, 95]
}

df = pd.DataFrame(data)

# Create model
X = df[["Hours"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

# User input
hours = st.number_input(
    "Enter number of hours studied:",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

# Prediction
if st.button("Predict Marks"):
    prediction = model.predict([[hours]])

    st.success(f"Predicted Marks: {prediction[0]:.2f}")

# Show data
st.subheader("Training Data")
st.dataframe(df)