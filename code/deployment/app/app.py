import streamlit as st
import requests

api_url = "http://api:8000/predict"
st.title("Diamond Price Prediction")

st.write("Enter the features for prediction (carat, depth, table, x, y, z):")
features = st.text_input(
    "Features in csv format",
    "0.23,61.5,55,3.95,3.98,2.43"
)

if st.button("Make prediction"):
    try:
        feature_list = [float(x.strip()) for x in features.split(",")]
        if len(feature_list) != 6:
            st.write("Please enter exactly 6 features: carat, depth, table, x, y, z.")
        else:
            payload = {"features": feature_list}
            response = requests.post(api_url, json=payload)

            if response.status_code == 200:
                prediction = response.json().get("prediction")
                st.write(f"Predicted price: ${prediction:.2f}")
            else:
                st.write("Error in response from the API")
    except ValueError:
        st.write("Please enter valid numbers for features.")
