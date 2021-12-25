import streamlit
import requests
import json
import re


def run():
    streamlit.title("CO2 Prediction")
    Engine_Size = streamlit.text_input("Engine_Size")
    Cylinders = streamlit.text_input("Cylinders")
    Fuel_Consumption_City = streamlit.text_input("Fuel_Consumption_City")
    Fuel_Consumption_Hwy = streamlit.text_input("Fuel_Consumption_Hwy")
    Fuel_Consumption_Comb = streamlit.text_input("Fuel_Consumption_Comb")


    data = {
        'Engine_Size': Engine_Size,
        'Cylinders': Cylinders,
        'Fuel_Consumption_City': Fuel_Consumption_City,
        'Fuel_Consumption_Hwy': Fuel_Consumption_Hwy,
        'Fuel_Consumption_Comb': Fuel_Consumption_Comb,


    }

    if streamlit.button("Predict"):

        response = requests.post("http://127.0.0.1:8000/predict", json=data)
        prediction = response.json()
        streamlit.success(f"The prediction from model: {prediction}")



if __name__ == '__main__':
    # by default it will run at 8501 port
    run()
