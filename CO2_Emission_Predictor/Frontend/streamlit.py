import streamlit
import requests
import json
import re
import psycopg2
connection = psycopg2.connect(user="postgres",
                              password="1234",
                              host="127.0.0.1",
                              port="5432",
                              database="postgres")
cursor = connection.cursor()

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
        piIn = str(prediction)
        patn = re.sub(r"[\([{})\]]", "", piIn)
        postgres_insert_query = """ INSERT INTO testdb (engine_size, cylinders,  fuel_consumption_city, fuel_consumption_hwy, fuel_consumption_comb, co2_emissions) VALUES (%s,%s,%s,%s,%s,%s,%s)"""
        record_to_insert = (Engine_Size, Cylinders, Fuel_Consumption_City, Fuel_Consumption_Hwy, Fuel_Consumption_Comb, patn)
        cursor.execute(postgres_insert_query, record_to_insert)

        connection.commit()
        count = cursor.rowcount
        print(count, "Record inserted successfully into mobile table")


if __name__ == '__main__':
    # by default it will run at 8501 port
    run()
