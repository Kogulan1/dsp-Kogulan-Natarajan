import uvicorn
import pickle
from pydantic import BaseModel
import psycopg2
import re

# FastAPI libray
from fastapi import FastAPI

# Initiate app instance
app = FastAPI(title='CO2 Analytics', version='1.0',
              description='Lr model is used for prediction')

# Initialize model artifacte files. This will be loaded at the start of FastAPI model server.
pickle_in = open("model_pickle.pkl", "rb")
model = pickle.load(pickle_in)

# Initiate DB CONNECTION

connection = psycopg2.connect(user="postgres",
                              password="1234",
                              host="127.0.0.1",
                              port="5432",
                              database="postgres")
cursor = connection.cursor()

class Data(BaseModel):
    Engine_Size: float
    Cylinders: float
    Fuel_Consumption_City: float
    Fuel_Consumption_Hwy: float
    Fuel_Consumption_Comb: float
    Fuel_Consumption_Comb_mpg: float


# Api root or home endpoint
@app.get('/')
@app.get('/home')
def read_home():
    """
     Home endpoint which can be used to test the availability of the application.
     """
    return {'message': 'System is healthy'}


# ML API endpoint for making prediction against the request received from client
@app.post("/predict")
def predict(data: Data):
    print(data)
    # Extract data in correct order
    prediction = model.predict([[data.Engine_Size,
                                 data.Cylinders,
                                 data.Fuel_Consumption_City,
                                 data.Fuel_Consumption_Hwy,
                                 data.Fuel_Consumption_Comb,
                                 data.Fuel_Consumption_Comb_mpg]])

    #STORE IN DB
    piIn = str(prediction)
    patn = re.sub(r"[\([{})\]]", "", piIn)
    postgres_insert_query = """ INSERT INTO testdb (engine_size, cylinders,  fuel_consumption_city, fuel_consumption_hwy, fuel_consumption_comb, co2_emissions) VALUES (%s,%s,%s,%s,%s,%s,%s)"""
    record_to_insert = (
    Engine_Size, Cylinders, Fuel_Consumption_City, Fuel_Consumption_Hwy, Fuel_Consumption_Comb, patn)
    cursor.execute(postgres_insert_query, record_to_insert)

    connection.commit()
    count = cursor.rowcount
    print(count, "Record inserted successfully into mobile table")
    # return {"prediction": prediction}
    return {"result": prediction[0]}


if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
