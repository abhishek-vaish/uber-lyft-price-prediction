# load the required library
from fastapi import FastAPI
import pickle
import sklearn
from datetime import datetime
from pydantic import BaseModel
import random
import pandas as pd

# initialize the fastapi
app = FastAPI()
# load the machine learning model
model = pickle.load(open('../model/random_forest_regressor.pkl', 'rb'))

# surge_multiplier list
surge_multiplier_list = [1.00, 1.25, 1.50, 1.75, 2.00, 2.50]
# surge_multiplier variable
surge_multiplier = 0.0
# cab type variable
cab_type = 0.0

# Model Class
class DataField(BaseModel):
    distance: float
    cab_type: str
    time_stamp: str
    destination: str
    source: str
    product_id: str
    name: str

@app.get("/")
def welcome():
    return {
        "Welcome": "Welcome To FastAPI"
    }

# ---------------------------------prediction call-------------------------------------
@app.post("/predict")
def data_from_user(data_field: DataField):
    """
    This function is used to call the prediction api 
    to predict the uber/lyft price by taking user input.
    data_field: DataField Base Model
    return: dictionary contains the prediction value.
    """
    data_frame = data_manipulation(data_field)
    prediction = model.predict(data_frame)
    return {
        "cab_type": data_field.cab_type,
        "time_stamp": data_field.time_stamp,
        "destination": data_field.destination,
        "source": data_field.source,
        "name": data_field.name,
        "prediction": prediction[0]
    }

# --------------------------------data prepration----------------------------------------
def data_manipulation(data_field: DataField):
    """
    This function is used to prepare the data for the machine learning model to predict the price.
    data_field: DataField Base Model
    return: DataFrame
    """

    # ------------------------------product_id----------------------------------------------
    product_id = [0.0 for i in range(12)]

    product_id_label = ['55c66225-fbe7-4fd5-9072-eab1ece5e23e',
       '6c84fd89-3f11-4782-9b50-97c468b19529',
       '6d318bcc-22a3-4af6-bddd-b409bfce1546',
       '6f72dfc5-27f1-42e8-84db-ccc7a75f6969',
       '997acbb5-e102-41e1-b155-9df7de0a73f2',
       '9a0e7b09-b92b-4c41-9779-2ad22b4d779d', 'lyft', 'lyft_line', 'lyft_lux',
       'lyft_luxsuv', 'lyft_plus', 'lyft_premier']   
    
    if data_field.product_id in product_id_label:
        product_id[product_id_label.index(data_field.product_id)] = 1.0


    # ---------------------------------------------Source Destination-------------------------------------
    source_destination = [0.0 for i in range(72)]

    source_destination_label = ['Back Bay-Boston University', 'Back Bay-Fenway',
       'Back Bay-Haymarket Square', 'Back Bay-North End',
       'Back Bay-Northeastern University', 'Back Bay-South Station',
       'Beacon Hill-Boston University', 'Beacon Hill-Fenway',
       'Beacon Hill-Haymarket Square', 'Beacon Hill-North End',
       'Beacon Hill-Northeastern University', 'Beacon Hill-South Station',
       'Boston University-Back Bay', 'Boston University-Beacon Hill',
       'Boston University-Financial District',
       'Boston University-North Station', 'Boston University-Theatre District',
       'Boston University-West End', 'Fenway-Back Bay', 'Fenway-Beacon Hill',
       'Fenway-Financial District', 'Fenway-North Station',
       'Fenway-Theatre District', 'Fenway-West End',
       'Financial District-Boston University', 'Financial District-Fenway',
       'Financial District-Haymarket Square', 'Financial District-North End',
       'Financial District-Northeastern University',
       'Financial District-South Station', 'Haymarket Square-Back Bay',
       'Haymarket Square-Beacon Hill', 'Haymarket Square-Financial District',
       'Haymarket Square-North Station', 'Haymarket Square-Theatre District',
       'Haymarket Square-West End', 'North End-Back Bay',
       'North End-Beacon Hill', 'North End-Financial District',
       'North End-North Station', 'North End-Theatre District',
       'North End-West End', 'North Station-Boston University',
       'North Station-Fenway', 'North Station-Haymarket Square',
       'North Station-North End', 'North Station-Northeastern University',
       'North Station-South Station', 'Northeastern University-Back Bay',
       'Northeastern University-Beacon Hill',
       'Northeastern University-Financial District',
       'Northeastern University-North Station',
       'Northeastern University-Theatre District',
       'Northeastern University-West End', 'South Station-Back Bay',
       'South Station-Beacon Hill', 'South Station-Financial District',
       'South Station-North Station', 'South Station-Theatre District',
       'South Station-West End', 'Theatre District-Boston University',
       'Theatre District-Fenway', 'Theatre District-Haymarket Square',
       'Theatre District-North End',
       'Theatre District-Northeastern University',
       'Theatre District-South Station', 'West End-Boston University',
       'West End-Fenway', 'West End-Haymarket Square', 'West End-North End',
       'West End-Northeastern University', 'West End-South Station']

    source_destination_user_input = data_field.source + '-' + data_field.destination

    if source_destination_user_input in source_destination_label:
        source_destination[source_destination_label.index(source_destination_user_input)] = 1.0


    # -----------------------------------------------name------------------------------------------------
    name = [0.0 for i in range(12)]

    name_label = ['Black',
       'Black SUV', 'Lux', 'Lux Black', 'Lux Black XL', 'Lyft', 'Lyft XL',
       'Shared', 'UberPool', 'UberX', 'UberXL', 'WAV']

    if data_field.name in name_label:
        name[name_label.index(data_field.name)] = 1.0

    # -------------------------------------------Cab Type------------------------------------
    if data_field.cab_type == 'Lyft':
        cab_type = 0.0
    else:
        cab_type = 1.0

    # ------------------------------------------TimeStamp-------------------------------------
    element = datetime.strptime(data_field.time_stamp, "%d/%m/%Y,%H:%M")
    time_stamp = datetime.timestamp(element)

    # -----------------------------------------Surge Multiplier-------------------------------
    if(data_field.cab_type == 'Uber'):
        surge_multiplier = 1.0
    else:
        surge_multiplier = surge_multiplier_list[random.randrange(start=0, stop=len(surge_multiplier_list))]

    # ----------------------------------------Remaining Lable----------------------------------
    remainder = [data_field.distance, surge_multiplier, time_stamp, cab_type]
    data = product_id + source_destination + name + remainder

    # ----------------------------------------DataFrame----------------------------------------
    data_frame = pd.DataFrame(data).T

    # ---------------------------------------Return--------------------------------------------
    return data_frame
    

