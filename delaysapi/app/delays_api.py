#!/usr/bin/env python
# coding: utf-8

# import statements
from fastapi import FastAPI, HTTPException, Form, Request
import json
import numpy as np
import pickle
import datetime
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import uvicorn
from pydantic import BaseModel
from typing import Annotated
from starlette.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Import the airport encodings file
f = open('./app/airport_encodings.json')

# returns JSON object as a dictionary
airports = json.load(f)


def create_airport_encoding(airport: str, airports: dict) -> np.array:
    """
    create_airport_encoding is a function that creates an array the length of all arrival airports from the chosen
    departure airport.  The array consists of all zeros except for the specified arrival airport, which is a 1.

    Parameters
    ----------
    airport : str
        The specified arrival airport code as a string
    airports: dict
        A dictionary containing all the arrival airport codes served from the chosen departure airport

    Returns
    -------
    np.array
        A NumPy array the length of the number of arrival airports.  All zeros except for a single 1
        denoting the arrival airport.  Returns None if arrival airport is not found in the input list.
        This is a one-hot encoded airport array.

    """
    temp = np.zeros(len(airports))
    if airport in airports:
        temp[airports.get(airport)] = 1
        temp = temp.T
        return temp
    else:
        return None

# TODO:  write the back-end logic to provide a prediction given the inputs
# requires finalized_model.pkl to be loaded
# the model must be passed a NumPy array consisting of the following:
# (polynomial order, encoded airport array, departure time as seconds since midnight, arrival time as seconds since midnight)
# the polynomial order is 1 unless you changed it during model training in Task 2
# YOUR CODE GOES HERE

def format_hour(string: str):
    if pd.isnull(string):
        raise HTTPException(status_code=404, detail=f"Input required")
    else:
        if string == 2400: string = 0
        string = "{0:04d}".format(int(string))
        t = datetime.time(int(string[0:2]), int(string[2:4]))
        seconds = (t.hour * 3600) + (t.minute * 60) + t.second
        return seconds

with open('./app/finalized_model.pkl', 'rb') as m:
    model = pickle.load(m)

class AirportInput(BaseModel):
    arrival_airport: str
    departure_time: str
    arrival_time: str

# TODO:  write the API endpoints.
# YOUR CODE GOES HERE
app: FastAPI = FastAPI(title="ORD Airport Flight Delays")
templates = Jinja2Templates(directory="templates")

@app.get("/")
def welcome():
    return {"message": "Welcome to the flight delay estimation service. API is working."}

@app.get("/predict/delays", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/predict/search")
async def predict(data: Annotated[AirportInput, Form()]):
    arrival_airport: str = create_airport_encoding(data.arrival_airport, airports)
    departure_time: int = format_hour(data.departure_time)
    arrival_time: int = format_hour(data.arrival_time)

    polynomial_order = 1
    b = np.array([departure_time, arrival_time])
    a = np.hstack([arrival_airport, b])
    input_data = np.reshape(a,(1, -1))

    poly = PolynomialFeatures(degree=polynomial_order)
    input_fit = poly.fit_transform(input_data)

    prediction = model.predict(input_fit)
    result = ",".join(str(x) for x in prediction)

    txt = "Your predicted delay time is {delay} minutes."
    delay = txt.format(delay = result)
    return delay

if __name__ == "__main__":
    uvicorn.run("predict_delays:app", reload=True)