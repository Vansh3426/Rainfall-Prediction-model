import pandas as pd
import joblib
import numpy as np


data = {
    "Date": "2008-12-05",
    "Location": "Albury",
    "MinTemp": 17.5,
    "MaxTemp": 32.3,
    "Rainfall": 1,
    "Evaporation": "NA",
    "Sunshine": "NA",
    "WindGustDir": "W",
    "WindGustSpeed": 41,
    "WindDir9am": "ENE",
    "WindDir3pm": "NW",
    "WindSpeed9am": 7,
    "WindSpeed3pm": 20,
    "Humidity9am": 82,
    "Humidity3pm": 33,
    "Pressure9am": 1010.8,
    "Pressure3pm": 1006,
    "Cloud9am": 7,
    "Cloud3pm": 8,
    "Temp9am": 17.8,
    "Temp3pm": 29.7,
    "RainToday": "No",
    "RainTomorrow": "No"
}

df = pd.DataFrame([data])
df.replace("NA", np.nan, inplace=True)


model = joblib.load("Final_Pipeline.pkl")

prediction = model.predict(df)

print(prediction)