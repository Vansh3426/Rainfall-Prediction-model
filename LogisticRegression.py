import numpy as np
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler




df = pd.read_csv("weatherAUS.csv")

# print(df.describe())

input1= df.drop(columns=['Date','RainTomorrow'])

target = df["RainTomorrow"]


#  Encoding and imputing categorical data


input2 =input1.select_dtypes(include ='object')

# print(input2)
imputer_c = SimpleImputer(strategy='most_frequent')
input3 = imputer_c.fit_transform(input2)

encoder = OneHotEncoder(sparse_output = False)
input3 = pd.DataFrame(encoder.fit_transform(input2))


# Imputing numeric data
input4 = input1.select_dtypes(include = "number")
imputer = SimpleImputer(strategy='mean')
input5 = pd.DataFrame(imputer.fit_transform(input4))

imputer_t = SimpleImputer(strategy='most_frequent')
target1 = imputer_t.fit_transform(target.to_frame())
# scaling

scaler = MinMaxScaler()
input6 = pd.DataFrame(scaler.fit_transform(input5))

# final input 

input_final = pd.concat([input3,input6],axis=1)

# model training

model = LogisticRegression()

train_model = model.fit(input_final,target1)

# predictions
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

example = pd.DataFrame([data])

example1= example.drop(columns=['Date','RainTomorrow'])

example1.replace("NA", np.nan, inplace=True)

example2 =example1.select_dtypes(include ='object')
# print(example2)

example2 = imputer_c.transform(example2)

example2= pd.DataFrame(encoder.transform(example2))


example3 = example1.select_dtypes(include = "number")

example4 = pd.DataFrame(imputer.transform(example3))

example4 = pd.DataFrame(scaler.transform(example4))

example_f= pd.concat([example2,example4],axis=1)



print(example_f)
predcition = train_model.predict(example_f)

print(predcition)