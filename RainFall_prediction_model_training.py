import numpy as np
import pandas as pd 
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer




df = pd.read_csv("weatherAUS.csv")
df =df.dropna(subset ="RainTomorrow")
numeric_col = df.select_dtypes(include = "number").columns.tolist()
categorical_col = df.select_dtypes(include ="object").columns.tolist()


y= df["RainTomorrow"]


nums1 =Pipeline([("imputer",SimpleImputer(strategy ="mean")),
                  ("scaling",MinMaxScaler())])

cat1 =Pipeline([("imputer",SimpleImputer(strategy ="most_frequent")),
                  ("encoding",OneHotEncoder(handle_unknown="ignore"))])

preprocessor =ColumnTransformer(
                                transformers = [("nums",nums1,numeric_col),
                                               ("cat",cat1,categorical_col)],
                                remainder = "drop")

final_pipeline = Pipeline([("preprocessing",preprocessor),
                  ("model",LogisticRegression())])

final_pipeline.fit(df,y)

joblib.dump(final_pipeline,"Final_Pipeline.pkl")
print("model train and save successfully")

