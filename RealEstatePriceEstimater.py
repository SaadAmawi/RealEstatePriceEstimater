import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

RealEstate = pd.read_csv('realtor-data.zip.csv')
print(RealEstate.head())

X=RealEstate[['bed','state','house_size']]
y=RealEstate[['price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

Estimate=model.predict(X_test)
print(Estimate)