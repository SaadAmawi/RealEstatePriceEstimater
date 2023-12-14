import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
import seaborn as sns




RealEstate = pd.read_csv('realtor-data.zip.csv')
RealEstate=RealEstate.drop(columns=['status','bath','acre_lot','city','zip_code','prev_sold_date'])
RealEstate=RealEstate.dropna()
# print(RealEstate)
Q1 = RealEstate['price'].quantile(0.25)
print(Q1)
Q3 = RealEstate['price'].quantile(0.75)
print(Q3)
IQR = Q3 - Q1
print(IQR)

RealEstate = RealEstate[(RealEstate['price'] >= Q1 + 1.5*IQR) & (RealEstate['price'] <= Q3 + 1.5*IQR)]
RealEstate.info()
X = pd.get_dummies(RealEstate.drop(['price'], axis=1), columns=['state'], drop_first=True)
y=RealEstate[['price']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
train_data=X_train.join(y_train)
# train_data.hist()

# print(y)
# print(X)

# sns.boxplot(x=RealEstate['price'])
# plt.show()



# model = LinearRegression()
# model.fit(X_train, y_train)

# Estimate=model.predict(X_test)
# print(Estimate)
# print((model.score(X_test,y_test))*100,"%")

bedroom_price_distribution = RealEstate.groupby('house_size')['price'].mean()
bedroom_price_distribution.plot(kind='bar')
plt.title('Average Price for Each Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Average Price')
plt.show()
