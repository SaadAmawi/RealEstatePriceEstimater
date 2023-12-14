import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
import seaborn as sns
from sklearn.preprocessing import StandardScaler



RealEstate = pd.read_csv('realtor-data.zip.csv')
RealEstate=RealEstate.drop(columns=['status','acre_lot','city','zip_code','prev_sold_date'])
RealEstate=RealEstate.dropna()
Q1 = RealEstate['price'].quantile(0.25)
Q3 = RealEstate['price'].quantile(0.75)
IQR = Q3 - Q1

RealEstate = RealEstate[(RealEstate['price'] >= Q1 - 1.5*IQR) & (RealEstate['price'] <= Q3 + 1.5*IQR)]
# RealEstate.info()

X=RealEstate.drop(columns=['price'])
y=RealEstate[['price']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
train_data=X_train.join(y_train)
# X,y,train_data

train_data['bed']=np.log(train_data['bed']+1)
train_data['bath']=np.log(train_data['bath']+1)
train_data['house_size']=np.log(train_data['house_size']+1)



train_data=train_data.join(pd.get_dummies(train_data.state).astype(int)).drop(['state'],axis=1)




scaler=StandardScaler()



X_trainer=train_data.drop(columns=['price','West Virginia'])
X_train_scaled=scaler.fit_transform(X_trainer)
y_trainer=train_data['price']
print(X_trainer)
model = LinearRegression()
model.fit(X_trainer,y_trainer)


X_test=X_test.join(pd.get_dummies(X_test.state).astype(int)).drop(['state','West Virginia'],axis=1)
test_data=X_test.join(y_test)
test_data['bed']=np.log(test_data['bed']+1)
test_data['bath']=np.log(test_data['bath']+1)
test_data['house_size']=np.log(test_data['house_size']+1)
print(X_test)

score=model.score(X_test,y_test)
print(score)