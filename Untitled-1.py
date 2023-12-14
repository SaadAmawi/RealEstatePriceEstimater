#!/usr/bin/env python
# coding: utf-8

# In[273]:


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
RealEstate=RealEstate.drop(columns=['status','acre_lot','city','zip_code','prev_sold_date'])
# RealEstate=RealEstate.drop(columns=['status','city','prev_sold_date','state'])
RealEstate=RealEstate.dropna()
# print(RealEstate)
Q1 = RealEstate['price'].quantile(0.25)
print(Q1)
Q3 = RealEstate['price'].quantile(0.75)
print(Q3)
IQR = Q3 - Q1
print(IQR)

RealEstate = RealEstate[(RealEstate['price'] >= Q1 - 1.5*IQR) & (RealEstate['price'] <= Q3 + 1.5*IQR)]
RealEstate.info()
# X = pd.get_dummies(RealEstate.drop(['price'], axis=1), columns=['state'], drop_first=True)
X=RealEstate.drop(columns=['price'])
y=RealEstate[['price']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
train_data=X_train.join(y_train)
X,y,train_data

# print(y)
# print(X)

# sns.boxplot(x=RealEstate['price'])
# plt.show()



# model = LinearRegression()
# model.fit(X_train, y_train)

# Estimate=model.predict(X_test)
# print(Estimate)
# print((model.score(X_test,y_test))*100,"%")



# In[274]:


train_data.hist(figsize=(15,6))


# In[275]:


bedroom_price_distribution = RealEstate.groupby('bed')['price'].mean()
bedroom_price_distribution.plot(kind='bar')
plt.title('Average Price for Each Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Average Price')
plt.show()


# In[279]:


plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")


# In[280]:


train_data['bed']=np.log(train_data['bed']+1)
train_data['bath']=np.log(train_data['bath']+1)
train_data['house_size']=np.log(train_data['house_size']+1)


# In[281]:


train_data.hist()


# In[300]:


# train_data=train_data.join(pd.get_dummies(train_data.state).astype(int)).drop(['state'],axis=1)

train_data


# In[262]:


plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")
# train_data.corr()
# train_data.dtypes


# In[289]:


# training_data['bed2bathRatio']=train_data['bed']+train_data['bath']
# training_data['bed2bathRatio']
# training_data.drop(['bed2bathRatio'],axis=1)
train_data.info()
X_train
train_data


# In[341]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()



X_trainer=train_data.drop(columns=['price'])
X_train_scaled=scaler.fit_transform(X_trainer)
y_trainer=train_data['price']
model = LinearRegression()
model.fit(X_train_scaled,y_trainer)


# In[349]:


test_data=X_test.join(y_test)
test_data['bed']=np.log(test_data['bed']+1)
test_data['bath']=np.log(test_data['bath']+1)
test_data['house_size']=np.log(test_data['house_size']+1)
X_test
# test_data




# In[342]:


X_test_scaled = scaler.transform(X_test)


# In[350]:


print(model.score(X_test_scaled,y_test))


# In[352]:


from sklearn.ensemble import RandomForestRegressor

forest= RandomForestRegressor()
forest.fit(X_train_scaled,y_trainer)


# In[353]:


print(forest.score(X_test_scaled,y_test))


# In[355]:


from sklearn.model_selection import GridSearchCV

forest=RandomForestRegressor()

param_grid={
    "n_estimators":[3,10,30],
    "max_features":[2,4,6,8],
    
}

grid_search=GridSearchCV(forest,param_grid,cv=5,scoring="neg_mean_squared_error",return_train_score=True)
grid_search.fit(X_train_scaled, y_trainer)


# In[356]:


best_forest=grid_search.best_estimator_


# In[360]:


best_forest.score(X_test_scaled,y_test)

