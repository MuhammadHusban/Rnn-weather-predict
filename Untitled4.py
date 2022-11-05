#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from time import time
import random
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('D:\\weatherHistory.csv')


# In[ ]:


df.shape


# In[18]:


df.head()


# In[19]:


df.info()


# In[20]:


df=df.loc[1:50000]


# In[30]:


training_set=df.iloc[:,3:4].values


# In[31]:


#Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)


# In[32]:


x_train = []
y_train = []
n_future = 4 # next 4 days temperature forecast
n_past = 30 # Past 30 days 
for i in range(0,len(training_set_scaled)-n_past-n_future+1):
    x_train.append(training_set_scaled[i : i + n_past , 0])     
    y_train.append(training_set_scaled[i + n_past : i + n_past + n_future , 0 ])
x_train , y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0] , x_train.shape[1], 1) )


# In[ ]:


from keras.models import Sequential
from tensorflow.keras.layers import Bidirectional
from keras.layers import LSTM,Dense ,Dropout
# Fitting RNN to training set using Keras Callbacks. Read Keras callbacks docs for more info.

regressor = Sequential()
regressor.add(Bidirectional(LSTM(units=30, return_sequences=True, input_shape = (x_train.shape[1],1) ) ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 30 , return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 30 , return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 30))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = n_future,activation='linear'))
#I have used Adam optimizer because it is computationally efficient.
regressor.compile(optimizer='adam', loss='mean_squared_error',metrics=['acc'])
regressor.fit(x_train, y_train, epochs=30,batch_size=32 )


# In[ ]:


# read test dataset
testdataset = pd.read_csv('data (12).csv')
#get only the temperature column
testdataset = testdataset.iloc[:30,3:4].values
real_temperature = pd.read_csv('data (12).csv')
real_temperature = real_temperature.iloc[30:,3:4].values
testing = sc.transform(testdataset)
testing = np.array(testing)
testing = np.reshape(testing,(testing.shape[1],testing.shape[0],1))


# In[ ]:


predicted_temperature = regressor.predict(testing)
predicted_temperature = sc.inverse_transform(predicted_temperature)
predicted_temperature = np.reshape(predicted_temperature,(predicted_temperature.shape[1],predicted_temperature.shape[0]))

