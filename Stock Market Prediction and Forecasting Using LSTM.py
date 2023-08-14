#!/usr/bin/env python
# coding: utf-8

# # Stock Market Prediction And Forecasting Using LSTM

# In[170]:


import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[171]:


df=pd.read_csv("https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv")
df=df.iloc[::-1]
df.head()


# In[172]:


df.tail()


# In[173]:


df.shape


# In[174]:


df.info()


# In[175]:


df.columns


# In[176]:


df.describe()


# In[177]:


#Data Preprocessing
df.isnull().sum()


# In[178]:


duplicates=df.duplicated()
duplicates.value_counts()


# In[179]:


plt.figure(figsize=(6,6))
sns.heatmap(df.corr(),annot=True)


# In[180]:


df_high=df.reset_index()['High']
plt.plot(df_high)


# In[181]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df_high =scaler.fit_transform(np.array(df_high).reshape(-1,1))


# In[182]:


df_high.shape


# In[183]:


df_high


# In[184]:


#split the data into train and test split
train_size=int(len(df_high)*0.75)
test_size=len(df_high)-train_size
train_data,test_data=df_high[0:train_size,:],df_high[train_size:len(df_high),:1]


# In[185]:


train_size,test_size


# In[186]:


#convert an array of values into a dataset matrix
def create_dataset (dataset,time_step=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step-1])
        return np.array(dataX),np.array(dataY)


# In[187]:


time_step=100
xtrain,ytrain=create_dataset(train_data,time_step)
xtest,ytest=create_dataset(test_data,time_step)


# In[188]:


xtrain=xtrain.reshape(xtrain.shape[0],xtrain.shape[1],1)
xtest=xtest.reshape(xtest.shape[0],xtest.shape[1],1)


# In[189]:


print(xtrain.shape)
print(ytrain.shape)


# In[190]:


print(xtest.shape)
print(ytest.shape)


# In[191]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[192]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[193]:


model.summary()


# In[194]:


model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=100,batch_size=64,verbose=1)


# In[195]:


train_predict=model.predict(xtrain)
test_predict=model.predict(xtest)


# In[196]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[197]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(ytrain,train_predict))


# In[198]:


math.sqrt(mean_squared_error(ytest,test_predict))


# In[199]:


#plotting
#shift train prediction for plotting
look_back=100
trainPredictPlot=np.empty_like(df_high)
trainPredictPlot[:,:]=np.nan
trainPredictPlot[look_back:len(train_predict)+look_back,:]=train_predict

#shift test prediction for plotting
testPredictPlot=np.empty_like(df_high)
testPredictPlot[:,:]=np.nan
testPredictPlot[look_back:len(train_predict)+look_back,:]=test_predict


# In[200]:


plt.plot(scaler.inverse_transform(df_high))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[201]:


len(test_data),xtest.shape


# In[202]:


x_input=test_data[409:].reshape(1,-1)
x_input.shape


# In[203]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()

temp_input


# # predict for next 28 days

# In[204]:


temp_input = list(x_input)
temp_input = temp_input[0].tolist()

lst_output=[]
n_steps=100
nextNumberOfDays = 28
i=0

while(i<nextNumberOfDays):
    
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    
print(lst_output)


# In[205]:


day_new=np.arange(1,101)
day_pred=np.arange(101,129)


# In[206]:


day_new.shape


# In[207]:


day_pred.shape


# In[208]:


len(df_high)


# In[209]:


plt.plot(day_new, scaler.inverse_transform(df_high[1935:]))
plt.plot(day_pred, scaler.inverse_transform(lst_output))


# In[210]:


data_new=df_high.tolist()
data_new.extend(lst_output)
plt.plot(data_new[2000:])


# In[211]:


data_new=scaler.inverse_transform(data_new).tolist()


# In[212]:


plt.plot(data_new)

