import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
# import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import datetime

def stockpred(ticker,day):
# Reading dataset
   start="2012-01-01"
   print(day)
   end = day+datetime.timedelta(days=1)
   stock_data=yf.download(ticker,start,end)
   stock_data["Date"]=stock_data.index
   std_2=stock_data.reset_index(drop=True)
   std_2=std_2.drop(["Open","High","Low","Adj Close","Volume"],axis=1)

# normalizing the close/last price column
   from sklearn.preprocessing import MinMaxScaler
   scaler=MinMaxScaler(feature_range=(0,1))
   df1=scaler.fit_transform(np.array(std_2["Close"]).reshape(-1,1))

##splitting dataset into train and test split
   training_size=int(len(df1)*0.7)
   test_size=len(df1)-training_size
   train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
# convert an array of values into a dataset matrix
   def create_dataset(dataset, time_step=1):
	   dataX, dataY = [], []
	   for i in range(len(dataset)-time_step-1):
		    a = dataset[i:(i+time_step), 0]   
		    dataX.append(a)
		    dataY.append(dataset[i + time_step, 0])
	   return np.array(dataX), np.array(dataY)
# reshape into X=t,t+1,t+2,t+3 and Y=t+4
   time_step = 15
   X_train, y_train = create_dataset(train_data, time_step)
   X_test, ytest = create_dataset(test_data, time_step)
# reshape input to be [samples, time steps, features] which is required for LSTM
   X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
   X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
### Create the Stacked LSTM model

   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense
   from tensorflow.keras.layers import LSTM

   model=Sequential()
   model.add(LSTM(50,return_sequences=True,input_shape=(15,1)))
   model.add(LSTM(50,return_sequences=True))
   model.add(LSTM(50))
   model.add(Dense(1))
   model.compile(loss='mean_squared_error',optimizer='adam')
   model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=50,batch_size=15,verbose=1)
   def date(day):
     k=std_2[std_2["Date"]==str(day)].index.values
     x=k[0]
     new_set=std_2.iloc[x-14:x+1,:]
     y= scaler.fit_transform(np.array(new_set[["Close"]]))
     return y
   x_input=date(day).reshape(1,-1)
   temp_input=list(x_input)
   temp_input=temp_input[0].tolist()

# demonstrate prediction for next 10 days
   from numpy import array

   lst_output=[]
   n_steps=15
   i=0
   while(i<7):
    
    if(len(temp_input)>15):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
      #   print(x_input)
        yhat = model.predict(x_input, verbose=0)
      #   print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
      #   print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
      #   print(yhat[0])
        temp_input.extend(yhat[0].tolist())
      #   print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
   return scaler.inverse_transform(lst_output)


