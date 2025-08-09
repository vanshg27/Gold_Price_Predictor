import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose

from dateutil import parser

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

import matplotlib.dates as mdates

import warnings
warnings.filterwarnings('ignore')
import pickle
import os
def load_data():
    """
    Load dataset from a CSV file. """
    data = pd.read_csv(f"ML_Model\Gold (2).csv")
    return data
#print(load_data())
df = load_data()
#print(df.info())
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.dropna(inplace=True)
#print(df.head())
print(plt.style.available)

plt.style.use('seaborn-v0_8-dark-palette')
plt.figure(figsize=(14, 7))

plt.figure(figsize=(14, 7))
plt.plot(df['Close/Last'], label='Closing Price')
plt.title('Gold Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
#plt.show()
decomposition = seasonal_decompose(df['Close/Last'], model='multiplicative', period=365)
#decomposition.plot()
#plt.show()
# Split the data into train and test sets
train = df['Close/Last'][:int(0.8 * len(df))]
test = df['Close/Last'][int(0.8 * len(df)):]
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))
test.index = forecast.index  # Ensure test and forecast indices match
plt.figure(figsize=(14, 7))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(forecast, label='Forecast')
plt.legend()
plt.show()
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f'RMSE: {rmse}')
df['SMA_50'] = df['Close/Last'].rolling(window=50).mean()
df['SMA_200'] = df['Close/Last'].rolling(window=200).mean()
plt.figure(figsize=(14, 7))
plt.plot(df['Close/Last'], label='Close Price')
plt.plot(df['SMA_50'], label='50-Day SMA')
plt.plot(df['SMA_200'], label='200-Day SMA')
plt.title('Gold Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()