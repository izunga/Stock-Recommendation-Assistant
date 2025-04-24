import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def prepare_data(ticker, start_date, end_date, feature_column='Close', time_step=60):
    # Download the stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data_scaled = scaler.fit_transform(stock_data[[feature_column]])

    # Create the sequences for LSTM
    X, y = [], []
    for i in range(time_step, len(stock_data_scaled)):
        X.append(stock_data_scaled[i-time_step:i, 0])
        y.append(stock_data_scaled[i, 0])

    X, y = np.array(X), np.array(y)

    # Reshape X to be [samples, time steps, features] which is required for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return stock_data, X, y, scaler
