import dash
from dash import html
from dash import dcc
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras import Sequential
from keras.src.layers import Embedding, LSTM, Dense
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go

def prediction(stock_code, n_days):
    # Load data
    df = yf.download(stock_code, period='60d')
    df.reset_index(inplace=True)
    close_prices = df['Close'].values.reshape(-1, 1)
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    # Prepare sequences
    X, y = [], []
    for i in range(len(scaled_data) - n_days):
        X.append(scaled_data[i:i+n_days])
        y.append(scaled_data[i+n_days])
    X, y = np.array(X), np.array(y)
    
    # Define and train LSTM model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(units=50, return_sequences=False),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32)
    
    # Make predictions
    forecast = []
    for i in range(n_days):
        if i == 0:
            input_data = scaled_data[-n_days:].reshape(1, n_days, 1)
        else:
            input_data = np.append(input_data[:, 1:, :], [[scaled_data[-1]]], axis=1)
        prediction = model.predict(input_data)
        forecast.append(prediction[0][0])
        scaled_data = np.append(scaled_data, prediction, axis=0)
    
    # Inverse scaling
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    
    # Generate dates for forecast
    last_date = df['Date'].iloc[-1]
    prediction_dates = [(last_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, n_days+1)]
    
    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prediction_dates, y=forecast.flatten(), mode='lines+markers', name='Predicted Data'))
    fig.update_layout(title="Predicted Close Price of next " + str(n_days) + " days",
                      xaxis_title="Date",
                      yaxis_title="Closed Price")
    
    return fig

