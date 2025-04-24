import matplotlib.pyplot as plt
import pandas as pd

def visualize_predictions(stock_data, predictions, ticker):
    plt.figure(figsize=(12, 6))
    
    # Actual stock prices
    plt.plot(stock_data.index, stock_data['Close'], label='Actual Prices', color='blue')
    
    # Predicted trend
    future_dates = stock_data.index[:len(predictions)]
    plt.plot(future_dates, predictions, label='Predicted Trend', color='red', linestyle='--')
    
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()
