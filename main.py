import yfinance as yf
import pandas as pd
import numpy as np
from model import build_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Predefined list of stocks
stocks = {'AAPL': 'Apple', 'SBUX': 'Starbucks', 'NFLX': 'Netflix', 'AMZN': 'Amazon', 'TSLA': 'Tesla'}

# Function to collect user data
def collect_user_data():
    print("Welcome to the Stock Recommendation Chatbot!")
    investment = float(input("Enter your investment amount ($): "))
    risk_tolerance = float(input("Rate your risk tolerance (1-10): "))
    goal = input("Enter your financial goal (e.g., retirement, buying a house): ")
    print("\nAvailable stocks:")
    for ticker, name in stocks.items():
        print(f"{ticker}: {name}")
    selected_stock = input("\nEnter the stock ticker you are interested in: ").upper()
    while selected_stock not in stocks:
        print("Invalid ticker. Please choose from the available stocks.")
        selected_stock = input("Enter the stock ticker you are interested in: ").upper()
    return investment, risk_tolerance, goal, selected_stock

# Fetch and preprocess stock data
def fetch_and_preprocess_data(ticker):
    print(f"\nFetching data for {ticker} ({stocks[ticker]})...")
    stock_data = yf.download(ticker, start="2023-01-01", end="2023-12-31")
    scaler = MinMaxScaler()
    # Normalize the stock data (only columns Open, High, Low, Close, and Volume)
    normalized_data = scaler.fit_transform(stock_data[['Open', 'High', 'Low', 'Close', 'Volume']])
    normalized_columns = ['Normalized_Open', 'Normalized_High', 'Normalized_Low', 'Normalized_Close', 'Normalized_Volume']
    stock_data[normalized_columns] = normalized_data
    return stock_data, scaler

# Train and predict stock trends
def train_and_predict(stock_data):
    model = build_model()
    X = stock_data[['Normalized_Open', 'Normalized_High', 'Normalized_Low', 'Normalized_Close', 'Normalized_Volume']].values
    y = stock_data['Normalized_Close'].shift(-1).dropna().values
    X = X[:-1]  # Adjust to match size of y
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    predictions = model.predict(X)
    return predictions

# Provide recommendations
def provide_recommendations(investment, risk_tolerance, predictions, actual_prices, ticker):
    # De-normalize predictions and actual prices
    predicted_growth = predictions[-1] - actual_prices[-1]
    expected_roi = investment * (1 + predicted_growth / actual_prices[-1])

    print(f"\n--- Personalized Recommendation for {ticker} ---")
    if predicted_growth > 0:
        print(f"Stock {ticker} is expected to rise by {predicted_growth:.2f}.")
        if risk_tolerance >= 7:
            print(f"Recommendation: Buy {ticker}. This aligns with your high risk tolerance.")
        else:
            print(f"Recommendation: Hold {ticker} for now. Consider increasing investment cautiously.")
    else:
        print(f"Stock {ticker} shows a potential decline of {abs(predicted_growth):.2f}.")
        if risk_tolerance <= 3:
            print(f"Recommendation: Avoid investing in {ticker} given your low risk tolerance.")
        else:
            print(f"Recommendation: Hold {ticker} if you already own it, but avoid new investments.")

    print(f"Expected ROI based on prediction: ${expected_roi:.2f}\n")

# Plotting function
def plot_predictions(actual_prices, predictions, ticker):
    plt.figure(figsize=(10,6))
    plt.plot(actual_prices, label='Actual Prices', color='blue')
    plt.plot(predictions, label='Predicted Prices', color='red', linestyle='--')
    plt.title(f"Stock Price Prediction vs Actual for {ticker}")
    plt.xlabel('Days')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.show()

# Main chatbot function
def chatbot():
    # Collect user data
    investment, risk_tolerance, goal, selected_stock = collect_user_data()
    
    # Fetch and preprocess data for the selected stock
    stock_data, scaler = fetch_and_preprocess_data(selected_stock)

    # Train and predict for the selected stock
    predictions = train_and_predict(stock_data)

    # De-normalize predictions and actual prices for the selected stock
    normalized_values = stock_data[['Normalized_Open', 'Normalized_High', 'Normalized_Low', 'Normalized_Close', 'Normalized_Volume']].values
    denormalized_values = scaler.inverse_transform(normalized_values)

    # Extract actual close prices
    actual_prices = denormalized_values[:, 3]  # 'Normalized_Close' corresponds to index 3

    # Rescale predictions for proper inverse transform and only transform Close column
    predictions = predictions.reshape(-1, 1)  # Reshape predictions for proper inverse transform
    predictions = scaler.inverse_transform(np.hstack([np.zeros((predictions.shape[0], 4)), predictions]))[:, 4]  # Only transform Close column

    # Dynamically scale predictions based on the stock
    if selected_stock == "TSLA" or selected_stock == 'AMZN':
        predictions = predictions / 1000000
        
    elif selected_stock == "SBUX":
        predictions = predictions / 500000
        
    else:
        predictions = predictions / 30000
        


    # Provide personalized recommendations
    provide_recommendations(investment, risk_tolerance, predictions, actual_prices, selected_stock)

    # Plot the actual vs predicted prices
    plot_predictions(actual_prices, predictions, selected_stock)

if __name__ == "__main__":
    chatbot()








