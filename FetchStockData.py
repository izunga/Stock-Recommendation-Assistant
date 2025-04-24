import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define the stock tickers and date range
tickers = ["AAPL", "SBUX", "NFLX", "AMZN", "TSLA"]  # Apple, Starbucks, Netflix, Amazon, Tesla
start_date = "2023-01-01"
end_date = "2023-12-31"

# Initialize a dictionary to store the normalized data for each stock
normalized_data_dict = {}

# Loop over each ticker to download and process the data
for ticker in tickers:
    # Download the stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Check and print the columns to ensure correct data
    print(f"Columns for {ticker}: {stock_data.columns}")
    
    # Ensure the necessary columns exist
    if all(col in stock_data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        # Normalize the columns using MinMaxScaler
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(stock_data[['Open', 'High', 'Low', 'Close', 'Volume']])

        # Add the normalized data to the DataFrame
        stock_data[['Normalized_Open', 'Normalized_High', 'Normalized_Low', 'Normalized_Close', 'Normalized_Volume']] = normalized_data
        
        # Save the normalized data to the dictionary
        normalized_data_dict[ticker] = stock_data
        
        # Print the first few rows to verify the normalized columns
        print(stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 
                          'Normalized_Open', 'Normalized_High', 'Normalized_Low', 
                          'Normalized_Close', 'Normalized_Volume']].head())
    else:
        print(f"Missing necessary columns for {ticker}. Skipping...")

# Save all stock data to CSV files
for ticker, data in normalized_data_dict.items():
    output_file = f"{ticker}_normalized_stock_data.csv"
    data.to_csv(output_file)
    print(f"Normalized data for {ticker} saved to {output_file}")
