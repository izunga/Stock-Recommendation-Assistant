import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Example: Ask user for financial data (Investment, Risk Tolerance, etc.)
def get_user_input():
    investment_amount = float(input("Enter your investment amount ($): "))
    risk_tolerance = float(input("Rate your risk tolerance (1-10): "))
    goal = input("Enter your financial goal (e.g., 'retirement', 'buying a house'): ")
    return investment_amount, risk_tolerance, goal

# Simulate market data trends (in a real scenario, you would fetch live data)
def generate_trend_data():
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    values = np.random.normal(loc=0.1, scale=1, size=100).cumsum()  # Simulated trend
    market_data = pd.DataFrame({"Date": dates, "Trend": values})
    return market_data

# Build a basic model to predict investment outcomes
def investment_model(investment, risk_tolerance):
    # Example of a very basic regression model
    market_data = generate_trend_data()
    X = np.array(range(len(market_data))).reshape(-1, 1)  # Time in days
    y = market_data["Trend"].values
    model = LinearRegression().fit(X, y)

    # Predict future market trend based on user investment and risk tolerance
    predicted_trend = model.predict([[len(market_data) + 30]])  # 30 days ahead prediction
    print(f"Predicted market trend in 30 days: {predicted_trend[0]:.2f}")

    # Estimate user’s portfolio value based on risk tolerance
    estimated_value = investment * (1 + (risk_tolerance / 100) * predicted_trend[0])
    print(f"Estimated portfolio value in 30 days: ${estimated_value:.2f}")

# Run the process
if __name__ == "__main__":
    investment, risk_tolerance, goal = get_user_input()
    investment_model(investment, risk_tolerance)
