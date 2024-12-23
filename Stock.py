import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import yfinance as yf

# Configure Streamlit app
st.set_page_config(page_title="Stock Price Predictor")
st.title("Yearly Stock Predictor for NSE/BSE")

# Sidebar for user inputs
st.sidebar.header("Stock Selection")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE.NS, TCS.BO):", "RELIANCE.NS")
days_to_predict = st.sidebar.slider("Days to Predict:", min_value=30, max_value=365, value=365)

st.sidebar.header("Prediction Configuration")
train_size = st.sidebar.slider("Training Data Percentage:", min_value=60, max_value=90, value=80)

# Fetch historical stock data
@st.cache_data
def fetch_stock_data(symbol):
    try:
        data = yf.download(symbol, period="1y")  # Fetch 1 year of data
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# Predict stock prices
def predict_stock_prices(data, days):
    data["Day"] = np.arange(len(data))

    # Feature and target variables
    X = data[["Day"]]
    y = data["Close"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - train_size) / 100, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict future prices
    future_days = np.arange(len(data), len(data) + days).reshape(-1, 1)
    future_prices = model.predict(future_days)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return future_prices, mse, model

# Main app logic
if st.button("Predict Stock Prices"):
    stock_data = fetch_stock_data(stock_symbol)

    if stock_data is not None:
        future_prices, mse, model = predict_stock_prices(stock_data, days_to_predict)

        st.subheader("Prediction Results")
        st.write(f"Mean Squared Error: {mse:.2f}")

        future_dates = pd.date_range(stock_data["Date"].iloc[-1] + pd.Timedelta(days=1), periods=days_to_predict)
        prediction_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_prices})

        st.write(prediction_df)
    else:
        st.error("Unable to fetch stock data. Please check the symbol and try again.")
