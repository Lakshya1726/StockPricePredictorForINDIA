import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import yfinance as yf

# Configure Streamlit app
st.set_page_config(page_title="Stock Price Predictor")
st.title("Daily Stock Predictor for NSE/BSE")

# Sidebar for user inputs
st.sidebar.header("Stock Selection")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE.NS, TCS.BO):", "RELIANCE.NS")
analysis_duration = st.sidebar.selectbox("Select Analysis Duration:", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
days_to_predict = st.sidebar.slider("Days to Predict:", min_value=1, max_value=365, value=30)

st.sidebar.header("Prediction Configuration")
train_size = st.sidebar.slider("Training Data Percentage:", min_value=60, max_value=90, value=80)
additional_feature = st.sidebar.selectbox("Additional Feature for Analysis:", ["Volume", "High", "Low"], index=0)

# Fetch historical stock data
@st.cache_data
def fetch_stock_data(symbol, period):
    try:
        data = yf.download(symbol, period=period)  # Fetch data for the selected period
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
    if additional_feature in data.columns:
        X[additional_feature] = data[additional_feature]
    y = data["Close"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - train_size) / 100, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict future prices
    future_days = np.arange(len(data), len(data) + days).reshape(-1, 1)
    future_prices = model.predict(future_days).flatten()  # Ensure 1-dimensional output

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return future_prices, mse, model

# Main app logic
if st.button("Predict Stock Prices"):
    stock_data = fetch_stock_data(stock_symbol, analysis_duration)

    if stock_data is not None:
        st.subheader(f"Historical Data for {stock_symbol} ({analysis_duration})")
        st.dataframe(stock_data.tail())

        st.subheader("Closing Price Trend")
        st.line_chart(stock_data["Close"])

        if additional_feature in stock_data.columns:
            st.subheader(f"{additional_feature} Trend")
            st.line_chart(stock_data[additional_feature])

        future_prices, mse, model = predict_stock_prices(stock_data, days_to_predict)

        st.subheader("Prediction Results")
        st.write(f"Mean Squared Error: {mse:.2f}")

        future_dates = pd.date_range(stock_data["Date"].iloc[-1] + pd.Timedelta(days=1), periods=days_to_predict)
        prediction_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_prices})

        st.write(prediction_df)
        st.line_chart(prediction_df.set_index("Date"))

    else:
        st.error("Unable to fetch stock data. Please check the symbol and try again.")
