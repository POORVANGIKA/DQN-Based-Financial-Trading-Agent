import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def train_arima(input_csv='data/nifty100_10years.csv'):
    print("Loading raw data for ARIMA...")
    df = pd.read_csv(input_csv)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    close_prices = df['Close'].values

    train_size = int(0.8 * len(close_prices))
    train, test = close_prices[:train_size], close_prices[train_size:]

    print("Fitting ARIMA model (order=(5,1,0))...")
    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()

    print("Forecasting on test data...")
    predictions = model_fit.forecast(steps=len(test))

    position = 0
    cash = 0
    for i in range(len(predictions) - 1):
        # Buy if next predicted price > current predicted price and no position
        if predictions[i+1] > predictions[i] and position == 0:
            position = 1
            buy_price = test[i]
        # Sell if next predicted price < current and holding position
        elif predictions[i+1] < predictions[i] and position == 1:
            position = 0
            cash += test[i] - buy_price

    print(f"Total profit from ARIMA trading strategy: {cash:.2f}")

if __name__ == "__main__":
    train_arima()

