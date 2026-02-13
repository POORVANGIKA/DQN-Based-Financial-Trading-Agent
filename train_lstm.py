import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_sequences(data, seq_length=50):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length][3])  # Close price index
    return np.array(xs), np.array(ys)

def train_lstm(processed_data_file='data/nifty100_processed.feather'):
    print("Loading processed data...")
    data = pd.read_feather(processed_data_file).values

    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    seq_length = 50
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    print(f"Training LSTM model on {len(X_train)} sequences...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, data.shape[1])),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=2)

    print("Predicting on test set...")
    preds = model.predict(X_test).flatten()

    position = 0
    cash = 0
    for i in range(len(preds) - 1):
        # Buy signal
        if preds[i+1] > preds[i] and position == 0:
            position = 1
            buy_price = test_data[i + seq_length][3]
        # Sell signal
        elif preds[i+1] < preds[i] and position == 1:
            position = 0
            sell_price = test_data[i + seq_length][3]
            cash += sell_price - buy_price

    print(f"Total profit from LSTM trading strategy: {cash:.2f}")

if __name__ == "__main__":
    train_lstm()
