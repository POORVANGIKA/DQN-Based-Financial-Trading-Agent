import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ta

def preprocess(input_csv='data/nifty100_10years.csv', output_file='data/nifty100_processed.feather'):
    print("Loading raw data...")

    # Read CSV with multi-level header (2 rows)
    df = pd.read_csv(input_csv, header=[0,1], index_col=0)

    # Convert index to datetime (Date)
    df.index = pd.to_datetime(df.index)

    # Flatten multi-index columns to simple names like 'Price_Close'
    df.columns = ['_'.join(col).strip() for col in df.columns.values]

    df.sort_index(inplace=True)

    # Select relevant columns
    features = ['Price_Open', 'Price_High', 'Price_Low', 'Price_Close', 'Price_Volume']
    data = df[features].copy()

    print("Calculating technical indicators...")
    data['rsi'] = ta.momentum.rsi(data['Price_Close'], window=14)
    data['sma_10'] = ta.trend.sma_indicator(data['Price_Close'], window=10)
    data['sma_50'] = ta.trend.sma_indicator(data['Price_Close'], window=50)

    data.fillna(method='bfill', inplace=True)

    print("Scaling data...")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    pd.DataFrame(data_scaled, columns=data.columns).to_feather(output_file)

    print(f"Preprocessing complete. Processed data saved to: {output_file}")

if __name__ == "__main__":
    preprocess()
