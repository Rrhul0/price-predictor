import pandas as pd
import time
import numpy as np
from data_collection import fetch_binance_data
from model_training import train_model
from backtesting import backtest
import os

if __name__ == "__main__":
    csv_file = 'historical_data.csv'

    if not os.path.exists(csv_file):
        # Data Collection
        symbol = 'BTCUSDT'
        interval = '1h'
        start_time = int(time.mktime(time.strptime('2020-01-01', '%Y-%m-%d')) * 1000)
        end_time = int(time.mktime(time.strptime('2024-12-31', '%Y-%m-%d')) * 1000)
        df = fetch_binance_data(symbol, interval, start_time, end_time)
        df.to_csv(csv_file)


    # Model Training
    df = pd.read_csv(csv_file, index_col='timestamp', parse_dates=True)
    model, scaler, test_predict, Y_test = train_model(df)
    np.save('test_predict.npy', test_predict)
    np.save('Y_test.npy', Y_test)
    np.save('scaler.npy', scaler)

    # Backtesting
    results = backtest(test_predict, scaler.inverse_transform(Y_test.reshape(-1, 1)))
    print(f"Backtest Accuracy: {results['accuracy'] * 100:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Profit: ${results['profit'].item():.2f}")
    print(f"Profit Percentage: {results['profit_percentage'].item():.2f}%")
    print(f"Final Balance: ${results['final_balance'].item():.2f}")