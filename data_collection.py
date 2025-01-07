import requests
import pandas as pd
import time

def fetch_binance_data(symbol, interval, start_time, end_time):
    url = f"https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1000
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

if __name__ == "__main__":
    symbol = 'BTCUSDT'
    interval = '1h'
    start_time = int(time.mktime(time.strptime('2020-01-01', '%Y-%m-%d')) * 1000)
    end_time = int(time.mktime(time.strptime('2024-12-31', '%Y-%m-%d')) * 1000)

    df = fetch_binance_data(symbol, interval, start_time, end_time)
    df.to_csv('historical_data.csv')
    print(df.head())