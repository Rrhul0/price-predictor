import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def backtest(predictions, actual, threshold=0.5):
    correct = 0
    total_trades = 0
    profit = 0
    initial_balance = 1000
    balance = initial_balance
    position = 0

    for i in range(1, len(predictions)):
        if predictions[i] > predictions[i-1] and position == 0:
            position = 1
            buy_price = actual[i]
            total_trades += 1
        elif predictions[i] < predictions[i-1] and position == 1:
            position = 0
            sell_price = actual[i]
            profit += sell_price - buy_price
            balance += (sell_price - buy_price) * (balance / buy_price)
            total_trades += 1

        if (predictions[i] > predictions[i-1] and actual[i] > actual[i-1]) or \
           (predictions[i] < predictions[i-1] and actual[i] < actual[i-1]):
            correct += 1

    accuracy = correct / len(predictions)
    profit_percentage = (balance - initial_balance) / initial_balance * 100

    return {
        'accuracy': accuracy,
        'total_trades': total_trades,
        'profit': profit,
        'profit_percentage': profit_percentage,
        'final_balance': balance
    }

if __name__ == "__main__":
    test_predict = np.load('test_predict.npy')
    Y_test = np.load('Y_test.npy')
    scaler = np.load('scaler.npy', allow_pickle=True).item()

    results = backtest(test_predict, scaler.inverse_transform(Y_test.reshape(-1, 1)))
    print(f"Backtest Accuracy: {results['accuracy'] * 100:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Profit: ${results['profit']:.2f}")
    print(f"Profit Percentage: {results['profit_percentage']:.2f}%")
    print(f"Final Balance: ${results['final_balance']:.2f}")

    df = pd.read_csv('historical_data.csv', index_col='timestamp', parse_dates=True)
    plt.figure(figsize=(14, 5))
    plt.plot(df.index[-len(test_predict):], scaler.inverse_transform(Y_test.reshape(-1, 1)), color='blue', label='Actual Price')
    plt.plot(df.index[-len(test_predict):], test_predict, color='red', label='Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()