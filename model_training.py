import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))
    return scaled_data, scaler

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

def train_model(df):
    scaled_data, scaler = preprocess_data(df)
    time_step = 60
    X, Y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, batch_size=1, epochs=1)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    train_rmse = np.sqrt(np.mean(((train_predict - Y_train) ** 2)))
    test_rmse = np.sqrt(np.mean(((test_predict - Y_test) ** 2)))

    print(f'Train RMSE: {train_rmse}')
    print(f'Test RMSE: {test_rmse}')

    model.save('model.h5')

    return model, scaler, test_predict, Y_test

if __name__ == "__main__":
    df = pd.read_csv('historical_data.csv', index_col='timestamp', parse_dates=True)
    model, scaler, test_predict, Y_test = train_model(df)
    np.save('test_predict.npy', test_predict)
    np.save('Y_test.npy', Y_test)
    np.save('scaler.npy', scaler)