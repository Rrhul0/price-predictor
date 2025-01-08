from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the trained model and scaler
model = load_model('model.h5')
scaler = np.load('scaler.npy', allow_pickle=True).item()

def preprocess_data(df):
    # Select only the 'close' column for scaling
    close_prices = df[['close']].values
    scaled_data = scaler.transform(close_prices)
    return scaled_data

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    df = pd.DataFrame(data)
    scaled_data = preprocess_data(df)
    X = scaled_data.reshape(1, scaled_data.shape[0], 1)
    prediction = model.predict(X)
    prediction = scaler.inverse_transform(prediction)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)