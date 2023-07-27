
from prediction_model import prediction_model
import pandas as pd

def signal_gen():

    model, stock_data = prediction_model()
    last_row = stock_data.iloc[-1].drop(['close', 'ds'])  # Remove 'Close' and 'Signal' columns as they are not features

    X_new = pd.DataFrame([last_row])
    predicted_features = model.predict(X_new)
    short_window = 20
    predicted_close_price = predicted_features[0]  # Assuming the predicted feature corresponds to 'Close' price
    historical_short_ma = stock_data['close'].rolling(window=short_window).mean().iloc[-1]
    if predicted_close_price > historical_short_ma:
        signal = 1  # Buy signal
    else:
        signal = -1  # Sell signal (or no action, depending on the strategy)

    print("Predicted Close Price:", predicted_close_price)
    print("Short-Term Moving Average (Historical):", historical_short_ma)
    print("Signal:", signal)  # 1 for Buy, -1 for Sell (or no action)

signal_gen()
