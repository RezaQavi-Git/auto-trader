from prediction_models import prediction_models
import pandas as pd


def gen_moving_average_crossover_signal(
    stock_data, predicted_close_price, short_window=20, long_window=50
):
    signal = 0
    short_ma = stock_data["close"].rolling(window=short_window).mean()
    long_ma = stock_data["close"].rolling(window=long_window).mean()

    if predicted_close_price > short_ma.iloc[-1]:
        signal = 1
    elif predicted_close_price < long_ma.iloc[-1]:
        signal = -1

    return signal


def gen_bollinger_bands_signal(stock_data, predicted_close_price):
    signal = 0

    # Calculate Bollinger Bands
    ma = stock_data["close"].rolling(window=20).mean()
    std = stock_data["close"].rolling(window=20).std()
    upper_band = ma.iloc[-1] + 2 * std.iloc[-1]
    lower_band = ma.iloc[-1] - 2 * std.iloc[-1]

    if predicted_close_price < lower_band:
        print("+")
        signal = 1
    elif predicted_close_price > upper_band:
        print("-")
        signal = -1

    return signal


def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1]


def gen_rsi_signal(stock_data, lower_threshold=30, upper_threshold=70):
    signal = 0
    rsi = calculate_rsi(stock_data["close"])

    if rsi > lower_threshold:
        signal = 1
    elif rsi < upper_threshold:
        signal = -1

    return signal


def generate_signal(model, model_name, stock_data, time):
    last_row = stock_data.iloc[-1]
    X_new = pd.DataFrame([last_row.drop(["close", "ds"])])
    predicted_features = model.predict(X_new)
    predicted_close_price = predicted_features[0]
    # 1. Moving Average Crossover Strategy
    moving_average_signal = gen_moving_average_crossover_signal(
        stock_data, predicted_close_price
    )
    print("Moving Average Crossover Signal:", moving_average_signal)

    # 2. Bollinger Bands Strategy
    bollinger_bands_signal = gen_bollinger_bands_signal(
        stock_data, predicted_close_price
    )
    print("Bollinger Bands Signal:", bollinger_bands_signal)

    # 3. RSI Strategy
    rsi_signal = gen_rsi_signal(stock_data)
    print("RSI Signal:", rsi_signal)

    signal_logger(
        time=time,
        model=model_name,
        close=last_row["close"],
        prediction=predicted_close_price,
        s_ma=moving_average_signal,
        s_bb=bollinger_bands_signal,
        s_rsi=rsi_signal,
    )


def signal_logger(time, model, close, prediction, s_ma, s_bb, s_rsi):
    line_pattern = "{ds},{model},{close},{prediction},{s_ma},{s_bb},{s_rsi}\n"
    with open("{0}.txt".format("signals"), "a") as out:
        out.write(
            line_pattern.format(
                ds=str(time),
                model=model,
                close=close,
                prediction=prediction,
                s_ma=s_ma,
                s_bb=s_bb,
                s_rsi=s_rsi,
            )
        )
        out.close()
