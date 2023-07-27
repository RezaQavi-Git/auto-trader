from data_loader import load_real_data
import numpy as np




def data_processing():
    print("data_processing start ...")
    stock_data = load_real_data()
    # Data preprocessing (For simplicity, we'll just handle missing values by dropping them)
    stock_data = add_technical_indicators(stock_data)
    stock_data = add_average_true_range(stock_data)
    stock_data = add_commodity_channel_index(stock_data)
    stock_data = add_rate_of_change(stock_data)

    stock_data = add_williams_percent_r(stock_data)
    stock_data = add_chaikin_oscillator(stock_data)

    stock_data = stock_data.dropna()

    print("data_processing end ...")
    return stock_data

# Feature engineering (You can create technical indicators or other relevant features here)
# Example of adding RSI and MACD as features
def add_technical_indicators(data):
    # Calculate Relative Strength Index (RSI)
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi

    # Calculate Moving Average Convergence Divergence (MACD)
    short_window = 12
    long_window = 26
    signal_window = 9
    ema_short = data['close'].ewm(span=short_window, min_periods=1, adjust=False).mean()
    ema_long = data['close'].ewm(span=long_window, min_periods=1, adjust=False).mean()
    data['MACD'] = ema_short - ema_long
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, min_periods=1, adjust=False).mean()
    return data


def add_average_true_range(data):
    data['H-L'] = data['high'] - data['low']
    data['H-PC'] = abs(data['high'] - data['close'].shift(1))
    data['L-PC'] = abs(data['low'] - data['close'].shift(1))
    data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    data['ATR'] = data['TR'].rolling(window=14).mean()
    data.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1, inplace=True)
    return data


def add_commodity_channel_index(data):
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    data['CCI'] = (typical_price - typical_price.rolling(window=20).mean()) / (0.015 * typical_price.rolling(window=20).std())
    return data


def add_rate_of_change(data):
    data['ROC'] = (data['close'] - data['close'].shift(12)) / data['close'].shift(12) * 100
    return data



def add_williams_percent_r(data):
    highest_high = data['high'].rolling(window=14).max()
    lowest_low = data['low'].rolling(window=14).min()
    data['Williams %R'] = (highest_high - data['close']) / (highest_high - lowest_low) * -100
    return data


def add_chaikin_oscillator(data):
    adl = (2 * data['close'] - data['low'] - data['high']) / (data['high'] - data['low']) * data['volume']
    adl = adl.fillna(0)
    data['Chaikin Oscillator'] = adl.rolling(window=3).mean() - adl.rolling(window=10).mean()
    return data




# # Based on the predictions from the model, create a trading strategy that generates buy/sell signals
# # For simplicity, let's assume a basic strategy based on moving averages
# def generate_signals(data):
#     data['Signal'] = 0
#     data.loc[data['close'] > data['MA_50'], 'Signal'] = 1  # Buy signal
#     data.loc[data['close'] < data['MA_50'], 'Signal'] = -1  # Sell signal
#     return data

# # Apply the strategy to the stock data
# stock_data = generate_signals(stock_data)


# print(stock_data[stock_data['Signal'] < 1])