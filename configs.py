import os
from datetime import datetime

MARKET_URL = 'https://api.nobitex.ir/market/udf/history'

SYMBOL = 'TRXUSDT'
API_RESOLUTION = '5'
TIME_FRAME = '5m'

CURRENT_DATE = datetime.now()
LONG_PERIOD = 720 # days
MID_PERIOD = 180 # days
SHORT_PERIOD = 1 # days


TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_FILE_PATH = './models/{name}.sav'
MODELS = ['RF', 'GB', 'SVM']
TRADE_RANGE = 500
SLEEP_TIME = 600 # second
MOCK_D_PATH = os.getcwd() + '/../mockD/'


ENV = 'PROD'
GEN_MOCK = False


