import requests
import pandas as pd
from datetime import datetime, timedelta

from configs import (
    MARKET_URL,
)


def market_api_call(symbol: str, resolution: str, from_ts: int, to_ts: int):
    params = {
        "symbol": symbol,
        "resolution": resolution,
        "from": str(from_ts),
        "to": str(to_ts),
    }

    print(params)
    response = requests.get(MARKET_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Error:", response.status_code)
        return None


def json_to_df(json) -> pd.DataFrame:

    df = pd.DataFrame(json)
    df = df[['t', 'c', 'o', 'h', 'l', 'v']]
    df.columns = ['time', 'close', 'open', 'high', 'low', 'volume']
    df['ds'] = pd.to_datetime(df['time'], unit='s') 
    return df


def convert_period(period):

    return (period * 24)