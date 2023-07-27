from datetime import datetime, timedelta
import pandas as pd

from configs import *
from utils import market_api_call, json_to_df, convert_period


def load_data(period):
    print("Start: Loading Data")
    now = datetime.now()
    data = market_api_call(
        symbol=SYMBOL,
        resolution=API_RESOLUTION,
        from_ts=str(round((now - timedelta(hours=convert_period(period))).timestamp())),
        to_ts=str(round(now.timestamp())),
    )
    df = json_to_df(data)
    print("End: Loading Data")
    return df
