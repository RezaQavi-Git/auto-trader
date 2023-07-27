from datetime import datetime, timedelta
import pandas as pd

from configs import *
from utils import market_api_call, json_to_df, convert_period


def load_real_data():
    print("Start: Loading realtime data")
    now = datetime.now()
    data = market_api_call(
        symbol=SYMBOL, 
        resolution=API_RESOLUTION, 
        from_ts=str(round((now - timedelta(hours=convert_period(SHORT_PERIOD))).timestamp())),
        to_ts=str(round(now.timestamp())),
    )
    df = json_to_df(data)
    print("End: Loading realtime data")

    return df
